# run_real_server.py
import argparse
import socket
import torch
import logging
import copy
from fed_socket import send_data, recv_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("FedServer")


def fed_avg(updates):
    """
    带样本量加权的 FedAvg 聚合算法
    updates: list of dict, 每个 dict 包含 {'weights': state_dict, 'num_samples': int}
    """
    num_clients = len(updates)
    if num_clients == 0:
        return None

    # 1. 计算总样本量
    total_samples = sum(u['num_samples'] for u in updates)

    # 取出第一个客户端的权重作为模板（用于获取 keys 和 tensor 形状）
    first_weights = updates[0]['weights']
    weighted_weights = copy.deepcopy(first_weights)

    # 初始化为 0
    for k in weighted_weights.keys():
        weighted_weights[k] = torch.zeros_like(weighted_weights[k], dtype=torch.float32)

    # 2. 加权累加
    for update in updates:
        n_k = update['num_samples']
        w_k = update['weights']

        # 计算该客户端的权重系数 (n_k / total)
        # 注意：这里我们直接乘进去，或者也可以先 sum(w_k * n_k) 最后除 total
        coefficient = n_k / total_samples

        for k in weighted_weights.keys():
            # w_global += w_local * (n_local / n_total)
            weighted_weights[k] += w_k[k] * coefficient

    logger.info(f"Aggregated {num_clients} clients. Total samples: {total_samples}")
    return weighted_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--num_clients", type=int, default=4)  # 对应你的4个集群
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    # 建立 Socket Server
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('0.0.0.0', args.port))
    server_sock.listen(args.num_clients)

    logger.info(f"Server started on port {args.port}, waiting for {args.num_clients} clients...")

    clients = []
    while len(clients) < args.num_clients:
        client_sock, addr = server_sock.accept()
        logger.info(f"Client connected: {addr}")
        clients.append(client_sock)

    global_weights = None

    for r in range(args.rounds):
        logger.info(f"=== Global Round {r + 1}/{args.rounds} ===")

        # 1. 广播全局模型
        logger.info("Broadcasting global model...")
        broadcast_payload = {
            'weights': global_weights,  # 第一轮为 None，客户端会自己初始化
            'round': r
        }
        for c in clients:
            send_data(c, broadcast_payload)

        # 2. 接收更新
        updates = []
        logger.info("Waiting for client updates...")
        for i, c in enumerate(clients):
            data = recv_data(c)
            updates.append(data)
            logger.info(f"Received update from client index {i}")

        # 3. 聚合
        logger.info("Aggregating weights...")
        global_weights = fed_avg(updates)

    # 结束信号
    logger.info("Training finished. Sending shutdown signal.")
    for c in clients:
        send_data(c, {'stop': True})
        c.close()
    server_sock.close()


if __name__ == "__main__":
    main()