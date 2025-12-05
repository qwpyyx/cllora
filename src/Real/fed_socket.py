# fed_socket.py
import socket
import struct
import pickle
import logging
import torch

logger = logging.getLogger("FedSocket")


def send_data(sock, data):
    """序列化并发送数据（先发长度，再发内容）"""
    # 将 Tensor 转为 CPU 以减小序列化体积并防止设备不兼容
    if isinstance(data, dict) and 'weights' in data:
        data['weights'] = {k: v.cpu() for k, v in data['weights'].items()}

    serialized_data = pickle.dumps(data)
    # 发送 4字节的大端整数表示长度
    sock.sendall(struct.pack('>I', len(serialized_data)))
    sock.sendall(serialized_data)


def recv_data(sock):
    """接收并反序列化数据"""
    # 先读 4字节长度
    raw_msglen = _recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # 再读内容
    data = _recvall(sock, msglen)
    return pickle.loads(data)


def _recvall(sock, n):
    """辅助函数：确保接收 n 个字节"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data