import os
import json
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'new_order_1')
METRICS = ['predict_exact_match', 'predict_rouge1', 'predict_rougeL']
TASK_ORDER = ['dbpedia', 'amazon', 'yahoo', 'agnews']


def find_algorithm_dirs(base_dir):
    algs = []
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        tasks = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d[0].isdigit()]
        if len(tasks) == 4:
            algs.append((name, path))
        else:
            for sub in os.listdir(path):
                sub_path = os.path.join(path, sub)
                if not os.path.isdir(sub_path):
                    continue
                sub_tasks = [d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d)) and d[0].isdigit()]
                if len(sub_tasks) == 4:
                    algs.append((f"{name}_{sub}", sub_path))
    return algs


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_matrix(path, matrix):
    with open(path, 'w') as f:
        for row in matrix:
            line = ' '.join(f"{v:.4f}" if v is not None else '0' for v in row)
            f.write(line + '\n')


def main():
    alg_dirs = find_algorithm_dirs(BASE_DIR)
    final_values = {m: {} for m in METRICS}
    per_task_values = {m: {} for m in METRICS}

    for alg_name, alg_path in alg_dirs:
        task_dirs = sorted(
            [d for d in os.listdir(alg_path) if os.path.isdir(os.path.join(alg_path, d)) and d[0].isdigit()],
            key=lambda x: int(x.split('-')[0])
        )
        matrices = {m: [[0 for _ in range(len(TASK_ORDER))] for _ in range(len(TASK_ORDER))] for m in METRICS}

        for i, task_dir in enumerate(task_dirs):
            res = load_json(os.path.join(alg_path, task_dir, 'predict_results.json'))
            for j, t in enumerate(task_dirs[:i+1]):
                dataset = t.split('-', 1)[1]
                for m in METRICS:
                    matrices[m][i][j] = res.get(f"{m}_for_{dataset}", 0)

        # write txt files
        for m in METRICS:
            out_file = os.path.join(alg_path, f"{m}.txt")
            save_matrix(out_file, matrices[m])

        # collect metric values for each training step
        step_metrics = {m: [] for m in METRICS}
        for t in task_dirs:
            res = load_json(os.path.join(alg_path, t, 'predict_results.json'))
            for m in METRICS:
                step_metrics[m].append(res.get(m, 0))

        # collect final metric values from the last task (4-agnews)
        final_res = load_json(os.path.join(alg_path, '4-agnews', 'predict_results.json'))
        for m in METRICS:
            final_values[m][alg_name] = final_res.get(m, 0)
            per_task_values[m][alg_name] = step_metrics[m]

    # plot results for each metric
    plots_dir = os.path.join(os.path.dirname(__file__), 'order1_plots')
    os.makedirs(plots_dir, exist_ok=True)
    x = range(1, len(TASK_ORDER) + 1)
    for m in METRICS:
        plt.figure()
        for alg, values in per_task_values[m].items():
            plt.plot(x, values, marker='o', label=alg)
        plt.xticks(x, TASK_ORDER)
        plt.xlabel('Tasks')
        plt.ylabel(m)
        plt.title(f'{m} across tasks')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{m}.png'))
        plt.close()

    # print final aggregated values
    for m in METRICS:
        print(f"\nFinal {m}:")
        for alg, val in final_values[m].items():
            print(f"{alg}: {val:.4f}")


if __name__ == '__main__':
    main()