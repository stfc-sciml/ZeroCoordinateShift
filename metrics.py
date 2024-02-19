"""
Metrics stat
"""
import argparse
import json

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metrics',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('directory')
    args = parser.parse_args()
    with open(args.directory + '/outputs/config.json') as fs:
        config = json.load(fs)
    seeds = config['seeds']
    methods = config['methods']
    metrics = ['data_error', 'pde_error',
               'peak_mem', 'graph_mem',
               'time_total', 'time_input', 'time_forward',
               'time_loss', 'time_backprop', 'time_pde']
    mean_std_dict = {method: {
        metric: {'mean': 0., 'std': 0.} for metric in metrics + ['time_bc']
    } for method in methods}

    for method in methods:
        # read
        metric_dict = {metric: [] for metric in metrics}
        for seed in seeds:
            fname = f'{args.directory}/outputs/{method}_{seed}.txt'
            lines = np.genfromtxt(fname, str, comments='==', delimiter=':',
                                  skip_header=config['line_argv'])
            for i, metric in enumerate(metrics):
                metric_dict[metric].append(float(lines[i][1].replace('MB', '')))
        for i, metric in enumerate(metrics):
            metric_dict[metric] = np.array(metric_dict[metric])
        # bc
        metric_dict['time_bc'] = metric_dict['time_loss'] - metric_dict['time_pde']
        # mean, std
        for i, metric in enumerate(metrics + ['time_bc']):
            mean_std_dict[method][metric]['mean'] = metric_dict[metric].mean()
            mean_std_dict[method][metric]['std'] = metric_dict[metric].std()

    for method in methods:
        print(method)
        for i, metric in enumerate(metrics + ['time_bc']):
            print(f"  {metric} {mean_std_dict[method][metric]['mean']} ± "
                  f"{mean_std_dict[method][metric]['std']}")

    # for latex table in paper
    epochs = {
        'diffusion_reaction': 10000,
        'burgers': 1000,
        'KL_plates': 200,
        'stokes': 100
    }
    pde = args.directory.replace('/', '')
    methods_latex = {
        '\\texttt{FuncLoop}': 'aligned_baseline',
        '\\texttt{DataVect}': 'unaligned_baseline',
        '\\texttt{ZCS} (ours)': 'aligned_ZCS',
    }
    print('Latex table contents')
    for method_show, method in methods_latex.items():
        try:
            print(f"& {method_show} & "
                  f"{mean_std_dict[method]['graph_mem']['mean'] / 1000:.2f} & "
                  f"{mean_std_dict[method]['peak_mem']['mean'] / 1000:.2f} & "
                  f"{round(mean_std_dict[method]['time_input']['mean'] / epochs[pde] * 1000)} & "
                  f"{round(mean_std_dict[method]['time_forward']['mean'] / epochs[pde] * 1000)} & "
                  f"{round(mean_std_dict[method]['time_loss']['mean'] / epochs[pde] * 1000)} & "
                  f"{round(mean_std_dict[method]['time_backprop']['mean'] / epochs[pde] * 1000)} & "
                  f"{round(mean_std_dict[method]['time_total']['mean'] / epochs[pde] * 1000)} & "
                  f"{mean_std_dict[method]['data_error']['mean'] * 100:.1f}"
                  f"$\pm$"  # noqa
                  f"{mean_std_dict[method]['data_error']['std'] * 100:.1f}\% \\\\")
        except KeyError:
            pass
