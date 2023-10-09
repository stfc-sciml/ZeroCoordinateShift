"""
Scaling analysis
"""

import matplotlib.pyplot as plt
import numpy as np

# set plot env
plt.style.use(['seaborn-paper'])
plt.rcParams.update({
    "font.family": "Times"
})

# set TeX
plt.rcParams["text.usetex"] = True
try:
    plt.text(0, 0, '$x$')
    plt.close()
except:  # noqa
    # if latex is not installed, disable
    plt.rcParams["text.usetex"] = False

# set size
SIZE = 10.5
plt.rc('font', size=SIZE)
plt.rc('axes', titlesize=SIZE)
plt.rc('axes', labelsize=SIZE)
plt.rc('xtick', labelsize=SIZE - 1)
plt.rc('ytick', labelsize=SIZE - 1)
plt.rc('legend', fontsize=SIZE)
plt.rc('figure', titlesize=SIZE)


def read_single(method, n_f, n_p, n_o):
    mem, time = 0., 0.
    for repeat in repeats:
        run_name = f'{method}_{n_f}_{n_p}_{n_o}_{repeat}'
        try:
            data = np.loadtxt(f'outputs/{run_name}.txt')
        except FileNotFoundError:
            # failed for GPU-memory limit
            continue
        mem += data[1]
        time += data[3]
    mem /= len(repeats)
    time /= len(repeats)
    if mem == 0.:
        mem = np.nan
    if time == 0.:
        time = np.nan
    return mem, time


def read_all(varying):
    mems_all, times_all = {}, {}
    for method in methods:
        mems, times = [], []
        if varying == 'n_functions':
            for n_f in n_functions_list:
                mem, time = read_single(method, n_f, n_points_fix, max_order_fix)
                mems.append(mem)
                times.append(time)
        elif varying == 'n_points':
            for n_p in n_points_list:
                mem, time = read_single(method, n_functions_fix, n_p, max_order_fix)
                mems.append(mem)
                times.append(time)
        else:
            for n_o in max_order_list:
                mem, time = read_single(method, n_functions_fix, n_points_fix, n_o)
                mems.append(mem)
                times.append(time)
        mems_all[method] = mems
        times_all[method] = times
    return mems_all, times_all


def plot(varying, ax_time, ax_mem, xl_time=False, yl_time=False, xl_mem=False, yl_mem=False):
    mems_all, times_all = read_all(varying)
    if varying == 'n_functions':
        x_values = np.array(n_functions_list)
        x_ticks = 10 ** np.log2(x_values / 10)
        x_log = True
        x_label = 'Number of functions, $M$'
    elif varying == 'n_points':
        x_values = np.array(n_points_list) // 100
        x_ticks = 10 ** np.log2(x_values / 50)
        x_log = True
        x_label = 'Number of points, $N$ (hundred)'
    else:
        x_values = max_order_list
        x_ticks = max_order_list
        x_log = False
        x_label = 'Maximum order of PDE, $P$'
    # time
    for method, method_show, marker, color in zip(methods, methods_show,
                                                  ['o', 's', 'D'],
                                                  ['r', 'b', 'g']):
        ax_time.plot(x_ticks, np.array(times_all[method]) * 10,
                     label=method_show, marker=marker, color=color, markersize=5)
        if x_log:
            ax_time.set_xscale('log')
        ax_time.set_yscale('log')
        if xl_time:
            ax_time.set_xlabel(x_label)
        ax_time.set_xticks(x_ticks, x_values)
        ax_time.tick_params(axis='y', which='major', pad=2)
        if yl_time:
            ax_time.set_ylabel('Time per 1000 batches (sec)')
    # memory
    for method, method_show, marker, color in zip(methods, methods_show,
                                                  ['o', 's', 'D'],
                                                  ['r', 'b', 'g']):
        ax_mem.plot(x_ticks, np.array(mems_all[method]) / 1e9,
                    label=method_show, marker=marker, color=color, markersize=5)
        if x_log:
            ax_mem.set_xscale('log')
        ax_mem.set_yscale('log')
        if xl_mem:
            ax_mem.set_xlabel(x_label)
        ax_mem.set_xticks(x_ticks, x_values)
        ax_mem.tick_params(axis='y', which='major', pad=2)
        if yl_mem:
            ax_mem.set_ylabel('Peak GPU Memory (GB)')
    ax_mem.axhline(80, xmin=.75, c='k', ls='--')
    ax_mem.set_ylim(.02, 250)
    ax_mem.text(0.97, 0.98, '80 GB', ha='right', va='top', transform = ax_mem.transAxes)


if __name__ == "__main__":
    # computing parameters
    methods = ['aligned_baseline', 'unaligned_baseline', 'aligned_ZCS']
    methods_show = ['$\\texttt{FuncLoop}$',
                    '$\\texttt{DataVect}$',
                    '$\\texttt{ZCS}$ (ours)']
    n_functions_list = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    n_points_list = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
    max_order_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    n_functions_fix = 80
    n_points_fix = 8000
    max_order_fix = 3
    repeats = range(5)

    # plot
    fig = plt.figure(dpi=200, figsize=(10, 5))
    plt.subplots_adjust(hspace=.25)
    ax00 = fig.add_subplot(2, 3, 1)
    ax10 = fig.add_subplot(2, 3, 4, sharex=ax00)
    ax01 = fig.add_subplot(2, 3, 2, sharey=ax00)
    ax11 = fig.add_subplot(2, 3, 5, sharex=ax01, sharey=ax10)
    ax02 = fig.add_subplot(2, 3, 3, sharey=ax00)
    ax12 = fig.add_subplot(2, 3, 6, sharex=ax02, sharey=ax10)
    plot('n_functions', ax10, ax00, yl_time=True, yl_mem=True, xl_time=True)
    plot('n_points', ax11, ax01, xl_time=True)
    plot('max_power', ax12, ax02, xl_time=True)
    ax12.legend(borderpad=.3, handlelength=1.5, handletextpad=.6,
                borderaxespad=.2, labelspacing=0.4)
    ax00.text(.05, .95, '$N=8000, P=3$', va='top', ha='left', transform = ax00.transAxes)
    ax10.text(.05, .95, '$N=8000, P=3$', va='top', ha='left', transform = ax10.transAxes)
    ax01.text(.05, .95, '$M=80, P=3$', va='top', ha='left', transform = ax01.transAxes)
    ax11.text(.05, .95, '$M=80, P=3$', va='top', ha='left', transform = ax11.transAxes)
    ax02.text(.05, .95, '$N=8000, M=80$', va='top', ha='left', transform = ax02.transAxes)
    ax12.text(.05, .95, '$N=8000, M=80$', va='top', ha='left', transform = ax12.transAxes)
    plt.savefig('../figs/scaling.pdf', bbox_inches='tight', pad_inches=0.02,
                facecolor='w')
