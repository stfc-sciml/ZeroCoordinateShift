"""
Scaling analysis
"""

import os
import sys


def run(method, n_functions, n_points, max_order, repeat, device):
    cmd = (f"CUDA_VISIBLE_DEVICES={device} "
           f"python run_single.py {method} {n_functions} {n_points} {max_order} {repeat}")
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    method_name = sys.argv[1]
    try:
        device_no = sys.argv[2]
    except IndexError:
        device_no = 0
    assert method_name in ['unaligned_baseline', 'aligned_baseline', 'aligned_ZCS']

    n_functions_list = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    n_points_list = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
    max_order_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    n_functions_fix = 80
    n_points_fix = 8000
    max_order_fix = 3

    n_repeats = 3
    for i in range(n_repeats):
        for n_f in n_functions_list:
            run(method_name, n_f, n_points_fix, max_order_fix, i, device_no)
        for n_p in n_points_list:
            run(method_name, n_functions_fix, n_p, max_order_fix, i, device_no)
        for n_o in max_order_list:
            run(method_name, n_functions_fix, n_points_fix, n_o, i, device_no)
