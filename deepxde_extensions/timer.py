"""
Timing utils
"""

from time import time

from deepxde.backend import torch


def get_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time()


class ContextTimer:
    """ context timer """

    def __init__(self, wct_dict, key, activated):
        self.t0 = 0.
        self.wct_dict = wct_dict
        self.key = key
        self.activated = activated

    def __enter__(self):
        if self.activated:
            self.t0 = get_time()

    def __exit__(self, exception_type, exception_value, traceback):
        if self.activated:
            self.wct_dict[self.key] += get_time() - self.t0


class FunctionTimer:
    """ function timer """

    def __init__(self, func):
        self.wct = 0.
        self.func = func

    def reset_time(self):
        self.wct = 0.

    def func_timed(self, *args):
        t0 = get_time()
        out = self.func(*args)
        self.wct += get_time() - t0
        return out
