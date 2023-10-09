"""
Operator learning for Kirchhoff-Love plates
"""

import argparse
import sys

import deepxde as dde
import numpy as np
from deepxde.backend import torch

from physics import GL_baseline, GL_ZCS, compute_q, compute_w

# import deepxde extensions
sys.path.append("..")
import deepxde_extensions as ddex  # noqa: E402


def set_seed(seed):
    """ Set seed for reproducibility """
    torch.manual_seed(seed)
    np.random.seed(seed)


class SinLoadFunctionSpace(dde.data.function_spaces.FunctionSpace):
    """ Function space for load q """

    def __init__(self, order):
        self.order = order
        self.dtype = dde.data.function_spaces.config.real(np)

    def random(self, size):
        return np.random.randn(size, self.order ** 2).astype(self.dtype)

    def eval_one(self, feature, xy):
        return compute_q(np.expand_dims(feature, axis=0),
                         np.expand_dims(xy, axis=0)).astype(self.dtype)[0, 0]

    def eval_batch(self, features, xys):
        # here we have a workaround to return features (a-coefficients) as branch input
        if len(xys) == args.n_orders_q ** 2:
            # branch input
            return features
        else:
            # auxiliary variables
            return compute_q(features, xys).astype(self.dtype)


def output_transform(inputs, outputs):
    """ enforce bc """
    x, y = inputs[1][:, 0], inputs[1][:, 1]
    return outputs * (torch.sin(np.pi * x) * torch.sin(np.pi * y))[None, :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PINO Kirchhoff-Love Plates',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ZCS', action='store_true',
                        help='enable ZCS')
    parser.add_argument('--n-functions-train', type=int, default=1080,
                        help='number of functions to train')
    parser.add_argument('--n-functions-test', type=int, default=100,
                        help='number of functions to test during training')
    parser.add_argument('--n-orders-q', type=int, default=10,
                        help='Fourier order of load')
    parser.add_argument('--n-points-train', type=int, default=10000,
                        help='number of training points in domain')
    parser.add_argument('--n-points-test', type=int, default=500,
                        help='number of testing points in domain')
    parser.add_argument('--batch-size', type=int, default=36,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=50000,
                        help='epochs')
    parser.add_argument('--decay_epochs', type=int, default=10000,
                        help='epochs for lr decay')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='rate for lr decay')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for reproducibility')
    args = parser.parse_args(args=[] if 'ipykernel' in sys.modules else None)

    # seed
    set_seed(args.seed)

    # geometry
    geom = dde.geometry.Rectangle((0, 0), (1, 1))

    # PDE
    pde_with_timer = ddex.timer.FunctionTimer(GL_ZCS if args.ZCS else GL_baseline)
    pde_GL = dde.data.PDE(
        geometry=geom,
        pde=pde_with_timer.func_timed,
        bcs=[],  # no bc
        # use 'pseudo' for reproducibility because the other
        # distributions are based on skopt.sampler, whose
        # `random_state` is not used by deepxde.geometry
        train_distribution='pseudo',
        num_domain=args.n_points_train,
        num_boundary=0,
        num_test=args.n_points_test
    )

    # function space
    func_space = SinLoadFunctionSpace(order=args.n_orders_q)

    # data
    eval_pts_train = np.full((args.n_orders_q ** 2, 2), np.nan)  # values unused
    CLS = ddex.operator.PDEOperatorCartesianProdZCS if args.ZCS \
        else dde.data.PDEOperatorCartesianProd
    data = CLS(pde_GL, func_space, eval_pts_train, num_function=args.n_functions_train,
               num_test=args.n_functions_test, batch_size=args.batch_size)

    # network
    net = dde.nn.DeepONetCartesianProd(
        [args.n_orders_q ** 2, 128, 128, 128],
        [2, 128, 128, 128],
        "tanh",
        "Glorot normal",
    )
    net.apply_output_transform(output_transform)

    # model
    ModelClass = ddex.model.ModelZCS if args.ZCS else ddex.model.Model
    model = ModelClass(data, net)
    decay = ("inverse time", args.decay_epochs, args.decay_rate)
    model.compile("adam", lr=args.lr, decay=decay)

    # train
    torch.cuda.reset_peak_memory_stats()
    model.activate_timer(True)
    loss_history, train_state = model.train(iterations=args.epochs,
                                            batch_size=args.batch_size)
    model.activate_timer(False)
    max_mem_train = torch.cuda.max_memory_allocated()

    #######################
    # BELOW IS VALIDATION #
    #######################

    # ground truth
    a_validate = func_space.random(args.batch_size)
    n_side = int(np.sqrt(args.n_points_train))
    assert n_side ** 2 == args.n_points_train
    grid_x = np.linspace(0, 1, num=n_side)
    grid_y = np.linspace(0, 1, num=n_side)
    eval_pts_validate = np.stack(np.meshgrid(grid_x, grid_y,
                                             indexing='ij'), axis=-1).reshape(-1, 2)
    q_validate = compute_q(a_validate, eval_pts_validate)
    w_true = compute_w(a_validate, eval_pts_validate)

    # branch input
    v_branch = torch.from_numpy(a_validate).to(dtype=torch.float32, device='cuda')

    # trunk input
    x_trunk = torch.from_numpy(eval_pts_validate).to(dtype=torch.float32, device='cuda')
    x_trunk.requires_grad_()
    zcs_scalars_pred = None
    if args.ZCS:
        x_trunk, zcs_scalars_pred = ddex.model.trunk_inputs_to_ZCS(x_trunk)

    # data error
    w_pred = model.net((v_branch, x_trunk))
    data_error = dde.metrics.l2_relative_error(w_true, w_pred.detach().cpu().numpy())

    # PDE error and graph size
    q_validate = torch.from_numpy(q_validate).to(dtype=torch.float32, device='cuda')
    torch.cuda.empty_cache()
    m0 = torch.cuda.memory_allocated()
    if args.ZCS:
        pde_pred = GL_ZCS(zcs_scalars_pred, w_pred, q_validate)
    else:
        pde_pred = torch.empty_like(w_pred)
        for index in range(len(w_pred)):
            pde_pred[index] = GL_baseline(x_trunk, w_pred[index][:, None],
                                          q_validate[index][:, None])[:, 0]
    # graph size
    torch.cuda.empty_cache()
    m1 = torch.cuda.memory_allocated()
    mem_graph_validate = m1 - m0
    # error
    pde_pred = pde_pred.detach().cpu().numpy()
    pde_error = dde.metrics.mean_squared_error(np.zeros_like(pde_pred), pde_pred)

    # report
    print('argv: ' + ' '.join(sys.argv))
    print(f'== ERRORS ==')
    print(f'* Relative data error: {data_error}')
    print(f'* Absolute PDE error: {pde_error}')
    print(f'== GPU MEMORY ==')
    print(f'* Peak during training: {max_mem_train / 1e6} MB')
    print(f'* PDE graph during validation: {mem_graph_validate / 1e6} MB')
    print(f'== WALL TIME ==')
    print(f'* Total: {model.wct_dict["train_step"]}')
    print(f'* Prep inputs: {model.wct_dict["inputs"]}')
    print(f'* Forward: {model.wct_dict["forward"]}')
    print(f'* Losses: {model.wct_dict["losses"]}')
    wct_backprop = (model.wct_dict["train_step"] - model.wct_dict["inputs"]
                    - model.wct_dict["forward"] - model.wct_dict["losses"])
    print(f'* Backprop: {wct_backprop}')
    print(f'* PDE function: {pde_with_timer.wct}')
