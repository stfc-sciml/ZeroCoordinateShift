"""
Operator learning for Burgers equation
"""

import argparse
import sys

import deepxde as dde
import numpy as np
import scipy.io as io
from deepxde.backend import torch

# import deepxde extensions
sys.path.append("..")
import deepxde_extensions as ddex  # noqa: E402


def set_seed(seed):
    """ Set seed for reproducibility """
    torch.manual_seed(seed)
    np.random.seed(seed)


def burgers_baseline(x, u, _):
    """ Burgers PDE with original deepxde """
    visc = 0.01
    u_t = dde.grad.jacobian(u, x, j=0)
    u_x = dde.grad.jacobian(u, x, j=1)
    u_xx = dde.grad.jacobian(u_x, x, j=1)
    return u_t + u * u_x - visc * u_xx


def burgers_ZCS(zcs_scalars, u_, _):
    """ Burgers PDE with ZCS """
    visc = 0.01
    zcs_t, zcs_x = zcs_scalars
    # pseudo sum
    dummy = torch.ones_like(u_).requires_grad_()
    u_ps = (u_ * dummy).sum()
    # all-scalar AD
    u_t, u_x = torch.autograd.grad(u_ps, (zcs_t, zcs_x), create_graph=True)
    u_xx = torch.autograd.grad(u_x, zcs_x, create_graph=True)[0]
    # linear part
    lin = u_t - visc * u_xx
    lin = torch.autograd.grad(lin, dummy, create_graph=True)[0]
    # nonlinear part
    u_x = torch.autograd.grad(u_x, dummy, create_graph=True)[0]
    non_lin = u_ * u_x
    return lin + non_lin


class DataDefinedFunctionSpace(dde.data.function_spaces.FunctionSpace):
    """ Data-defined function space """

    def __init__(self, u0_data):
        self.data = u0_data

    def random(self, size):
        # return index
        return np.random.choice(len(self.data), size)

    def eval_one(self, idx, xy):
        return self.data[idx]

    def eval_batch(self, idxes, xys):
        assert len(xys) >= self.data.shape[1]
        # fill with nan
        res = np.full((len(idxes), len(xys)), np.nan, dtype=self.data.dtype)
        # fill the first N_X boundary points
        res[:, :self.data.shape[1]] = self.data[idxes, :]
        return res


def bc_func_baseline(_, outputs, aux, beg, end):
    """ BC function with original deepxde """
    # outputs: n_points, n_component=1
    # aux: n_points, n_component=1
    return outputs[beg:end, 0] - aux[beg:end, 0]


def bc_func_ZCS(_, outputs, aux, beg, end):
    """ BC function with ZCS """
    # outputs: n_data, n_points, n_component=1
    # aux: n_data, n_points
    return outputs[:, beg:end, 0] - aux[:, beg:end]


def periodic(tx):
    """ Periodic boundary condition along x-dimension """
    t, x = tx[:, 0], tx[:, 1]
    x = 2 * np.pi * x
    return torch.stack(
        [torch.cos(x),
         torch.sin(x),
         torch.cos(2 * x),
         torch.sin(2 * x), t], dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PINO Burgers',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--unaligned', action='store_true',
                        help='unaligned data arrangement')
    parser.add_argument('--ZCS', action='store_true',
                        help='enable ZCS')
    parser.add_argument('--n-data-train', type=int, default=950,
                        help='number of functions to train')
    parser.add_argument('--n-points-train', type=int, default=12672,  # 128 * 101 - 128 * 2
                        help='number of training points in domain')
    parser.add_argument('--n-points-test', type=int, default=500,
                        help='number of testing points in domain')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=100000,
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

    # data
    try:
        mat_data = io.loadmat("burgers_pino.mat")
    except FileNotFoundError:
        raise FileNotFoundError('\nPlease download `burgers_pino.mat` from '
                                '\nhttps://github.com/neuraloperator/physics_informed '
                                '\nand put it under `Burgers/`.')
    u0_input = mat_data['input'].astype(np.float32)
    u_output = mat_data['output'].astype(np.float32)
    N_DATA, N_T, N_X = u_output.shape

    # geometry
    geom = dde.geometry.Rectangle((0, 0), (1, 1))

    # boundary condition
    x_crds = np.arange(N_X) / N_X
    t_crds = np.zeros_like(x_crds)
    bc_points = np.stack([t_crds, x_crds], axis=1)
    bc_operator = ddex.operator.PointSetOperatorBCUsingAux(
        points=bc_points, values=0.,
        func=bc_func_ZCS if args.ZCS and not args.unaligned else bc_func_baseline)

    # PDE
    pde_with_timer = ddex.timer.FunctionTimer(burgers_ZCS
                                              if args.ZCS else burgers_baseline)
    pde_burgers = dde.data.PDE(
        geometry=geom,
        pde=pde_with_timer.func_timed,
        bcs=[bc_operator],
        # use 'pseudo' for reproducibility because the other
        # distributions are based on skopt.sampler, whose
        # `random_state` is not used by deepxde.geometry
        train_distribution='pseudo',
        num_domain=args.n_points_train,
        num_boundary=N_X,
        num_test=args.n_points_test
    )

    # function space
    func_space = DataDefinedFunctionSpace(u0_input)

    # data
    if args.unaligned:
        CLS = ddex.operator.PDEOperatorZCS if args.ZCS else ddex.operator.PDEOperatorBatch
    else:
        CLS = ddex.operator.PDEOperatorCartesianProdZCS if args.ZCS \
            else dde.data.PDEOperatorCartesianProd
    data = CLS(pde_burgers, func_space, x_crds, num_function=args.n_data_train,
               num_test=args.batch_size, batch_size=args.batch_size)

    # network
    NetClass = dde.nn.DeepONet if args.unaligned else dde.nn.DeepONetCartesianProd
    net = NetClass(
        [N_X, 128, 128, 128],
        [5, 128, 128, 128],
        "tanh",
        "Glorot normal",
    )
    net.apply_feature_transform(periodic)

    # model
    ModelClass = ddex.model.ModelZCS if args.ZCS else ddex.model.Model
    model = ModelClass(data, net)
    decay = ("inverse time", args.decay_epochs, args.decay_rate)
    model.compile("adam", lr=args.lr, decay=decay)

    # train
    torch.cuda.reset_peak_memory_stats()
    model.activate_timer(True)
    loss_history, train_state = model.train(iterations=args.epochs)
    model.activate_timer(False)
    max_mem_train = torch.cuda.max_memory_allocated()

    #######################
    # BELOW IS VALIDATION #
    #######################
    # trunk input
    x_crds = np.arange(N_X) / N_X
    t_crds = np.arange(N_T) / (N_T - 1)
    x_trunk = np.stack(np.meshgrid(t_crds, x_crds, indexing='ij'),
                       axis=2).reshape(-1, 2)
    if args.unaligned:
        x_trunk = np.tile(x_trunk[None, :, :], (args.batch_size, 1, 1)).reshape(-1, 2)
    x_trunk = torch.from_numpy(x_trunk).to(
        dtype=torch.float32, device='cuda')
    x_trunk.requires_grad_()
    zcs_scalars_pred = None
    if args.ZCS:
        x_trunk, zcs_scalars_pred = ddex.model.trunk_inputs_to_ZCS(x_trunk)

    # branch input
    v_branch = u0_input[-args.batch_size:]
    if args.unaligned:
        v_branch = np.tile(
            v_branch[:, None, :], (1, N_T * N_X, 1)).reshape(-1, N_X)
    v_branch = torch.from_numpy(v_branch).to(
        dtype=torch.float32, device='cuda')

    # data error
    u_pred = model.net((v_branch, x_trunk))
    if args.unaligned:
        u_pred = u_pred.reshape(args.batch_size, N_T * N_X)
    u_true = u_output[-args.batch_size:].reshape(u_pred.shape)
    data_error = dde.metrics.l2_relative_error(u_true,
                                               u_pred.detach().cpu().numpy())

    # PDE error and graph size
    torch.cuda.empty_cache()
    m0 = torch.cuda.memory_allocated()
    if args.unaligned:
        if args.ZCS:
            pde_pred = burgers_ZCS(zcs_scalars_pred, u_pred.reshape(-1, 1), 0)
        else:
            pde_pred = burgers_baseline(x_trunk, u_pred.reshape(-1, 1), 0)
    else:
        if args.ZCS:
            pde_pred = burgers_ZCS(zcs_scalars_pred, u_pred, 0)
        else:
            pde_pred = torch.empty_like(u_pred)
            for index in range(len(u_pred)):
                pde_pred[index] = burgers_baseline(x_trunk,
                                                   u_pred[index][:, None], 0)[:, 0]
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
