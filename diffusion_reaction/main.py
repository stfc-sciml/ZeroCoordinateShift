"""
Operator learning for diffusion reaction

Extended from
https://github.com/lululxvi/deepxde/blob/master/examples/operator/diff_rec_aligned_pideeponet.py
"""

import argparse
import sys

import deepxde as dde
import numpy as np
from deepxde.backend import torch

from ADR_solver import solve_ADR

# import deepxde extensions
sys.path.append("..")
import deepxde_extensions as ddex  # noqa: E402


def set_seed(seed):
    """ Set seed for reproducibility """
    torch.manual_seed(seed)
    np.random.seed(seed)


def DR_baseline(x_, u_, v_):
    """ DR PDE with original deepxde """
    D = 0.01
    k = 0.01
    du_t = dde.grad.jacobian(u_, x_, j=1)
    du_xx = dde.grad.hessian(u_, x_, i=0, j=0)
    return du_t - D * du_xx + k * u_ ** 2 - v_


def DR_ZCS(zcs_scalars, u_, v_):
    """ DR PDE with ZCS """
    D = 0.01
    k = 0.01
    zcs_x, zcs_t = zcs_scalars
    # pseudo sum
    dummy = torch.ones_like(u_).requires_grad_()
    u_ps = (u_ * dummy).sum()
    # all-scalar AD
    u_x_ps, u_t_ps = torch.autograd.grad(u_ps, (zcs_x, zcs_t), create_graph=True)
    u_xx_ps = torch.autograd.grad(u_x_ps, zcs_x, create_graph=True)[0]
    # pseudo sum of differential terms
    diff_ps = u_t_ps - D * u_xx_ps
    # field obtained by AD w.r.t. dummy
    diff = torch.autograd.grad(diff_ps, dummy, create_graph=True)[0]
    return diff + k * u_ ** 2 - v_


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PINO Diffusion Reaction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--unaligned', action='store_true',
                        help='unaligned data arrangement')
    parser.add_argument('--ZCS', action='store_true',
                        help='enable ZCS')
    parser.add_argument('--n-functions-train', type=int, default=1000,
                        help='number of functions to train')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='batch size')
    parser.add_argument('--n-samples-fs-train', type=int, default=50,
                        help='number of samples from function space for training')
    parser.add_argument('--n-samples-fs-validate', type=int, default=40,
                        help='number of samples from function space for validation')

    parser.add_argument('--n-points-train', type=int, default=1000,
                        help='number of training points in domain')
    parser.add_argument('--n-points-bc', type=int, default=200,
                        help='number of training points on BC')
    parser.add_argument('--n-points-ic', type=int, default=100,
                        help='number of training points on IC')
    parser.add_argument('--n-points-test', type=int, default=500,
                        help='number of testing points in domain')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='epochs')
    parser.add_argument('--decay_epochs', type=int, default=2000,
                        help='epochs for lr decay')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='rate for lr decay')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for reproducibility')
    args = parser.parse_args(args=[] if 'ipykernel' in sys.modules else None)

    # seed
    set_seed(args.seed)

    # geometry
    geom = dde.geometry.Interval(0, 1)
    time_domain = dde.geometry.TimeDomain(0, 1)
    geom_time = dde.geometry.GeometryXTime(geom, time_domain)

    # icbc
    if args.ZCS and not args.unaligned:
        bc = ddex.operator.CollectiveDirichletBC(geom_time, lambda _: 0, lambda _, on_boundary: on_boundary)
        ic = ddex.operator.CollectiveIC(geom_time, lambda _: 0, lambda _, on_initial: on_initial)
    else:
        bc = dde.icbc.DirichletBC(geom_time, lambda _: 0, lambda _, on_boundary: on_boundary)
        ic = dde.icbc.IC(geom_time, lambda _: 0, lambda _, on_initial: on_initial)

    # PDE
    pde_with_timer = ddex.timer.FunctionTimer(DR_ZCS if args.ZCS else DR_baseline)
    pde_DR = dde.data.TimePDE(
        geometryxtime=geom_time,
        pde=pde_with_timer.func_timed,
        ic_bcs=[bc, ic],
        # use 'pseudo' for reproducibility because the other
        # distributions are based on skopt.sampler, whose
        # `random_state` is not used by deepxde.geometry
        train_distribution='pseudo',
        num_domain=args.n_points_train,
        num_boundary=args.n_points_bc,
        num_initial=args.n_points_ic,
        num_test=args.n_points_test
    )

    # function space
    func_space = dde.data.GRF(length_scale=0.2)

    # data
    eval_pts = np.linspace(0, 1, num=args.n_samples_fs_train)[:, None]
    if args.unaligned:
        CLS = ddex.operator.PDEOperatorZCS if args.ZCS else ddex.operator.PDEOperatorBatch
    else:
        CLS = ddex.operator.PDEOperatorCartesianProdZCS if args.ZCS \
            else dde.data.PDEOperatorCartesianProd
    data = CLS(pde_DR, func_space, eval_pts, num_function=args.n_functions_train,
               function_variables=[0], num_test=args.batch_size,
               batch_size=args.batch_size)

    # network
    NetClass = dde.nn.DeepONet if args.unaligned else dde.nn.DeepONetCartesianProd
    net = NetClass(
        [args.n_samples_fs_train, 128, 128, 128],
        [2, 128, 128, 128],
        "tanh",
        "Glorot normal",
    )

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

    # ground truth
    # make sure the size is the same as training
    n_functions_validate = args.batch_size
    func_feats = func_space.random(n_functions_validate)
    vs = func_space.eval_batch(
        func_feats, np.linspace(0, 1, args.n_samples_fs_validate)[:, None])
    x, t, us_true = None, None, []
    for v in vs:
        x, t, u_true = solve_ADR(
            xmin=0,
            xmax=1,
            tmin=0,
            tmax=1,
            k=lambda x_: 0.01 * np.ones_like(x_),
            v=lambda x_: np.zeros_like(x_),
            g=lambda u_: 0.01 * u_ ** 2,
            dg=lambda u_: 0.02 * u_,
            f=lambda x_, t_: np.tile(v[:, None], (1, len(t_))),
            u0=lambda x_: np.zeros_like(x_),
            Nx=args.n_samples_fs_validate,
            Nt=args.n_samples_fs_validate,
        )
        us_true.append(u_true.T)
    us_true = np.array(us_true).reshape(len(us_true), -1)

    # branch input
    v_branch = func_space.eval_batch(
        func_feats, np.linspace(0, 1, args.n_samples_fs_train)[:, None])
    if args.unaligned:
        v_branch = np.tile(
            v_branch[:, None, :], (1, args.n_samples_fs_validate ** 2, 1)).reshape(
            -1, args.n_samples_fs_train)
    v_branch = torch.from_numpy(v_branch).to(dtype=torch.float32, device='cuda')

    # trunk input
    xv, tv = np.meshgrid(x, t)
    x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
    if args.unaligned:
        x_trunk = np.tile(x_trunk[None, :, :], (n_functions_validate, 1, 1)).reshape(-1, 2)
    x_trunk = torch.from_numpy(x_trunk).to(dtype=torch.float32, device='cuda')
    x_trunk.requires_grad_()
    zcs_scalars_pred = None
    if args.ZCS:
        x_trunk, zcs_scalars_pred = ddex.model.trunk_inputs_to_ZCS(x_trunk)

    # data error
    u_pred = model.net((v_branch, x_trunk))
    if args.unaligned:
        u_pred = u_pred.reshape(n_functions_validate, args.n_samples_fs_validate ** 2)
    data_error = dde.metrics.l2_relative_error(us_true, u_pred.detach().cpu().numpy())

    # PDE error and graph size
    aux_var = np.tile(vs[:, None, :], (1, args.n_samples_fs_validate, 1)).reshape(
        n_functions_validate, -1)
    aux_var = torch.from_numpy(aux_var).to(dtype=torch.float32, device='cuda')
    torch.cuda.empty_cache()
    m0 = torch.cuda.memory_allocated()
    if args.unaligned:
        if args.ZCS:
            pde_pred = DR_ZCS(zcs_scalars_pred, u_pred.reshape(-1, 1), aux_var.reshape(-1, 1))
        else:
            pde_pred = DR_baseline(x_trunk, u_pred.reshape(-1, 1), aux_var.reshape(-1, 1))
    else:
        if args.ZCS:
            pde_pred = DR_ZCS(zcs_scalars_pred, u_pred, aux_var)
        else:
            pde_pred = torch.empty_like(u_pred)
            for index in range(len(u_pred)):
                pde_pred[index] = DR_baseline(x_trunk, u_pred[index][:, None],
                                              aux_var[index][:, None])[:, 0]
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
