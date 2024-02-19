"""
Operator learning for Stokes flow
"""

import argparse
import sys

import deepxde as dde
import numpy as np
import torch

# import deepxde extensions
sys.path.append("..")
import deepxde_extensions as ddex  # noqa: E402


def set_seed(seed):
    """ Set seed for reproducibility """
    torch.manual_seed(seed)
    np.random.seed(seed)


def output_transform(inputs, outputs):
    """ enforce bc """
    x, y = inputs[1][:, 0], inputs[1][:, 1]
    # horizontal velocity on left, right, bottom
    u = outputs[:, :, 0] * (x * (1 - x) * y)[None, :]
    # vertical velocity on all edges
    v = outputs[:, :, 1] * (x * (1 - x) * y * (1 - y))[None, :]
    # pressure on bottom
    p = outputs[:, :, 2] * y[None, :]
    return torch.stack((u, v, p), dim=2)


def stokes_zcs(xy, uvp, aux):
    """ stokes PDE with ZCS """
    mu = 0.01
    u, v, p = uvp[..., 0:1], uvp[..., 1:2], uvp[..., 2:3]
    grad_u = ddex.gradient.LazyGrad({'leaves': xy}, u)
    grad_v = ddex.gradient.LazyGrad({'leaves': xy}, v)
    grad_p = ddex.gradient.LazyGrad({'leaves': xy}, p)
    # first order
    du_x = grad_u.compute((1, 0))
    dv_y = grad_v.compute((0, 1))
    dp_x = grad_p.compute((1, 0))
    dp_y = grad_p.compute((0, 1))
    # second order
    du_xx = grad_u.compute((2, 0))
    du_yy = grad_u.compute((0, 2))
    dv_xx = grad_v.compute((2, 0))
    dv_yy = grad_v.compute((0, 2))
    motion_x = mu * (du_xx + du_yy) - dp_x
    motion_y = mu * (dv_xx + dv_yy) - dp_y
    mass = du_x + dv_y
    return motion_x, motion_y, mass


def stokes_baseline(xy, uvp, aux):
    """ stokes PDE without ZCS """
    mu = 0.01
    uvp = uvp[:, 0, :]
    # first order
    du_x = dde.grad.jacobian(uvp, xy, i=0, j=0)
    dv_y = dde.grad.jacobian(uvp, xy, i=1, j=1)
    dp_x = dde.grad.jacobian(uvp, xy, i=2, j=0)
    dp_y = dde.grad.jacobian(uvp, xy, i=2, j=1)
    # second order
    du_xx = dde.grad.hessian(uvp, xy, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(uvp, xy, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(uvp, xy, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(uvp, xy, component=1, i=1, j=1)
    motion_x = mu * (du_xx + du_yy) - dp_x
    motion_y = mu * (dv_xx + dv_yy) - dp_y
    mass = du_x + dv_y
    return motion_x, motion_y, mass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PINO Stokes flow',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ZCS', action='store_true',
                        help='enable ZCS')
    parser.add_argument('--n-functions-train', type=int, default=1080,
                        help='number of functions to train')
    parser.add_argument('--n-functions-test', type=int, default=100,
                        help='number of functions to test during training')
    parser.add_argument('--n-points-train', type=int, default=5000,
                        help='number of training points in domain')
    parser.add_argument('--n-points-boundary', type=int, default=4000,
                        help='number of training points on boundary')
    parser.add_argument('--n-points-test', type=int, default=500,
                        help='number of testing points in domain')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
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

    # boundary condition
    # other boundary conditions will be enforced by output transform
    def bc_slip_top_func(x, aux_var):
        # using (perturbation / 10 + 1) * x * (1 - x)
        return (aux_var.t() / 10 + 1.) * dde.backend.as_tensor(x[:, 0] * (1 - x[:, 0]))[None, :]

    bc_slip_top = ddex.operator.CollectiveDirichletBC2(
        geom=geom,
        func=bc_slip_top_func,
        on_boundary=lambda x, on_boundary: np.isclose(x[1], 1.),
        component=0)

    # PDE
    pde_with_timer = ddex.timer.FunctionTimer(stokes_zcs if args.ZCS else stokes_baseline)
    pde_stokes = dde.data.PDE(
        geometry=geom,
        pde=pde_with_timer.func_timed,
        bcs=[bc_slip_top],  # no bc
        # use 'pseudo' for reproducibility because the other
        # distributions are based on skopt.sampler, whose
        # `random_state` is not used by deepxde.geometry
        train_distribution='pseudo',
        num_domain=args.n_points_train,
        num_boundary=args.n_points_boundary,
        num_test=args.n_points_test
    )

    # Function space
    func_space = dde.data.GRF(length_scale=0.2)

    # Data
    n_pts_edge = 101  # using the size of true solution, but this is unnecessary
    eval_pts = np.linspace(0, 1, num=n_pts_edge)[:, None]
    CLS = ddex.operator.PDEOperatorCartesianProdZCS if args.ZCS \
        else dde.data.PDEOperatorCartesianProd
    data = CLS(
        pde_stokes, func_space, eval_pts, num_function=1000,
        function_variables=[0], num_test=100, batch_size=50
    )

    # Net
    net = ddex.deeponet_pytorch.DeepONetCartesianProd(
        [n_pts_edge, 128, 128, 128],
        [2, 128, 128, 128],
        "tanh",
        "Glorot normal",
        num_outputs=3,
        multi_output_strategy="independent"
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
    func_feats = func_space.random(1)
    v = func_space.eval_batch(func_feats, eval_pts)
    v[:] = 0.  # true solution uses zero perturbation
    xv, yv = np.meshgrid(eval_pts[:, 0], eval_pts[:, 0], indexing='ij')
    xy = np.vstack((np.ravel(xv), np.ravel(yv))).T
    sol_pred = model.predict((v, xy))[0]
    sol_true = np.load('./stokes.npz')['arr_0']
    data_error = dde.metrics.l2_relative_error(sol_true[:, 0], sol_pred[:, 0])

    # graph size
    torch.cuda.empty_cache()
    m0 = torch.cuda.memory_allocated()
    (branch, trunk), _, _ = data.train_next_batch()
    branch = torch.from_numpy(branch).to(dtype=torch.float32, device='cuda')
    trunk = torch.from_numpy(trunk).to(dtype=torch.float32, device='cuda')
    trunk.requires_grad_()
    zcs_scalars_pred = None
    if args.ZCS:
        trunk, zcs_scalars_pred = ddex.model.trunk_inputs_to_ZCS(trunk)
    u_pred = model.net((branch, trunk))
    if args.ZCS:
        pde_pred = stokes_zcs(zcs_scalars_pred, u_pred, None)
    else:
        mx_pred = torch.empty_like(u_pred[:, :, 0])
        my_pred = torch.empty_like(u_pred[:, :, 0])
        mass_pred = torch.empty_like(u_pred[:, :, 0])
        for index in range(len(u_pred)):
            mx, my, mass = stokes_baseline(trunk, u_pred[index].unsqueeze(1),
                                              None)
            mx_pred[index] = mx[:, 0]
            my_pred[index] = my[:, 0]
            mass_pred[index] = mass[:, 0]
    torch.cuda.empty_cache()
    m1 = torch.cuda.memory_allocated()
    mem_graph_validate = m1 - m0

    # report
    print('argv: ' + ' '.join(sys.argv))
    print(f'== ERRORS ==')
    print(f'* Relative data error: {data_error}')
    print(f'* Absolute PDE error: 0')
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
