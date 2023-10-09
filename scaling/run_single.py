"""
Scaling analysis
"""

import sys

import deepxde as dde
import numpy as np
from deepxde.backend import torch

# import deepxde extensions
sys.path.append("..")
import deepxde_extensions as ddex  # noqa: E402


def run(method, n_functions, n_points, max_order, repeat):
    ZCS = 'ZCS' in method
    unaligned = 'unaligned' in method

    # geometry
    geom = dde.geometry.Rectangle((0, 0), (1, 1))

    # PDE
    if not ZCS:
        def pde_func(x_, u_, _):
            """ PDE with original deepxde """
            sum_ = u_.clone()
            for order in range(max_order):
                du = dde.grad.jacobian(u_, x_, j=0) + dde.grad.jacobian(u_, x_, j=1)
                sum_ += du
                u_ = du
            return sum_
    else:
        def pde_func(zcs_scalars, u_, _):
            """ PDE with ZCS """
            zcs_x, zcs_y = zcs_scalars
            # pseudo sum
            dummy = torch.ones_like(u_).requires_grad_()
            omega = (u_ * dummy).sum()
            sum_ = omega.clone()
            # all-scalar AD
            for order in range(max_order):
                dx, dy = torch.autograd.grad(omega, (zcs_x, zcs_y), create_graph=True)
                d_omega = dx + dy
                sum_ += d_omega
                omega = d_omega
            # field obtained by AD w.r.t. dummy
            return torch.autograd.grad(sum_, dummy, create_graph=True)[0]
    pde_with_timer = ddex.timer.FunctionTimer(pde_func)
    pde = dde.data.PDE(
        geometry=geom,
        pde=pde_with_timer.func_timed,
        bcs=[],
        train_distribution='pseudo',
        num_domain=n_points
    )

    # function space
    func_space = dde.data.GRF(length_scale=0.2)

    # data
    eval_pts = np.linspace(0, 1, num=50)[:, None]
    if unaligned:
        CLS = ddex.operator.PDEOperatorZCS if ZCS else dde.data.PDEOperator
    else:
        CLS = ddex.operator.PDEOperatorCartesianProdZCS if ZCS \
            else dde.data.PDEOperatorCartesianProd
    data = CLS(pde, func_space, eval_pts, num_function=n_functions,
               function_variables=[0], num_test=1)

    # network
    NetClass = dde.nn.DeepONet if unaligned else dde.nn.DeepONetCartesianProd
    net = NetClass(
        [50, 128, 128, 128],
        [2, 128, 128, 128],
        "tanh",
        "Glorot normal",
    )

    # model
    ModelClass = ddex.model.ModelZCS if ZCS else ddex.model.Model
    model = ModelClass(data, net)
    model.compile("adam", lr=0.0001)

    # train
    torch.cuda.reset_peak_memory_stats()
    model.activate_timer(True)
    _, _ = model.train(iterations=100)
    model.activate_timer(False)
    max_mem_train = torch.cuda.max_memory_allocated()

    #################################
    # BELOW IS TO OBTAIN GRAPH SIZE #
    #################################

    # branch input
    v_branch = torch.from_numpy(data.train_x[0]).to(dtype=torch.float32, device='cuda')

    # trunk input
    x_trunk = torch.from_numpy(data.train_x[1]).to(dtype=torch.float32, device='cuda')
    x_trunk.requires_grad_()
    zcs_scalars_pred = None
    if ZCS:
        x_trunk, zcs_scalars_pred = ddex.model.trunk_inputs_to_ZCS(x_trunk)

    # forward and pde
    u_pred = model.net((v_branch, x_trunk))
    torch.cuda.empty_cache()
    m0 = torch.cuda.memory_allocated()
    if unaligned:
        if ZCS:
            pde_pred = pde_func(zcs_scalars_pred, u_pred.reshape(-1, 1), 0)
        else:
            pde_pred = pde_func(x_trunk, u_pred.reshape(-1, 1), 0)
    else:
        if ZCS:
            pde_pred = pde_func(zcs_scalars_pred, u_pred, 0)
        else:
            pde_pred = torch.empty_like(u_pred)
            for index in range(len(u_pred)):
                pde_pred[index] = pde_func(x_trunk, u_pred[index][:, None], 0)[:, 0]
    # graph size
    torch.cuda.empty_cache()
    m1 = torch.cuda.memory_allocated()
    mem_graph = m1 - m0
    # error
    pde_pred = pde_pred.detach().cpu().numpy()
    pde_error = dde.metrics.mean_squared_error(np.zeros_like(pde_pred), pde_pred)

    # report
    wct_backprop = (model.wct_dict["train_step"] - model.wct_dict["inputs"]
                    - model.wct_dict["forward"] - model.wct_dict["losses"])
    out = np.array([pde_error,
                    max_mem_train,
                    mem_graph,
                    model.wct_dict["train_step"],
                    model.wct_dict["inputs"],
                    model.wct_dict["forward"],
                    model.wct_dict["losses"],
                    wct_backprop,
                    pde_with_timer.wct])
    run_name = f'{method}_{n_functions}_{n_points}_{max_order}_{repeat}'
    np.savetxt(f'outputs/{run_name}.txt', out)  # noqa


if __name__ == "__main__":
    # We are unable to do for-loop over the arguments here because
    # torch.cuda.OutOfMemoryError cannot be handled properly
    # https://discuss.pytorch.org/t/how-to-clean-gpu-memory-after-a-runtimeerror/28781/2?u=ptrblck
    # Call this externally
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
