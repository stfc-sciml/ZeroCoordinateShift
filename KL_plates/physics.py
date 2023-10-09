"""
Physics of Kirchhoff-Love plates
"""

import sys

import deepxde as dde
import numpy as np
from deepxde.backend import torch

# import deepxde extensions
sys.path.append("..")
import deepxde_extensions as ddex  # noqa: E402

# flexural rigidity
D = 0.01


def GL_baseline(x, w, q):
    """ Germain-Lagrange PDE with original deepxde """
    w_xx = dde.grad.hessian(w, x, i=0, j=0)
    w_yy = dde.grad.hessian(w, x, i=1, j=1)
    w_xxxx = dde.grad.hessian(w_xx, x, i=0, j=0)
    w_yyyy = dde.grad.hessian(w_yy, x, i=1, j=1)
    w_xxyy = dde.grad.hessian(w_xx, x, i=1, j=1)
    return w_xxxx + 2 * w_xxyy + w_yyyy - q / D


def GL_ZCS(zcs_scalars, w, q):
    """ Germain-Lagrange PDE with ZCS """
    zcs_x, zcs_y = zcs_scalars
    # pseudo sum
    dummy = torch.ones_like(w).requires_grad_()
    w_ps = (w * dummy).sum()
    # all-scalar AD
    w_x, w_y = torch.autograd.grad(w_ps, (zcs_x, zcs_y), create_graph=True)
    w_xx = torch.autograd.grad(w_x, zcs_x, create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, zcs_y, create_graph=True)[0]
    w_xxx, w_xxy = torch.autograd.grad(w_xx, (zcs_x, zcs_y), create_graph=True)
    w_yyy = torch.autograd.grad(w_yy, zcs_y, create_graph=True)[0]
    w_xxxx = torch.autograd.grad(w_xxx, zcs_x, create_graph=True)[0]
    w_yyyy = torch.autograd.grad(w_yyy, zcs_y, create_graph=True)[0]
    w_xxyy = torch.autograd.grad(w_xxy, zcs_y, create_graph=True)[0]
    # pseudo sum of differential terms
    diff_ps = w_xxxx + 2 * w_xxyy + w_yyyy
    # field obtained by AD w.r.t. dummy
    diff = torch.autograd.grad(diff_ps, dummy, create_graph=True)[0]
    return diff - q / D


def compute_sin_terms(a, xy):
    """ compute sin terms in q and w """
    backend = np if isinstance(a, np.ndarray) else torch
    order = int(round(np.sqrt(a.shape[1])))
    m = backend.arange(1, order + 1, dtype=xy.dtype)
    sin_pi_mx = backend.sin(np.pi * m[:, None] * xy[:, 0][None, :])
    sin_pi_ny = backend.sin(np.pi * m[:, None] * xy[:, 1][None, :])
    sin_xy = backend.einsum('mx,nx->mnx', sin_pi_mx, sin_pi_ny)
    return sin_xy, order, backend


def compute_q(a, xy):
    """ compute load q """
    sin_xy, order, backend = compute_sin_terms(a, xy)
    a = a.reshape((-1, order, order))
    return backend.einsum('bmn,mnx->bx', a, sin_xy)


def compute_w(a, xy):
    """ compute solution w """
    sin_xy, order, backend = compute_sin_terms(a, xy)
    a = a.reshape((-1, order, order))
    m = backend.arange(1, order + 1, dtype=xy.dtype)
    m2_n2 = ((m ** 2)[:, None] + (m ** 2)[None, :]) ** 2
    w = a / (m2_n2[None, :, :] * D * np.pi ** 4)
    return backend.einsum('bmn,mnx->bx', w, sin_xy)


if __name__ == '__main__':
    a_ = torch.randn(1, 100, dtype=torch.float64)
    x_ = torch.linspace(0, 1, 100, dtype=torch.float64)
    y_ = torch.linspace(0, 1, 100, dtype=torch.float64)
    xy_ = torch.stack(torch.meshgrid(x_, y_, indexing='ij'), dim=-1).reshape(-1, 2)

    # non-ZCS
    xy_.requires_grad_(True)
    w_ = compute_w(a_, xy_)
    q_ = compute_q(a_, xy_)
    pde_ = GL_baseline(xy_, w_[0][:, None], q_[0][:, None])
    print(pde_.abs().max())

    # ZCS
    xy_, zcs = ddex.model.trunk_inputs_to_ZCS(xy_)
    w_ = compute_w(a_, xy_)
    pde_ = GL_ZCS(zcs, w_, q_)
    print(pde_.abs().max())
