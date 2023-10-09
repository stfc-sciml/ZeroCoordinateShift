"""
PDE operator extensions
"""

import deepxde as dde
import numpy as np


class PDEOperatorBatch(dde.data.Data):
    """
    Refactored PDEOperator that supports batches

    This is needed for fair time and memory comparison.
    """

    def __init__(self, pde, function_space, evaluation_points,
                 num_function, function_variables=None, num_test=None,
                 batch_size=None):
        self.pde = pde
        self.func_space = function_space
        self.eval_pts = evaluation_points
        self.num_func = num_function
        self.func_vars = (
            function_variables
            if function_variables is not None
            else list(range(pde.geom.dim))
        )
        self.num_test = num_test
        ################
        # handle batch #
        ################
        self.batch_size = batch_size
        if self.batch_size is None:
            self.batch_size = self.num_func
        self.train_sampler = dde.data.sampler.BatchSampler(num_function, shuffle=True)
        self.num_bcs = [n * self.batch_size for n in self.pde.num_bcs]
        # otherwise self.num_bcs is wrong
        assert self.num_func % self.batch_size == 0, \
            "[caveat] num_function must divide batch_size"

        self.train_bc = None
        self.train_x = None
        self.train_y = None
        self.train_aux_vars = None
        self.test_x = None
        self.test_y = None
        self.test_aux_vars = None
        self.train_next_batch()
        self.test()

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        bkd = dde.backend

        f = []
        if self.pde.pde is not None:
            f = self.pde.pde(inputs[1], outputs, model.net.auxiliary_vars)
            if not isinstance(f, (list, tuple)):
                f = [f]

        bcs_start = np.cumsum([0] + self.num_bcs)
        error_f = [fi[bcs_start[-1]:] for fi in f]
        losses = [loss_fn(bkd.zeros_like(error), error) for error in error_f]
        for i, bc in enumerate(self.pde.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            # The same BC points are used for training and testing.
            error = bc.error(
                self.train_x[1],
                inputs[1],
                outputs,
                beg,
                end,
                aux_var=model.net.auxiliary_vars,
            )
            losses.append(loss_fn(bkd.zeros_like(error), error))
        return losses

    def train_next_batch(self, batch_size=None):
        if self.train_x is None:
            func_feats = self.func_space.random(self.num_func)
            func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)
            v, x, vx = self.bc_inputs(func_feats, func_vals)
            if self.pde.pde is not None:
                v_pde, x_pde, vx_pde = self.gen_inputs(
                    func_feats, func_vals, self.pde.train_x_all
                )
                v = np.concatenate((v, v_pde), axis=1)
                x = np.concatenate((x, x_pde), axis=1)
                vx = np.concatenate((vx, vx_pde), axis=1)
            self.train_x = (v, x)
            self.train_aux_vars = vx

        ################
        # handle batch #
        ################
        # sample
        func_index = self.train_sampler.get_next(self.batch_size)
        v_flat, x_flat, a_flat = self._struct_to_flat(self.train_x[0][func_index],
                                                      self.train_x[1][func_index],
                                                      self.train_aux_vars[func_index])
        return (v_flat, x_flat), self.train_y, a_flat

    def test(self):
        if self.num_test is None:
            raise NotImplementedError('Undefined behavior. Please specify num_test.')
        func_feats = self.func_space.random(self.num_test)
        func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)
        v, x, vx = self.bc_inputs(func_feats, func_vals)
        if self.pde.pde is not None:
            v_pde, x_pde, vx_pde = self.gen_inputs(
                func_feats, func_vals, self.pde.test_x[sum(self.pde.num_bcs):]
            )
            v = np.concatenate((v, v_pde), axis=1)
            x = np.concatenate((x, x_pde), axis=1)
            vx = np.concatenate((vx, vx_pde), axis=1)
        v_flat, x_flat, a_flat = self._struct_to_flat(v, x, vx)
        self.test_x, self.test_aux_vars = (v_flat, x_flat), a_flat
        return self.test_x, self.test_y, self.test_aux_vars

    def _struct_to_flat(self, v_struct, x_struct, a_struct):
        if a_struct.ndim == 2:
            a_struct = a_struct[:, :, None]
        new_a_shape = list(a_struct.shape)[1:]
        new_a_shape[0] = -1

        # put bc points at first to be compatible with original DeepXDE
        v_flat = []
        x_flat = []
        a_flat = []
        bcs_start = np.cumsum([0] + self.pde.num_bcs)
        for i, _ in enumerate(self.pde.num_bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            v_flat.append(v_struct[:, beg:end, :].reshape(-1, v_struct.shape[-1]))
            x_flat.append(x_struct[:, beg:end, :].reshape(-1, x_struct.shape[-1]))
            a_flat.append(a_struct[:, beg:end, :].reshape(new_a_shape))
        v_flat.append(v_struct[:, bcs_start[-1]:, :].reshape(-1, v_struct.shape[-1]))
        x_flat.append(x_struct[:, bcs_start[-1]:, :].reshape(-1, x_struct.shape[-1]))
        a_flat.append(a_struct[:, bcs_start[-1]:, :].reshape(new_a_shape))
        v_flat = np.concatenate(v_flat, axis=0)
        x_flat = np.concatenate(x_flat, axis=0)
        a_flat = np.concatenate(a_flat, axis=0)
        return v_flat, x_flat, a_flat

    def gen_inputs(self, func_feats, func_vals, points):
        # n_func, n_pnt, n_feat
        v = np.tile(func_vals[:, None, :], (1, len(points), 1))
        # n_func, n_pnt, n_dim
        x = np.tile(points[None, :, :], (len(func_feats), 1, 1))
        # n_func, n_pnt, [n_comp]
        vx = self.func_space.eval_batch(func_feats, points[:, self.func_vars])
        assert len(vx) == len(func_feats), \
            "[caveat] function space must not change the first dimension"
        return v, x, vx

    def bc_inputs(self, func_feats, func_vals):
        config = dde.data.pde_operator.config
        if not self.pde.bcs:
            self.train_bc = (
                np.empty((len(func_feats), 0, len(self.eval_pts)), dtype=config.real(np)),
                np.empty((len(func_feats), 0, self.pde.geom.dim), dtype=config.real(np)),
                np.empty((len(func_feats), 0), dtype=config.real(np)),
            )
            return self.train_bc
        v, x, vx = [], [], []
        bcs_start = np.cumsum([0] + self.pde.num_bcs)
        for i, _ in enumerate(self.pde.num_bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            vi, xi, vxi = self.gen_inputs(
                func_feats, func_vals, self.pde.train_x_bc[beg:end]
            )
            v.append(vi)
            x.append(xi)
            vx.append(vxi)
        self.train_bc = (np.concatenate(v, axis=1),
                         np.concatenate(x, axis=1),
                         np.concatenate(vx, axis=1))
        return self.train_bc


class PDEOperatorZCS(PDEOperatorBatch):
    """
    Derived PDEOperator that supports ZCS
    """

    def __init__(self, pde, function_space, evaluation_points, num_function,
                 function_variables=None, num_test=None, batch_size=None):
        super().__init__(pde, function_space, evaluation_points, num_function,
                         function_variables, num_test, batch_size)

    def losses(self, targets, outputs, loss_fn, inputs, model, zcs_scalars=None):
        # caller must send zcs_scalars
        assert zcs_scalars is not None
        bkd = dde.backend

        f = []
        if self.pde.pde is not None:
            f = self.pde.pde(zcs_scalars, outputs, model.net.auxiliary_vars)
            if not isinstance(f, (list, tuple)):
                f = [f]

        bcs_start = np.cumsum([0] + self.num_bcs)
        error_f = [fi[bcs_start[-1]:] for fi in f]
        losses = [loss_fn(bkd.zeros_like(error), error) for error in error_f]
        for i, bc in enumerate(self.pde.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            # The same BC points are used for training and testing.
            error = bc.error(
                self.train_x[1],
                inputs[1],
                outputs,
                beg,
                end,
                aux_var=model.net.auxiliary_vars,
            )
            losses.append(loss_fn(bkd.zeros_like(error), error))
        return losses

    def losses_train(self, targets, outputs, loss_fn, inputs, model, zcs_scalars=None):
        # caller must send zcs_scalars
        assert zcs_scalars is not None
        return self.losses(targets, outputs, loss_fn, inputs, model, zcs_scalars=zcs_scalars)

    def losses_test(self, targets, outputs, loss_fn, inputs, model, zcs_scalars=None):
        # caller must send zcs_scalars
        assert zcs_scalars is not None
        return self.losses(targets, outputs, loss_fn, inputs, model, zcs_scalars=zcs_scalars)


class PDEOperatorCartesianProdZCS(dde.data.PDEOperatorCartesianProd):
    """
    Derived PDEOperatorCartesianProd that supports ZCS
    """

    def __init__(self, pde, function_space, evaluation_points, num_function,
                 function_variables=None, num_test=None, batch_size=None):
        super().__init__(pde, function_space, evaluation_points, num_function,
                         function_variables, num_test, batch_size)

    def _losses(self, outputs, loss_fn, inputs, model, num_func,
                zcs_scalars=None):
        # caller must send zcs_scalars
        assert zcs_scalars is not None
        bkd = dde.backend

        #########################################################################
        # thanks to ZCS, the following part is mostly copied from PDEOperatorZCS
        # the outstanding advantage is that we have avoided the branch for-loop

        # PDE
        f = []
        if self.pde.pde is not None:
            f = self.pde.pde(zcs_scalars, outputs, model.net.auxiliary_vars)
            if not isinstance(f, (list, tuple)):
                f = [f]
        bcs_start = np.cumsum([0] + self.pde.num_bcs)
        error_f = [fi[:, bcs_start[-1]:] for fi in f]
        losses = [loss_fn(bkd.zeros_like(error), error) for error in error_f]

        # BC
        for i, bc in enumerate(self.pde.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            error = bc.error(
                self.train_x[1],
                inputs[1],
                # add component dim for CollectiveDirichletBC and CollectiveIC
                outputs[:, :, None],
                beg,
                end,
                aux_var=model.net.auxiliary_vars,
            )
            losses.append(loss_fn(bkd.zeros_like(error), error))
        #########################################################################
        return losses

    def losses_train(self, targets, outputs, loss_fn, inputs, model,
                     zcs_scalars=None):
        # caller must send zcs_scalars
        assert zcs_scalars is not None
        num_func = self.num_func if self.batch_size is None else self.batch_size
        return self._losses(outputs, loss_fn, inputs, model, num_func,
                            zcs_scalars=zcs_scalars)

    def losses_test(self, targets, outputs, loss_fn, inputs, model,
                    zcs_scalars=None):
        # caller must send zcs_scalars
        assert zcs_scalars is not None
        return self._losses(outputs, loss_fn, inputs, model, len(self.test_x[0]),
                            zcs_scalars=zcs_scalars)


class CollectiveDirichletBC(dde.icbc.DirichletBC):
    """ derived DirichletBC that supports collective computation """

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, func, on_boundary, component)

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        return outputs[:, beg:end, self.component:self.component + 1] - values


class CollectiveIC(dde.icbc.IC):
    """ derived IC that supports collective computation """

    def __init__(self, geom, func, on_initial, component=0):
        super().__init__(geom, func, on_initial, component)

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        return outputs[:, beg:end, self.component:self.component + 1] - values


class PointSetOperatorBCUsingAux(dde.icbc.PointSetOperatorBC):
    """ derived PointSetOperatorBC using auxiliary variable """

    def __init__(self, points, values, func):
        super().__init__(points, values, func)

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        return self.func(inputs, outputs, aux_var, beg, end) - self.values
