"""
Model extensions
"""

import deepxde as dde
import deepxde.gradients as grad
from deepxde.backend import torch

from deepxde_extensions.timer import ContextTimer


def trunk_inputs_to_ZCS(trunk_inputs):
    """ change trunk inputs to ZCS format """
    # ZCS is needed only when grad is needed
    if not trunk_inputs.requires_grad:
        return trunk_inputs
    # create ZCS scalars
    n_dim_crds = trunk_inputs.shape[1]
    zcs_scalars = [torch.as_tensor(0.).requires_grad_()
                   for _ in range(n_dim_crds)]
    # disable inputs grad
    trunk_inputs.requires_grad_(False)
    # add ZCS to inputs
    for i_dim in range(n_dim_crds):
        trunk_inputs[:, i_dim] += zcs_scalars[i_dim]
    return trunk_inputs, zcs_scalars


class Model(dde.Model):
    """
    Derived Model class with timers
    """

    def __init__(self, data, net):
        super().__init__(data, net)
        self.wct_dict = {
            'inputs': 0.,
            'forward': 0.,
            'losses': 0.,
            'train_step': 0.,
        }
        self.timer_activated = False

    def activate_timer(self, on_off):
        self.timer_activated = on_off

    def _compile_pytorch(self, lr, loss_fn, decay, loss_weights):
        """ implemented only for pytorch at this moment """
        super()._compile_pytorch(lr, loss_fn, decay, loss_weights)

        def outputs_losses(training, inputs, targets, auxiliary_vars, losses_fn):
            with ContextTimer(self.wct_dict, 'inputs', self.timer_activated):
                self.net.auxiliary_vars = None
                if auxiliary_vars is not None:
                    self.net.auxiliary_vars = torch.as_tensor(auxiliary_vars)
                self.net.train(mode=training)
                if isinstance(inputs, tuple):
                    inputs = tuple(
                        map(lambda x: torch.as_tensor(x).requires_grad_(), inputs)
                    )
                else:
                    inputs = torch.as_tensor(inputs)
                    inputs.requires_grad_()

            with ContextTimer(self.wct_dict, 'forward', self.timer_activated):
                outputs_ = self.net(inputs)

            with ContextTimer(self.wct_dict, 'losses', self.timer_activated):
                # Data losses
                if targets is not None:
                    targets = torch.as_tensor(targets)
                losses = losses_fn(targets, outputs_, loss_fn, inputs, self)

                if not isinstance(losses, list):
                    losses = [losses]
                losses = torch.stack(losses)
                # Weighted losses
                if loss_weights is not None:
                    losses *= torch.as_tensor(loss_weights)
            # Clear cached Jacobians and Hessians.
            grad.clear()
            return outputs_, losses

        def outputs_losses_train(inputs, targets, auxiliary_vars):
            return outputs_losses(
                True, inputs, targets, auxiliary_vars, self.data.losses_train
            )

        def outputs_losses_test(inputs, targets, auxiliary_vars):
            return outputs_losses(
                False, inputs, targets, auxiliary_vars, self.data.losses_test
            )

        def train_step(inputs, targets, auxiliary_vars):
            def closure():
                losses = outputs_losses_train(inputs, targets, auxiliary_vars)[1]
                total_loss = torch.sum(losses)
                self.opt.zero_grad()
                total_loss.backward()
                return total_loss

            with ContextTimer(self.wct_dict, 'train_step', self.timer_activated):
                self.opt.step(closure)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        #######################
        # overwrite callables #
        #######################
        self.outputs_losses_train = outputs_losses_train
        self.outputs_losses_test = outputs_losses_test
        self.train_step = train_step


class ModelZCS(dde.Model):
    """
    Derived Model class that supports ZCS (and timers)
    """

    def __init__(self, data, net):
        super().__init__(data, net)
        # ZCS scalars
        self.zcs_scalars = None

        # timers
        self.wct_dict = {
            'inputs': 0.,
            'forward': 0.,
            'losses': 0.,
            'train_step': 0.,
        }
        self.timer_activated = False

    def activate_timer(self, on_off):
        self.timer_activated = on_off

    def _compile_pytorch(self, lr, loss_fn, decay, loss_weights):
        """ implemented only for pytorch at this moment """
        super()._compile_pytorch(lr, loss_fn, decay, loss_weights)

        def outputs_ZCS(training, inputs):
            self.net.train(mode=training)
            with torch.no_grad():
                if isinstance(inputs, tuple):
                    inputs = list(
                        map(lambda x: torch.as_tensor(x).requires_grad_(), inputs)
                    )
                    ##############
                    # create ZCS #
                    ##############
                    inputs[1], self.zcs_scalars = trunk_inputs_to_ZCS(inputs[1])
                    inputs = tuple(inputs)
                else:
                    inputs = torch.as_tensor(inputs)
                    inputs.requires_grad_()
                    ##############
                    # create ZCS #
                    ##############
                    inputs, self.zcs_scalars = trunk_inputs_to_ZCS(inputs)
            grad.clear()
            return self.net(inputs)

        def outputs_losses_ZCS(training, inputs, targets, auxiliary_vars, losses_fn):
            with ContextTimer(self.wct_dict, 'inputs', self.timer_activated):
                self.net.auxiliary_vars = None
                if auxiliary_vars is not None:
                    self.net.auxiliary_vars = torch.as_tensor(auxiliary_vars)
                self.net.train(mode=training)
                if isinstance(inputs, tuple):
                    inputs = list(
                        map(lambda x: torch.as_tensor(x).requires_grad_(), inputs)
                    )
                    ##############
                    # create ZCS #
                    ##############
                    inputs[1], self.zcs_scalars = trunk_inputs_to_ZCS(inputs[1])
                    inputs = tuple(inputs)
                else:
                    inputs = torch.as_tensor(inputs)
                    inputs.requires_grad_()
                    ##############
                    # create ZCS #
                    ##############
                    inputs, self.zcs_scalars = trunk_inputs_to_ZCS(inputs)

            with ContextTimer(self.wct_dict, 'forward', self.timer_activated):
                outputs_ = self.net(inputs)

            with ContextTimer(self.wct_dict, 'losses', self.timer_activated):
                # Data losses
                if targets is not None:
                    targets = torch.as_tensor(targets)
                ####################
                # send ZCS to loss #
                ####################
                losses = losses_fn(targets, outputs_, loss_fn, inputs, self,
                                   zcs_scalars=self.zcs_scalars)
                if not isinstance(losses, list):
                    losses = [losses]
                losses = torch.stack(losses)
                # Weighted losses
                if loss_weights is not None:
                    losses *= torch.as_tensor(loss_weights)
            # Clear cached Jacobians and Hessians.
            grad.clear()
            return outputs_, losses

        def outputs_losses_train_ZCS(inputs, targets, auxiliary_vars):
            return outputs_losses_ZCS(
                True, inputs, targets, auxiliary_vars, self.data.losses_train
            )

        def outputs_losses_test_ZCS(inputs, targets, auxiliary_vars):
            return outputs_losses_ZCS(
                False, inputs, targets, auxiliary_vars, self.data.losses_test
            )

        def train_step_ZCS(inputs, targets, auxiliary_vars):
            def closure():
                losses = outputs_losses_train_ZCS(inputs, targets, auxiliary_vars)[1]
                total_loss = torch.sum(losses)
                self.opt.zero_grad()
                total_loss.backward()
                return total_loss

            with ContextTimer(self.wct_dict, 'train_step', self.timer_activated):
                self.opt.step(closure)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        #######################
        # overwrite callables #
        #######################
        self.outputs = outputs_ZCS
        self.outputs_losses_train = outputs_losses_train_ZCS
        self.outputs_losses_test = outputs_losses_test_ZCS
        self.train_step = train_step_ZCS
