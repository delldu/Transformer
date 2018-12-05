'''A wrapper class for optimizer '''
import numpy as np
import pdb

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

        # pdb.set_trace()
        # (Pdb) a
        # self = <transformer.Optim.ScheduledOptim object at 0x7fb960e680f0>
        # optimizer = Adam (
        # Parameter Group 0
        #     amsgrad: False
        #     betas: (0.9, 0.98)
        #     eps: 1e-09
        #     lr: 0.001
        #     weight_decay: 0
        # )
        # d_model = 512
        # n_warmup_steps = 4000

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()
        # pdb.set_trace()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

