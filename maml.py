from torch.optim import Optimizer


class MAML(Optimizer):

    def __init__(self, params, base_lr, meta_lr):
        self.params = params
        self.base_lr = base_lr
        self.meta_lr = meta_lr

        defaults = {
            'params': params,
            'base_lr': base_lr,
            'meta_lr': meta_lr
        }
        super(MAML, self).__init__(params, defaults)

    def step(self, closure):
        pass