import numpy as np
from deepspeed.utils import log_dist


class ProgressiveLayerDrop(object):
    r""" Progressive Layer Dropping (PLD) for model training.
        This implements the PLD technique for compressed model training
        from this paper: https://arxiv.org/pdf/2010.13369.pdf
    Args:
        theta (float): a hyper-parameter that controls the trade-off between training time and robustness.
        The lower the theta value, the faster the training speed. Default value: 0.5.
        gamma (float): a hyper-parameter that controls how fast the drop ratio increases. Default value: 0.001.
    """
    def __init__(self, theta=0.5, gamma=0.001):
        super().__init__()

        self.theta = theta
        self.gamma = gamma
        self.current_theta = 1.0
        log_dist(f'Enabled progressive layer dropping (theta = {self.theta})', ranks=[0])

    def get_state(self):
        kwargs = {'progressive_layer_drop': True, 'pld_theta': self.get_theta()}
        return kwargs

    def get_theta(self):
        return self.current_theta

    def update_state(self, global_step):
        def _prob(x, gamma, p):
            return (1. - p) * np.exp(-gamma * x) + p

        self.current_theta = _prob(global_step, self.gamma, self.theta)
