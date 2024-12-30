import torch

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

from .DG.AbstractCAD import AbstractCAD
from .DG.AbstractDANN import AbstractDANN
from .DG.ERM import ERM
from .DG.IRM import IRM
from .DG.Mixup import Mixup
from .DG.CDANN import CDANN
from .DG.SagNet import SagNet
from .DG.IB_ERM import IB_ERM
from .DG.CondCAD import CondCAD
from .DG.EQRM import EQRM
from .MML.LFM import LFM
from .MML.OGM import OGM
from .MML.Concat import Concat

ALGORITHMS = [
    # OOD Algorithms >>> 
    "ERM",
    "IRM",
    "Mixup",
    "CDANN",
    "SagNet",
    "IB_ERM",
    "CondCAD",
    "EQRM",
    # OOD Algorithms <<< 
    
    # MM Algorithms >>>
    "OGM",
    "LFM",
    "Concat"
    # MM Algorithms <<<
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

