from enum import Enum

# define optuna distribution pattern for params
class Distribution(Enum):
    CHOICE = 0
    UNIFORM = 1
    INTUNIFORM = 2
    QUNIFORM = 3
    LOGUNIFORM = 4
    DISCRETEUNIFORM = 5
    NORMAL = 6
    QNORMAL = 7
    LOGNORMAL = 8

OPTUNA_DISTRIBUTIONS_MAP = {Distribution.CHOICE: "suggest_categorical",
                            Distribution.UNIFORM: "suggest_uniform",
                            Distribution.LOGUNIFORM: "suggest_loguniform",
                            Distribution.INTUNIFORM: "suggest_int",
                            Distribution.DISCRETEUNIFORM: "suggest_discrete_uniform"}

class SearchSpace:
    distribution_type: Distribution = None
    params: dict = {}

    def __init__(self, distribution_type: Distribution, *args, **kwargs):
        self.distribution_type = distribution_type
        self.params = kwargs