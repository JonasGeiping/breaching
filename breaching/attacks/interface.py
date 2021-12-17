"""Load attacker code and instantiate appropriate objects."""


from .optimization_based_attack import OptimizationBasedAttacker
from .multiscale_optimization_attack import MultiScaleOptimizationAttacker
from .analytic_attack import AnalyticAttacker, ImprintAttacker
from .recursive_attack import RecursiveAttacker


def prepare_attack(model, loss, cfg_attack, setup):
    if cfg_attack.attack_type == "optimization":
        attacker = OptimizationBasedAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "multiscale":
        attacker = MultiScaleOptimizationAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "analytic":
        attacker = AnalyticAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "imprint-readout":
        attacker = ImprintAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "recursive":
        attacker = RecursiveAttacker(model, loss, cfg_attack, setup)
    else:
        raise ValueError(f"Invalid type of attack {cfg_attack.attack_type} given.")

    return attacker
