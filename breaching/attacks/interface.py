"""Load attacker code and instantiate appropriate objects."""


from .optimization_based_attack import OptimizationBasedAttack
from .analytic_attack import AnalyticAttacker

def prepare_attack(model, loss, cfg_attack, setup):
    if cfg_attack.type in ['invertinggradients', 'beyond-infering', 'deep-leakage', 'see-through-gradients']:
        attacker = OptimizationBasedAttack(model, loss, cfg_attack, setup)
    elif cfg_attack.type == 'analytic':
        attacker = AnalyticAttacker(model, loss, cfg_attack, setup)
    else:
        raise ValueError('Invalid attacker')

    return attacker
