"""Load attacker code and instantiate appropriate objects."""


from .gradient_inversion import OptimizationBasedAttack

def prepare_attack(model, loss, cfg_attack, setup):
    if cfg_attack.type in ['invertinggradients', 'beyond-infering', 'deep-leakage', 'see-through-gradients']:
        attacker = OptimizationBasedAttack(model, loss, cfg_attack, setup)
    else:
        raise ValueError('Invalid attacker')

    return attacker
