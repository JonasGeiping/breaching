# Modular Attacks against privacy in federated learning scenarios.

## Repo Setup:
Subfolders:
- config   -> contains all configurations
- cases    -> code to train and present neural networks to the attacker, provides "user" and "server" objects
- attacks  -> implementations of attacks

Design Tenets:
- Attacks should be as case-agnostic as possible, based on transmission of captured user data and server data, e.g. gradients
- To implement a new use-case or threat model should only require creation of a new "case" with the appropriate config and possible novel code
- To implement a new attack (or attack variation) should only require new attack code and configuration

Package Requirements:
- PyTorch 1.9.*
- hydra-core 1.1  [install from pip]
- lpips  [install from pip]
- in the future probably a cool FL package
