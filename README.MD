# Cool attacks against privacy in federated learning scenarios.

## Repo Setup:
Subfolders:
- config   -> contains all configurations
- cases    -> code to train and present neural networks to the attacker
- attacks  -> implementations of attacks
- scripts  -> shell scripts that contain experimental evaluations of attacks and cases
- _models   -> pure data folder to store prepared models for convenience reasons [not in .git]

Requirements:
- Attacks should be as case-agnostic as possible, based on only a single (for now) transmission of captured user data, e.g. gradients, and the model state
- To implement a new use-case or threat model should only require creation of a new "case" with the appropriate config and possible novel code
- To implement a new attack (or attack variation) should only require new attack code and configuration
