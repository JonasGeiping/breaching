"""Implementation for basic gradient inversion attacks.

This covers optimization-based reconstruction attacks as in Wang et al. "Beyond Infer-
ring Class Representatives: User-Level Privacy Leakage From Federated Learning."
and convers subsequent developments such as
* Zhu et al., "Deep Leakage from gradients",
* Geiping et al., "Inverting Gradients - How easy is it to break privacy in FL"
* ?
"""

import torch
import time

from .base_attack import _BaseAttacker
from .auxiliaries.objectives_and_regularizers import regularizer_lookup, Euclidean, CosineSimilarity, TotalVariation


class OptimizationBasedAttack(_BaseAttacker):
    """Implements a wide spectrum of optimization-based attacks."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device('cpu'))):
        super().__init__(model, loss_fn, cfg_attack, setup)
        if self.cfg.objective.type == 'cosine-similarity':
            self.objective = CosineSimilarity(self.cfg.objective.scale)
        else:
            self.objective = Euclidean(self.cfg.objective.scale)
        self.regularizers = []
        for key in self.cfg.regularization.keys():
            if self.cfg.regularization[key].scale > 0:
                self.regularizers += [regularizer_lookup[key](setup, **self.cfg.regularization[key])]

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        # Main reconstruction loop starts here:
        scores = torch.zeros(self.cfg.restarts.num_trials)
        candidate_solutions = []
        try:
            for trial in range(self.cfg.restarts.num_trials):
                candidate_solutions += [self._run_trial(rec_models, shared_data, labels, stats, trial, dryrun)]
                scores[trial] = self._score_trial(candidate_solutions[trial], labels, rec_models, shared_data)
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass
        optimal_solution = self._select_optimal_reconstruction(candidate_solutions, scores, stats)
        reconstructed_data = dict(data=optimal_solution, labels=labels)

        return reconstructed_data, stats

    def _run_trial(self, rec_model, shared_data, labels, stats, trial, dryrun=False):
        """Run a single reconstruction trial."""
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model)

        candidate = self._initialize_data([shared_data['num_data_points'], *self.data_shape])
        optimizer, scheduler = self._init_optimizer(candidate)
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):

                closure = self._objective_function(candidate, labels, rec_model, optimizer, shared_data)
                objective_value = optimizer.step(closure)

                scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.cfg.optim.boxed:
                        candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)

                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    print(f'It: {iteration + 1}. Rec. loss: {objective_value.item():2.4f}. T: {timestamp - current_wallclock:4.2f}s')
                    current_wallclock = timestamp

                if not torch.isfinite(objective_value):
                    print(f'Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!')
                    break

                stats[f'Trial_{trial}_Val'].append(objective_value.item())

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass

        return candidate.detach()

    def _objective_function(self, candidate, labels, rec_model, optimizer, shared_data):
        def closure():
            optimizer.zero_grad()

            total_objective = 0
            for model, shared_grad in zip(rec_model, shared_data['gradients']):
                model.zero_grad()
                spoofed_loss = self.loss_fn(model(candidate), labels)
                gradient = torch.autograd.grad(spoofed_loss, model.parameters(), create_graph=True)

                total_objective += self.objective(gradient, shared_grad)

            for regularizer in self.regularizers:
                total_objective += regularizer(candidate)
            total_objective.backward()

            if self.cfg.optim.signed:
                candidate.grad.sign_()
            if self.cfg.optim.langevin_noise > 0:
                candidate.grad += self.cfg.optim.langevin_noise * torch.randn_like(candidate.grad)
            return total_objective
        return closure

    def _score_trial(self, candidate, labels, rec_model, shared_data):
        """Score candidate solutions based on some criterion."""
        def _compute_objective_fast(candidate, objective):
            val = 0
            for model, shared_grad in zip(rec_model, shared_data['gradients']):
                model.zero_grad()
                spoofed_loss = self.loss_fn(model(candidate), labels)
                gradient = torch.autograd.grad(spoofed_loss, model.parameters(), create_graph=False)
                val += objective(gradient, shared_grad)
            return val

        if self.cfg.restarts.scoring == 'euclidean':
            score = _compute_objective_fast(candidate, Euclidean())
        elif self.cfg.restarts.scoring == 'cosine-similarity':
            score = _compute_objective_fast(candidate, CosineSimilarity())
        elif self.cfg.restarts.scoring in ['TV', 'total-variation']:
            score = TotalVariation(scale=1.0)(candidate)
        else:
            raise ValueError(f'Scoring mechanism {self.cfg.scoring} not implemented.')
        return score if score.isfinite() else float('inf')

    def _select_optimal_reconstruction(self, candidate_solutions, scores, stats):
        """Choose one of the candidate solutions based on their scores (for now).

        More complicated combinations are possible in the future."""
        optimal_val, optimal_index = torch.min(scores, dim=0)
        optimal_solution = candidate_solutions[optimal_index]
        stats['opt_value'] = optimal_val.item()
        if optimal_val.isfinite():
            print(f'Optimal condidate solution with rec. loss {optimal_val.item():2.4f} selected.')
            return optimal_solution
        else:
            print('No valid reconstruction could be found.')
            return torch.zeros_like(optimal_solution)
