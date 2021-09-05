"""Implementation for basic gradient inversion attacks.

This covers optimization-based reconstruction attacks as in Wang et al. "Beyond Infer-
ring Class Representatives: User-Level Privacy Leakage From Federated Learning."
and convers subsequent developments such as
* Zhu et al., "Deep Leakage from gradients",
* Geiping et al., "Inverting Gradients - How easy is it to break privacy in FL"
* ?
"""

import torch
from collections import defaultdict

from .base_attacker import _BaseAttacker
from .utils import CosineSimilarity, Euclidean, TotalVariation


class OptimizationBasedAttack(_BaseAttacker):
    """Implements a wide spectrum of optimization-based attacks."""

    def __init__(model, loss, cfg_attack, setup=dict(dtype=torch.float, device=torch.device('cpu'))):
        self.cfg = cfg_attack
        self.setup = setup

        self.model_template = copy.deepcopy(model)
        self.loss_fn = copy.deepcopy(loss)

        self.objective = CosineSimilarity() if self.cfg.objective == 'cosine-similarity' else Euclidean()
        if self.cfg.regularization.total_variation > 0:
            self.regularizer = TotalVariation(scale=self.cfg.regularization.total_variation)
        else:
            self.regularizer = None

    def reconstruct(self, server_payload, shared_data, dryrun=False):
        # Initialize stats module for later usage:
        stats = defaultdict(list)

        # Load preprocessing constants:
        self.dm = torch.as_tensor(server_payload['data'].mean, **self.setup)[None, :, None, None]
        self.ds = torch.as_tensor(server_payload['data'].std, **self.setup)[None, :, None, None]
        self.data_shape = server_payload['data'].shape

        # Load server_payload into state:
        rec_model = self._construct_models_from_payload(server_payload)

        # Consider label information
        if shared_data['labels'] is None:
            labels = self._recover_label_information(shared_data)
        else:
            labels = shared_data['labels']

        # Main reconstruction loop starts here:
        scores = torch.zeros(self.cfg.restarts.num_trials)
        candidate_solutions = []
        for trial in range(self.cfg.restarts.num_trials):
            candidate_solutions += [self._run_trial(rec_model, shared_data, labels, stats, trial, dryrun)]
            scores[trial] = self._score_trial(candidate_solutions[trial], labels, rec_model, shared_data)

        optimal_solution = self._select_optimal_reconstruction(candidate_solutions, scores, stats)

        return reconstructed_data, stats

    def _run_trial(self, rec_model, shared_data, labels, stats, trial, dryrun=False):
        """Run a single reconstruction trial."""

        candidate = self._initialize_data([shared_data['num_data_points'], *self.data_shape])
        optimizer, scheduler = self._init_optimizer(candidate)

        for iteration in range(cfg.optim.max_iterations):

            closure = self._objective_function(self, candidate, labels, rec_model, optimizer, shared_data)
            objective_value = optimizer.step(closure)

            scheduler.step()

            with torch.no_grad():
                # Project into image space
                if self.cfg.optim.boxed:
                    candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)

            if iteration + 1 == max_iterations or iteration % 100 == 0:
                print(f'It: {iteration + 1}. Rec. loss: {objective_value.item():2.4f}.')

            if not torch.isfinite(objective_value):
                print(f'Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!')
                break

            stats[f'Trial_{trial}_Val'].append(objective_value.item())

            if dryrun:
                break

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

            if self.regularizer is not None:
                total_objective += self.regularizer(candidate)
            total_objective.backward()

            if self.cfg.optim.signed:
                candidate.grad.sign_()
            return total_objective
        return closure

    def _score_trial(self, candidate, labels, rec_model, shared_data):
        """Score candidate solutions based on some criterion."""
        def _compute_objective_fast(candidate, obective):
            objective = 0
            for model, shared_grad in zip(rec_model, shared_data['gradients']):
                model.zero_grad()
                spoofed_loss = self.loss_fn(model(candidate), labels)
                gradient = torch.autograd.grad(spoofed_loss, model.parameters(), create_graph=False)
            return gradient

        if self.cfg.scoring == 'euclidean':
            score = _compute_objective_fast(candidate, Euclidean())
        elif self.cfg.scoring == 'cosine-similarity':
            score = _compute_objective_fast(candidate, CosineSimilarity())
        elif self.cfg.scoring in ['TV', 'total-variation']:
            score = TotalVariation(scale=1.0)(candidate)
        else:
            raise ValueError(f'Scoring mechanism {self.cfg.scoring} not implemented.')
        return score if scores.isfinite() else float('inf')

    def _select_optimal_reconstruction(self, candidate_solutions, scores, stats):
        """Choose one of the candidate solutions based on their scores (for now).

        More complicated combinations are possible in the future."""
        optimal_index = torch.argmin(scores)
        optimal_solution = candidate_solutions[optimal_index]
        return optimal_solution
