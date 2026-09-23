"""Bounded, opt-in research observations; no numerical admission or timing claim."""
from collections import deque

import numpy as np
import torch

from benchmarks import qualify_cuda as common
from clipp1d.cuda.solver import inner_certificate_diagnostics


def save_tensors(journal, name, tensors):
    """Preserve binary64 bits (including nonfinite values), then bind readback."""
    arrays = {key: value.detach().cpu().numpy().copy()
              for key, value in tensors.items() if value is not None}
    path = journal.root / name
    with path.open('xb') as stream:
        np.savez(stream, **arrays)
    with np.load(path, allow_pickle=False) as restored:
        if set(restored.files) != set(arrays) or any(
                restored[key].dtype != value.dtype or restored[key].shape != value.shape
                or restored[key].tobytes() != value.tobytes() for key, value in arrays.items()):
            raise AssertionError('Research tensor archive failed exact readback')
    journal.bind(path)
    return dict(path=str(path.relative_to(journal.receipt_path.parent)),
                sha256=common.sha(path),
                tensors={key: None if value is None else dict(
                    shape=list(value.shape), dtype=str(value.dtype),
                    tensor_sha256=common.tensor_sha(value)) for key, value in tensors.items()})


class SurrogateTrace:
    """Retain first 128 and last 16 trials per start, plus every failed QP.

    The fixture path/start/iteration budgets bound the number of starts and
    failures. Counts explicitly identify gaps in the retained trial history.
    Timing includes device synchronization, copies and artifact writes.
    """
    def __init__(self, journal, index, lam, policy, *, head_limit=128, tail_limit=16):
        self.journal, self.index, self.policy = journal, index, policy
        self.lam = float(lam)
        self.head_limit, self.tail = head_limit, deque(maxlen=tail_limit)
        self.head, self.failures = [], []
        self.seen = self.accepted = self.rejected = 0

    def __call__(self, event, state):
        inflation = state['inflation']
        row = dict(event=event, iteration=state['iteration'],
                   backtrack_index=state['backtrack_index'],
                   inflation=inflation.detach().cpu().tolist()
                   if isinstance(inflation, torch.Tensor) else inflation,
                   base_curvature_range=[float(state['base_h'].min()), float(state['base_h'].max())],
                   curvature_range=[float(state['h'].min()), float(state['h'].max())])
        if event == 'unresolved_qp':
            fitted = state['fitted']
            tensors = {key: state[key] for key in
                       ('h', 'target', 'lower', 'upper', 'caps', 'start', 'dual', 'base_h')}
            tensors.update(returned_x=fitted.x, returned_dual=fitted.dual,
                           inflation=state['h'].new_tensor(inflation)
                           if not isinstance(inflation, torch.Tensor) else inflation)
            row.update(problem_and_returned_state=save_tensors(
                self.journal, f'start-{self.index:03d}-failed-qp.npz', tensors),
                certificate=inner_certificate_diagnostics(fitted, self.policy),
                qp_iterations=fitted.iterations)
            self.failures.append(row)
            self.journal.artifact(f'start-{self.index:03d}-failed-qp.json', row)
            return
        major = state['losses'] + state['gradient_step'] + state['quadratic_step']
        margin = 64 * torch.finfo(major.dtype).eps * (
            1 + state['losses'].abs() + state['gradient_step'].abs()
            + state['quadratic_step'].abs() + state['trial_losses'].abs())
        mask = (~torch.isfinite(state['trial_losses']) | ~torch.isfinite(major)
                | ~torch.isfinite(margin) | (state['trial_losses'] > major + margin))
        row.update(sequence=self.seen, accepted=state['accepted'],
                   finite_gate=state['finite_gate'],
                   rejected_trial_mask=None if state['accepted'] else mask.cpu().tolist(),
                   global_fallback=not state['accepted'] and not bool(mask.any()),
                   **{key: float(state[key]) for key in
                      ('margin', 'majorization_slack', 'surrogate_slack', 'objective_slack')})
        self.seen += 1
        self.accepted += state['accepted']
        self.rejected += not state['accepted']
        if len(self.head) < self.head_limit:
            self.head.append(row)
        else:
            self.tail.append(row)

    def finish(self):
        rows = self.head + list(self.tail)
        return self.journal.artifact(f'start-{self.index:03d}-surrogate-trace.json', dict(
            lambda_value=self.lam, start_index=self.index,
            observed_trials=self.seen, accepted_trials=self.accepted, rejected_trials=self.rejected,
            retained_trials=len(rows), omitted_trials=self.seen-len(rows),
            head_limit=self.head_limit, tail_limit=self.tail.maxlen,
            trials=rows, unresolved_qps=self.failures,
            scope='Diagnostic observations only; original numerical gates unchanged'))
