"""Bounded likelihood batches for the existing ordered finite audit proposals."""

from collections import OrderedDict
from dataclasses import dataclass
import struct

import numpy as np

from .model import loss_at_rows


MAX_BATCH_PROPOSALS = 64
MAX_BATCH_ROWS = 4096
MAX_CACHE_ENTRIES = 4096


@dataclass(frozen=True)
class FiniteProposal:
    start: int
    stop: int
    value: object
    margin: float
    step: float | None = None

    def key(self):
        value = self.value
        if np.ndim(value):
            if not np.all(value == value[0]):
                return None
            value = value[0]
        # Bit identity preserves signed zero and avoids any approximate merging.
        return self.start, self.stop, struct.pack("=d", float(value))


class ProposalLossMemo:
    """Bounded memo, owned by one immutable audit's model/x/loss quantities.

    Only likelihood deltas are stored: anchor boxes and incident TV penalties
    are never cached. Mutating this memo cannot mutate the audit's arrays.
    """

    __slots__ = ("_values",)

    def __init__(self):
        self._values = OrderedDict()

    def get(self, key):
        result = self._values.get(key)
        if result is not None:
            self._values.move_to_end(key)
        return result

    def remember(self, key, value):
        if key is None:
            return
        self._values[key] = value
        self._values.move_to_end(key)
        if len(self._values) > MAX_CACHE_ENTRIES:
            self._values.popitem(last=False)


def _batch_deltas(context, batch):
    """Evaluate unique misses, retaining scalar per-interval reduction order."""
    entries, missing, positions = [], [], {}
    for proposal in batch:
        key = proposal.key()
        cached = context.proposal_losses.get(key) if key is not None else None
        if cached is not None:
            entries.append((cached, None))
        elif key is not None and key in positions:
            entries.append((None, positions[key]))
        else:
            index = len(missing)
            entries.append((None, index))
            if key is not None:
                positions[key] = index
            missing.append((proposal, key))
    deltas = []
    if missing:
        size = sum(p.stop - p.start for p, _ in missing)
        rows = np.empty(size, dtype=np.intp)
        values = np.empty(size, dtype=np.float64)
        cursor = 0
        for proposal, _ in missing:
            end = cursor + proposal.stop - proposal.start
            rows[cursor:end] = np.arange(proposal.start, proposal.stop)
            values[cursor:end] = proposal.value
            cursor = end
        losses = loss_at_rows(context.model, rows, values)
        cursor = 0
        for proposal, key in missing:
            end = cursor + proposal.stop - proposal.start
            # Do not sum a padded matrix or subtract whole-array prefixes: this
            # is the original contiguous scalar interval reduction, bit for bit.
            delta = float(np.sum(losses[cursor:end] - context.losses[proposal.start:proposal.stop]))
            deltas.append(delta)
            context.proposal_losses.remember(key, delta)
            cursor = end
    return [(proposal, cached if index is None else deltas[index])
            for proposal, (cached, index) in zip(batch, entries)]


def iter_proposal_deltas(context, proposals, *, initial_batch_size=MAX_BATCH_PROPOSALS):
    """Yield in proposal order, with bounded lookahead and bounded row storage.

    A single interval longer than MAX_BATCH_ROWS is evaluated alone; otherwise
    each likelihood call contains at most that many mutation rows. Thus storage
    is O(max(M, MAX_BATCH_ROWS) * multiplicity_count), never all intervals at once.
    Directional backtracking may request a first batch of one to keep the common
    immediate-acceptance path cheap. No acceptance decisions occur here.
    """
    proposals = iter(proposals)
    pending = None
    batch_limit = initial_batch_size
    while True:
        batch, rows = [], 0
        while len(batch) < batch_limit:
            proposal = pending if pending is not None else next(proposals, None)
            pending = None
            if proposal is None:
                break
            count = proposal.stop - proposal.start
            if batch and rows + count > MAX_BATCH_ROWS:
                pending = proposal
                break
            batch.append(proposal)
            rows += count
            if rows >= MAX_BATCH_ROWS:
                break
        if not batch:
            return
        yield from _batch_deltas(context, batch)
        batch_limit = MAX_BATCH_PROPOSALS
