"""Compact, explicit identities for independently qualified partition ancestry."""
import hashlib


def membership_hash(labels):
    return hashlib.sha256(labels.detach().cpu().numpy().astype('<i8').tobytes()).hexdigest()


def refit_identity(refit):
    upper = float(refit.score)
    gap = float(refit.gap)
    return dict(membership_sha256=membership_hash(refit.labels),
                n=int(refit.labels.numel()), k=int(refit.centers.numel()),
                score=upper, score_lower=upper-2*gap, score_upper=upper,
                refit_gap=gap, refit_qualified=True)


def ancestry_step(operation, parent, child, details, minimum_decrease):
    before, after = refit_identity(parent), refit_identity(child)
    margin = max(minimum_decrease, 128 * 2.220446049250313e-16 * (1+abs(before['score'])))
    return dict(operation=operation, parent=before, child=after, details=details,
                published_score_improved=after['score'] < before['score']-margin,
                refit_order_certified=after['score_upper'] < before['score_lower']-margin,
                comparison_margin=margin)
