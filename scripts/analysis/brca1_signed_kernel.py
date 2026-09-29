#!/usr/bin/env python3
"""Train-only normalized linear kernels for the frozen BRCA1 signed probe.

No data paths or phenotype loading. All transforms receive explicit training
indices. Raw Gram matrices contain only dot products, never global centering.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from scipy.linalg import eigh

LAMBDAS = np.array([1e-4, 1e-3, .01, .1, 1., 10.])


class FeatureGram:
    """Retain features for stable fallback when raw-moment subtraction cancels."""
    def __init__(self, values):
        self.values = np.array(values, dtype=np.float64, copy=True)
        self.values.flags.writeable = False
        self.matrix = self.values @ self.values.T

    def __getitem__(self, item):
        return self.matrix[item]


def raw_gram(values, unit_direction=False):
    x = np.asarray(values, dtype=np.float64)
    assert x.ndim == 2 and np.isfinite(x).all()
    if unit_direction:
        norm = np.linalg.norm(x, axis=1)
        x = np.divide(x, norm[:, None], out=np.zeros_like(x), where=norm[:, None] != 0)
    return FeatureGram(x)


def center_raw_gram(gram, train, test):
    """Equivalent to explicitly centering features with the training mean."""
    gtt = gram[np.ix_(train, train)]
    gxt = gram[np.ix_(test, train)]
    mean = gtt.mean(axis=0)
    overall = gtt.mean()
    tt = gtt - mean[None, :] - mean[:, None] + overall
    xt = gxt - gxt.mean(axis=1, keepdims=True) - mean[None, :] + overall
    tt = (tt + tt.T) * .5
    scale2 = float(np.trace(tt) / len(train))
    raw_energy = float(np.trace(gtt) / len(train))
    risky = scale2 <= max(raw_energy, 0.) * 1e-4 or scale2 <= 0
    if isinstance(gram, FeatureGram):
        a, b = gram.values[train], gram.values[test]
        constant = not np.any(np.ptp(a, axis=0))
        if constant:
            return np.zeros_like(tt), np.zeros_like(xt), False
        if risky:
            # Mathematically identical transform, without subtracting large
            # raw second moments. Never discard nonconstant features because
            # their raw Gram matrix has lost the small variance numerically.
            mean_feature = a.mean(axis=0)
            a, b = a - mean_feature, b - mean_feature
            scale2 = float(np.mean(np.sum(a*a, axis=1)))
            assert scale2 > 0 and np.isfinite(scale2)
            tt, xt = a @ a.T, b @ a.T
    elif risky:
        raise FloatingPointError('Raw Gram cancellation risk: original features required for explicit train-centered fallback')
    if scale2 <= 0:
        return np.zeros_like(tt), np.zeros_like(xt), False
    return tt / scale2, xt / scale2, True


def baseline_kernel(values, train, test):
    x = np.asarray(values, dtype=np.float64)
    a, b = x[train], x[test]
    mean = a.mean(axis=0)
    std = a.std(axis=0, ddof=0)
    active = np.ptp(a, axis=0) != 0
    if not active.any():
        return np.zeros((len(train), len(train))), np.zeros((len(test), len(train))), False
    assert np.all(std[active] > 0)
    a = (a[:, active] - mean[active]) / std[active]
    b = (b[:, active] - mean[active]) / std[active]
    a /= np.sqrt(active.sum())
    b /= np.sqrt(active.sum())
    return a @ a.T, b @ a.T, True


def average_active(blocks):
    active = [(a, b) for a, b, valid in blocks if valid]
    if not active:
        a, b, _ = blocks[0]
        return np.zeros_like(a), np.zeros_like(b), False
    return sum(x[0] for x in active) / len(active), sum(x[1] for x in active) / len(active), True


def kernels(baseline, grams, model, train, test):
    base = baseline_kernel(baseline, train, test)
    if model == 'scalar':
        return base[:2]
    if model == 'duplicate_scalar':
        return average_active([base, base])[:2]
    assert model in ['signed', 'REF', 'unit_direction']
    assert len(grams[model]) == 2
    vector = average_active([center_raw_gram(g, train, test) for g in grams[model]])
    return average_active([base, vector])[:2]


def predict_grid(ktt, kxt, train_y, lambdas=LAMBDAS):
    """Uniform-observation loss: (K + n*lambda*I)^-1(y-mean(y))."""
    y = np.asarray(train_y, dtype=np.float64)
    assert ktt.shape == (len(y), len(y)) and kxt.shape[1] == len(y)
    eigen, vectors = eigh((ktt + ktt.T) * .5, check_finite=True)
    tolerance = 1e-10 * max(float(np.abs(eigen).max()), 1.)
    assert eigen.min() >= -tolerance, 'Kernel is not positive semidefinite within floating point tolerance'
    # Tiny negative eigenvalues from numerical roundoff are set to zero only in
    # the statistical solver; model activations and saved deltas are untouched.
    eigen = np.maximum(eigen, 0.)
    weights = vectors.T @ (y - y.mean())
    return (kxt @ vectors) @ (weights[:, None] / (eigen[:, None] + len(y) * lambdas[None, :])) + y.mean()


def nested_loeo(y, groups, positions, baseline, grams, model):
    y, groups, positions = np.asarray(y), np.asarray(groups), np.asarray(positions)
    assert np.isfinite(y).all() and np.isfinite(baseline).all()
    unique = np.unique(groups)
    for p in np.unique(positions):
        assert len(np.unique(groups[positions == p])) == 1
    predictions = np.full(len(y), np.nan)
    inner_rows, selected, audits = [], [], []
    for outer in unique:
        train = np.flatnonzero(groups != outer)
        test = np.flatnonzero(groups == outer)
        assert set(positions[train]).isdisjoint(positions[test])
        losses = []
        for inner in unique[unique != outer]:
            a = np.flatnonzero((groups != outer) & (groups != inner))
            b = np.flatnonzero(groups == inner)
            assert set(a).isdisjoint(test) and set(b).isdisjoint(test)
            assert set(positions[a]).isdisjoint(positions[b])
            tt, xt = kernels(baseline, grams, model, a, b)
            pred = predict_grid(tt, xt, y[a])
            loss = np.abs(pred - y[b, None]).mean(axis=0)
            losses.append(loss)
            for penalty, value in zip(LAMBDAS, loss):
                inner_rows.append({'outer_exon': int(outer), 'inner_exon': int(inner),
                                   'lambda': float(penalty), 'validation_MAE_pp': float(value),
                                   'n_train': len(a), 'n_validation': len(b)})
        mean_loss = np.mean(losses, axis=0)
        best = np.flatnonzero(mean_loss <= mean_loss.min() + 1e-12)[-1]
        penalty = LAMBDAS[best]
        tt, xt = kernels(baseline, grams, model, train, test)
        pred = predict_grid(tt, xt, y[train], np.array([penalty]))[:, 0]
        # Independent direct solve verifies every outer prediction; eigensolver
        # roundoff may introduce a tiny difference, measured rather than hidden.
        direct = xt @ np.linalg.solve(tt + len(train) * penalty * np.eye(len(train)), y[train] - y[train].mean()) + y[train].mean()
        error = float(np.max(np.abs(direct - pred)))
        assert error < 1e-7
        predictions[test] = pred
        selected.append({'outer_exon': int(outer), 'lambda': float(penalty),
                         'inner_macro_exon_MAE_pp': float(mean_loss[best]), 'n_train': len(train), 'n_test': len(test)})
        audits.append({'outer_exon': int(outer), 'direct_solve_max_abs_difference': error,
                       'heldout_position_overlap': 0, 'transforms_fitted_only_on_train': True})
    assert np.isfinite(predictions).all()
    return predictions, inner_rows, selected, audits


def self_check():
    rng = np.random.default_rng(20260916)
    x = rng.normal(size=(19, 37))
    train, test = np.arange(12), np.arange(12, 19)
    g = raw_gram(x)
    tt, xt, active = center_raw_gram(g, train, test)
    a, b = x[train] - x[train].mean(axis=0), x[test] - x[train].mean(axis=0)
    scale = np.mean(np.sum(a * a, axis=1))
    errors = {'center_train': float(np.max(np.abs(tt - a @ a.T / scale))),
              'center_test': float(np.max(np.abs(xt - b @ a.T / scale)))}
    assert active and max(errors.values()) < 1e-13
    # Changing held-out vectors cannot change the training transform/kernel.
    shifted = x.copy(); shifted[test] += 1000
    altered, _, _ = center_raw_gram(raw_gram(shifted), train, test)
    assert np.array_equal(tt, altered)
    baseline = rng.normal(size=(19, 7)); baseline[:, -1] = 3.
    grams = {key: [raw_gram(x, key == 'unit_direction'), raw_gram(x * 2, key == 'unit_direction')]
             for key in ['signed', 'REF', 'unit_direction']}
    kt, kx = kernels(baseline, grams, 'scalar', train, test)
    dt, dx = kernels(baseline, grams, 'duplicate_scalar', train, test)
    assert np.array_equal(kt, dt) and np.array_equal(kx, dx)
    yt = rng.normal(size=12)
    for penalty, pred in zip(LAMBDAS, predict_grid(kt, kx, yt).T):
        direct = kx @ np.linalg.solve(kt + len(train)*penalty*np.eye(len(train)), yt-yt.mean()) + yt.mean()
        assert np.max(np.abs(direct-pred)) < 1e-10
    const = raw_gram(np.ones((19, 3)))
    assert center_raw_gram(const, train, test)[2] is False
    z = np.zeros((19, 3));z[0, 0] = 1
    assert np.isfinite(raw_gram(z, unit_direction=True).matrix).all()
    for offset in [100., 1e8]:
        large = offset + rng.normal(scale=.01, size=(19, 37))
        kt, kx, active = center_raw_gram(raw_gram(large), train, test)
        aa, bb = large[train]-large[train].mean(axis=0), large[test]-large[train].mean(axis=0)
        energy = np.mean(np.sum(aa*aa, axis=1))
        assert active
        np.testing.assert_array_equal(kt, aa@aa.T/energy)
        np.testing.assert_array_equal(kx, bb@aa.T/energy)
    group = np.repeat(np.arange(4), 5)
    base = rng.normal(size=(20, 3)); yy = rng.normal(size=20)
    pred, _, _, audit = nested_loeo(yy, group, np.arange(20), base, {}, 'scalar')
    # Held-out labels must not influence their predictions or inner tuning.
    changed_y = yy.copy();changed_y[group == 0] += 100
    pred2, _, selected2, _ = nested_loeo(changed_y, group, np.arange(20), base, {}, 'scalar')
    np.testing.assert_array_equal(pred[group == 0], pred2[group == 0])
    errors['outer_direct_solve'] = max(v['direct_solve_max_abs_difference'] for v in audit)
    return {'status': 'passed', 'max_abs_differences': errors,
            'heldout_features_do_not_affect_train_transform': True,
            'heldout_labels_do_not_affect_own_OOF_predictions': True,
            'duplicate_kernel_exactly_identical': True,
            'constant_and_zero_direction_controls': True}


if __name__ == '__main__':
    import json
    print(json.dumps(self_check(), indent=2))
