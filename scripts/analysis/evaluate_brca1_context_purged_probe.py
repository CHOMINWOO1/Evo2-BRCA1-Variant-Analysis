#!/usr/bin/env python3
"""Context-purged outer LOEO, ordinary inner LOEO: descriptive stress test.

--prepare reads code, frozen protocol metadata and coordinate manifest only.
--run requires the completed original signed probe and its full193 QC gate.
Never imports the original evaluator or calls its mutating load_complete().
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import os

for _key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[_key] = '1'

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / 'results/brca1_grch38/external_functional_validation'
BENCH = EXT / 'seqsplice_complete_benchmark_20260916'
ORIGINAL = EXT / 'seqsplice_signed_probe_20260916'
INFERENCE = ROOT / 'results/brca1_grch38/seqsplice_signed_layers_20260916'
GEOMETRY = EXT / 'seqsplice_context_overlap_audit_20260916'
OUT = EXT / 'context_purged_probe_20260916'
KERNEL = Path(__file__).with_name('brca1_signed_kernel.py')
REPORT_GATE = Path(__file__).with_name('report_brca1_signed_probe.py')
LAYERS = ['blocks.0', 'blocks.7', 'blocks.14', 'blocks.21', 'blocks.28', 'norm']
REGIONS = ['variant_pm20', 'assayed_exon', 'donor_pm20', 'acceptor_pm20', 'whole_32k']
MODELS = ['scalar_matched7', 'scalar_plus_signed', 'scalar_plus_REF', 'SpliceAI_only',
          'scalar_common5', 'scalar_plus_unit_direction', 'exact_duplicate_scalar_control']
CELL_N = {'MDA_MB_231': 193, 'HS578T': 191}
MODEL_PAIRS = [('scalar_matched7', 'scalar_plus_signed'), ('scalar_matched7', 'scalar_plus_REF'),
               ('scalar_plus_REF', 'scalar_plus_signed'), ('scalar_matched7', 'scalar_plus_unit_direction'),
               ('scalar_matched7', 'exact_duplicate_scalar_control'), ('SpliceAI_only', 'scalar_matched7'),
               ('scalar_common5', 'scalar_matched7')]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def prepare():
    """No phenotype/activation/prediction file reads, including content hashes."""
    prior = read_json(ORIGINAL / 'protocol.json')['specification']
    geometry = read_json(GEOMETRY / 'validation.json')
    require(geometry['status'] == 'PASS_GEOMETRY_ONLY', 'Geometry audit is not complete.')
    require(geometry['counts']['exon_window_union_connected_components'] == 1, 'Geometry scope changed.')
    require(prior['primary_pair'] == ['norm', 'assayed_exon'], 'Original primary pair changed.')
    require(prior['lambda_grid'] == [.0001, .001, .01, .1, 1., 10.], 'Original lambda grid changed.')
    sources = [Path(__file__), KERNEL, REPORT_GATE,
               Path(__file__).with_name('evaluate_brca1_signed_probe.py'),
               ORIGINAL / 'protocol.json', ORIGINAL / 'implementation_clarification.json',
               BENCH / 'benchmark_manifest_193.csv', INFERENCE / 'execution_config.json',
               GEOMETRY / 'validation.json', GEOMETRY / 'outer_purged_LOEO_12.csv',
               GEOMETRY / 'inner_purged_LOEO_candidates_132.csv']
    spec = {
        'status': 'Exploratory descriptive context-purge stress test, designed after scalar results and input geometry were known; no claim of external preregistration.',
        'preparation_reads': 'Code, fixed protocol metadata, coordinate manifest only; outcome and activation files are not opened or hashed during preparation.',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources},
        'inherited_original_input_sha256': prior['inputs'],
        'expected_execution_fingerprint': prior['expected_execution_fingerprint'],
        'cohort': CELL_N, 'measured_variant_cells': 384, 'prediction_rows': 2688,
        'model_cell_groups': 14, 'outer_fits': 168, 'primary_pair': ['norm', 'assayed_exon'],
        'models': MODELS, 'primary_cell': 'MDA_MB_231', 'supporting_cell': 'HS578T',
        'outcome': prior['outcome'],
        'outer_split': 'For each assay exon, hold all its variants. Remove every other variant whose GRCh38 1-based closed32768bp window overlaps any test window by at least1bp. Retain every test variant.',
        'inner_split': 'Ordinary leave-one-assay-exon-out solely within the outer-purged train. No further inner context purge. Never use outer-test or outer-removed observations for tuning.',
        'same_position_rule': 'All alternate alleles at one genomic position have one exon/window and remain together in every outer and inner split.',
        'common_scalar5': prior['common_scalar5'], 'matched_scalar7': prior['matched_scalar7'],
        'baseline_kernel': prior['baseline_kernel'], 'vector_kernel': prior['vector_kernel'],
        'unit_direction': prior['unit_direction'], 'combined_kernel': prior['combined_kernel'],
        'solver': prior['solver'], 'lambda_grid': prior['lambda_grid'],
        'lambda_selection': 'Mean inner exon MAE, training-only transformations in each split. Ties within1e-12 choose largest lambda; same kernels(), predict_grid(), raw_gram(), LAMBDAS from pinned original helper.',
        'metrics': ['variant_weighted_MAE_pp', 'macro_exon_MAE_pp', 'RMSE_pp', 'Spearman'],
        'per_exon': 'All12 outer exons: test errors and retained/excluded train variant/position/exon counts, exact zero test/train overlap, chosen lambda and direct-solve audit.',
        'model_comparison_pairs': [list(p) for p in MODEL_PAIRS],
        'ordinary_LOEO_comparison': 'Same allele and outcome point differences for every model/cell; purged minus original. This combines removal of training examples with changed tuning cohorts, not an estimate of leakage magnitude.',
        'bootstrap_draws': 0, 'p_values': False,
        'uncertainty': 'No bootstrap confidence intervals or p-values: one connected genomic-context component. Point summaries are descriptive stress tests.',
        'execution_gate': 'Original signed probe complete with report_brca1_signed_probe.gate(): full193 x2 activation QC, audit_all PASS386, original fit/source provenance intact. Original evaluator.load_complete() is prohibited.',
        'activation_loading': 'Independent read-only NPZ helper: norm/assayed_exon REFmean float64 and signed delta float32, original whole32k relativeL2 mean and signed sequence penalties. No original provenance file writes.',
        'limits': ['Inner folds retain context overlap; tuning distribution differs from outer test.',
                   'Training size falls substantially and differs across outer exons.',
                   'Cells share alleles, and a single gene does not establish cross-gene generalization.',
                   'Input overlap alone does not prove held-out labels entered training.',
                   'No change to frozen original signed-probe or cross-assay protocols.'],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / 'protocol.json'
    if protocol.exists():
        require(read_json(protocol)['specification'] == spec, 'Frozen preparation changed; preserve a revision explicitly.')
    else:
        require(not (ORIGINAL / 'heldout_predictions.csv').exists(), 'Original OOF already exists; cannot claim pre-OOF freeze.')
        require(not (ORIGINAL / 'heldout_predictions.partial.csv').exists(), 'Original OOF has started; cannot claim pre-OOF freeze.')
        write_json(protocol, {'frozen_utc': datetime.now(timezone.utc).isoformat(),
                             'original_OOF_absent_at_freeze': True, 'specification': spec})
    return spec


def membership_hash(ids):
    return hashlib.sha256(json.dumps(sorted(map(str, ids)), separators=(',', ':')).encode()).hexdigest()


def purged_nested_loeo(y, groups, positions, starts, ends, baseline, grams, model, run_ids):
    """Pure CPU core: explicit arrays only, no files. Ordinary inner tuning."""
    import numpy as np
    from brca1_signed_kernel import kernels, predict_grid, LAMBDAS
    y, groups, positions = np.asarray(y), np.asarray(groups), np.asarray(positions)
    starts, ends, run_ids = np.asarray(starts), np.asarray(ends), np.asarray(run_ids)
    baseline = np.asarray(baseline)
    n = len(y)
    require(all(len(a) == n for a in [groups, positions, starts, ends, baseline, run_ids]), 'Array lengths differ.')
    require(np.isfinite(y).all() and np.isfinite(baseline).all(), 'Nonfinite fit input.')
    require(np.all(ends - starts + 1 == 32768), 'Unexpected genomic window length.')
    require(np.all(positions - starts == 16384), 'Unexpected window placement.')
    require(len(set(run_ids)) == n, 'Duplicate observation ID within cell.')
    for pos in np.unique(positions):
        idx = positions == pos
        require(len(np.unique(groups[idx])) == len(np.unique(starts[idx])) == len(np.unique(ends[idx])) == 1, 'Same-position ALT group split.')
    predictions = np.full(n, np.nan)
    tuning, chosen, audit, inner_audit = [], [], [], []
    for outer in np.unique(groups):
        test = np.flatnonzero(groups == outer)
        intersects = np.any((starts[:, None] <= ends[test][None, :]) & (ends[:, None] >= starts[test][None, :]), axis=1)
        initial = np.flatnonzero(groups != outer)
        train = np.flatnonzero((groups != outer) & ~intersects)
        removed = np.flatnonzero((groups != outer) & intersects)
        require(len(train) > 0 and len(np.unique(groups[train])) >= 2, 'No viable ordinary inner exon split after outer purge.')
        overlap_pairs = int(np.count_nonzero((starts[train, None] <= ends[test]) & (ends[train, None] >= starts[test])))
        require(overlap_pairs == 0 and set(positions[train]).isdisjoint(positions[test]), 'Outer train/test context overlap.')
        losses = []
        for inner in np.unique(groups[train]):
            a, b = train[groups[train] != inner], train[groups[train] == inner]
            require(len(a) > 0 and len(b) > 0, 'Empty ordinary inner fold.')
            require(set(a).isdisjoint(b) and set(a).union(b) == set(train), 'Inner membership differs from outer training.')
            require(set(a).isdisjoint(test) and set(b).isdisjoint(test), 'Outer test entered tuning.')
            require(set(a).isdisjoint(removed) and set(b).isdisjoint(removed), 'Outer-purged data entered tuning.')
            require(set(positions[a]).isdisjoint(positions[b]), 'Inner same-position overlap.')
            tt, xt = kernels(baseline, grams, model, a, b)
            pred = predict_grid(tt, xt, y[a])
            require(np.isfinite(pred).all(), 'Nonfinite inner prediction.')
            loss = np.abs(pred - y[b, None]).mean(axis=0)
            losses.append(loss)
            for penalty, value in zip(LAMBDAS, loss):
                tuning.append({'outer_exon': int(outer), 'inner_exon': int(inner), 'lambda': float(penalty),
                               'validation_MAE_pp': float(value), 'n_train': len(a), 'n_validation': len(b)})
            inner_audit.append({'outer_exon': int(outer), 'inner_exon': int(inner),
                                'n_train': len(a), 'n_validation': len(b),
                                'train_exons': '|'.join(map(str, np.unique(groups[a]))),
                                'train_ids_json': json.dumps(run_ids[a].tolist()),
                                'validation_ids_json': json.dumps(run_ids[b].tolist()),
                                'train_ids_sha256': membership_hash(run_ids[a]),
                                'validation_ids_sha256': membership_hash(run_ids[b]),
                                'outer_test_and_removed_excluded': True,
                                'transforms_fitted_only_on_inner_train': True,
                                'inner_position_overlap': 0,
                                'inner_window_overlap_pairs': int(np.count_nonzero((starts[a, None] <= ends[b]) & (ends[a, None] >= starts[b])))})
        mean_loss = np.mean(losses, axis=0)
        best = np.flatnonzero(mean_loss <= mean_loss.min() + 1e-12)[-1]
        penalty = LAMBDAS[best]
        tt, xt = kernels(baseline, grams, model, train, test)
        pred = predict_grid(tt, xt, y[train], np.array([penalty]))[:, 0]
        direct = xt @ np.linalg.solve(tt + len(train) * penalty * np.eye(len(train)), y[train] - y[train].mean()) + y[train].mean()
        error = float(np.max(np.abs(direct - pred)))
        require(np.isfinite(pred).all() and np.isfinite(error) and 0 <= error < 1e-7, 'Outer direct solve failed.')
        predictions[test] = pred
        chosen.append({'outer_exon': int(outer), 'lambda': float(penalty), 'inner_macro_exon_MAE_pp': float(mean_loss[best]),
                       'n_train': len(train), 'n_test': len(test), 'inner_folds': len(losses)})
        audit.append({'outer_exon': int(outer), 'initial_train_n': len(initial), 'excluded_overlap_n': len(removed),
                      'retained_train_n': len(train), 'retained_train_positions': len(np.unique(positions[train])),
                      'retained_train_exons': len(np.unique(groups[train])),
                      'retained_train_exon_ids': '|'.join(map(str, np.unique(groups[train]))),
                      'train_ids_json': json.dumps(run_ids[train].tolist()), 'test_ids_json': json.dumps(run_ids[test].tolist()),
                      'removed_ids_json': json.dumps(run_ids[removed].tolist()),
                      'train_ids_sha256': membership_hash(run_ids[train]), 'test_ids_sha256': membership_hash(run_ids[test]),
                      'outer_test_train_window_overlap_pairs': overlap_pairs, 'heldout_position_overlap': 0,
                      'transforms_fitted_only_on_train': True, 'all_inner_data_within_outer_train': True,
                      'direct_solve_max_abs_difference': error})
    require(np.isfinite(predictions).all(), 'Missing outer prediction.')
    return predictions, tuning, chosen, audit, inner_audit


def load_activation_features_readonly(manifest, fingerprint):
    """Independent reader; returns provenance without writing existing files."""
    import numpy as np
    refs = np.empty((len(manifest), 2, 4096), dtype=np.float64)
    deltas = np.empty_like(refs)
    scores, l2 = np.empty((len(manifest), 2)), np.empty((len(manifest), 2))
    provenance = []
    for i, row in manifest.iterrows():
        for view_index, view in enumerate(['forward', 'rc']):
            marker = INFERENCE / 'variants' / row.run_id / (view + '.complete.json')
            meta = read_json(marker)
            require(meta['execution_fingerprint'] == fingerprint and meta['status'] == 'passed', 'Activation QC failed.')
            require(meta['run_id'] == row.run_id and meta['orientation'] == view, 'Activation identity mismatch.')
            require(meta['layers'] == LAYERS and meta['regions'] == REGIONS, 'Activation schema changed.')
            require(meta['REF_sequence_sha256'] == row[view + '_ref_sha256'] and meta['ALT_sequence_sha256'] == row[view + '_alt_sha256'], 'Activation sequence hash mismatch.')
            qc = meta['QC']
            require(qc['all_selected_layer_prefixes_exactly_equal'] is True and qc['reference_control_passed'] is True, 'Exact QC failed.')
            require(len(qc['layer_prefix_checks']) == 6 and all(x['exactly_equal'] is True for x in qc['layer_prefix_checks']), 'Layer prefix QC incomplete.')
            controlp = Path(qc['reference_control_file'])
            require(sha(controlp) == qc['reference_control_sha256'], 'Control hash changed.')
            control = read_json(controlp)
            require(control['passed'] is True and control['execution_fingerprint'] == fingerprint, 'Reference control failed.')
            require(int(control['position']) == row.position_grch38_1based and control['orientation'] == view, 'Reference control identity mismatch.')
            data = marker.parent / meta['npz_filename']
            require(sha(data) == meta['npz_sha256'], 'Activation NPZ hash changed.')
            with np.load(data, allow_pickle=False) as z:
                a, b, d = z['REF_mean_float64'], z['ALT_mean_float64'], z['delta_float32']
                require(a.shape == b.shape == d.shape == (6, 5, 4096), 'Activation shape changed.')
                require(a.dtype == b.dtype == np.float64 and d.dtype == np.float32, 'Activation dtype changed.')
                require(np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(d).all(), 'Nonfinite activation.')
                require(np.array_equal((b - a).astype(np.float32), d), 'Delta storage relation changed.')
                require(np.count_nonzero(z['prefix_mismatch_counts']) == 0, 'Stored prefix QC failed.')
                rel = z['relative_l2']
                require(rel.shape == (32768,) and np.isfinite(rel).all(), 'RelativeL2 schema changed.')
                require(np.count_nonzero(rel[:meta['variant_index_0based']]) == 0, 'Prefix relativeL2 nonzero.')
                refs[i, view_index] = a[LAYERS.index('norm'), REGIONS.index('assayed_exon')]
                deltas[i, view_index] = d[LAYERS.index('norm'), REGIONS.index('assayed_exon')]
                l2[i, view_index] = float(rel.mean())  # Exact original scalar aggregation/dtype.
                require(l2[i, view_index] == meta['mean_norm_relative_l2'], 'L2 scalar differs from original.')
            scores[i, view_index] = -meta['delta_score']
            provenance.append({'run_id': row.run_id, 'orientation': view, 'complete_marker': str(marker),
                               'marker_sha256': sha(marker), 'npz_sha256': meta['npz_sha256'], 'REF_control_sha256': sha(controlp)})
    require(np.isfinite(scores).all() and np.isfinite(l2).all(), 'Nonfinite scalar feature.')
    return refs, deltas, scores, l2, provenance


def metrics(frame):
    import numpy as np
    from scipy.stats import spearmanr
    error = np.abs(frame.prediction_pp.to_numpy() - frame.full_length_loss_pp.to_numpy())
    macro = frame.assign(absolute_error=error).groupby('Exon_legacy').absolute_error.mean().mean()
    x, y = frame.prediction_pp.to_numpy(), frame.full_length_loss_pp.to_numpy()
    rho = float(spearmanr(x, y).statistic) if np.ptp(x) and np.ptp(y) else float('nan')
    return {'n': len(frame), 'exons': frame.Exon_legacy.nunique(), 'MAE_pp': float(error.mean()),
            'macro_exon_MAE_pp': float(macro), 'RMSE_pp': float(np.sqrt(np.mean(error ** 2))),
            'spearman': rho, 'spearman_defined': bool(np.isfinite(rho))}


def model_inputs(common, delta, ref):
    import numpy as np
    from brca1_signed_kernel import raw_gram
    baseline = np.column_stack([common, np.linalg.norm(delta, axis=-1)])
    require(common.shape[1] == 5 and baseline.shape[1] == 7, 'Scalar dimensions changed.')
    grams = {'signed': [raw_gram(delta[:, v]) for v in range(2)],
             'REF': [raw_gram(ref[:, v]) for v in range(2)],
             'unit_direction': [raw_gram(delta[:, v], unit_direction=True) for v in range(2)]}
    models = [('scalar_matched7', 'scalar', baseline), ('scalar_plus_signed', 'signed', baseline),
              ('scalar_plus_REF', 'REF', baseline), ('SpliceAI_only', 'scalar', common[:, :1]),
              ('scalar_common5', 'scalar', common), ('scalar_plus_unit_direction', 'unit_direction', baseline),
              ('exact_duplicate_scalar_control', 'duplicate_scalar', baseline)]
    return grams, models


def summarize_and_compare(predictions, original, outer_audit):
    import numpy as np
    import pandas as pd
    results, per_exon, contrasts, differences, joined = [], [], [], [], []
    expected_groups = {(cell, model) for cell in CELL_N for model in MODELS}
    require(set(predictions.groupby(['cell_line', 'model']).groups) == expected_groups, 'New model/cell groups incomplete.')
    require(set(original.groupby(['cell_line', 'model']).groups) == expected_groups, 'Original model/cell groups incomplete.')
    for (cell, model), frame in predictions.groupby(['cell_line', 'model']):
        metadata = {'cell_line': cell, 'layer': 'norm', 'region': 'assayed_exon', 'model': model}
        require(len(frame) == CELL_N[cell] and frame.Exon_legacy.nunique() == 12 and frame.run_id.is_unique, 'New prediction cohort mismatch.')
        results.append({**metadata, **metrics(frame)})
        for exon, sub in frame.groupby('Exon_legacy'):
            fold = outer_audit[(outer_audit.cell_line == cell) & (outer_audit.model == model) & (outer_audit.outer_exon == exon)]
            require(len(fold) == 1, 'Outer audit fold missing.')
            per_exon.append({**metadata, 'outer_exon': int(exon), **metrics(sub),
                             **fold[['initial_train_n', 'excluded_overlap_n', 'retained_train_n', 'retained_train_positions', 'retained_train_exons', 'outer_test_train_window_overlap_pairs']].iloc[0].to_dict()})
        old = original[(original.cell_line == cell) & (original.model == model)]
        require(len(old) == len(frame) and old.run_id.is_unique and old.Exon_legacy.nunique() == 12, 'Original cohort mismatch.')
        keys = ['run_id', 'position_grch38_1based', 'ref', 'alt', 'Exon_legacy']
        pair = frame.merge(old[keys + ['full_length_loss_pp', 'prediction_pp']], on=keys, suffixes=('', '_original'), validate='one_to_one')
        require(len(pair) == len(frame), 'Original comparison changed exact alleles.')
        require(np.allclose(pair.full_length_loss_pp, pair.full_length_loss_pp_original, atol=1e-12, rtol=0), 'Original measured endpoint changed.')
        pair['prediction_difference_purged_minus_original_pp'] = pair.prediction_pp - pair.prediction_pp_original
        pair['absolute_error_difference_purged_minus_original_pp'] = abs(pair.prediction_pp - pair.full_length_loss_pp) - abs(pair.prediction_pp_original - pair.full_length_loss_pp)
        joined.append(pair)
        now, previous = metrics(frame), metrics(old)
        d = {**metadata, 'n': len(frame), 'interpretation': 'training exclusion and tuning cohort change; not leakage magnitude'}
        for k in ['MAE_pp', 'macro_exon_MAE_pp', 'RMSE_pp', 'spearman']:
            d.update({k + '_purged': now[k], k + '_original': previous[k], 'delta_' + k: now[k] - previous[k]})
        differences.append(d)
    lookup = {(r['cell_line'], r['model']): r for r in results}
    for cell in CELL_N:
        for base, added in MODEL_PAIRS:
            r = {'cell_line': cell, 'layer': 'norm', 'region': 'assayed_exon', 'baseline': base, 'combined': added, 'n': CELL_N[cell]}
            r.update({'delta_' + k: lookup[cell, added][k] - lookup[cell, base][k] for k in ['MAE_pp', 'macro_exon_MAE_pp', 'RMSE_pp', 'spearman']})
            contrasts.append(r)
    return {'metrics.csv': pd.DataFrame(results), 'per_exon_metrics.csv': pd.DataFrame(per_exon),
            'model_comparisons_points.csv': pd.DataFrame(contrasts), 'original_LOEO_point_differences.csv': pd.DataFrame(differences),
            'same_allele_original_and_purged_predictions.csv': pd.concat(joined, ignore_index=True)}


def run(spec):
    import numpy as np
    import pandas as pd
    require(not (OUT / 'execution_manifest.json').exists(), 'A run already started; preserve it and review instead of silently restarting.')
    require(not (OUT / 'validation.json').exists(), 'Completed result already exists.')
    for path, digest in spec['source_sha256'].items():
        require(sha(ROOT / path) == digest, 'Pinned code/metadata changed: ' + path)
    for path, digest in spec['inherited_original_input_sha256'].items():
        require(sha(ROOT / path) == digest, 'Original input changed: ' + path)
    from report_brca1_signed_probe import gate
    snapshot = gate()  # Original full193 inference, audit and signed fit must be complete.
    snapshot.update(spec['source_sha256'])
    snapshot[str((OUT / 'protocol.json').relative_to(ROOT))] = sha(OUT / 'protocol.json')
    manifest = pd.read_csv(BENCH / 'benchmark_manifest_193.csv')
    outcomes = pd.read_csv(BENCH / 'measured_outcomes/SeqSplice193_per_variant_cell_values.csv')
    require(len(manifest) == 193 and manifest.run_id.is_unique and manifest.chromosome.eq('chr17').all(), 'Manifest cohort mismatch.')
    require(outcomes.groupby('Cell_line').size().to_dict() == CELL_N, 'Outcome cell counts changed.')
    require(not outcomes.duplicated(['Cell_line', 'run_id']).any(), 'Repeated variant-cell outcome.')
    require(np.isfinite(outcomes.full_length_loss_pp).all(), 'Nonfinite measured outcome.')
    original = pd.read_csv(ORIGINAL / 'heldout_predictions.csv')
    original = original[(original.layer == 'norm') & (original.region == 'assayed_exon')].copy()
    require(len(original) == 2688 and not original.duplicated(['cell_line', 'model', 'run_id']).any(), 'Original primary predictions incomplete.')
    refs, deltas, sequence, l2, provenance = load_activation_features_readonly(manifest, spec['expected_execution_fingerprint'])
    require(len(provenance) == 386, 'Activation views incomplete.')
    pd.DataFrame(provenance).to_csv(OUT / 'activation_provenance.csv', index=False)
    write_json(OUT / 'execution_manifest.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
               'source_sha256': snapshot, 'all193_QC_and_original_signed_probe_complete_before_any_fit': True,
               'original_evaluator_load_complete_called': False, 'new_statistical_fit_scope': '168 context-purged outer ridge fits with ordinary inner tuning'})
    by_id = {rid: i for i, rid in enumerate(manifest.run_id)}
    predictions, tuning_rows, selected_rows, audit_rows, inner_rows = [], [], [], [], []
    for cell, sub in outcomes.groupby('Cell_line'):
        sub = sub.sort_values(['position_grch38_1based', 'ref', 'alt']).reset_index(drop=True)
        idx = np.array([by_id[r] for r in sub.run_id])
        geom = manifest.iloc[idx].reset_index(drop=True)
        for key in ['position_grch38_1based', 'ref', 'alt']:
            require(np.array_equal(sub[key].to_numpy(), geom[key].to_numpy()), 'Outcome/activation allele mismatch.')
        require(np.array_equal(sub.Exon_legacy.to_numpy(), geom.source_legacy_exon.to_numpy()), 'Outcome assay exon mismatch.')
        require(sub.Exon_legacy.nunique() == 12, 'Measured cell lacks an exon.')
        common = np.column_stack([sub.SpliceAI_max_delta_score, sequence[idx], l2[idx]])
        grams, models = model_inputs(common, deltas[idx], refs[idx])
        duplicates = {}
        for label, kind, baseline in models:
            pred, tuning, selected, audit, inner_audit = purged_nested_loeo(
                sub.full_length_loss_pp.to_numpy(), sub.Exon_legacy.to_numpy(), sub.position_grch38_1based.to_numpy(),
                geom.window_start_grch38_1based.to_numpy(), geom.window_end_grch38_1based.to_numpy(),
                baseline, grams, kind, sub.run_id.to_numpy())
            meta = {'cell_line': cell, 'layer': 'norm', 'region': 'assayed_exon', 'model': label, 'role': 'context_purged_stress_test'}
            values = sub[['run_id', 'HGVSc', 'position_grch38_1based', 'ref', 'alt', 'Exon_legacy', 'full_length_loss_pp']].copy()
            for key, value in meta.items():
                values[key] = value
            values['prediction_pp'] = pred
            predictions.append(values)
            for accumulator, rows in [(tuning_rows, tuning), (selected_rows, selected), (audit_rows, audit), (inner_rows, inner_audit)]:
                accumulator.extend([{**meta, **r} for r in rows])
            duplicates[label] = pred
            pd.concat(predictions, ignore_index=True).to_csv(OUT / 'heldout_predictions.partial.csv', index=False)
            print(json.dumps({'event': 'model_complete', **meta, 'n': len(pred)}, ensure_ascii=False), flush=True)
        np.testing.assert_array_equal(duplicates['scalar_matched7'], duplicates['exact_duplicate_scalar_control'])
    pred = pd.concat(predictions, ignore_index=True)
    audits = pd.DataFrame(audit_rows)
    require(len(pred) == 2688 and len(audits) == 168, 'Final prediction/fold count mismatch.')
    require(len(selected_rows) == 168 and len(inner_rows) == 868 and len(tuning_rows) == 5208, 'Inner/outer fit accounting changed.')
    tables = summarize_and_compare(pred, original, audits)
    tables.update({'heldout_predictions.csv': pred, 'outer_fit_audit.csv': audits,
                   'inner_fit_audit.csv': pd.DataFrame(inner_rows), 'selected_lambdas.csv': pd.DataFrame(selected_rows),
                   'inner_tuning_scores.csv': pd.DataFrame(tuning_rows)})
    for name, frame in tables.items():
        frame.to_csv(OUT / name, index=False)
    require(all(sha(ROOT / p) == d for p, d in snapshot.items()), 'An original source/result changed during the new fit.')
    require(np.isfinite(audits.direct_solve_max_abs_difference).all() and (audits.direct_solve_max_abs_difference < 1e-7).all(), 'Direct solve audit failed.')
    (OUT / 'heldout_predictions.partial.csv').unlink()
    write_json(OUT / 'validation.json', {'status': 'PASS_DESCRIPTIVE_CONTEXT_PURGE', 'completed_utc': datetime.now(timezone.utc).isoformat(),
               'variants': 193, 'measured_variant_cells': 384, 'prediction_rows': 2688, 'model_cell_groups': 14,
               'outer_fits': 168, 'inner_folds': len(inner_rows), 'inner_lambda_validation_rows': len(tuning_rows),
               'outer_test_train_window_overlap_pairs': int(audits.outer_test_train_window_overlap_pairs.sum()),
               'max_direct_solve_difference': float(audits.direct_solve_max_abs_difference.max()),
               'duplicate_predictions_exact': True, 'transforms_and_inner_tuning_train_only': True,
               'same_position_groups_preserved': True, 'all193_QC_and_original_fit_required': True,
               'original_sources_and_results_unchanged': True, 'bootstrap_draws': 0, 'p_values': False,
               'script_sha256': sha(Path(__file__)), 'kernel_sha256': sha(KERNEL),
               'output_sha256': {name: sha(OUT / name) for name in tables}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        print(json.dumps({'status': 'PREPARED_NO_ACTUAL_DATA_FIT', 'protocol': str(OUT / 'protocol.json')}))
    else:
        spec = read_json(OUT / 'protocol.json')['specification']
        run(spec)


if __name__ == '__main__':
    main()
