#!/usr/bin/env python3
"""Frozen exploratory evaluation of signed pooled BRCA1 representations.

--prepare freezes rules without opening new activation artifacts or fitting.
--run requires all193 variants in both orientations to pass numerical QC.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from brca1_signed_kernel import raw_gram, nested_loeo, self_check, LAMBDAS

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/brca1_grch38'
EXT = BASE / 'external_functional_validation'
BENCH = EXT / 'seqsplice_complete_benchmark_20260916'
INPUT = BASE / 'seqsplice_signed_layers_20260916'
OUT = EXT / 'seqsplice_signed_probe_20260916'
LAYERS = ['blocks.0', 'blocks.7', 'blocks.14', 'blocks.21', 'blocks.28', 'norm']
REGIONS = ['variant_pm20', 'assayed_exon', 'donor_pm20', 'acceptor_pm20', 'whole_32k']
PAIRS = [('norm', 'assayed_exon')] + [(l, 'assayed_exon') for l in LAYERS[:-1]] + [('norm', r) for r in REGIONS if r != 'assayed_exon']
SEED, BOOTSTRAPS = 20260916, 5000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [BENCH / 'benchmark_manifest_193.csv',
                    BENCH / 'measured_outcomes/SeqSplice193_per_variant_cell_values.csv',
                    BENCH / 'measured_outcomes/validation.json', INPUT / 'execution_config.json']
    config = json.loads((INPUT / 'execution_config.json').read_text())
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    spec = {
        'study_status': 'Exploratory protocol fixed after prior scalar results were known, before new signed-vector/outcome associations. Not external preregistration.',
        'inputs': {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        'expected_execution_fingerprint': fingerprint,
        'numerical_QC_gate': 'Require all193 x two orientations complete with passing exact selected-layer prefix checks and per-coordinate/reference-direction repeats. Never silently exclude numerical failures or replace prefix values.',
        'primary_cell': 'MDA_MB_231', 'supporting_cell': 'HS578T',
        'cell_dependence': 'Supporting cell shares variants and is not an independent variant replication cohort.',
        'outcome': 'Published paired replicate mean WT full-length minus ALT full-length percentage points, retain negative outcomes, no clipping of predictions.',
        'primary_pair': ['norm', 'assayed_exon'], 'layer_region_pairs': [list(x) for x in PAIRS],
        'primary_effect': 'MDA_MB_231 macro-exon MAE(scalar_plus_signed) minus macro-exon MAE(scalar_matched7), norm assayed_exon.',
        'supporting_controls': ['scalar_plus_REF', 'scalar_plus_unit_direction', 'exact_duplicate_scalar_control'],
        'common_scalar5': ['original_S1_SpliceAI_max_delta_score', 'negative_corrected_forward_delta_score', 'negative_corrected_rc_delta_score', 'corrected_forward_norm_whole32k_mean_relative_l2', 'corrected_rc_norm_whole32k_mean_relative_l2'],
        'matched_scalar7': 'common5 plus forward and RC Euclidean norms of the pooled signed delta of THIS layer/region. Thus every pair has its own matched baseline.',
        'secondary_pairs': 'Six fixed layers at assay exon plus five fixed regions at norm, ten unique pairs including primary. Report every pair; no overall best-layer claim from differing baselines.',
        'baseline_kernel': 'Featurewise training mean/std (ddof0), drop exact training constants; divide concatenated features by sqrt(number_active_columns).',
        'vector_kernel': 'Each orientation separately: center features by training mean, divide by sqrt(training mean squared Euclidean norm). Average the two active orientation Gram matrices.',
        'unit_direction': 'Normalize each raw per-variant per-orientation delta by its own Euclidean norm before training center/RMS scaling. Exactzero vectors remain zero.',
        'combined_kernel': 'Equal average of active scalar and vector kernels. If a block has no training variation it is inactive and remaining weights renormalize. Exact duplicate scalar kernel is identical to scalar kernel.',
        'raw_Gram': 'Raw label-free dot products may be precomputed, but centering and scales use training indices only. No full-cohort standardization, PCA or phenotype-based selection.',
        'solver': 'y intercept=train mean; c=(K + n_train*lambda*I)^(-1)(y-trainmean); prediction=Ktesttrain*c+trainmean. Uniform variant weights in fitting.',
        'lambda_grid': LAMBDAS.tolist(), 'lambda_selection': 'Inner leave-one-assay-exon-out mean of exon MAEs. Ties within1e-12 choose largest lambda. Transforms recalculated in every inner split.',
        'outer_split': 'Leave one of12 assay exons out. Every ALT at the same genomic position stays in one exon. Overlapping32k contexts and single gene limit independence.',
        'metrics': 'Primary macro-exon MAE; variant-weighted MAE, RMSE and Spearman are supporting. Larger exon (66variants) has more training weight, but same weight as other exons in primary evaluation.',
        'bootstrap': {'unit': 'assay_exon', 'draws': BOOTSTRAPS, 'seed': SEED, 'shared_draws_across_cells_models_pairs': True, 'interpretation': 'Percentile95% intervals conditional on fixed outer OOF predictions and12exons. Do not include retraining/tuning uncertainty. No multiplicity-adjusted discovery claims.'},
        'interpretation_limit': 'Information beyond specified linear scalar baseline, not beyond all nonlinear score relationships or anatomy. REF control does not remove all context confounding. Association is not mechanism or clinical pathogenicity.',
        'no_paid_API_or_new_model_training': 'Only small statistical ridge probes are fitted to frozen Evo2 outputs; checkpoint unchanged.',
    }
    p = OUT / 'protocol.json'
    if p.exists():
        old = json.loads(p.read_text())
        assert old['specification'] == spec, 'Frozen protocol changed; use a new analysis directory.'
    else:
        write_json(p, {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'specification': spec})
    write_json(OUT / 'kernel_self_check.json', self_check())
    return spec


def load_complete(spec):
    progress = json.loads((INPUT / 'progress.json').read_text())
    assert progress.get('complete_all193') is True and progress['status'] == 'complete_all193', 'Full corrected193 cohort has not completed QC; no fitting permitted.'
    assert progress['execution_fingerprint'] == spec['expected_execution_fingerprint']
    auditp = INPUT / 'audit_all/validation.json'
    assert auditp.exists(), 'Independent all-cohort artifact audit is required before fitting.'
    audit = json.loads(auditp.read_text())
    assert audit['status'] == 'PASS' and audit['cohort'] == 'all' and audit['cohort_complete'] is True
    assert audit['execution_fingerprint'] == spec['expected_execution_fingerprint']
    assert audit['expected_variant_views'] == audit['verified_variant_views'] == 386
    assert not audit['errors'] and not audit['quarantined_failures'] and not audit['missing_at_snapshot']
    assert audit['audit_script_sha256'] == sha(ROOT/'scripts/analysis/audit_brca1_signed_panel.py')
    for p in (INPUT/'quarantine').glob('**/*.failure.json'):
        assert json.loads(p.read_text()).get('execution_fingerprint') != spec['expected_execution_fingerprint'], 'Same-protocol numerical failure was quarantined; investigate and revise protocol before fitting.'
    manifest = pd.read_csv(BENCH / 'benchmark_manifest_193.csv')
    outcomes = pd.read_csv(BENCH / 'measured_outcomes/SeqSplice193_per_variant_cell_values.csv')
    assert len(manifest) == 193 and manifest.run_id.is_unique
    assert outcomes.groupby('Cell_line').size().to_dict() == {'HS578T': 191, 'MDA_MB_231': 193}
    shape = (193, 2, len(LAYERS), len(REGIONS), 4096)
    refs = np.empty(shape, dtype=np.float64)
    deltas = np.empty(shape, dtype=np.float64)
    scores = np.empty((193, 2)); l2 = np.empty((193, 2))
    provenance = []
    for i, row in manifest.iterrows():
        for v, view in enumerate(['forward', 'rc']):
            p = INPUT / 'variants' / row.run_id / f'{view}.complete.json'
            m = json.loads(p.read_text())
            assert m['execution_fingerprint'] == spec['expected_execution_fingerprint'] and m['status'] == 'passed'
            assert m['run_id'] == row.run_id and m['orientation'] == view
            assert m['layers'] == LAYERS and m['regions'] == REGIONS
            assert m['REF_sequence_sha256'] == row[view + '_ref_sha256']
            assert m['ALT_sequence_sha256'] == row[view + '_alt_sha256']
            qc = m['QC']; assert qc['all_selected_layer_prefixes_exactly_equal'] and qc['reference_control_passed']
            assert len(qc['layer_prefix_checks']) == 6 and all(q['exactly_equal'] for q in qc['layer_prefix_checks'])
            controlp = Path(qc['reference_control_file'])
            assert sha(controlp) == qc['reference_control_sha256']
            control = json.loads(controlp.read_text())
            assert control['passed'] and control['execution_fingerprint'] == spec['expected_execution_fingerprint']
            assert control['position'] == row.position_grch38_1based and control['orientation'] == view
            npzp = p.parent / m['npz_filename']; assert sha(npzp) == m['npz_sha256']
            with np.load(npzp, allow_pickle=False) as z:
                a, b, d = z['REF_mean_float64'], z['ALT_mean_float64'], z['delta_float32']
                assert a.shape == b.shape == d.shape == (6, 5, 4096)
                assert a.dtype == b.dtype == np.float64 and d.dtype == np.float32
                assert np.array_equal((b-a).astype(np.float32), d)
                assert np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(d).all()
                assert np.count_nonzero(z['prefix_mismatch_counts']) == 0
                assert np.count_nonzero(z['relative_l2'][:m['variant_index_0based']]) == 0
                refs[i, v], deltas[i, v] = a, d
                l2[i, v] = float(z['relative_l2'].mean())
                assert l2[i, v] == m['mean_norm_relative_l2']
            scores[i, v] = -m['delta_score']
            provenance.append({'run_id': row.run_id, 'orientation': view, 'complete_marker': str(p),
                               'marker_sha256': sha(p), 'npz_sha256': m['npz_sha256'], 'REF_control_sha256': sha(controlp)})
    assert np.isfinite(scores).all() and np.isfinite(l2).all()
    pd.DataFrame(provenance).to_csv(OUT / 'activation_provenance.csv', index=False)
    return manifest, outcomes, refs, deltas, scores, l2


def metrics(frame):
    error = (frame.prediction_pp - frame.full_length_loss_pp).abs()
    macro = error.groupby(frame.Exon_legacy).mean().mean()
    return {'n': len(frame), 'MAE_pp': error.mean(), 'macro_exon_MAE_pp': macro,
            'RMSE_pp': np.sqrt(np.mean(error**2)), 'spearman': spearmanr(frame.prediction_pp, frame.full_length_loss_pp).statistic}


def summarize(predictions):
    scores, per_exon = [], []
    group_cols = ['cell_line', 'layer', 'region', 'model']
    for keys, sub in predictions.groupby(group_cols, sort=False):
        label = dict(zip(group_cols, keys))
        scores.append({**label, **metrics(sub)})
        for exon, small in sub.groupby('Exon_legacy'):
            per_exon.append({**label, 'exon': int(exon), 'n': len(small),
                             'MAE_pp': float(np.mean(np.abs(small.prediction_pp-small.full_length_loss_pp)))})
    pd.DataFrame(scores).to_csv(OUT / 'metrics.csv', index=False)
    pd.DataFrame(per_exon).to_csv(OUT / 'per_exon_metrics.csv', index=False)
    exons = np.sort(predictions.Exon_legacy.unique())
    assert len(exons) == 12
    draws = np.random.default_rng(SEED).integers(0, len(exons), size=(BOOTSTRAPS, len(exons)))
    np.savez_compressed(OUT / 'shared_exon_bootstrap_draws.npz', exon_order=exons, draws=draws)
    pairs = [('scalar_matched7', 'scalar_plus_signed'), ('scalar_matched7', 'scalar_plus_REF'),
             ('scalar_plus_REF', 'scalar_plus_signed'), ('scalar_matched7', 'scalar_plus_unit_direction'),
             ('scalar_matched7', 'exact_duplicate_scalar_control'), ('SpliceAI_only', 'scalar_matched7'),
             ('scalar_common5', 'scalar_matched7')]
    comparisons = []
    for keys, sub in predictions.groupby(['cell_line', 'layer', 'region'], sort=False):
        label = dict(zip(['cell_line', 'layer', 'region'], keys))
        for base, added in pairs:
            if base not in set(sub.model) or added not in set(sub.model):
                continue
            cols = ['run_id', 'Exon_legacy', 'full_length_loss_pp', 'prediction_pp']
            a = sub[sub.model.eq(base)][cols]
            b = sub[sub.model.eq(added)][cols]
            joined = a.merge(b, on=['run_id', 'Exon_legacy', 'full_length_loss_pp'], suffixes=('_base', '_added'), validate='one_to_one')
            assert len(joined) == len(a) == len(b)
            difference = np.abs(joined.prediction_pp_added-joined.full_length_loss_pp) - np.abs(joined.prediction_pp_base-joined.full_length_loss_pp)
            sums = np.array([difference[joined.Exon_legacy.eq(e)].sum() for e in exons])
            counts = np.array([joined.Exon_legacy.eq(e).sum() for e in exons])
            assert np.all(counts > 0)
            exonmeans = sums/counts
            for weighting, estimate, bootstrap in [
                ('macro_exon', exonmeans.mean(), exonmeans[draws].mean(axis=1)),
                ('variant_weighted', sums.sum()/counts.sum(), sums[draws].sum(axis=1)/counts[draws].sum(axis=1)),
            ]:
                lo, hi = np.quantile(bootstrap, [.025, .975])
                comparisons.append({**label, 'baseline': base, 'combined': added, 'weighting': weighting,
                                    'delta_MAE_pp': estimate, 'ci_low': lo, 'ci_high': hi, 'n': len(joined), 'bootstrap_draws': BOOTSTRAPS})
    pd.DataFrame(comparisons).to_csv(OUT / 'paired_comparisons.csv', index=False)
    return scores, comparisons


def run(spec):
    manifest, outcome, refs, deltas, sequence, l2 = load_complete(spec)
    implementation = [Path(__file__), Path(__file__).with_name('brca1_signed_kernel.py'),
                      OUT/'protocol.json', OUT/'implementation_clarification.json', INPUT/'audit_all/validation.json']
    execution = {'input_code_and_rules_sha256': {str(p.relative_to(ROOT)): sha(p) for p in implementation},
                 'execution_fingerprint': spec['expected_execution_fingerprint'],
                 'numerical_cohort_complete_before_any_fit': True}
    execp = OUT/'analysis_execution_manifest.json'
    if execp.exists():
        assert json.loads(execp.read_text())['specification'] == execution, 'Previously started analysis implementation changed.'
    else:
        write_json(execp, {'started_utc': datetime.now(timezone.utc).isoformat(), 'specification': execution})
    by_id = {rid: i for i, rid in enumerate(manifest.run_id)}
    allpred, alltuning, allselected, allaudit = [], [], [], []
    for layer, region in PAIRS:
        li, ri = LAYERS.index(layer), REGIONS.index(region)
        for cell, sub in outcome.groupby('Cell_line'):
            sub = sub.sort_values(['position_grch38_1based', 'ref', 'alt']).reset_index(drop=True)
            idx = np.array([by_id[r] for r in sub.run_id])
            y = sub.full_length_loss_pp.to_numpy(); groups = sub.Exon_legacy.to_numpy()
            positions = sub.position_grch38_1based.to_numpy()
            delta = deltas[idx, :, li, ri]; ref = refs[idx, :, li, ri]
            norm = np.linalg.norm(delta, axis=-1)
            common = np.column_stack([sub.SpliceAI_max_delta_score, sequence[idx], l2[idx]])
            baseline = np.column_stack([common, norm])
            assert baseline.shape == (len(sub), 7) and np.isfinite(baseline).all()
            grams = {'signed': [raw_gram(delta[:, v]) for v in range(2)],
                     'REF': [raw_gram(ref[:, v]) for v in range(2)],
                     'unit_direction': [raw_gram(delta[:, v], unit_direction=True) for v in range(2)]}
            models = [('scalar_matched7', 'scalar', baseline), ('scalar_plus_signed', 'signed', baseline), ('scalar_plus_REF', 'REF', baseline)]
            if (layer, region) == PAIRS[0]:
                models += [('SpliceAI_only', 'scalar', common[:, :1]), ('scalar_common5', 'scalar', common),
                           ('scalar_plus_unit_direction', 'unit_direction', baseline), ('exact_duplicate_scalar_control', 'duplicate_scalar', baseline)]
            duplicate_check = {}
            for label, kind, x in models:
                pred, tuning, selected, audit = nested_loeo(y, groups, positions, x, grams, kind)
                metadata = {'cell_line': cell, 'layer': layer, 'region': region, 'model': label,
                            'role': 'primary_pair' if (layer, region) == PAIRS[0] else 'secondary_pair'}
                values = sub[['run_id', 'HGVSc', 'position_grch38_1based', 'ref', 'alt', 'Exon_legacy', 'full_length_loss_pp']].copy()
                for k, v in metadata.items(): values[k] = v
                values['prediction_pp'] = pred
                allpred.append(values)
                alltuning.extend([{**metadata, **r} for r in tuning])
                allselected.extend([{**metadata, **r} for r in selected])
                allaudit.extend([{**metadata, **r} for r in audit])
                duplicate_check[label] = pred
                # Save completed fits incrementally; no partial result is labelled complete.
                pd.concat(allpred, ignore_index=True).to_csv(OUT / 'heldout_predictions.partial.csv', index=False)
                print(json.dumps({'event': 'model_complete', **metadata, 'macro_exon_MAE_pp': metrics(values)['macro_exon_MAE_pp']}), flush=True)
            if (layer, region) == PAIRS[0]:
                np.testing.assert_array_equal(duplicate_check['scalar_matched7'], duplicate_check['exact_duplicate_scalar_control'])
    predictions = pd.concat(allpred, ignore_index=True)
    predictions.to_csv(OUT / 'heldout_predictions.csv', index=False)
    (OUT / 'heldout_predictions.partial.csv').unlink()
    pd.DataFrame(alltuning).to_csv(OUT / 'inner_tuning_scores.csv', index=False)
    pd.DataFrame(allselected).to_csv(OUT / 'selected_lambdas.csv', index=False)
    pd.DataFrame(allaudit).to_csv(OUT / 'outer_fit_audit.csv', index=False)
    scores, comparisons = summarize(predictions)
    validation = {'status': 'passed', 'variants': 193, 'measured_variant_cells': 384, 'layer_region_pairs': len(PAIRS),
                  'model_cell_pair_combinations': len(scores), 'outer_fits': len(allaudit),
                  'max_outer_direct_solve_difference': max(x['direct_solve_max_abs_difference'] for x in allaudit),
                  'all193_numerical_QC_required': True, 'duplicate_kernel_predictions_exactly_identical': True,
                  'source_hashes_unchanged': all(sha(ROOT/p)==v for p,v in spec['inputs'].items()),
                  'implementation_hashes_unchanged': all(sha(ROOT/p)==v for p,v in execution['input_code_and_rules_sha256'].items()),
                  'script_sha256': sha(Path(__file__)), 'kernel_script_sha256': sha(Path(__file__).with_name('brca1_signed_kernel.py')),
                  'completed_utc': datetime.now(timezone.utc).isoformat()}
    assert validation['source_hashes_unchanged'] and validation['implementation_hashes_unchanged']
    write_json(OUT / 'validation.json', validation)
    primary = [c for c in comparisons if c['cell_line']=='MDA_MB_231' and c['layer']=='norm' and c['region']=='assayed_exon' and c['baseline']=='scalar_matched7' and c['combined']=='scalar_plus_signed' and c['weighting']=='macro_exon']
    assert len(primary) == 1
    write_json(OUT / 'primary_effect.json', primary[0])
    print(json.dumps({'validation': validation, 'primary': primary[0]}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    args = p.parse_args()
    spec = freeze()
    if args.prepare:
        print(json.dumps({'status': 'protocol_frozen_no_new_vector_associations', 'protocol': str(OUT/'protocol.json')}))
        return
    run(spec)


if __name__ == '__main__':
    main()
