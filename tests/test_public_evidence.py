"""Public aggregate integrity and CPU-only mathematical checks; no model downloads."""
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def read_json(name):
    return json.loads((ROOT / 'docs/results' / name).read_text(encoding='utf-8'))


def test_selected_source_and_result_hashes():
    records = json.loads((ROOT / 'docs/SOURCE_FILES.json').read_text(encoding='utf-8'))['files']
    assert records
    for item in records:
        path = ROOT / item['public_path']
        assert path.resolve().is_relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['public_sha256'], item['public_path']
        if not item['changes']:
            assert item['source_sha256'] == item['public_sha256']


def test_selected_python_sources_parse():
    paths = list((ROOT / 'scripts').rglob('*.py'))
    assert len(paths) >= 32
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def test_primary_delta_matches_recorded_mae():
    effect = read_json('signed_primary_effect.json')
    with (ROOT / 'docs/results/signed_metrics.csv').open(encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    values = {r['model']:float(r['macro_exon_MAE_pp']) for r in rows
              if r['cell_line']==effect['cell_line'] and r['layer']==effect['layer']
              and r['region']==effect['region']}
    delta=values[effect['combined']]-values[effect['baseline']]
    assert delta == pytest.approx(effect['delta_MAE_pp'],abs=1e-12)
    assert effect['ci_low'] > 0  # The recorded primary outcome did not improve.
    assert effect['n'] == 193


def test_numerical_qc_scope_is_not_the_native_screen():
    qc=read_json('numerical_qc.json')
    assert qc['verified_variant_views'] == 386
    assert qc['selected_layer_full_prefix_checks_passed'] == 386*6
    assert qc['verified_distinct_REF_repeat_controls'] == 288
    assert qc['native_forward_comparison_n'] == 183
    assert qc['native_vs_new_descriptive']['new_pre_variant_mean_relative_l2']['max'] == 0
    assert read_json('native_screen_status.json')['variants'] == 13455


def test_context_purge_claims_match_audits():
    signed=read_json('context_purged_validation.json')
    pure=read_json('evo2_only_context_purged_validation.json')
    assert signed['outer_test_train_window_overlap_pairs'] == 0
    assert pure['outer_overlap_pairs'] == 0
    assert signed['p_values'] is False and pure['confidence_intervals'] is False
    for row in [signed,pure]:
        assert row['bootstrap_draws'] == 0


def test_cpu_kernel_train_only_and_direct_solve():
    path=ROOT / 'scripts/analysis/brca1_signed_kernel.py'
    spec=importlib.util.spec_from_file_location('public_kernel',path)
    kernel=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    result=kernel.self_check()
    assert result['status']=='passed'
    assert result['heldout_features_do_not_affect_train_transform']
    assert result['heldout_labels_do_not_affect_own_OOF_predictions']
    assert result['duplicate_kernel_exactly_identical']
    assert result['max_abs_differences']['outer_direct_solve'] < 1e-7
