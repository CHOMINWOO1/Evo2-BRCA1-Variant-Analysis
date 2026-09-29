#!/usr/bin/env python3
"""Fixed-case BRCA1 numerical diagnostic; five native GPU forwards, then CPU audit."""
from pathlib import Path
import datetime
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/brca1_grch38/numeric_validation_20260916'
SCREEN = ROOT / 'results/brca1_grch38/screen_32768'
CASE = 'cv_868688_43063900_C_A'
LAYERS = ['embedding_layer', 'blocks.0', 'blocks.1', 'blocks.2', 'norm']
os.environ['HF_HUB_OFFLINE'] = '1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(name, value):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def rc(seq):
    return seq.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=True)
    if (OUT / 'validation.json').exists():
        raise RuntimeError('Completed diagnostic already exists; preserve the recorded run.')
    cfg = json.loads((SCREEN / 'config.json').read_text())
    old = json.loads((SCREEN / 'variants' / (CASE + '.json')).read_text())
    assert cfg['model'] == 'evo2_7b' and cfg['length_bp'] == 32768
    with gzip.open(cfg['reference_fasta'], 'rt') as f:
        assert next(f).startswith('>chr17')
        chromosome = ''.join(line.strip().upper() for line in f)
    start0 = old['window_start_grch38_1based'] - 1
    length, variant = cfg['length_bp'], old['variant_index_0based']
    ref = chromosome[start0:start0 + length]
    assert variant == 16384 and ref[variant] == old['ref'] == 'C'
    alt = ref[:variant] + old['alt'] + ref[variant + 1:]
    assert old['alt'] == 'A' and len(ref) == len(alt) == 32768
    assert ref[:variant] == alt[:variant]
    assert hashlib.sha256(ref.encode()).hexdigest() == old['ref_sequence_sha256']
    assert hashlib.sha256(alt.encode()).hexdigest() == old['alt_sequence_sha256']
    assert sha(cfg['reference_fasta']) == cfg['reference_sha256']
    del chromosome
    sequences = {'forward_ref': ref, 'forward_alt': alt, 'forward_ref_repeat': ref,
                 'rc_ref': rc(ref), 'rc_alt': rc(alt)}
    assert len([i for i, (a, b) in enumerate(zip(rc(ref), rc(alt))) if a != b]) == 1
    assert rc(ref)[16383] == 'G' and rc(alt)[16383] == 'T'
    checkpoint_sha = sha(cfg['checkpoint'])
    protocol = {
        'frozen_before_model_load_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'case': CASE, 'assembly': cfg['assembly'], 'chromosome': 'chr17',
        'position_grch38_1based': int(old['position_grch38_1based']),
        'ref': 'C', 'alt': 'A', 'window_start_1based': start0 + 1,
        'window_end_1based': start0 + length, 'length_bp': length,
        'variant_index_0based': {'forward': 16384, 'reverse_complement': 16383},
        'model': cfg['model'], 'checkpoint': cfg['checkpoint'],
        'checkpoint_revision': cfg['checkpoint_revision'], 'checkpoint_sha256': checkpoint_sha,
        'reference_fasta_sha256': cfg['reference_sha256'],
        'sequence_sha256': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sequences.items()},
        'sequence_order': list(sequences), 'GPU_forward_count': 5,
        'precision': 'Existing native BF16 model; existing FP32 FFT operators; no model or operator modification',
        'layers': LAYERS, 'probes': 'First 256 and last 256 pre-variant positions in each input orientation',
        'operator_capture': 'First HCM fftconv_func (block 1), first 64 channels, same native calculation',
        'CPU_operator_tests': 'FP32 FFT; FP64 FFT; exact causal direct FIR in FP64 on first 256 positions; compare before and after BF16 conversion',
        'metrics': 'Original position_metrics function, same REF-denominator relative L2; saved in input order and RC reversed to forward genomic coordinates',
        'historical_baseline': str(SCREEN / 'variants' / (CASE + '.npz')),
        'historical_baseline_sha256': sha(SCREEN / 'variants' / (CASE + '.npz')),
        'interpretation': 'Numerical/directional diagnostic for one fixed variant. Unchanged-prefix differences are not evidence of biological propagation; RC is not an independent biological replicate.',
        'script_sha256': sha(__file__),
    }
    if (OUT / 'protocol.json').exists():
        previous = json.loads((OUT / 'protocol.json').read_text())
        assert all(previous[k] == v for k, v in protocol.items()
                   if k != 'frozen_before_model_load_utc'), 'Existing frozen protocol differs.'
    else:
        json_write('protocol.json', protocol)
    json_write('progress.json', {'status': 'loading_model', 'five_forwards': 5})
    print('Protocol frozen; loading model', flush=True)

    import numpy as np
    import torch
    import torch.nn.functional as F
    from evo2 import Evo2
    from evo2.scoring import logits_to_logprobs
    import vortex.model.engine as engine
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'GPU diagnostic requires GPU-accessible execution.'
    metric_spec = importlib.util.spec_from_file_location('original_screen', ROOT / 'scripts/inference/run_brca1_grch38_screen.py')
    screen = importlib.util.module_from_spec(metric_spec)
    metric_spec.loader.exec_module(screen)
    torch.cuda.reset_peak_memory_stats()
    t = time.monotonic()
    model = Evo2(cfg['model'], local_path=cfg['checkpoint'])
    model.model.eval()
    load_seconds = time.monotonic() - t
    assert not model.model.config.use_fp8_input_projections
    assert all(not module.training for module in model.model.modules())
    assert not any(getattr(model.model.blocks[i].filter.engine, 'use_hcm_kernel', False)
                   for i in model.model.config.hcm_layer_idxs)
    hooks, captured, operators, forward_records, layer_records = [], {}, {}, [], []
    active = {'label': None, 'variant': variant}
    original_fft = engine.fftconv_func

    def layer_hook(name):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            idx = active['variant']
            captured[active['label']][name] = {
                'first_256': output[:, :256, :].detach().cpu().clone(),
                'last_256_before_variant': output[:, idx - 256:idx, :].detach().cpu().clone(),
            }
        return hook

    def capture_fft(u, k, D, dropout_mask, **kwargs):
        result = original_fft(u, k, D, dropout_mask, **kwargs)
        if kwargs.get('layer_idx') == 1:
            assert active['label'] not in operators
            assert kwargs.get('bidirectional') is False and u.shape[1] == 4096
            idx = active['variant']
            operators[active['label']] = {
                'u': u[:, :64, :].detach().cpu().clone(),
                'k': k.squeeze()[:64].detach().cpu().clone(),
                'D': D[:64].detach().cpu().clone(),
                'native_first_256': result[:, :64, :256].detach().cpu().clone(),
                'native_last_256_before_variant': result[:, :64, idx - 256:idx].detach().cpu().clone(),
                'variant_index_0based': idx, 'native_kwargs': {key: kwargs.get(key) for key in ['layer_idx', 'gelu', 'bidirectional']},
            }
        return result

    def tensor_comparison(a, b):
        af, bf = a.float(), b.float()
        diff = bf - af
        return {'exactly_equal': bool(torch.equal(a, b)), 'max_abs_difference': float(diff.abs().max()),
                'unequal_elements': int((a != b).sum()), 'elements': a.numel(),
                'mean_relative_l2': float((diff.norm(dim=-1) / af.norm(dim=-1).clamp_min(1e-30)).mean())}

    def compare_layers(a, b, comparison):
        for layer in LAYERS:
            for region in captured[a][layer]:
                layer_records.append({'comparison': comparison, 'layer': layer, 'region': region,
                                      **tensor_comparison(captured[a][layer][region], captured[b][layer][region])})

    def forward(label):
        active.update(label=label, variant=16383 if label.startswith('rc_') else 16384)
        captured[label] = {}
        ids = torch.tensor(model.tokenizer.tokenize(sequences[label]), dtype=torch.long, device='cuda:0')[None]
        t = time.monotonic()
        with torch.inference_mode():
            outputs, embeddings = model(ids, return_embeddings=True, layer_names=['norm'])
            hidden = embeddings['norm'].detach().cpu()
            score = float(logits_to_logprobs(outputs[0], ids).float().mean().item())
        torch.cuda.synchronize()
        assert tuple(hidden.shape) == (1, length, 4096) and torch.isfinite(hidden).all()
        seconds = time.monotonic() - t
        del outputs, embeddings, ids
        forward_records.append({'label': label, 'seconds': seconds, 'score': score,
                                'hidden_dtype': str(hidden.dtype), 'variant_index_0based': active['variant']})
        json_write('progress.json', {'status': 'running', 'completed_forwards': len(forward_records),
                                    'current_label': label, 'elapsed_seconds': time.monotonic() - started})
        print(json.dumps(forward_records[-1]), flush=True)
        return hidden, score

    try:
        hooks = [model.model.get_submodule(name).register_forward_hook(layer_hook(name)) for name in LAYERS]
        engine.fftconv_func = capture_fft
        ref_hidden, ref_score = forward('forward_ref')
        alt_hidden, alt_score = forward('forward_alt')
        native_metrics = screen.position_metrics(ref_hidden, alt_hidden)
        np.savez_compressed(OUT / 'forward_ref_alt_metrics.npz', **native_metrics)
        del alt_hidden
        repeat, repeat_score = forward('forward_ref_repeat')
        repeat_metrics = screen.position_metrics(ref_hidden, repeat)
        np.savez_compressed(OUT / 'forward_ref_repeat_metrics.npz', **repeat_metrics)
        del repeat, ref_hidden
        compare_layers('forward_ref', 'forward_alt', 'forward_REF_vs_ALT')
        compare_layers('forward_ref', 'forward_ref_repeat', 'REF_before_vs_after_ALT')
        rc_ref_hidden, rc_ref_score = forward('rc_ref')
        rc_alt_hidden, rc_alt_score = forward('rc_alt')
        rc_metrics = screen.position_metrics(rc_ref_hidden, rc_alt_hidden)
        np.savez_compressed(OUT / 'rc_ref_alt_metrics_input_order.npz', **rc_metrics)
        np.savez_compressed(OUT / 'rc_ref_alt_metrics_forward_genomic_order.npz', **{k: v[::-1].copy() for k, v in rc_metrics.items()})
        del rc_ref_hidden, rc_alt_hidden
        compare_layers('rc_ref', 'rc_alt', 'RC_REF_vs_ALT')
    finally:
        for hook in hooks:
            hook.remove()
        engine.fftconv_func = original_fft
    peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
    gpu = torch.cuda.get_device_name(0)
    del model
    torch.cuda.empty_cache()
    torch.save(captured, OUT / 'layer_prefix_samples.pt')
    torch.save(operators, OUT / 'first_HCM_operator_samples.pt')
    json_write('layer_prefix_comparisons.json', layer_records)
    json_write('forward_runs.json', forward_records)

    def operator_delta(a, b):
        diff = (b - a).double()
        qa, qb = a.to(torch.bfloat16), b.to(torch.bfloat16)
        return {'max_abs_difference': float(diff.abs().max()),
                'rms_difference': float(diff.square().mean().sqrt()),
                'exactly_equal_before_BF16': bool(torch.equal(a, b)),
                'unequal_BF16_elements': int((qa != qb).sum()), 'elements': a.numel(),
                'exactly_equal_after_BF16': bool(torch.equal(qa, qb))}

    operator_reports = []
    for orientation, ref_name, alt_name in [('forward', 'forward_ref', 'forward_alt'), ('reverse_complement', 'rc_ref', 'rc_alt')]:
        pair = [operators[ref_name], operators[alt_name]]
        a, b = pair
        idx = a['variant_index_0based']
        assert torch.equal(a['k'], b['k']) and torch.equal(a['D'], b['D'])
        assert torch.equal(a['u'][..., :idx], b['u'][..., :idx])
        differences = torch.nonzero((a['u'] != b['u']).any(dim=1)[0]).flatten()
        report = {'orientation': orientation, 'sampled_channels': 64,
                  'prefix_operator_inputs_exactly_equal': True, 'filters_and_bias_exactly_equal': True,
                  'first_different_operator_input_index': int(differences[0]), 'variant_index_0based': idx,
                  'native_GPU_FP32_first256': operator_delta(a['native_first_256'], b['native_first_256']),
                  'native_GPU_FP32_last256_before_variant': operator_delta(a['native_last_256_before_variant'], b['native_last_256_before_variant']),
                  'CPU_recomputations': {}}
        direct = []
        for data in pair:
            # Only causal first-256 outputs: no suffix can enter this direct calculation.
            u = data['u'][..., :256].double()
            k = data['k'][..., :min(data['k'].shape[-1], 256)].double()
            y = F.conv1d(F.pad(u, (k.shape[-1] - 1, 0)), k.flip(-1)[:, None, :], groups=64)
            direct.append(y + u * data['D'].double()[None, :, None])
        report['CPU_recomputations']['direct_causal_FIR_FP64_first256'] = operator_delta(*direct)
        assert torch.equal(*direct)
        for dtype in [torch.float32, torch.float64]:
            results = []
            for data in pair:
                u, k, D = data['u'].to(dtype), data['k'].to(dtype), data['D'].to(dtype)
                fft_size = 2 * length
                y = torch.fft.irfft(torch.fft.rfft(u, n=fft_size) * (torch.fft.rfft(k, n=fft_size) / fft_size),
                                    n=fft_size, norm='forward')[..., :length]
                results.append(y + u * D[None, :, None])
            for name, sl in [('first256', slice(0, 256)), ('last256_before_variant', slice(idx - 256, idx))]:
                report['CPU_recomputations'][str(dtype) + '_' + name] = operator_delta(results[0][..., sl], results[1][..., sl])
            report['CPU_recomputations'][str(dtype) + '_max_error_to_direct_REF_first256'] = float((results[0][..., :256].double() - direct[0]).abs().max())
        operator_reports.append(report)
    json_write('operator_precision_comparisons.json', operator_reports)

    with np.load(SCREEN / 'variants' / (CASE + '.npz')) as baseline:
        baseline_comparison = {key: {'exactly_equal': bool(np.array_equal(native_metrics[key], baseline[key])),
                                    'max_abs_difference': float(np.max(np.abs(native_metrics[key] - baseline[key]))),
                                    'allclose_atol1e-7_rtol1e-6': bool(np.allclose(native_metrics[key], baseline[key], atol=1e-7, rtol=1e-6))}
                               for key in baseline.files}
    json_write('historical_baseline_comparison.json', baseline_comparison)

    def metrics_summary(values, idx):
        x = values['relative_l2']
        return {'mean_relative_l2': float(x.mean()), 'pre_variant_mean_relative_l2': float(x[:idx].mean()),
                'post_variant_mean_relative_l2': float(x[idx + 1:].mean()),
                'pre_variant_median_relative_l2': float(np.median(x[:idx])),
                'pre_variant_nonzero_positions': int(np.count_nonzero(x[:idx])),
                'max_relative_l2': float(x.max()), 'max_input_index_0based': int(np.argmax(x)),
                'max_offset_in_input_direction': int(np.argmax(x)) - idx}
    summary = {'case': CASE, 'model': cfg['model'], 'GPU': gpu, 'GPU_forwards': len(forward_records),
               'model_load_seconds': load_seconds, 'peak_allocated_GPU_GiB': peak_gib,
               'forward': metrics_summary(native_metrics, 16384), 'reverse_complement': metrics_summary(rc_metrics, 16383),
               'forward_REF_score': ref_score, 'forward_ALT_score': alt_score,
               'forward_REF_repeat_score': repeat_score, 'RC_REF_score': rc_ref_score, 'RC_ALT_score': rc_alt_score,
               'forward_REF_repeat_max_relative_l2': float(repeat_metrics['relative_l2'].max()),
               'score_exactly_matches_original': ref_score == old['ref_score'] and alt_score == old['alt_score'],
               'all_saved_position_metrics_exactly_match_original': all(x['exactly_equal'] for x in baseline_comparison.values()),
               'native_vs_RC_same_genomic_metric_correlation_descriptive': float(np.corrcoef(native_metrics['relative_l2'], rc_metrics['relative_l2'][::-1])[0, 1]),
               'total_seconds': time.monotonic() - started,
               'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    json_write('summary.json', summary)
    checks = {'status': 'PASSED', 'five_native_forwards': len(forward_records) == 5,
              'unchanged_prefix_and_single_SNV_inputs_verified': True, 'RC_index_and_alleles_verified': True,
              'all_operator_pair_prefix_inputs_identical': True, 'direct_causal_FIR_pair_exactly_equal': True,
              'REF_repeat_difference_metrics_exactly_zero': all(np.count_nonzero(repeat_metrics[key]) == 0 for key in ['relative_l2', 'rms_difference', 'max_abs_difference', 'changed_dimension_fraction']),
              'REF_repeat_cosine_distance_below1e-12': bool(np.max(repeat_metrics['cosine_distance']) < 1e-12),
              'historical_baseline_allclose': all(x['allclose_atol1e-7_rtol1e-6'] for x in baseline_comparison.values()),
              'runtime_libraries': {name: importlib.metadata.version(name) for name in ['torch', 'evo2', 'vtx', 'numpy']},
              'script_sha256': sha(__file__), 'protocol_sha256': sha(OUT / 'protocol.json')}
    if not checks['REF_repeat_difference_metrics_exactly_zero'] or not checks['REF_repeat_cosine_distance_below1e-12'] or not checks['historical_baseline_allclose']:
        checks['status'] = 'REVIEW_REQUIRED'
    json_write('validation.json', checks)
    layer_native = {(x['layer'], x['region']): x for x in layer_records if x['comparison'] == 'forward_REF_vs_ALT'}
    lines = ['# BRCA1 32k 수치·방향 진단', '',
             '고정 변이: GRCh38 chr17:43063900 C>A (cv_868688_43063900_C_A).',
             '기존 32,768bp 입력과 checkpoint를 그대로 사용한 REF → ALT → REF 및 reverse complement REF/ALT, 총 5회입니다.',
             '일반 추론 계산은 수정하지 않았습니다. 정밀도 비교는 저장한 첫 HCM 연산 64채널에서 CPU로 수행했습니다.', '',
             '## 확인 결과', '',
             f"- 기존 전체 위치 지표와 정확히 일치: {summary['all_saved_position_metrics_exactly_match_original']}",
             f"- REF 반복 최대 상대 L2: {summary['forward_REF_repeat_max_relative_l2']:.8g}",
             f"- Forward 변이 이전 평균 상대 L2: {summary['forward']['pre_variant_mean_relative_l2']:.8g}",
             f"- RC 입력에서 변이 이전 평균 상대 L2: {summary['reverse_complement']['pre_variant_mean_relative_l2']:.8g}",
             '', '| Layer | 동일한 첫 256bp 평균 상대 L2 |', '|---|---:|']
    for layer in LAYERS:
        lines.append(f"| {layer} | {layer_native[layer, 'first_256']['mean_relative_l2']:.8g} |")
    lines.extend(['', '## 읽는 방법과 제한', '',
                  'REF와 ALT에서 변이 이전 입력은 동일합니다. 이 구간의 차이를 생물학적 전달 또는 원거리 조절 효과로 해석하면 안 됩니다.',
                  '첫 HCM의 입력과 filter가 같은지 확인하고, 전 구간 FFT의 FP32·FP64 결과와 앞 256bp만 사용하는 직접 causal FIR을 비교했습니다. operator_precision_comparisons.json의 반올림 전 차이와 BF16 변환 후 차이를 함께 보세요.',
                  '이 검증은 고정 변이 한 개의 수치 현상입니다. 모든 변이의 결과가 오류라는 뜻도, 정확도를 교정한 모델을 얻었다는 뜻도 아닙니다. 전체 모델의 FFT를 고정밀도로 바꾼 실험은 포함하지 않았습니다.',
                  'RC는 같은 DNA 구간을 반대 방향으로 읽는 대조입니다. RC 입력 인덱스 16383의 변이는 원래 forward 인덱스 16384와 같은 좌표입니다. *_forward_genomic_order.npz는 RC 위치 축만 뒤집었으며, 각 방향에서 REF 대비 ALT 지표를 먼저 계산했습니다.',
                  '서로 다른 방향의 hidden vector를 직접 빼거나 두 결과를 독립적인 생물학적 반복으로 취급하지 않았습니다.', '',
                  '## 파일', '',
                  '- protocol.json: 추론 전 고정한 순서·입력·checkpoint SHA256·비교 방법.',
                  '- summary.json / validation.json: 핵심 지표·실행 시간·검증 결과.',
                  '- forward_runs.json: 각 추론 점수와 시간.',
                  '- historical_baseline_comparison.json: 기존 저장 NPZ의 모든 지표와 비교.',
                  '- layer_prefix_comparisons.json / layer_prefix_samples.pt: 두 동일 prefix 구간의 layer별 비교와 실제 샘플.',
                  '- first_HCM_operator_samples.pt / operator_precision_comparisons.json: 첫 FFT 연산의 실제 샘플과 CPU 정밀도 대조.',
                  '- *_metrics*.npz: 최종 norm의 기존 정의 그대로 계산한 위치별 지표.', '',
                  f"실행 시간: {summary['total_seconds']:.1f}초. 최대 GPU 할당: {peak_gib:.2f} GiB.", ''])
    (OUT / 'README.md').write_text('\n'.join(lines))
    artifacts = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in OUT.iterdir()
                 if p.is_file() and p.name not in ['progress.json', 'artifact_manifest.json']}
    json_write('artifact_manifest.json', artifacts)
    json_write('progress.json', {'status': 'complete', 'completed_forwards': 5,
                                'validation_status': checks['status'], 'total_seconds': time.monotonic() - started})
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(json.dumps(checks, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if OUT.exists():
            (OUT / 'failure_traceback.txt').write_text(traceback.format_exc())
            json_write('progress.json', {'status': 'failed', 'error': traceback.format_exc()})
        raise
