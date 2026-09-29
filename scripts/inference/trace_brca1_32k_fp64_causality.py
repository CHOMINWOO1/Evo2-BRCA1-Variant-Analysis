#!/usr/bin/env python3
"""Locate full-prefix numerical differences without changing the FP64 pilot math."""
from pathlib import Path
import datetime
import gzip
import hashlib
import importlib.util
import json
import os
import time
import traceback

from diagnose_brca1_32k_numeric import ROOT, SCREEN, CASE, sha, rc
from diagnose_brca1_32k_high_precision import FFT64Patch, CHUNK

BASE = ROOT / 'results/brca1_grch38/numeric_validation_20260916'
PILOT = BASE / 'high_precision_pilot'
OUT = BASE / 'high_precision_trace'
os.environ['HF_HUB_OFFLINE'] = '1'


def write_json(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def compare_prefix(a, b):
    """B,L,C tensors; exact per-position comparisons, with bounded float scratch."""
    import torch
    import numpy as np
    assert a.shape == b.shape and a.shape[0] == 1
    counts = np.zeros(a.shape[1], dtype=np.int32)
    maximum = 0.0
    first = None
    for start in range(0, a.shape[1], 256):
        stop = min(start + 256, a.shape[1])
        x, y = a[:, start:stop], b[:, start:stop]
        neq = x != y
        n = neq.sum(dim=-1)[0].numpy().astype(np.int32)
        counts[start:stop] = n
        if n.any():
            maximum = max(maximum, float((x.double() - y.double()).abs().max()))
            if first is None:
                position = int(np.flatnonzero(n)[0])
                channel = int(torch.nonzero(neq[0, position], as_tuple=False)[0, 0])
                first = {'input_position_0based': start + position, 'channel': channel,
                         'REF_value': float(x[0, position, channel]), 'ALT_value': float(y[0, position, channel])}
    summary = {'exactly_equal': first is None, 'changed_positions': int(np.count_nonzero(counts)),
               'changed_elements': int(counts.sum()), 'max_abs_difference': maximum, 'first_difference': first}
    return summary, counts


class OperatorTracePatch(FFT64Patch):
    def __init__(self, engine, model, torch):
        super().__init__(engine, model, torch)
        self.operator_data = {}
        self.current_kind = None
        self.current_layer = None
        self.original_fir = engine.HyenaInferenceEngine.parallel_fir

    def begin(self, label):
        super().begin(label)
        self.operator_data[label] = {}

    def mark(self, kind, layer):
        super().mark(kind, layer)
        self.current_kind, self.current_layer = kind, layer

    def convolution(self, u, h, dtype, input_transform='rfft'):
        if self.current_layer not in [1, 2]:
            return super().convolution(u, h, dtype, input_transform)
        torch = self.torch
        assert not self.inside_convolution
        self.inside_convolution = True
        variant = 16383 if self.current.startswith('rc') else 16384
        channels, length = u.shape[1:]
        filters = h.reshape(channels, h.shape[-1])
        data = self.operator_data[self.current].setdefault(str(self.current_layer), {})
        data.update({'kind': self.current_kind, 'variant_index_0based': variant,
                     'u': u.detach().cpu().clone(), 'filter': filters.detach().cpu().clone(),
                     'D': self.model.blocks[self.current_layer].filter.D.detach().cpu().clone(),
                     'output_dtype': str(dtype), 'input_transform': input_transform})
        pre_cast = torch.empty((1, channels, variant), dtype=torch.float64)
        result = torch.empty((1, channels, length), dtype=dtype, device=u.device)
        fft_size = 2 * length
        try:
            for start in range(0, channels, CHUNK):
                stop = min(start + CHUNK, channels)
                inputs = u[:, start:stop].to(torch.float64)
                kernel = filters[start:stop].to(torch.float64)
                if input_transform == 'rfft':
                    inputs_f = torch.fft.rfft(inputs, n=fft_size)
                else:
                    assert input_transform == 'fft'
                    inputs_f = torch.fft.fft(inputs, n=fft_size)[..., :fft_size // 2 + 1]
                kernel_f = torch.fft.rfft(kernel, n=fft_size) / fft_size
                product = inputs_f * kernel_f
                value = torch.fft.irfft(product, n=fft_size, norm='forward')[..., :length]
                pre_cast[:, start:stop] = value[..., :variant].detach().cpu()
                result[:, start:stop] = value.to(dtype)
                del inputs, kernel, inputs_f, kernel_f, product, value
        finally:
            self.inside_convolution = False
        data['convolution_FP64_prefix'] = pre_cast
        data['convolution_nativecast_prefix'] = result[..., :variant].detach().cpu().clone()
        return result

    def __enter__(self):
        super().__enter__()
        patched_hcm = self.engine.fftconv_func
        patched_hcl = self.engine.HyenaInferenceEngine.parallel_iir

        def hcm(u, k, D, dropout_mask, **kwargs):
            result = patched_hcm(u, k, D, dropout_mask, **kwargs)
            if kwargs.get('layer_idx') == 1:
                variant = 16383 if self.current.startswith('rc') else 16384
                data = self.operator_data[self.current]['1']
                data['GPU_after_skip_prefix'] = result[..., :variant].detach().cpu().clone()
                data['GPU_after_BF16_prefix'] = result[..., :variant].to(self.torch.bfloat16).detach().cpu().clone()
            return result

        def fir(engine_self, fir_fn, u, weight, bias, L, dims, **kwargs):
            if engine_self.layer_idx == 1 and kwargs.get('gate') and kwargs.get('fir_length') == 128:
                assert not kwargs.get('column_split_hyena') and not engine_self.hyena_flip_x1x2
                variant = 16383 if self.current.startswith('rc') else 16384
                self.operator_data[self.current].setdefault('1', {})['x2_prefix'] = u[:, :dims[0], :variant].detach().cpu().clone()
            result = self.original_fir(engine_self, fir_fn, u, weight, bias, L, dims, **kwargs)
            if engine_self.layer_idx == 1 and kwargs.get('gate') and kwargs.get('fir_length') == 128:
                self.operator_data[self.current]['1']['GPU_after_gate_prefix'] = result[0][..., :variant].detach().cpu().clone()
            return result

        def hcl(engine_self, z_pre, h, D, L, poles, residues, t, dims, layer_idx, **kwargs):
            if layer_idx == 2:
                variant = 16383 if self.current.startswith('rc') else 16384
                self.operator_data[self.current].setdefault('2', {})['x2_prefix'] = z_pre[:, :dims[0], :variant].detach().cpu().clone()
            result = patched_hcl(engine_self, z_pre, h, D, L, poles, residues, t, dims, layer_idx, **kwargs)
            if layer_idx == 2:
                self.operator_data[self.current]['2']['GPU_after_gate_prefix'] = result[:, :variant].permute(0, 2, 1).detach().cpu().clone()
            return result

        self.engine.fftconv_func = hcm
        self.engine.HyenaInferenceEngine.parallel_fir = fir
        self.engine.HyenaInferenceEngine.parallel_iir = hcl
        return self

    def __exit__(self, *args):
        self.engine.HyenaInferenceEngine.parallel_fir = self.original_fir
        return super().__exit__(*args)


def main():
    import numpy as np
    import torch
    from evo2 import Evo2
    from evo2.scoring import logits_to_logprobs
    import vortex.model.engine as engine
    started = time.monotonic()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'validation.json').exists():
        raise RuntimeError('Completed trace already exists; preserve it.')
    cfg = json.loads((SCREEN / 'config.json').read_text())
    old = json.loads((SCREEN / 'variants' / (CASE + '.json')).read_text())
    prior_protocol = json.loads((PILOT / 'protocol.json').read_text())
    with gzip.open(cfg['reference_fasta'], 'rt') as f:
        assert next(f).startswith('>chr17')
        chromosome = ''.join(line.strip().upper() for line in f)
    start0 = old['window_start_grch38_1based'] - 1
    ref = chromosome[start0:start0 + 32768]
    alt = ref[:16384] + 'A' + ref[16385:]
    assert ref[16384] == 'C'
    del chromosome
    sequences = {'forward_ref': ref, 'forward_alt': alt, 'rc_ref': rc(ref), 'rc_alt': rc(alt)}
    hashes = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sequences.items()}
    assert all(hashes[k] == prior_protocol['sequence_sha256'][k] for k in hashes)
    checkpoint_sha = sha(cfg['checkpoint'])
    assert checkpoint_sha == prior_protocol['checkpoint_sha256']
    protocol = {
        'frozen_before_model_load_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'case': CASE, 'assembly': 'GRCh38', 'length': 32768, 'forward_variant_index': 16384, 'RC_variant_index': 16383,
        'sequence_order': list(sequences), 'sequence_sha256': hashes, 'checkpoint_sha256': checkpoint_sha,
        'checkpoint_revision': cfg['checkpoint_revision'], 'script_sha256': sha(__file__),
        'FP64_patch_script_sha256': sha(ROOT / 'scripts/inference/diagnose_brca1_32k_high_precision.py'),
        'FP64_prior_protocol_sha256': sha(PILOT / 'protocol.json'), 'installed_engine_sha256': sha(engine.__file__),
        'intervention': 'Identical FP64 HCM/HCL128-channel convolution mathematics and native cast/skip/gate boundaries as prior pilot; only observability changes.',
        'capture': 'Whole unchanged input-prefix, all4096channels, every block + embedding/norm; REF cached on CPU, ALT compared in256-position chunks. Persist counts and first coordinates, not full hidden arrays.',
        'operators': 'First HCM(block1) and HCL(block2), all4096channels: actual input, filter, FP64 convolution prefix, actual native-cast prefix, D and x2; keep in hostmemory only until differences identified.',
        'saved_operator_limit': 'Up to16 channels per operator/orientation selected from first native-cast/skip/BF16/gate mismatches; at most4 first points per stage.',
        'operator_stage_comparison': 'Actual native-cast output; CPU replay of same-dtype elementwise D addition/BF16/gate using actual captured operands. Independent causal sums in a separate review.',
        'scope': 'FP64 pilot did NOT remove unchanged-prefix differences. This trace locates residual sources; no new broad screen or scientific correction is assumed.',
    }
    write_json('protocol.json', protocol)
    assert torch.cuda.is_available()
    spec = importlib.util.spec_from_file_location('native_metrics', ROOT / 'scripts/inference/run_brca1_grch38_screen.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    torch.cuda.reset_peak_memory_stats()
    model = Evo2(cfg['model'], local_path=cfg['checkpoint'])
    model.model.eval()
    assert not model.model.config.use_fp8_input_projections and not model.model.config.use_flashfft
    assert not model.model.config.column_split_hyena and not model.model.config.hyena_flip_x1x2
    versions = {k: p._version for k, p in model.model.named_parameters()}
    for i in model.model.config.hcm_layer_idxs + model.model.config.hcl_layer_idxs + model.model.config.hcs_layer_idxs:
        e = model.model.blocks[i].filter.engine
        assert not e.use_hcm_kernel and not e.use_hcl_kernel and not e.use_hcs_kernel
    layer_names = ['embedding_layer'] + [f'blocks.{i}' for i in range(32)] + ['norm']
    current = {'label': None, 'orientation': None, 'index': None}
    reference_prefixes, traces, counts, runs, saved_operators, operator_summaries, comparisons = {}, [], {}, [], {}, [], []

    def block_hook(name):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            prefix = output[:, :current['index']].detach().cpu().clone()
            if current['label'].endswith('_ref'):
                reference_prefixes[name] = prefix
            else:
                result, per_position = compare_prefix(reference_prefixes[name], prefix)
                traces.append({'orientation': current['orientation'], 'layer': name, **result})
                counts[current['orientation'] + '__' + name] = per_position
        return hook

    def run(label, patch):
        current.update(label=label, orientation='RC' if label.startswith('rc') else 'forward', index=16383 if label.startswith('rc') else 16384)
        patch.begin(label)
        write_json('progress.json', {'status': 'running', 'forward': label, 'completed': len(runs)})
        t = time.monotonic()
        ids = torch.tensor(model.tokenizer.tokenize(sequences[label]), dtype=torch.long, device='cuda:0')[None]
        with torch.inference_mode():
            output, embeddings = model(ids, return_embeddings=True, layer_names=['norm'])
            hidden = embeddings['norm'].detach().cpu()
            score = float(logits_to_logprobs(output[0], ids).float().mean().item())
        torch.cuda.synchronize()
        patch.check(label)
        del output, embeddings, ids
        runs.append({'label': label, 'seconds': time.monotonic() - t, 'score': score, 'coverage': patch.records[label]})
        write_json('forward_runs.json', runs)
        print(json.dumps({'label': label, 'seconds': runs[-1]['seconds'], 'score': score}), flush=True)
        return hidden

    def analyse_operator_pair(patch, ref_label, alt_label, orientation):
        for layer in ['1', '2']:
            a, b = patch.operator_data[ref_label][layer], patch.operator_data[alt_label][layer]
            idx = a['variant_index_0based']
            assert torch.equal(a['filter'], b['filter']) and torch.equal(a['D'], b['D'])
            stages = {}
            for side, data in [('REF', a), ('ALT', b)]:
                u = data['u'][..., :idx]
                y = data['convolution_nativecast_prefix']
                after_skip = y + u * data['D'][None, :, None]
                # Native HCM caller casts the FP32 skip result; HCL y and skip
                # already BF16. Both then use the original BF16 x2 gate.
                after_bf16 = after_skip.to(torch.bfloat16)
                assert data['D'].dtype == data['x2_prefix'].dtype == torch.bfloat16
                assert after_skip.dtype == (torch.float32 if layer == '1' else torch.bfloat16)
                data['after_skip_prefix'] = after_skip
                data['after_BF16_prefix'] = after_bf16
                data['after_gate_prefix'] = after_bf16 * data['x2_prefix']
                assert torch.equal(data['after_gate_prefix'], data['GPU_after_gate_prefix']), 'CPU replay differs from actual GPU gate output'
                if layer == '1':
                    assert torch.equal(after_skip, data['GPU_after_skip_prefix'])
                    assert torch.equal(after_bf16, data['GPU_after_BF16_prefix'])
                stages[side] = {'input': u, 'convolution_FP64': data['convolution_FP64_prefix'],
                                'convolution_nativecast': y, 'after_skip': after_skip,
                                'after_BF16': after_bf16, 'after_gate': data['after_gate_prefix']}
            channels, points = [], []
            for stage in stages['REF']:
                x, y = stages['REF'][stage], stages['ALT'][stage]
                result, per_position = compare_prefix(x.permute(0, 2, 1), y.permute(0, 2, 1))
                operator_summaries.append({'orientation': orientation, 'layer': layer, 'kind': a['kind'], 'stage': stage, **result})
                counts[orientation + '__operator_' + layer + '__' + stage] = per_position
                if stage in ['convolution_nativecast', 'after_skip', 'after_BF16', 'after_gate'] and not result['exactly_equal']:
                    added = 0
                    for position in np.flatnonzero(per_position):
                        for channel in torch.nonzero(x[0, :, position] != y[0, :, position]).flatten().tolist():
                            if channel not in channels and len(channels) < 16:
                                channels.append(channel)
                            if channel in channels:
                                points.append({'stage': stage, 'input_position_0based': int(position), 'channel_original': channel,
                                               'REF_value': float(x[0, channel, position]), 'ALT_value': float(y[0, channel, position])})
                                added += 1
                            if added >= 4:
                                break
                        if added >= 4:
                            break
            # In an exact operator, retain no unrelated channels just to make examples.
            selected = {'orientation': orientation, 'layer': layer, 'kind': a['kind'],
                        'variant_index_0based': idx, 'channels_original': channels, 'points': points,
                        'notes': 'Convolution values and gate output captured from GPU. HCM skip/BF16 also GPU-captured; CPU replay asserted exactly equal. HCL intermediate skip is CPU replay with actual BF16 operands; gate agrees exactly with GPU.',
                        'REF': {}, 'ALT': {}}
            for side, data in [('REF', a), ('ALT', b)]:
                for key in ['u', 'convolution_FP64_prefix', 'convolution_nativecast_prefix', 'x2_prefix', 'after_skip_prefix', 'after_BF16_prefix', 'after_gate_prefix']:
                    selected[side][key] = data[key][:, channels].clone()
                selected[side]['filter'] = data['filter'][channels].clone()
                selected[side]['D'] = data['D'][channels].clone()
                for key in ['GPU_after_skip_prefix', 'GPU_after_BF16_prefix', 'GPU_after_gate_prefix']:
                    if key in data:
                        selected[side][key] = data[key][:, channels].clone()
            saved_operators[orientation + '_block' + layer] = selected
            del stages
        torch.save(saved_operators, OUT / 'operator_selected_samples.pt')
        write_json('operator_stage_comparisons.json', operator_summaries)
        write_json('full_prefix_layer_comparisons.json', traces)
        np.savez_compressed(OUT / 'per_position_mismatch_counts.npz', **counts)
        del patch.operator_data[ref_label], patch.operator_data[alt_label]

    hooks = [model.model.get_submodule(name).register_forward_hook(block_hook(name)) for name in layer_names]
    try:
        with OperatorTracePatch(engine, model.model, torch) as patch:
            for orientation, ref_label, alt_label, filename in [('forward', 'forward_ref', 'forward_alt', 'forward_ref_alt_metrics.npz'), ('RC', 'rc_ref', 'rc_alt', 'rc_ref_alt_metrics_input_order.npz')]:
                reference = run(ref_label, patch)
                alternate = run(alt_label, patch)
                metrics = original.position_metrics(reference, alternate)
                np.savez_compressed(OUT / filename, **metrics)
                with np.load(PILOT / filename) as previous:
                    same = {key: bool(np.array_equal(metrics[key], previous[key])) for key in previous.files}
                comparisons.append({'orientation': orientation, 'exactly_equal_to_prior_FP64_pilot': same})
                del reference, alternate
                reference_prefixes.clear()
                analyse_operator_pair(patch, ref_label, alt_label, orientation)
            coverage = patch.records
    finally:
        for hook in hooks:
            hook.remove()
    assert versions == {k: p._version for k, p in model.model.named_parameters()}
    peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
    del model
    torch.cuda.empty_cache()
    first_layers = {orientation: next((row for row in traces if row['orientation'] == orientation and not row['exactly_equal']), None) for orientation in ['forward', 'RC']}
    summary = {'case': CASE, 'forwards': len(runs), 'total_seconds': time.monotonic() - started, 'peak_allocated_GPU_GiB': peak_gib,
               'first_divergent_layer_entire_prefix': first_layers, 'prior_pilot_comparison': comparisons,
               'FP64_is_not_a_complete_causal_numeric_remedy': True,
               'operator_stage_replay_caution': 'GPU convolution captures are actual. Skip/gate stage arithmetic is replayed on CPU from actual operands and must be distinguished from raw GPU stage capture.',
               'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json('summary.json', summary)
    write_json('FFT_layer_coverage.json', coverage)
    validation = {'status': 'PASSED', 'full_prefix_all4096_channels_at_all34_layer_outputs': True,
                  'operator_all4096_channels_compared_before_selection': True,
                  'saved_channels_at_most16_per_operator': all(len(v['channels_original']) <= 16 for v in saved_operators.values()),
                  'native_and_prior_pilot_preserved': True, 'parameter_versions_unchanged': True,
                  'engine_file_unchanged': sha(engine.__file__) == protocol['installed_engine_sha256'],
                  'trace_matches_prior_pilot_exactly': all(all(row['exactly_equal_to_prior_FP64_pilot'].values()) for row in comparisons),
                  'script_sha256': sha(__file__), 'protocol_sha256': sha(OUT / 'protocol.json')}
    if not validation['trace_matches_prior_pilot_exactly']:
        validation['status'] = 'REVIEW_REQUIRED'
    write_json('validation.json', validation)
    lines = ['# FP64 파일럿에 남은 차이의 전체 prefix 추적', '',
             'FP64 FFT로 바꾼 이전 파일럿에서도 변이 이전 차이가 남았습니다. 이번에는 일부 위치만 보지 않고, 변이 이전 전체 위치 × 모든 4096채널을 모든 32개 block과 embedding/norm에서 비교했습니다.',
             '앞선 FFT 고정밀 계산 방식은 그대로 유지하고 관측만 추가했습니다. 기존 native 실험과 FP64 파일럿은 보존했습니다.', '',
             '## 최초 차이', '']
    for orientation, row in first_layers.items():
        lines.append(f"- {orientation}: {row['layer']}, 최초 입력 위치 {row['first_difference']['input_position_0based']} (0부터 시작), 채널 {row['first_difference']['channel']}, 다른 값 {row['changed_elements']}개.")
    lines += ['', '## 파일', '',
              '- full_prefix_layer_comparisons.json: 모든 층의 정확한 불일치 수·최초 위치/채널·최대 차이.',
              '- operator_stage_comparisons.json: 첫 HCM/HCL의 입력, FP64 합성곱, native cast 및 skip/BF16/gate 단계별 전체 prefix 비교.',
              '- per_position_mismatch_counts.npz: 각 층과 연산 단계의 위치별 불일치 채널 수.',
              '- operator_selected_samples.pt: 최초 차이가 나타난 채널(최대16개)과 해당 지점, 전체 input/filter 및 GPU에서 캡처한 cast 전후 값.',
              '- after_skip, after_BF16, after_gate는 실제 GPU operand로 CPU에서 같은 dtype/계산 순서를 재현한 값입니다. GPU에서 직접 캡처한 convolution 값과 구분해야 합니다.',
              '- summary.json, validation.json: 앞선 FP64 파일럿과의 재현 비교 및 실행 확인.', '',
              '반올림 중간값 때문인지는 저장한 입력의 직접 causal 합과 반올림 경계 비교로 별도 확인해야 합니다. 극히 작은 오차가 원인일 수 있다는 가설만으로 기전을 확정하지 않습니다.',
              '변이 이전 차이를 강제로 0으로 만들거나 전체 데이터에 새 점수를 덮어쓰지 않았습니다.', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    write_json('artifact_manifest.json', {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in OUT.iterdir() if p.is_file() and p.name not in ['artifact_manifest.json', 'progress.json']})
    write_json('progress.json', {'status': 'complete', 'forwards': len(runs), 'validation_status': validation['status']})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'failure_traceback.txt').write_text(traceback.format_exc())
        write_json('progress.json', {'status': 'failed', 'error': traceback.format_exc()})
        raise
