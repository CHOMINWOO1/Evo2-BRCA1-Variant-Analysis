#!/usr/bin/env python3
"""One-case FP64 FFT pilot; runtime patches only, unchanged checkpoint and gating."""
from pathlib import Path
import csv
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import time
import traceback

from diagnose_brca1_32k_numeric import ROOT, SCREEN, CASE, sha, rc

BASE = ROOT / 'results/brca1_grch38/numeric_validation_20260916'
OUT = BASE / 'high_precision_pilot'
CHUNK = 128
os.environ['HF_HUB_OFFLINE'] = '1'


def write_json(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def chunked_fft64(u, filters, output_dtype, chunk_size=CHUNK, input_transform='rfft'):
    """Exact mathematical causal convolution, with FP64 FFT and native output cast."""
    import torch
    assert u.ndim == 3
    batch, channels, length = u.shape
    assert filters.numel() == channels * filters.shape[-1]
    filters = filters.reshape(channels, filters.shape[-1])
    assert filters.shape[-1] <= length
    result = torch.empty((batch, channels, length), dtype=output_dtype, device=u.device)
    fft_size = 2 * length
    for start in range(0, channels, chunk_size):
        end = min(start + chunk_size, channels)
        inputs = u[:, start:end, :].to(torch.float64)
        kernel = filters[start:end, :].to(torch.float64)
        if input_transform == 'rfft':
            inputs_f = torch.fft.rfft(inputs, n=fft_size)
        elif input_transform == 'fft':
            # Native HCL uses a full complex FFT followed by positive-frequency
            # slicing; preserve that transform family as well as its math.
            inputs_f = torch.fft.fft(inputs, n=fft_size)[..., :fft_size // 2 + 1]
        else:
            raise ValueError(input_transform)
        kernel_f = torch.fft.rfft(kernel, n=fft_size) / fft_size
        product = inputs_f * kernel_f
        value = torch.fft.irfft(product, n=fft_size, norm='forward')[..., :length]
        result[:, start:end, :] = value.to(output_dtype)
        del inputs, kernel, inputs_f, kernel_f, product, value
    return result


class FFT64Patch:
    """Cover all active stateless HCM/HCL FFT paths and reject untracked FFTs."""
    def __init__(self, engine, model, torch):
        self.engine, self.model, self.torch = engine, model, torch
        self.original_hcm = engine.fftconv_func
        self.original_hcl = engine.HyenaInferenceEngine.parallel_iir
        self.original_fft = {name: getattr(torch.fft, name) for name in ['rfft', 'irfft', 'fft']}
        self.hcm_layers = set(model.config.hcm_layer_idxs)
        self.hcl_layers = set(model.config.hcl_layer_idxs)
        self.current = None
        self.inside_convolution = False
        self.records = {}

    def begin(self, label):
        self.current = label
        self.records[label] = {'HCM': {}, 'HCL': {}, 'fft_calls': {'rfft': 0, 'irfft': 0, 'fft': 0}}

    def mark(self, kind, layer):
        record = self.records[self.current][kind]
        record[str(layer)] = record.get(str(layer), 0) + 1

    def convolution(self, u, h, dtype, input_transform='rfft'):
        assert not self.inside_convolution
        self.inside_convolution = True
        try:
            return chunked_fft64(u, h, dtype, input_transform=input_transform)
        finally:
            self.inside_convolution = False

    def guard(self, name):
        def wrapped(input, *args, **kwargs):
            assert self.inside_convolution, f'Unexpected FFT outside patched convolution: {name}'
            expected = self.torch.complex128 if name == 'irfft' else self.torch.float64
            assert input.dtype == expected, f'Unexpected FFT precision {input.dtype} for {name}'
            self.records[self.current]['fft_calls'][name] += 1
            return self.original_fft[name](input, *args, **kwargs)
        return wrapped

    def __enter__(self):
        torch = self.torch

        def hcm(u, k, D, dropout_mask, gelu=True, k_rev=None, bidirectional=False,
                print_activations=False, layer_idx=None, **kwargs):
            assert layer_idx in self.hcm_layers and not bidirectional and k_rev is None
            assert dropout_mask is None and gelu is False and not print_activations
            assert u.dtype == torch.float32 and k.dtype == torch.float32
            assert u.ndim == 3 and u.shape[0] == 1 and u.shape[1] == 4096
            self.mark('HCM', layer_idx)
            # Preserve the original HCM arithmetic boundary: y is FP32 before
            # FP32 addition of u * D. The original caller later casts to BF16.
            y = self.convolution(u, k, u.dtype)
            return (y + u * D.unsqueeze(-1)).to(u.dtype)

        def hcl(engine_self, z_pre, h, D, L, poles, residues, t, dims, layer_idx,
                inference_params=None, prefill_style='fft', fftconv_fn=None,
                padding_mask=None, use_flashfft=False, column_split_hyena=False,
                long_fir_threshold=None):
            assert layer_idx in self.hcl_layers
            assert inference_params is None and prefill_style == 'fft'
            assert padding_mask is None and not use_flashfft and long_fir_threshold is None
            assert not column_split_hyena and not engine_self.hyena_flip_x1x2
            assert not engine_self.use_hcl_kernel and not engine_self.print_activations
            hidden_size = dims[0]
            assert z_pre.ndim == 3 and z_pre.shape == (1, 3 * hidden_size, L)
            self.mark('HCL', layer_idx)
            x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)
            x1v = x1 * v
            # Preserve native HCL order: convolution -> BF16 -> skip -> gate.
            y = self.convolution(x1v, h, x1v.dtype, input_transform='fft')
            y = (y + x1v * D.unsqueeze(-1)) * x2
            return y.permute(0, 2, 1)

        self.engine.fftconv_func = hcm
        self.engine.HyenaInferenceEngine.parallel_iir = hcl
        for name in self.original_fft:
            setattr(torch.fft, name, self.guard(name))
        return self

    def check(self, label):
        record = self.records[label]
        assert record['HCM'] == {str(k): 1 for k in self.hcm_layers}
        assert record['HCL'] == {str(k): 1 for k in self.hcl_layers}
        chunks = (4096 + CHUNK - 1) // CHUNK
        convolutions = len(self.hcm_layers) + len(self.hcl_layers)
        assert record['fft_calls'] == {'rfft': (2 * len(self.hcm_layers) + len(self.hcl_layers)) * chunks,
                                       'irfft': convolutions * chunks, 'fft': len(self.hcl_layers) * chunks}

    def __exit__(self, *args):
        self.engine.fftconv_func = self.original_hcm
        self.engine.HyenaInferenceEngine.parallel_iir = self.original_hcl
        for name, original in self.original_fft.items():
            setattr(self.torch.fft, name, original)


def cpu_math_checks():
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(20260916)
    results = []
    for length, order in [(61, 7), (61, 61)]:
        u = torch.randn((2, 5, length), dtype=torch.float32, generator=generator)
        kernel = torch.randn((5, order), dtype=torch.float32, generator=generator)
        direct = F.conv1d(F.pad(u.double(), (order - 1, 0)), kernel.double().flip(-1)[:, None], groups=5)
        for dtype in [torch.float32, torch.bfloat16]:
            for transform in ['rfft', 'fft']:
                actual = chunked_fft64(u, kernel, dtype, chunk_size=2, input_transform=transform)
                assert torch.equal(actual, direct.to(dtype))
                results.append({'length': length, 'filter_length': order, 'output_dtype': str(dtype),
                                'input_transform': transform, 'same_as_direct_causal_FIR_after_cast': True})
    samples = torch.load(BASE / 'first_HCM_operator_samples.pt', map_location='cpu', weights_only=True)
    for names in [('forward_ref', 'forward_alt'), ('rc_ref', 'rc_alt')]:
        outputs = []
        for name in names:
            sample = samples[name]
            y = chunked_fft64(sample['u'], sample['k'], torch.float32)
            outputs.append((y + sample['u'] * sample['D'][None, :, None]).to(torch.bfloat16))
        idx = samples[names[0]]['variant_index_0based']
        identical = torch.equal(outputs[0][..., :idx], outputs[1][..., :idx])
        assert identical
        results.append({'captured_pair': list(names), 'whole_unchanged_prefix_equal_after_native_HCM_D_term_and_BF16': identical})
    return results


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'validation.json').exists():
        raise RuntimeError('Completed pilot exists; preserve it.')
    import gzip
    import numpy as np
    import torch
    from evo2 import Evo2
    from evo2.scoring import logits_to_logprobs
    import vortex.model.engine as engine
    torch.set_num_threads(4)
    cfg = json.loads((SCREEN / 'config.json').read_text())
    native_protocol = json.loads((BASE / 'protocol.json').read_text())
    native_summary = json.loads((BASE / 'summary.json').read_text())
    old = json.loads((SCREEN / 'variants' / (CASE + '.json')).read_text())
    with gzip.open(cfg['reference_fasta'], 'rt') as f:
        assert next(f).startswith('>chr17')
        chromosome = ''.join(line.strip().upper() for line in f)
    length, idx = 32768, 16384
    start0 = old['window_start_grch38_1based'] - 1
    ref = chromosome[start0:start0 + length]
    alt = ref[:idx] + 'A' + ref[idx + 1:]
    assert ref[idx] == 'C'
    del chromosome
    sequences = {'forward_ref': ref, 'forward_alt': alt, 'forward_ref_repeat': ref, 'rc_ref': rc(ref), 'rc_alt': rc(alt)}
    assert {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sequences.items()} == native_protocol['sequence_sha256']
    checkpoint_sha = sha(cfg['checkpoint'])
    assert checkpoint_sha == native_protocol['checkpoint_sha256']
    math_checks = cpu_math_checks()
    write_json('CPU_math_checks.json', math_checks)
    protocol = {
        'frozen_before_model_load_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'case': CASE, 'assembly': 'GRCh38', 'window_start_1based': start0 + 1, 'length_bp': length,
        'variant_index_0based': {'forward': 16384, 'reverse_complement': 16383},
        'sequence_sha256': native_protocol['sequence_sha256'], 'checkpoint': cfg['checkpoint'],
        'checkpoint_sha256': checkpoint_sha, 'checkpoint_revision': cfg['checkpoint_revision'],
        'original_native_protocol_sha256': sha(BASE / 'protocol.json'), 'script_sha256': sha(__file__),
        'helper_script_sha256': sha(ROOT / 'scripts/inference/diagnose_brca1_32k_numeric.py'),
        'installed_engine_sha256': sha(engine.__file__), 'channel_chunk': CHUNK,
        'patch': 'Runtime only. HCM9 input rfft, HCL9 input fft with positive-frequency slice; filters rfft; inverse irfft. All FP64/complex128, zero-padded n=2L, slice first L, native cast boundary retained. HCS9 direct convolution remains native.',
        'HCM_boundary': 'Convolution y cast to FP32 BEFORE original FP32 y + u*D; original caller casts to BF16 and gates.',
        'HCL_boundary': 'Convolution y cast to x1v BF16 BEFORE original BF16 (y + x1v*D)*x2.',
        'unchanged': 'Checkpoint, filter construction, short convolutions, attention, normalizations, projections, MLPs, skip/gate expressions, input lengths/orientations.',
        'coverage_audit': '9 HCM + 9 HCL calls exactly once per forward. Guard every torch.fft rfft/fft/irfft call; require patched convolution context and float64/complex128.',
        'mandatory_passes': ['forward_ref', 'forward_alt', 'forward_ref_repeat'],
        'RC_policy': 'Run rc_ref+rc_alt only if elapsed since model load start plus twice maximum observed forward seconds is <=600 seconds; otherwise record time-budget omission.',
        'precision_cost': 'FP64 FFT on RTX6000 Ada can be substantially slower; chunking limits FFT scratch memory, not arithmetic cost. No whole-model FP64 conversion.',
        'primary_checks': 'REF repeat exactness; whole unchanged-prefix final-norm L2 zero or residual measured; all-layer prefix probes; surviving post-variant metric and score differences.',
        'scope': 'One fixed candidate. Higher precision FFT is a numerical intervention, not a new checkpoint or an established corrected biological score. RC equality is not required for a causal model.',
    }
    if (OUT / 'protocol.json').exists():
        prior = json.loads((OUT / 'protocol.json').read_text())
        assert all(prior[k] == v for k, v in protocol.items() if k != 'frozen_before_model_load_utc')
    else:
        write_json('protocol.json', protocol)
    print('FP64 FFT pilot protocol frozen; CPU math checks passed', flush=True)
    assert torch.cuda.is_available()
    spec = importlib.util.spec_from_file_location('native_metrics', ROOT / 'scripts/inference/run_brca1_grch38_screen.py')
    metrics_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics_module)
    torch.cuda.reset_peak_memory_stats()
    inference_started = time.monotonic()
    model = Evo2(cfg['model'], local_path=cfg['checkpoint'])
    model.model.eval()
    model_load_seconds = time.monotonic() - inference_started
    assert not model.model.config.use_fp8_input_projections
    assert not model.model.config.use_flashfft
    assert not model.model.config.column_split_hyena and not model.model.config.hyena_flip_x1x2
    assert all(not m.training for m in model.model.modules())
    assert len(model.model.config.hcm_layer_idxs) == len(model.model.config.hcl_layer_idxs) == len(model.model.config.hcs_layer_idxs) == 9
    for layer in model.model.config.hcm_layer_idxs + model.model.config.hcl_layer_idxs + model.model.config.hcs_layer_idxs:
        native_engine = model.model.blocks[layer].filter.engine
        assert not native_engine.use_hcm_kernel and not native_engine.use_hcl_kernel and not native_engine.use_hcs_kernel
    parameter_versions = {name: p._version for name, p in model.model.named_parameters()}
    layer_names = ['embedding_layer'] + [f'blocks.{i}' for i in range(32)] + ['norm']
    hcs_calls, active, probes, scores, runs = {}, {'label': None, 'index': idx}, {}, {}, []

    def hook_for(name):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            index = active['index']
            probes[active['label']][name] = {
                'first_256': output[:, :256].detach().cpu().clone(),
                'last_256_before_variant': output[:, index - 256:index].detach().cpu().clone(),
            }
            if name.startswith('blocks.') and int(name.split('.')[1]) in model.model.config.hcs_layer_idxs:
                hcs_calls[active['label']].append(int(name.split('.')[1]))
        return hook

    def run_forward(label, patch):
        active.update(label=label, index=16383 if label.startswith('rc') else 16384)
        probes[label], hcs_calls[label] = {}, []
        patch.begin(label)
        write_json('progress.json', {'status': 'running', 'current_forward': label, 'completed_forwards': len(runs)})
        t = time.monotonic()
        ids = torch.tensor(model.tokenizer.tokenize(sequences[label]), dtype=torch.long, device='cuda:0')[None]
        with torch.inference_mode():
            output, embeddings = model(ids, return_embeddings=True, layer_names=['norm'])
            hidden = embeddings['norm'].detach().cpu()
            score = float(logits_to_logprobs(output[0], ids).float().mean().item())
        torch.cuda.synchronize()
        assert hidden.shape == (1, length, 4096) and torch.isfinite(hidden).all()
        del output, embeddings, ids
        patch.check(label)
        assert hcs_calls[label] == model.model.config.hcs_layer_idxs
        scores[label] = score
        runs.append({'label': label, 'seconds': time.monotonic() - t, 'score': score,
                     'dtype': str(hidden.dtype), 'coverage': patch.records[label], 'HCS_native_layers': hcs_calls[label]})
        write_json('forward_runs.json', runs)
        print(json.dumps({'label': label, 'seconds': runs[-1]['seconds'], 'score': score, 'HCM_HCL_coverage': '9+9 passed'}), flush=True)
        return hidden

    hooks = [model.model.get_submodule(name).register_forward_hook(hook_for(name)) for name in layer_names]
    comparison_records = []

    def compare_probes(a, b, comparison):
        for name in layer_names:
            for region in probes[a][name]:
                x, y = probes[a][name][region], probes[b][name][region]
                delta = y.float() - x.float()
                comparison_records.append({'comparison': comparison, 'layer': name, 'region': region,
                                           'exactly_equal': bool(torch.equal(x, y)), 'unequal_elements': int((x != y).sum()),
                                           'max_abs_difference': float(delta.abs().max()),
                                           'mean_relative_l2': float((delta.norm(dim=-1) / x.float().norm(dim=-1).clamp_min(1e-30)).mean())})

    rc_metrics = None
    try:
        with FFT64Patch(engine, model.model, torch) as patch:
            reference = run_forward('forward_ref', patch)
            alternate = run_forward('forward_alt', patch)
            forward_metrics = metrics_module.position_metrics(reference, alternate)
            np.savez_compressed(OUT / 'forward_ref_alt_metrics.npz', **forward_metrics)
            del alternate
            repeated = run_forward('forward_ref_repeat', patch)
            repeat_metrics = metrics_module.position_metrics(reference, repeated)
            np.savez_compressed(OUT / 'forward_ref_repeat_metrics.npz', **repeat_metrics)
            del repeated, reference
            compare_probes('forward_ref', 'forward_alt', 'forward_REF_vs_ALT')
            compare_probes('forward_ref', 'forward_ref_repeat', 'REF_before_vs_after_ALT')
            estimated_finish = time.monotonic() - inference_started + 2 * max(r['seconds'] for r in runs)
            if estimated_finish <= 600:
                rc_reference = run_forward('rc_ref', patch)
                rc_alternate = run_forward('rc_alt', patch)
                rc_metrics = metrics_module.position_metrics(rc_reference, rc_alternate)
                np.savez_compressed(OUT / 'rc_ref_alt_metrics_input_order.npz', **rc_metrics)
                np.savez_compressed(OUT / 'rc_ref_alt_metrics_forward_genomic_order.npz', **{k: v[::-1].copy() for k, v in rc_metrics.items()})
                del rc_reference, rc_alternate
                compare_probes('rc_ref', 'rc_alt', 'RC_REF_vs_ALT')
                rc_status = 'completed'
            else:
                rc_status = 'omitted_by_frozen_600_second_rule'
            patch_coverage = patch.records
    finally:
        for hook in hooks:
            hook.remove()
    assert parameter_versions == {name: p._version for name, p in model.model.named_parameters()}
    peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
    gpu = torch.cuda.get_device_name(0)
    del model
    torch.cuda.empty_cache()
    write_json('layer_prefix_comparisons.json', comparison_records)
    # Keep compact reproducibility samples, not full 4096-dim layer probes.
    compact = {label: {name: {region: tensor[..., :64].clone() for region, tensor in regions.items()}
                        for name, regions in layers.items()} for label, layers in probes.items()}
    torch.save(compact, OUT / 'all_layer_prefix_first64channels.pt')
    native_probes = torch.load(BASE / 'layer_prefix_samples.pt', map_location='cpu', weights_only=True)
    intervention = []
    for label in probes:
        for name in native_probes[label]:
            for region, original in native_probes[label][name].items():
                revised = probes[label][name][region]
                d = revised.float() - original.float()
                intervention.append({'input': label, 'layer': name, 'region': region,
                                     'native_vs_FP64_exactly_equal': bool(torch.equal(original, revised)),
                                     'mean_relative_l2': float((d.norm(dim=-1) / original.float().norm(dim=-1).clamp_min(1e-30)).mean())})
    write_json('precision_intervention_layer_changes.json', intervention)

    def describe(values, variant_index):
        x = values['relative_l2']
        return {'mean_relative_l2': float(x.mean()), 'pre_variant_mean_relative_l2': float(x[:variant_index].mean()),
                'pre_variant_max_relative_l2': float(x[:variant_index].max()),
                'pre_variant_nonzero_positions': int(np.count_nonzero(x[:variant_index])),
                'post_variant_mean_relative_l2': float(x[variant_index + 1:].mean()),
                'variant_pm20_mean_relative_l2': float(x[variant_index - 20:variant_index + 21].mean()),
                'variant_to_plus127_mean_relative_l2': float(x[variant_index:variant_index + 128].mean()),
                'max_relative_l2': float(x.max()), 'max_input_index_0based': int(np.argmax(x)),
                'max_offset_in_input_direction': int(np.argmax(x)) - variant_index}
    with np.load(BASE / 'forward_ref_alt_metrics.npz') as archive:
        native_forward = {k: archive[k].copy() for k in archive.files}
    with np.load(BASE / 'rc_ref_alt_metrics_input_order.npz') as archive:
        native_rc = {k: archive[k].copy() for k in archive.files}
    regions = {'forward_native': describe(native_forward, 16384), 'forward_FP64_FFT': describe(forward_metrics, 16384),
               'RC_native': describe(native_rc, 16383)}
    if rc_metrics is not None:
        regions['RC_FP64_FFT'] = describe(rc_metrics, 16383)
    summary = {'case': CASE, 'GPU': gpu, 'forwards': len(runs), 'RC_status': rc_status,
               'model_load_seconds': model_load_seconds, 'total_seconds': time.monotonic() - started,
               'peak_allocated_GPU_GiB': peak_gib, 'regions': regions,
               'native_scores': {k: native_summary[k] for k in ['forward_REF_score', 'forward_ALT_score', 'RC_REF_score', 'RC_ALT_score']},
               'FP64_FFT_scores': scores,
               'native_forward_delta_score': native_summary['forward_ALT_score'] - native_summary['forward_REF_score'],
               'FP64_FFT_forward_delta_score': scores['forward_alt'] - scores['forward_ref'],
               'REF_repeat_max_relative_l2': float(repeat_metrics['relative_l2'].max()),
               'forward_native_vs_FP64_position_metric_pearson_descriptive': float(np.corrcoef(native_forward['relative_l2'], forward_metrics['relative_l2'])[0, 1]),
               'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if rc_metrics is not None:
        summary['FP64_FFT_RC_delta_score'] = scores['rc_alt'] - scores['rc_ref']
        summary['FP64_FFT_forward_vs_RC_genomic_position_metric_pearson_descriptive'] = float(np.corrcoef(forward_metrics['relative_l2'], rc_metrics['relative_l2'][::-1])[0, 1])
    write_json('summary.json', summary)
    write_json('FFT_layer_coverage.json', patch_coverage)
    with (OUT / 'per_position_native_vs_FP64.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['position_grch38_1based', 'offset_from_variant_genomic', 'forward_native_relative_l2', 'forward_FP64_FFT_relative_l2', 'RC_native_relative_l2_genomic_order', 'RC_FP64_FFT_relative_l2_genomic_order'])
        for i in range(length):
            writer.writerow([start0 + i + 1, i - 16384, native_forward['relative_l2'][i], forward_metrics['relative_l2'][i], native_rc['relative_l2'][length - 1 - i], rc_metrics['relative_l2'][length - 1 - i] if rc_metrics is not None else ''])
    validation = {'status': 'PASSED', 'CPU_math_checks_passed': True, 'all_active_HCM_HCL_FFT_paths_covered': True,
                  'HCS_layers_remained_native': True, 'no_untracked_FP32_torch_FFT_calls': True,
                  'parameter_version_counters_unchanged': True, 'engine_file_unchanged': sha(engine.__file__) == protocol['installed_engine_sha256'],
                  'REF_repeat_relative_l2_exactly_zero': bool(np.count_nonzero(repeat_metrics['relative_l2']) == 0),
                  'forward_unchanged_prefix_exactly_zero': bool(np.count_nonzero(forward_metrics['relative_l2'][:16384]) == 0),
                  'RC_unchanged_prefix_exactly_zero': bool(np.count_nonzero(rc_metrics['relative_l2'][:16383]) == 0) if rc_metrics is not None else None,
                  'script_sha256': sha(__file__), 'protocol_sha256': sha(OUT / 'protocol.json'),
                  'runtime_libraries': {name: importlib.metadata.version(name) for name in ['torch', 'evo2', 'vtx', 'numpy']}}
    # A nonzero ALT prefix is a scientific outcome to report, not a failed execution.
    if not validation['REF_repeat_relative_l2_exactly_zero'] or not validation['engine_file_unchanged']:
        validation['status'] = 'REVIEW_REQUIRED'
    write_json('validation.json', validation)
    lines = ['# 동일 BRCA1 변이의 FFT 정밀도 파일럿', '',
             'GRCh38 chr17:43063900 C>A, 기존과 동일한 32,768bp·checkpoint입니다.',
             '전체 모델을 재학습하거나 weight를 바꾸지 않았습니다. 활성 HCM 9층과 HCL 9층의 FFT 합성곱만 런타임에서 FP64로 계산했습니다.',
             '채널 128개씩 처리해 중간 배열 메모리를 제한했고, 기존 skip/gate 및 BF16/FP32 변환 경계는 유지했습니다. HCS·attention·MLP·정규화는 기존 계산입니다.', '',
             '## 결과', '', '| 조건 | 변이 이전 평균 상대 L2 | 변이 이후 평균 상대 L2 | 전체 평균 상대 L2 |', '|---|---:|---:|---:|']
    for label, values in regions.items():
        lines.append(f"| {label} | {values['pre_variant_mean_relative_l2']:.9g} | {values['post_variant_mean_relative_l2']:.9g} | {values['mean_relative_l2']:.9g} |")
    lines += ['', f"REF 반복 최대 상대 L2: {summary['REF_repeat_max_relative_l2']:.9g}",
              f"기존 forward ALT−REF 서열 점수: {summary['native_forward_delta_score']:.9g}",
              f"FP64 FFT forward ALT−REF 서열 점수: {summary['FP64_FFT_forward_delta_score']:.9g}",
              f"전체 실행 시간 {summary['total_seconds']:.1f}초, GPU 최대 {peak_gib:.2f} GiB, 추론 {len(runs)}회, RC {rc_status}.", '',
              '## 해석 범위', '',
              '변이 이전은 모델 입력 순서이며 생물학적인 upstream을 뜻하지 않습니다. 같은 입력 prefix의 비영(0이 아닌) 차이가 감소하는지는 수치 오차를 확인하는 대조입니다.',
              '반대 방향 입력의 위치별 결과가 서로 같아야 하는 모델은 아닙니다. RC 일치 여부 자체를 수치 교정 성공 조건으로 쓰지 않았습니다.',
              'FP64 FFT만 바꾸어도 미세한 오차와 BF16 반올림이 완전히 없어지는지는 실제 결과로 판단해야 합니다. validation.json의 prefix 항목과 모든 층의 layer_prefix_comparisons.json을 확인하세요.',
              '이 파일럿 한 개만으로 모든 변이를 교정했다고 주장할 수 없습니다. 변이 이후 남는 변화가 곧 실제 생물학적 기전이라는 뜻도 아닙니다.', '',
              '## 재현 파일', '',
              '- protocol.json: 추론 전 고정한 패치 범위·캐스팅 순서·입력·checkpoint hash·실행 순서.',
              '- CPU_math_checks.json: 짧고 긴 filter의 직접 causal FIR와 비교, 기존 실제 HCM 입력에 대한 사전 검사.',
              '- FFT_layer_coverage.json: 모든 추론에서 HCM/HCL 9+9층 및 FP64 FFT 호출 수 검증.',
              '- layer_prefix_comparisons.json: 모든 32개 block과 embedding/norm의 동일 prefix 비교.',
              '- precision_intervention_layer_changes.json: 동일 입력에서 기존 정밀도와 FP64 FFT의 layer probe 차이.',
              '- summary.json, validation.json, forward_runs.json: 결과와 검증·실행 시간.',
              '- *_metrics*.npz, per_position_native_vs_FP64.csv: 위치별 원래 정의의 지표.',
              '- all_layer_prefix_first64channels.pt: 모든 층의 prefix에서 64채널만 저장한 재현용 소형 샘플.', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    manifest = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in OUT.iterdir() if p.is_file() and p.name not in ['artifact_manifest.json', 'progress.json']}
    write_json('artifact_manifest.json', manifest)
    write_json('progress.json', {'status': 'complete', 'completed_forwards': len(runs), 'validation_status': validation['status']})
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(json.dumps(validation, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'failure_traceback.txt').write_text(traceback.format_exc())
        write_json('progress.json', {'status': 'failed', 'error': traceback.format_exc()})
        raise
