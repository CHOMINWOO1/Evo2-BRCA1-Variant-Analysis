"""Measure a single unchunked REF/ALT pair in a fresh Evo2 7B process."""
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/brca1_test'
os.environ['HF_HUB_OFFLINE'] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    if not 2 <= args.length <= 1048576 or args.length % 2:
        parser.error('length must be even, in [2, 1048576]')
    dest = ROOT / f'results/length_benchmark/runs/bp_{args.length}_rep{args.repeat}'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'summary.json').exists():
        raise FileExistsError(f'Result already exists: {dest}; choose a new repeat ID')
    start = time.monotonic()
    import numpy as np
    import torch
    from Bio import SeqIO
    from evo2 import Evo2

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    checkpoint = ROOT / '.cache/huggingface/hub/models--arcinstitute--evo2_7b/snapshots/bda0089f92582d5baabf0f22d9fc85f3588f6b58/evo2_7b.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    row = next(csv.DictReader((DATA / 'selected_variants.csv').open()))
    with gzip.open(DATA / 'official/GRCh37.p13_chr17.fna.gz', 'rt') as handle:
        record = next(SeqIO.parse(handle, 'fasta'))
        chr17 = str(record.seq)
    # Reuse the official function, changing only the window length.
    notebook = json.loads((DATA / 'official/brca1_zero_shot_vep.ipynb').read_text())
    source = ''.join(notebook['cells'][6]['source'])
    source = source[source.index('def parse_sequences('):source.index('# Parse sequences for the first variant')]
    namespace = {'seq_chr17': chr17, 'WINDOW_SIZE': args.length}
    exec(compile(source, 'official_parse_sequences', 'exec'), namespace)
    ref, alt = namespace['parse_sequences'](int(row['pos']), row['ref'], row['alt'])
    assert len(ref) == len(alt) == args.length
    assert sum(a != b for a, b in zip(ref, alt)) == 1
    assert ref[args.length // 2] == row['ref'] and alt[args.length // 2] == row['alt']
    (dest / 'inputs.fasta').write_text(f'>REF_chr17_{row["pos"]}\n{ref}\n>ALT_{row["ref"]}_to_{row["alt"]}\n{alt}\n')
    samples = []
    stop = threading.Event()
    physical_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]

    def sample_gpu():
        result = subprocess.run(['nvidia-smi', '-i', physical_gpu,
            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        if result.returncode:
            return {'error': result.stderr}
        fields = next(csv.reader([result.stdout.strip()]))
        return dict(elapsed_seconds=time.monotonic() - start, used_mib=float(fields[2]),
                    total_mib=float(fields[3]), utilization_percent=float(fields[4]))

    baseline = sample_gpu()

    def monitor():
        while not stop.is_set():
            try:
                samples.append(sample_gpu())
            except Exception as error:
                samples.append({'error': str(error)})
            stop.wait(0.2)

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    torch.cuda.reset_peak_memory_stats()
    summary = dict(status='running', length_bp=args.length, repeat=args.repeat,
        variant_count=1, variant='chr17:41276135 T>G', genome='GRCh37.p13 / hg19',
        model='evo2_7b', checkpoint=str(checkpoint), checkpoint_revision=checkpoint.parent.name,
        gpu=torch.cuda.get_device_name(0), gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/1024**3,
        batch_size=1, sequence_count=2, chunking=False, teacher_forcing=False,
        reduce_method='mean', prepend_bos=False, average_reverse_complement=False,
        allocator_config=os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default'),
        use_kernels=False, baseline_gpu=baseline,
        versions={x: importlib.metadata.version(x) for x in ['evo2', 'vtx', 'torch', 'numpy']},
        input_hashes={k: hashlib.sha256(v.encode()).hexdigest() for k, v in [('REF', ref), ('ALT', alt)]},
        stage='model_load')
    active_side = 'REF'
    (dest / 'progress.json').write_text(json.dumps(summary, indent=2))
    try:
        load_start = time.monotonic()
        model = Evo2('evo2_7b', local_path=str(checkpoint))
        model.model.eval()
        torch.cuda.synchronize()
        summary['model_load_seconds'] = time.monotonic() - load_start
        summary['use_fp8_input_projections'] = model.model.config.use_fp8_input_projections

        def block_pre(index):
            def hook(module, inputs):
                summary['stage'] = f'{active_side}/forward/block_{index}'
            return hook

        for index, block in enumerate(model.model.blocks):
            block.register_forward_pre_hook(block_pre(index))

        def forward_done(module, inputs, outputs):
            summary['stage'] = f'{active_side}/logprob_reduction'

        model.model.register_forward_hook(forward_done)
        pair_start = time.monotonic()
        for side, sequence in [('REF', ref), ('ALT', alt)]:
            active_side = side
            summary['stage'] = f'{side}/tokenize'
            (dest / 'progress.json').write_text(json.dumps(summary, indent=2))
            score_start = time.monotonic()
            # Same official scoring API and defaults; one entire window per forward.
            value = model.score_sequences([sequence], batch_size=1)[0]
            torch.cuda.synchronize()
            assert np.isfinite(value)
            summary[f'{side.lower()}_score'] = float(value)
            summary[f'{side.lower()}_seconds'] = time.monotonic() - score_start
            print(f'{side}: length={args.length}, score={value}, seconds={summary[f"{side.lower()}_seconds"]:.3f}', flush=True)
        summary['pair_scoring_seconds'] = time.monotonic() - pair_start
        summary['delta_score'] = float(np.float32(summary['alt_score']) - np.float32(summary['ref_score']))
        summary['status'] = 'success'
        summary['stage'] = 'complete'
    except Exception as error:
        cause = error
        chain = []
        while cause is not None:
            chain.append(cause)
            cause = cause.__cause__ or cause.__context__
        is_oom = any(isinstance(e, torch.cuda.OutOfMemoryError) for e in chain)
        summary['status'] = 'oom' if is_oom else 'error'
        summary['error'] = str(error)
        summary['failure_elapsed_seconds'] = time.monotonic() - start
        summary['allocated_at_failure_gib'] = torch.cuda.memory_allocated()/1024**3
        summary['reserved_at_failure_gib'] = torch.cuda.memory_reserved()/1024**3
        (dest / 'traceback.txt').write_text(traceback.format_exc())
        (dest / 'cuda_memory.txt').write_text(torch.cuda.memory_summary())
    finally:
        summary['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
        summary['peak_reserved_gib'] = torch.cuda.max_memory_reserved()/1024**3
        summary['total_seconds'] = time.monotonic() - start
        stop.set()
        worker.join(timeout=12)
        summary['sampled_peak_device_gib'] = max((v['used_mib']/1024 for v in samples if 'used_mib' in v), default=None)
        (dest / 'gpu_samples.json').write_text(json.dumps(samples, indent=2))
        (dest / 'summary.json').write_text(json.dumps(summary, indent=2))
        print('RESULT ' + json.dumps(summary), flush=True)
    return 0 if summary['status'] == 'success' else (42 if summary['status'] == 'oom' else 1)


if __name__ == '__main__':
    raise SystemExit(main())
