"""Run the official BRCA1 notebook's scoring cells on the first N variants."""
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/brca1_test'
OFFICIAL = OUT / 'official'
sys.path.insert(0, str(OUT / 'deps'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', type=int, default=10)
    args = parser.parse_args()
    if args.variants < 1:
        parser.error('--variants must be positive')
    started = time.monotonic()
    import numpy as np
    import pandas as pd
    import torch
    from Bio import SeqIO
    from evo2.models import Evo2

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; execute in a GPU-accessible shell.')
    checkpoint = ROOT / '.cache/huggingface/hub/models--arcinstitute--evo2_7b/snapshots/bda0089f92582d5baabf0f22d9fc85f3588f6b58/evo2_7b.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    notebook = json.loads((OFFICIAL / 'brca1_zero_shot_vep.ipynb').read_text())
    manifest = json.loads((OFFICIAL / 'manifest.json').read_text())
    for file in manifest['files']:
        assert hashlib.sha256((OFFICIAL / file['name']).read_bytes()).hexdigest() == file['sha256']

    ns = dict(os=os, gzip=gzip, SeqIO=SeqIO, np=np, pd=pd, OFFICIAL=OFFICIAL)
    def cell(index):
        source = ''.join(notebook['cells'][index]['source'])
        source = source.replace("os.path.join('notebooks', 'brca1', ", 'os.path.join(str(OFFICIAL), ')
        exec(compile(source, f'official_notebook_cell_{index}', 'exec'), ns)

    cell(2)
    dataset_count = len(ns['brca1_df'])
    cell(4)
    ns['brca1_df'] = ns['brca1_df'].head(args.variants).copy()
    cell(6)
    df = ns['brca1_df']
    df.to_csv(OUT / 'selected_variants.csv', index=False)
    with (OUT / 'input_sequences.fasta').open('w') as handle:
        for i, row in df.iterrows():
            ref, alt = ns['parse_sequences'](row['pos'], row['ref'], row['alt'])
            assert len(ref) == len(alt) == 8192
            assert sum(a != b for a, b in zip(ref, alt)) == 1
            for label, seq in [('REF', ref), ('ALT', alt)]:
                handle.write(f'>{i}|chr{row["chrom"]}:{row["pos"]}:{row["ref"]}>{row["alt"]}|{label}\n{seq}\n')

    samples = []
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            samples.append({'elapsed_seconds': time.monotonic() - started, 'returncode': result.returncode, 'output': result.stdout.strip()})
            stop.wait(0.5)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    torch.cuda.reset_peak_memory_stats()
    summary = dict(status='running', gpu=torch.cuda.get_device_name(0), model='evo2_7b',
                   checkpoint=str(checkpoint), checkpoint_revision=checkpoint.parent.name,
                   checkpoint_bytes=checkpoint.stat().st_size, official_commit=manifest['commit'],
                   variant_count=len(df), full_dataset_count=dataset_count, selection='first N rows in official spreadsheet',
                   window_bp=8192, batch_size=1, prepend_bos=False, reduce_method='mean',
                   average_reverse_complement=False, delta_definition='ALT mean log-likelihood minus REF mean log-likelihood',
                   python=sys.version, python_executable=sys.executable,
                   versions={x: importlib.metadata.version(x) for x in ['evo2','vtx','torch','numpy','pandas','openpyxl']},
                   result_file=str(OUT / 'variant_scores.csv'), stage='model_load')
    try:
        before_load = time.monotonic()
        model = Evo2('evo2_7b', local_path=str(checkpoint))
        model.model.eval()
        ns['model'] = model
        torch.cuda.synchronize()
        summary['model_load_seconds'] = time.monotonic() - before_load
        summary['use_fp8_input_projections'] = model.model.config.get('use_fp8_input_projections', None)
        summary['stage'] = 'REF_then_ALT_scoring'
        score_start = time.monotonic()
        cell(9)
        torch.cuda.synchronize()
        summary['scoring_seconds'] = time.monotonic() - score_start
        cell(11)
        df['ref_score'] = np.array(ns['ref_scores'])[ns['ref_seq_indexes']]
        df['alt_score'] = ns['var_scores']
        assert np.isfinite(df[['ref_score','alt_score','evo2_delta_score']].to_numpy()).all()
        df.to_csv(OUT / 'variant_scores.csv', index=False)
        summary['unique_reference_count'] = len(ns['ref_seqs'])
        summary['status'] = 'complete'
        summary['stage'] = 'complete'
        print(df.to_string(index=False), flush=True)
    except Exception as error:
        summary['status'] = 'failed'
        summary['error'] = repr(error)
        (OUT / 'error_traceback.txt').write_text(traceback.format_exc())
        (OUT / 'cuda_memory_on_error.txt').write_text(torch.cuda.memory_summary())
        raise
    finally:
        summary['peak_allocated_gpu_gib'] = torch.cuda.max_memory_allocated() / 1024**3
        summary['peak_reserved_gpu_gib'] = torch.cuda.max_memory_reserved() / 1024**3
        summary['elapsed_seconds_including_imports_data_and_model'] = time.monotonic() - started
        stop.set()
        thread.join(timeout=5)
        (OUT / 'gpu_samples.json').write_text(json.dumps(samples, indent=2))
        gpu_index = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
        memories = []
        for sample in samples:
            for line in sample['output'].splitlines():
                fields = next(csv.reader([line]))
                if len(fields) == 5 and int(fields[0]) == gpu_index:
                    memories.append(float(fields[2]))
        summary['sampled_peak_device_memory_mib_including_other_processes'] = max(memories, default=None)
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
