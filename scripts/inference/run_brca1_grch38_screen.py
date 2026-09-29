"""Resumable BRCA1 ClinVar SNV screen: 32k Evo2 scores and position metrics."""
import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[2]
CAT = ROOT / 'results/brca1_grch38'
os.environ['HF_HUB_OFFLINE'] = '1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def key(row):
    return (row['chromosome'], int(row['position_grch38_1based']), row['ref'], row['alt'])


def position_metrics(ref, alt):
    import numpy as np
    length = ref.shape[1]
    names = ['relative_l2', 'rms_difference', 'cosine_distance', 'max_abs_difference',
             'changed_dimension_fraction', 'reference_l2']
    result = {name: np.empty(length, dtype=np.float32) for name in names}
    for start in range(0, length, 256):
        end = min(start + 256, length)
        r, a = ref[0, start:end].float().numpy(), alt[0, start:end].float().numpy()
        d = a - r
        nr = np.sqrt(np.einsum('ij,ij->i', r, r, dtype=np.float64))
        na = np.sqrt(np.einsum('ij,ij->i', a, a, dtype=np.float64))
        nd = np.sqrt(np.einsum('ij,ij->i', d, d, dtype=np.float64))
        result['relative_l2'][start:end] = nd / np.maximum(nr, 1e-30)
        result['rms_difference'][start:end] = nd / np.sqrt(r.shape[1])
        result['cosine_distance'][start:end] = np.clip(1 - np.einsum('ij,ij->i', r, a, dtype=np.float64) / np.maximum(nr * na, 1e-30), 0, 2)
        result['max_abs_difference'][start:end] = np.max(np.abs(d), axis=1)
        result['changed_dimension_fraction'][start:end] = np.mean(d != 0, axis=1)
        result['reference_l2'][start:end] = nr
    assert all(np.isfinite(value).all() for value in result.values())
    return result


def save_metrics(path, values):
    import numpy as np
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'results/brca1_grch38/screen_32768')
    p.add_argument('--length', type=int, default=32768)
    p.add_argument('--limit', type=int, default=0, help='0 runs the complete SNV catalog')
    p.add_argument('--retain-raw-first', type=int, default=10)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    if args.length < 2 or args.length % 2 or args.limit < 0 or args.retain_raw_first < 0:
        p.error('Invalid length/count')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for folder in ('variants', 'controls', 'raw', 'dataset'):
        (out / folder).mkdir(exist_ok=True)
    sources = ['experiment_ready_snvs.csv', 'splice_nearby_snvs_20bp.csv',
               'splice_dinucleotide_snvs.csv', 'mane_select_junctions.csv', 'manifest.json']
    rows = read_csv(CAT / sources[0])
    junctions = read_csv(CAT / sources[3])
    site_annotations = {}
    for annotation in read_csv(CAT / sources[1]):
        site_annotations.setdefault(key(annotation), []).append({name: annotation[name] for name in
            ('junction_id', 'site_type', 'site_exonic_base_1based', 'offset_from_site_in_transcript_direction', 'site_category')})
    essential = {key(r) for r in read_csv(CAT / sources[2])}
    nearby = {key(r) for r in read_csv(CAT / sources[1])}
    assert len({key(r) for r in rows}) == len(rows)
    for r in rows:
        assert r['gene'] == 'BRCA1' and r['chromosome'] == 'chr17'
        assert len(r['ref']) == len(r['alt']) == 1 and r['ref'] != r['alt']
        assert r['ref_matches_hg38'] == 'True'
        r['screen_group'] = 'splice_dinucleotide' if key(r) in essential else 'splice_nearby_20bp' if key(r) in nearby else 'other_snv'
    priority = {name: i for i, name in enumerate(('splice_dinucleotide', 'splice_nearby_20bp', 'other_snv'))}
    rows.sort(key=lambda r: (priority[r['screen_group']], int(r['position_grch38_1based']), r['alt']))
    if args.limit:
        rows = rows[:args.limit]
    checkpoint = Path(json.loads((ROOT / 'results/embedding_comparison/summary.json').read_text())['checkpoint'])
    reference_path = ROOT / 'results/brca_grch38/sources/chr17.fa.gz'
    reference_manifest = json.loads((ROOT / 'results/brca_grch38/manifest.json').read_text())
    reference_sha = sha(reference_path)
    assert reference_sha == reference_manifest['sources']['chr17.fa.gz']['sha256']
    config = dict(assembly='GRCh38/hg38', gene='BRCA1', transcript='NM_007294.4',
                  length_bp=args.length, variant_index_0based=args.length // 2,
                  model='evo2_7b', checkpoint=str(checkpoint), checkpoint_revision=checkpoint.parent.name,
                  layer='norm', orientation='forward genomic', batch_size=1,
                  precision='BF16 model fallback', limit=args.limit,
                  retain_raw_first=args.retain_raw_first, dataset_sha256={name: sha(CAT / name) for name in sources},
                  reference_fasta=str(reference_path), reference_sha256=reference_sha,
                  clinvar_file_date=reference_manifest['clinvar_file_date'], mane_release=reference_manifest['mane_release'],
                  variants=len(rows), unique_positions=len({key(r)[:2] for r in rows}),
                  scope='ClinVar VCF reference-validated BRCA1 SNVs only; excludes indels, large/complex/imprecise variants and other databases',
                  score='mean next-token log probability, full vocabulary, no BOS, ALT minus REF',
                  metrics='float32 CPU differences; float64 norm accumulation; stored float32 per position',
                  controls='repeat REF at first group and every 100 coordinate groups per invocation',
                  raw_policy='first N variant REF/ALT embeddings retained; all variants retain position metrics and scores')
    if (out / 'config.json').exists():
        assert json.loads((out / 'config.json').read_text()) == config, 'Run config changed; use a new output directory'
    else:
        atomic_json(out / 'config.json', config)
        for name in sources:
            shutil.copy2(CAT / name, out / 'dataset' / name)
    with gzip.open(reference_path, 'rt') as f:
        assert next(f).startswith('>chr17')
        genome = ''.join(line.strip().upper() for line in f)
    for i, r in enumerate(rows):
        r['run_id'] = f"cv_{r['clinvar_variation_id']}_{r['position_grch38_1based']}_{r['ref']}_{r['alt']}"
        r['retain_raw'] = i < args.retain_raw_first
        pos = int(r['position_grch38_1based']) - 1
        assert genome[pos] == r['ref'] and pos - args.length // 2 >= 0
        assert pos - args.length // 2 + args.length <= len(genome)
    with (out / 'run_plan.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    progress = dict(status='prepared', total_variants=len(rows), completed_variants=0, pid=os.getpid())
    completed = [r for r in rows if (out / 'variants' / (r['run_id'] + '.json')).exists()
                 and (out / 'variants' / (r['run_id'] + '.npz')).exists()]
    done_ids = {r['run_id'] for r in completed}
    pending = [r for r in rows if r['run_id'] not in done_ids]
    progress['completed_variants'] = len(completed)
    atomic_json(out / 'progress.json', progress)
    print(json.dumps(dict(event='prepared', **progress, unique_positions=config['unique_positions'])), flush=True)
    if args.prepare_only or not pending:
        if not pending:
            progress['status'] = 'complete'; atomic_json(out / 'progress.json', progress)
        return
    import numpy as np
    import torch
    from evo2 import Evo2
    from evo2.scoring import logits_to_logprobs
    torch.set_num_threads(4)
    forward_times, processing_times = [], []
    started = time.monotonic()

    def forward(sequence):
        t = time.monotonic()
        ids = torch.tensor(model.tokenizer.tokenize(sequence), dtype=torch.long, device='cuda:0')[None]
        with torch.inference_mode():
            outputs, embeddings = model(ids, return_embeddings=True, layer_names=['norm'])
            hidden = embeddings['norm'].detach().cpu()
            score = float(logits_to_logprobs(outputs[0], ids).float().mean().item())
        torch.cuda.synchronize()
        assert tuple(hidden.shape) == (1, args.length, 4096)
        assert torch.isfinite(hidden).all() and np.isfinite(score)
        del outputs, embeddings, ids
        forward_times.append(time.monotonic() - t)
        return hidden, score

    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; run in GPU-accessible shell')
        progress.update(status='running', stage='model_load', gpu=torch.cuda.get_device_name(0),
                        started_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(out / 'progress.json', progress)
        torch.cuda.reset_peak_memory_stats()
        model = Evo2('evo2_7b', local_path=str(checkpoint)); model.model.eval()
        progress['use_fp8_input_projections'] = bool(model.model.config.use_fp8_input_projections)
        groups = {}
        for r in pending:
            groups.setdefault(int(r['position_grch38_1based']), []).append(r)
        for group_index, (position, group) in enumerate(groups.items()):
            start0 = position - 1 - args.length // 2
            reference = genome[start0:start0 + args.length]
            progress.update(stage='reference_forward', current_position=position)
            atomic_json(out / 'progress.json', progress)
            ref_hidden, ref_score = forward(reference)
            if group_index % 100 == 0:
                repeat, repeat_score = forward(reference)
                control = position_metrics(ref_hidden, repeat)
                save_metrics(out / 'controls' / f'ref_{position}.npz', control)
                atomic_json(out / 'controls' / f'ref_{position}.json', dict(position=position,
                            max_relative_l2=float(control['relative_l2'].max()), delta_score=repeat_score-ref_score))
                del repeat, control
            for r in group:
                run_id = r['run_id']
                progress.update(stage='alternate_forward', current_variant=run_id)
                atomic_json(out / 'progress.json', progress)
                alternate = reference[:args.length // 2] + r['alt'] + reference[args.length // 2 + 1:]
                assert reference[args.length // 2] == r['ref'] and len(alternate) == args.length
                alt_hidden, alt_score = forward(alternate)
                t = time.monotonic()
                metrics = position_metrics(ref_hidden, alt_hidden)
                save_metrics(out / 'variants' / f'{run_id}.npz', metrics)
                if r['retain_raw']:
                    torch.save({'embedding': ref_hidden, 'sequence_sha256': hashlib.sha256(reference.encode()).hexdigest()}, out / 'raw' / f'{run_id}_ref_norm.pt')
                    torch.save({'embedding': alt_hidden, 'sequence_sha256': hashlib.sha256(alternate.encode()).hexdigest()}, out / 'raw' / f'{run_id}_alt_norm.pt')
                maximum = int(np.argmax(metrics['relative_l2']))
                result = dict(r, window_start_grch38_1based=start0 + 1, window_end_grch38_1based=start0 + args.length,
                              variant_index_0based=args.length // 2, ref_score=ref_score, alt_score=alt_score,
                              delta_score=alt_score-ref_score, mean_relative_l2=float(metrics['relative_l2'].mean()),
                              max_relative_l2=float(metrics['relative_l2'][maximum]), max_offset_bp=maximum-args.length//2,
                              ref_sequence_sha256=hashlib.sha256(reference.encode()).hexdigest(),
                              alt_sequence_sha256=hashlib.sha256(alternate.encode()).hexdigest(),
                              completed_utc=datetime.now(timezone.utc).isoformat())
                result['nearby_splice_sites'] = site_annotations.get(key(r), [])
                result['window_splice_boundaries'] = [dict(junction_id=j['junction_id'], site_type=kind,
                    position_grch38_1based=int(j[column]), input_index_0based=int(j[column])-1-start0)
                    for j in junctions for kind, column in [('donor','donor_exonic_base_1based'), ('acceptor','acceptor_exonic_base_1based')]
                    if start0 < int(j[column]) <= start0 + args.length]
                atomic_json(out / 'variants' / f'{run_id}.json', result)
                del alt_hidden, metrics
                processing_times.append(time.monotonic()-t)
                done_ids.add(run_id)
                remaining = [x for x in pending if x['run_id'] not in done_ids]
                remaining_ref = len({int(x['position_grch38_1based']) for x in remaining} - {position})
                eta_seconds = float(np.mean(forward_times[-100:])) * (len(remaining) + remaining_ref + remaining_ref / 100) + float(np.mean(processing_times[-100:])) * len(remaining)
                progress.update(completed_variants=len(done_ids), stage='variant_complete',
                                elapsed_seconds=time.monotonic()-started,
                                measured_mean_forward_seconds=float(np.mean(forward_times[-100:])),
                                measured_mean_processing_seconds=float(np.mean(processing_times[-100:])),
                                estimated_remaining_seconds=eta_seconds,
                                estimated_finish_kst=(datetime.now(timezone(timedelta(hours=9)))+timedelta(seconds=eta_seconds)).isoformat(),
                                peak_allocated_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
                atomic_json(out / 'progress.json', progress)
                print(json.dumps(dict(event='variant_complete', **progress)), flush=True)
            del ref_hidden
        progress.update(status='complete', stage='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        progress.update(status='failed', error=repr(error))
        (out / 'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        atomic_json(out / 'progress.json', progress)


if __name__ == '__main__':
    main()
