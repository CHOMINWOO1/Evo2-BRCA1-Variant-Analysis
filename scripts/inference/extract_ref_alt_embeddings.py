"""Extract final normalized Evo2 hidden states for the recorded real REF/ALT pair."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/embedding_comparison'
INPUT = ROOT / 'results/length_benchmark/runs/bp_40960_rep1'
os.environ['HF_HUB_OFFLINE'] = '1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'summary.json').exists():
        raise FileExistsError('Embedding results already exist; use a new output folder for a new run.')
    start = time.monotonic()
    import torch
    from Bio import SeqIO
    from evo2 import Evo2
    from evo2.scoring import logits_to_logprobs
    original = json.loads((INPUT / 'summary.json').read_text())
    seqs = [str(r.seq) for r in SeqIO.parse(INPUT / 'inputs.fasta', 'fasta')]
    assert len(seqs) == 2 and all(len(s) == 40960 for s in seqs)
    assert [i for i,(a,b) in enumerate(zip(*seqs)) if a != b] == [20480]
    for label, seq in zip(['REF', 'ALT'], seqs):
        assert hashlib.sha256(seq.encode()).hexdigest() == original['input_hashes'][label]
    (OUT / 'inputs.fasta').write_bytes((INPUT / 'inputs.fasta').read_bytes())
    summary = dict(status='running', model=original['model'], checkpoint=original['checkpoint'],
        checkpoint_revision=original['checkpoint_revision'], input_hashes=original['input_hashes'],
        layer='norm', layer_description='final normalized hidden states, before vocabulary projection',
        shape=[1,40960,4096], variant_index_zero_based=20480,
        variant_position_one_based=20481, variant_genomic_position_hg19=41276135,
        genome='GRCh37.p13', ref_base='T', alt_base='G', batch_size=1,
        gpu=torch.cuda.get_device_name(0), stage='model_load', extraction_runs=[])
    samples=[]
    stop=threading.Event()

    def monitor():
        while not stop.is_set():
            r=subprocess.run(['nvidia-smi','-i',os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0],
                '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                capture_output=True,text=True)
            if r.returncode==0:
                f=next(csv.reader([r.stdout.strip()]))
                samples.append({'seconds':time.monotonic()-start,'used_mib':float(f[0]),'utilization':float(f[1])})
            stop.wait(.2)

    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    torch.cuda.reset_peak_memory_stats()
    try:
        t=time.monotonic()
        model=Evo2('evo2_7b',local_path=original['checkpoint'])
        model.model.eval()
        torch.cuda.synchronize()
        summary['model_load_seconds']=time.monotonic()-t
        summary['use_fp8_input_projections']=model.model.config.use_fp8_input_projections
        for label,seq in [('ref',seqs[0]),('alt',seqs[1]),('ref_repeat',seqs[0])]:
            summary['stage']=label+'/forward_and_capture'
            (OUT/'progress.json').write_text(json.dumps(summary,indent=2))
            t=time.monotonic()
            ids=torch.tensor(model.tokenizer.tokenize(seq),dtype=torch.long,device='cuda:0').unsqueeze(0)
            with torch.inference_mode():
                outputs,embeddings=model(ids,return_embeddings=True,layer_names=['norm'])
                hidden=embeddings['norm'].detach().cpu()
                logits=outputs[0]
                score=float(logits_to_logprobs(logits,ids).float().mean().item())
            torch.cuda.synchronize()
            compute=time.monotonic()-t
            assert list(hidden.shape)==summary['shape']
            assert torch.isfinite(hidden).all().item()
            # Release every GPU reference before the next full-length forward.
            del outputs,embeddings,logits,ids
            summary['stage']=label+'/save_cpu_tensor'
            path=OUT/f'{label}_norm.pt'
            torch.save({'embedding':hidden,'layer':'norm','sequence_sha256':hashlib.sha256(seq.encode()).hexdigest()},path)
            summary['extraction_runs'].append({'label':label,'forward_capture_and_score_seconds':compute,
                'seconds_including_save':time.monotonic()-t,'dtype':str(hidden.dtype),
                'shape':list(hidden.shape),'score':score,'file':path.name,'bytes':path.stat().st_size})
            del hidden
            print(json.dumps(summary['extraction_runs'][-1]),flush=True)
        summary['status']='complete';summary['stage']='complete'
    except Exception as e:
        summary['status']='failed';summary['error']=str(e)
        (OUT/'traceback.txt').write_text(traceback.format_exc())
        (OUT/'cuda_memory.txt').write_text(torch.cuda.memory_summary())
        raise
    finally:
        summary['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
        summary['peak_reserved_gib']=torch.cuda.max_memory_reserved()/1024**3
        summary['total_seconds']=time.monotonic()-start
        stop.set();thread.join(timeout=10)
        summary['sampled_peak_device_gib']=max((s['used_mib']/1024 for s in samples),default=None)
        (OUT/'gpu_samples.json').write_text(json.dumps(samples,indent=2))
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
