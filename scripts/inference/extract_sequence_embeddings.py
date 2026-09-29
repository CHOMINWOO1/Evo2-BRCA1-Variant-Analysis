"""Extract norm embeddings from a prepared FASTA with one sequence per forward."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
os.environ['HF_HUB_OFFLINE']='1'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True)
    args=parser.parse_args()
    out=Path(args.directory).resolve()
    if (out/'extraction_summary.json').exists():raise FileExistsError(out/'extraction_summary.json')
    start=time.monotonic()
    import torch
    from Bio import SeqIO
    from evo2 import Evo2
    meta=json.loads((ROOT/'results/embedding_comparison/summary.json').read_text())
    records=list(SeqIO.parse(out/'inputs.fasta','fasta'))
    summary={'status':'running','model':'evo2_7b','checkpoint':meta['checkpoint'],'layer':'norm',
             'batch_size':1,'gpu':torch.cuda.get_device_name(0),'runs':[],'stage':'model_load'}
    samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            r=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i','0'],capture_output=True,text=True)
            if r.returncode==0:samples.append({'seconds':time.monotonic()-start,'used_mib':float(r.stdout.strip())})
            stop.wait(.2)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    torch.cuda.reset_peak_memory_stats()
    try:
        model=Evo2('evo2_7b',local_path=meta['checkpoint']);model.model.eval()
        summary['use_fp8_input_projections']=model.model.config.use_fp8_input_projections
        for record in records:
            name=record.id
            if not name.replace('_','').isalnum():raise ValueError(name)
            seq=str(record.seq)
            summary['stage']=name+'/forward';t=time.monotonic()
            (out/'extraction_progress.json').write_text(json.dumps(summary,indent=2))
            ids=torch.tensor(model.tokenizer.tokenize(seq),dtype=torch.long,device='cuda:0')[None]
            with torch.inference_mode():
                outputs,emb=model(ids,return_embeddings=True,layer_names=['norm'])
                cpu=emb['norm'].detach().cpu()
            torch.cuda.synchronize()
            assert tuple(cpu.shape)==(1,len(seq),4096) and torch.isfinite(cpu).all()
            del outputs,emb,ids
            sha=hashlib.sha256(seq.encode()).hexdigest()
            torch.save({'embedding':cpu,'layer':'norm','sequence_sha256':sha},out/f'{name}_norm.pt')
            summary['runs'].append({'name':name,'shape':list(cpu.shape),'dtype':str(cpu.dtype),
                'seconds_including_capture_and_save':time.monotonic()-t,'sequence_sha256':sha})
            del cpu
            print(json.dumps(summary['runs'][-1]),flush=True)
        summary['status']='complete';summary['stage']='complete'
    except Exception as e:
        summary['status']='failed';summary['error']=str(e)
        (out/'traceback.txt').write_text(traceback.format_exc())
        (out/'cuda_memory.txt').write_text(torch.cuda.memory_summary())
        raise
    finally:
        summary['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
        summary['peak_reserved_gib']=torch.cuda.max_memory_reserved()/1024**3
        summary['total_seconds']=time.monotonic()-start
        stop.set();thread.join(timeout=10)
        summary['sampled_peak_device_gib']=max((x['used_mib']/1024 for x in samples),default=None)
        (out/'gpu_samples.json').write_text(json.dumps(samples,indent=2))
        (out/'extraction_summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
