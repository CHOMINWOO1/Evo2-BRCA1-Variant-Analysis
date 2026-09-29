"""Isolate the first FFT convolution and compare float32, float64 and direct FIR."""
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/embedding_comparison'
os.environ['HF_HUB_OFFLINE']='1'


def main():
    import torch
    import torch.nn.functional as F
    from Bio import SeqIO
    from evo2 import Evo2
    import vortex.model.engine as engine
    torch.set_num_threads(4)
    start=time.monotonic()
    meta=json.loads((OUT/'summary.json').read_text())
    seqs=[str(r.seq) for r in SeqIO.parse(OUT/'inputs.fasta','fasta')]
    model=Evo2('evo2_7b',local_path=meta['checkpoint']);model.model.eval()
    original=engine.fftconv_func
    captured={};label='ref'
    class Captured(Exception):pass
    def wrapper(u,k,D,dropout_mask,**kwargs):
        if kwargs.get('layer_idx')!=1:return original(u,k,D,dropout_mask,**kwargs)
        output=original(u,k,D,dropout_mask,**kwargs)
        assert u.ndim==3 and u.shape[1]==4096
        captured[label]={'u':u[:,:64,:].detach().cpu().clone(),
                         'k':k.squeeze()[:64].detach().cpu().clone(),
                         'D':D[:64].detach().cpu().clone(),
                         'observed_prefix':output[:,:64,:256].detach().cpu().clone(),
                         'kwargs':{x:kwargs.get(x) for x in ['layer_idx','gelu','bidirectional']}}
        raise Captured()
    engine.fftconv_func=wrapper
    try:
        for label,seq in zip(['ref','alt'],seqs):
            ids=torch.tensor(model.tokenizer.tokenize(seq),dtype=torch.long,device='cuda:0')[None]
            try:
                with torch.inference_mode():model(ids)
            except Captured:pass
            del ids
    finally:engine.fftconv_func=original
    del model
    torch.cuda.empty_cache()
    torch.save(captured,OUT/'fft_operator_samples.pt')
    a,b=captured['ref'],captured['alt']
    assert torch.equal(a['k'],b['k']) and torch.equal(a['D'],b['D'])
    assert torch.equal(a['u'][:,:,:20480],b['u'][:,:,:20480])
    first_diff=torch.nonzero((a['u']!=b['u']).any(dim=1)[0]).flatten()
    report={'layer':'blocks.1 / fftconv_func','sampled_channels':64,'prefix_positions':[1,256],
            'operator_inputs_identical_before_variant':True,'filters_identical':True,
            'first_different_operator_input_index':int(first_diff[0]),'tests':{}}
    native_delta=(a['observed_prefix']-b['observed_prefix']).abs()
    report['native_float32_prefix_max_difference']=float(native_delta.max())
    for dtype in [torch.float32,torch.float64]:
        values=[]
        for data in [a,b]:
            with torch.inference_mode():
                z=original(data['u'].to(device='cuda',dtype=dtype),
                    data['k'].to(device='cuda',dtype=dtype),data['D'].to(device='cuda',dtype=dtype),None,
                    gelu=False,bidirectional=False)
                values.append(z[:,:,:256].cpu().double())
                del z
        diff=(values[0]-values[1]).abs()
        report['tests'][str(dtype)]={'prefix_max_abs_difference':float(diff.max()),
                'prefix_rms_difference':float(diff.square().mean().sqrt()),
                'exactly_equal':torch.equal(*values)}
    values=[]
    for data in [a,b]:
        u=data['u'][:,:,:256].double();k=data['k'].double()
        z=F.conv1d(F.pad(u,(k.shape[-1]-1,0)),k.flip(-1)[:,None,:],groups=64)
        z=z+u*data['D'].double()[None,:,None]
        values.append(z)
    report['tests']['direct_causal_FIR_float64']={'prefix_max_abs_difference':float((values[0]-values[1]).abs().max()),
                                               'exactly_equal':torch.equal(*values)}
    report['seconds']=time.monotonic()-start
    (OUT/'fft_roundoff_diagnostic.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
