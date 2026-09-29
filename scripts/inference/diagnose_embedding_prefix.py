"""Locate differences in a small unchanged prefix across the first model blocks."""
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/embedding_comparison'
os.environ['HF_HUB_OFFLINE']='1'


def main():
    import torch
    from Bio import SeqIO
    from evo2 import Evo2
    s=json.loads((OUT/'summary.json').read_text())
    seqs=[str(r.seq) for r in SeqIO.parse(OUT/'inputs.fasta','fasta')]
    assert seqs[0][:256]==seqs[1][:256]
    start=time.monotonic()
    model=Evo2('evo2_7b',local_path=s['checkpoint']);model.model.eval()
    layers=['embedding_layer','blocks.0','blocks.1','blocks.2','norm']
    cache={};side='ref'
    def capture(name):
        def hook(module,inputs,output):
            if isinstance(output,tuple):output=output[0]
            cache[side][name]=output[:,:256,:].detach().cpu().clone()
        return hook
    handles=[model.model.get_submodule(name).register_forward_hook(capture(name)) for name in layers]
    for side,seq in zip(['ref','alt'],seqs):
        cache[side]={}
        ids=torch.tensor(model.tokenizer.tokenize(seq),dtype=torch.long,device='cuda:0')[None]
        with torch.inference_mode(): output=model(ids)
        torch.cuda.synchronize()
        del output,ids
    for h in handles:h.remove()
    torch.save(cache,OUT/'prefix_diagnostic.pt')
    report={'prefix_positions_1based':[1,256],'variant_position_1based':20481,
            'unchanged_input_prefix':True,'layers':{},'seconds':time.monotonic()-start}
    for name in layers:
        a=cache['ref'][name].float();b=cache['alt'][name].float();d=b-a
        report['layers'][name]={'exactly_equal':bool(torch.equal(a,b)),
            'max_abs_difference':float(d.abs().max()),
            'mean_relative_l2':float((d.norm(dim=-1)/a.norm(dim=-1)).mean())}
    (OUT/'prefix_diagnostic.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
