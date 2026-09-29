"""Prepare the ten user-specified hg19 SNVs at three centered window sizes."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/requested_variants'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lengths',type=int,nargs='+',default=[8192,16384,32768])
    args=parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fasta = ROOT/'results/brca1_test/official/GRCh37.p13_chr17.fna.gz'
    with gzip.open(fasta, 'rt') as f:
        seq = ''.join(line.strip() for line in f if not line.startswith('>')).upper()
    variants = []
    for pos, ref, alts in [(41276113,'T','ACG'), (41276112,'A','TCG'),
                           (41276111,'C','AGT'), (41256984,'A','C')]:
        assert seq[pos-1] == ref
        for alt in alts:
            variants.append(dict(id=f'v{len(variants)+1:02d}', chromosome='chr17',
                position_hg19=pos, ref=ref, alt=alt))
    with (OUT/'variants.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(variants[0]));w.writeheader();w.writerows(variants)
    for length in args.lengths:
        directory=OUT/f'bp_{length}';directory.mkdir(exist_ok=True)
        records={};rows=[]
        for v in variants:
            p=v['position_hg19']-1; start=p-length//2
            ref=seq[start:start+length];alt=ref[:length//2]+v['alt']+ref[length//2+1:]
            assert len(ref)==length and sum(a!=b for a,b in zip(ref,alt))==1
            ref_id=f"ref_{v['position_hg19']}"
            records[ref_id]=ref;records[v['id']]=alt
            rows.append(dict(**v,ref_id=ref_id,window_start_hg19=start+1,
                window_end_hg19=start+length,variant_index=length//2))
        records['ref_repeat']=records[rows[0]['ref_id']]
        with (directory/'inputs.fasta').open('w') as f:
            for name,s in records.items(): f.write(f'>{name}\n{s}\n')
        meta=dict(length_bp=length,orientation='forward',variants=rows,
            unique_refs=4,alts=10,repeat_controls=1,genome_length_bp=len(seq),
            source_fasta=str(fasta),source='user-supplied hg19 SNVs; NCBI GRCh37.p13 chr17',
            sequence_sha256={k:hashlib.sha256(s.encode()).hexdigest() for k,s in records.items()})
        (directory/'inputs.json').write_text(json.dumps(meta,indent=2))
        print(f'{length} bp: {len(records)} validated sequences',flush=True)

if __name__=='__main__': main()
