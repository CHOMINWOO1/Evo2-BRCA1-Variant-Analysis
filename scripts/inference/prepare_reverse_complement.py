"""Prepare exact reverse complements of the previously evaluated REF/ALT pair."""
import hashlib
import json
from pathlib import Path
from Bio import SeqIO

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/reverse_complement_comparison'
OUT.mkdir(parents=True,exist_ok=True)
records=list(SeqIO.parse(ROOT/'results/embedding_comparison/inputs.fasta','fasta'))
ref,alt=[str(r.seq.reverse_complement()) for r in records]
assert len(ref)==len(alt)==40960
assert [i for i,(a,b) in enumerate(zip(ref,alt)) if a!=b]==[20479]
content=f'>ref_rc\n{ref}\n>alt_rc\n{alt}\n'
path=OUT/'inputs.fasta'
if path.exists():assert path.read_text()==content
else:path.write_text(content)
manifest={'experiment':'reverse_complement','source':'results/embedding_comparison/inputs.fasta',
    'lengths':[40960,40960],'variant_index_zero_based':20479,'genomic_variant_hg19':41276135,
    'ref':ref[20479],'alt':alt[20479],'genomic_window_start_1based':41255655,
    'genomic_window_end_1based':41296614,
    'sequence_hashes':[hashlib.sha256(s.encode()).hexdigest() for s in [ref,alt]],
    'plot_mapping':'reverse positional metric array to increasing hg19 coordinates; negate only for mirrored display'}
(OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
print(path)
