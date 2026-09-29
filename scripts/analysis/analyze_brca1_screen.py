"""Validate and summarize BRCA1 SNV screens without GPU inference."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/inference'))
from run_brca1_grch38_screen import read_csv, key, sha, atomic_json
import numpy as np

BASE = ROOT/'results/brca1_grch38'
RUNS = [BASE/'screen_32768', BASE/'splice_saturation_20bp/screen_32768']
METRICS = {'relative_l2','rms_difference','cosine_distance','max_abs_difference',
           'changed_dimension_fraction','reference_l2'}


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


def region_mean(values, center, radius=20):
    """Only summarize complete windows; never silently clip or use negative indices."""
    lo, hi = center-radius, center+radius+1
    return float(values[lo:hi].mean()) if lo >= 0 and hi <= len(values) else None


def load_metrics(path, length):
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == METRICS, str(path)
        data = {k:z[k] for k in z.files}
    for k,a in data.items():
        assert a.shape == (length,) and np.isfinite(a).all(), (path,k)
        assert (a >= 0).all(), (path,k)
    assert (data['cosine_distance'] <= 2).all()
    assert (data['changed_dimension_fraction'] <= 1).all()
    return data


def choose_candidates(rows, n):
    chosen = {}
    for metric,label in [('mean_relative_l2','top_global'),('nearest_site_mean_relative_l2','top_nearest_site')]:
        ranked = sorted((r for r in rows if r[metric] is not None),
                        key=lambda r:(-r[metric],r['run_id']))[:n]
        for r in ranked:
            chosen.setdefault(r['run_id'],(r,[]))[1].append(label)
    # Prefer alternate alleles at the same locus, then same boundary/category.
    tops = list(chosen.values())
    for r,_ in tops:
        pool = [x for x in rows if x['run_id'] not in chosen
                and x['screen_group'] == r['screen_group']
                and x['nearest_site_position'] == r['nearest_site_position']]
        if pool:
            c = min(pool, key=lambda x:(x['position_grch38_1based'] != r['position_grch38_1based'],
                                        x['mean_relative_l2'], x['run_id']))
            chosen[c['run_id']] = (c,['comparison_for:'+r['run_id']])
    return [dict(r,selection_reason=';'.join(labels)) for r,labels in chosen.values()]


def plots(out, rows, candidates, junctions, preview):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = out/'figures'; folder.mkdir(exist_ok=True)
    title = 'PREVIEW — completed subset only' if preview else 'BRCA1 complete SNV screen'
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for group in sorted({r['screen_group'] for r in rows}):
        subset=[r for r in rows if r['screen_group']==group]
        for ax,metric in zip(axes,['mean_relative_l2','nearest_site_mean_relative_l2']):
            a=np.sort([r[metric] for r in subset if r[metric] is not None])
            if len(a): ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',label=f'{group} (n={len(a)})')
    for ax,name in zip(axes,['32k mean relative L2','Nearest splice boundary ±20 bp mean relative L2']):
        ax.set(xlabel=name,ylabel='Cumulative fraction',xlim=(0,None)); ax.legend(fontsize=7)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    fig.suptitle(title);fig.savefig(folder/'distributions.png',dpi=160);fig.savefig(folder/'distributions.pdf');plt.close(fig)
    for r in candidates:
        with np.load(r['metrics_path']) as z: y=z['relative_l2']
        start=r['window_start_grch38_1based'];x=np.arange(start,start+len(y))
        fig,axes=plt.subplots(2,1,figsize=(12,6),constrained_layout=True)
        # Plot every base; no peak-losing downsampling.
        for ax in axes:
            ax.plot(x,y,lw=.6,color='#225ea8')
            ax.axvline(r['position_grch38_1based'],color='black',ls='--',lw=1,label='Variant')
            for j in junctions:
                for kind,column,color in [('donor','donor_exonic_base_1based','#d95f02'),('acceptor','acceptor_exonic_base_1based','#1b9e77')]:
                    p=int(j[column])
                    if start<=p<start+len(y):ax.axvline(p,color=color,alpha=.4,lw=.8)
            ax.set(xlabel='GRCh38 chr17 position (1-based; genomic forward)',ylabel='Relative L2 (ratio)')
            ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        axes[0].set_xlim(x[0],x[-1]);axes[0].set_title('Full 32k; orange=donor, green=acceptor')
        c=r['nearest_site_position'] if r['nearest_site_mean_relative_l2'] is not None else r['position_grch38_1based']
        axes[1].set_xlim(max(start,c-200),min(x[-1],c+200));axes[1].set_title('Nearest boundary ±200 bp (variant-centered if boundary unavailable)')
        fig.suptitle(f"{title}\n{r['run_id']} | {r['selection_reason']}",fontsize=10)
        fig.savefig(folder/(r['run_id']+'.png'),dpi=160);fig.savefig(folder/(r['run_id']+'.pdf'));plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--wait',action='store_true',help='Wait for both screens to complete; no GPU use')
    parser.add_argument('--preview',type=int,default=0,help='Completed examples PER RUN; explicitly non-final output')
    parser.add_argument('--top',type=int,default=10)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    if args.preview<0 or args.top<1 or (args.preview and args.wait):parser.error('Invalid preview/top/wait combination')
    out=args.output or BASE/('analysis_preview' if args.preview else 'analysis_32768')
    if out.exists() and any(out.iterdir()):parser.error('Output must be empty; choose a new --output')
    while True:
        states=[json.loads((r/'progress.json').read_text()) for r in RUNS]
        if args.preview or all(s['status']=='complete' for s in states):break
        if any(s['status']=='failed' for s in states):raise RuntimeError('Source run failed; analysis stopped')
        if not args.wait:parser.error('Both runs must be complete; use --wait or --preview N')
        print('Waiting: '+str([(s['completed_variants'],s['total_variants']) for s in states]),flush=True)
        time.sleep(30)
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'status.json',dict(status='validating',preview=bool(args.preview)))
    try:
        configs=[json.loads((p/'config.json').read_text()) for p in RUNS]
        for field in ('model','checkpoint','checkpoint_revision','length_bp','layer','orientation','precision','reference_sha256'):
            assert configs[0][field]==configs[1][field], field
        assert sha(Path(configs[0]['reference_fasta']))==configs[0]['reference_sha256']
        with gzip.open(configs[0]['reference_fasta'],'rt') as f:
            next(f);genome=''.join(line.strip().upper() for line in f)
        junctions=read_csv(RUNS[0]/'dataset/mane_select_junctions.csv')
        sites=[(int(j[c]),j['junction_id'],kind) for j in junctions for kind,c in
               [('donor','donor_exonic_base_1based'),('acceptor','acceptor_exonic_base_1based')]]
        rows=[];seen=set();control_rows=[];config_hashes={}
        for run,config,state in zip(RUNS,configs,states):
            config_hashes[str(run)]=sha(run/'config.json')
            for name,digest in config['dataset_sha256'].items():assert sha(run/'dataset'/name)==digest,name
            plan=read_csv(run/'run_plan.csv')
            assert len(plan)==config['variants']==state['total_variants']
            if not args.preview:
                assert state['completed_variants']==len(plan)
                expected={r['run_id'] for r in plan}
                for ext in ('json','npz'):assert {p.stem for p in (run/'variants').glob('*.'+ext)}==expected
            available=[r for r in plan if all((run/'variants'/(r['run_id']+ext)).exists() for ext in ('.json','.npz'))]
            selected=available[:args.preview] if args.preview else plan
            for planned in selected:
                p=run/'variants'/(planned['run_id']+'.json');r=json.loads(p.read_text())
                assert key(r)==key(planned) and key(r) not in seen;seen.add(key(r))
                length=config['length_bp'];index=config['variant_index_0based'];pos=int(r['position_grch38_1based']);start=pos-index
                assert r['window_start_grch38_1based']==start and r['window_end_grch38_1based']==start+length-1
                assert r['variant_index_0based']==index
                ref=genome[start-1:start-1+length];assert len(ref)==length and ref[index]==r['ref']
                alt=ref[:index]+r['alt']+ref[index+1:]
                assert hashlib.sha256(ref.encode()).hexdigest()==r['ref_sequence_sha256']
                assert hashlib.sha256(alt.encode()).hexdigest()==r['alt_sequence_sha256']
                assert all(math.isfinite(r[k]) for k in ('ref_score','alt_score','delta_score'))
                assert math.isclose(r['delta_score'],r['alt_score']-r['ref_score'],abs_tol=1e-10)
                metrics=load_metrics(p.with_suffix('.npz'),length);v=metrics['relative_l2']
                assert np.isclose(v.mean(),r['mean_relative_l2'],rtol=1e-5,atol=1e-9)
                assert np.isclose(v.max(),r['max_relative_l2'],rtol=1e-5,atol=1e-9)
                boundary,jid,kind=min(sites,key=lambda x:(abs(x[0]-pos),x[1],x[2]))
                mean=region_mean(v,boundary-start)
                core=any(pos in [int(j[c]) for c in ('donor_plus1_1based','donor_plus2_1based','acceptor_minus2_1based','acceptor_minus1_1based')] for j in junctions)
                group='splice_dinucleotide' if core else 'splice_nearby_20bp' if abs(pos-boundary)<=20 else 'other_snv'
                assert group==r['screen_group']
                row={k:r.get(k,'') for k in ('run_id','chromosome','ref','alt','clinvar_variation_id','clinvar_germline_classification','clinvar_review_status','screen_group','ref_score','alt_score','delta_score','mean_relative_l2','max_relative_l2','max_offset_bp')}
                row.update(position_grch38_1based=pos,dataset='ClinVar' if run==RUNS[0] else 'splice_nonoverlap',
                           window_start_grch38_1based=start,nearest_junction_id=jid,nearest_site_type=kind,
                           nearest_site_position=boundary,distance_to_nearest_site_bp=abs(pos-boundary),
                           nearest_site_mean_relative_l2=mean,nearest_site_window_available=mean is not None,
                           variant_20bp_mean_relative_l2=region_mean(v,index),
                           pre_variant_mean_relative_l2=float(v[:index].mean()),
                           post_variant_mean_relative_l2=float(v[index+1:].mean()),
                           mean_cosine_distance=float(metrics['cosine_distance'].mean()),
                           mean_rms_difference=float(metrics['rms_difference'].mean()),metrics_path=str(p.with_suffix('.npz')))
                rows.append(row)
                if len(rows)%500==0:print(f'Validated {len(rows)} variants',flush=True)
            for p in sorted((run/'controls').glob('*.json')):
                c=json.loads(p.read_text());a=load_metrics(p.with_suffix('.npz'),config['length_bp'])
                assert np.isclose(a['relative_l2'].max(),c['max_relative_l2']) and math.isfinite(c['delta_score'])
                control_rows.append(dict(run=str(run),position=c['position'],max_relative_l2=c['max_relative_l2'],delta_score=c['delta_score']))
        if not args.preview:assert len(rows)==13455
        candidates=choose_candidates(rows,args.top)
        write_csv(out/'variant_summary.csv',rows);write_csv(out/'selected_candidates.csv',candidates);write_csv(out/'ref_controls.csv',control_rows)
        summaries=[]
        for dataset in sorted({r['dataset'] for r in rows}):
            for group in sorted({r['screen_group'] for r in rows}):
                subset=[r for r in rows if r['dataset']==dataset and r['screen_group']==group]
                for metric in ('mean_relative_l2','nearest_site_mean_relative_l2','delta_score'):
                    a=np.array([r[metric] for r in subset if r[metric] is not None])
                    if len(a):summaries.append(dict(dataset=dataset,group=group,metric=metric,n=len(a),median=float(np.median(a)),p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)),maximum=float(a.max())))
        write_csv(out/'group_summary.csv',summaries)
        report=dict(status='passed',preview=bool(args.preview),variants=len(rows),config_sha256=config_hashes,
                    ref_controls=len(control_rows),max_control_relative_l2=max((r['max_relative_l2'] for r in control_rows),default=None),
                    max_abs_control_score_delta=max((abs(r['delta_score']) for r in control_rows),default=None),
                    nearest_site_unavailable=sum(r['nearest_site_mean_relative_l2'] is None for r in rows),
                    numpy_version=np.__version__,script_sha256=sha(Path(__file__)),created_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(out/'validation.json',report)
        if not args.no_plots:plots(out,rows,candidates,junctions,bool(args.preview))
        (out/'README.md').write_text('# BRCA1 SNV analysis\n\n'+('PREVIEW: completed subset only; not representative of the full dataset.\n\n' if args.preview else '')+
            'variant_summary.csv: one row per unique SNV; blank nearest-site values mean the full ±20 bp site window is unavailable, not zero effect.\n\n'
            'selected_candidates.csv: top global/site mean relative L2 candidates and lower-change comparisons within the same nearest boundary and group; same-locus alternate alleles preferred. Selection is exploratory, not an independent test set or pathogenicity ranking.\n\n'
            'group_summary.csv and figures/: distributions and genomic position profiles. Ratios are not percentages.\n\n'
            'ref_controls.csv / validation.json: data integrity and repeated REF numerical controls; nonzero controls require inspection. Forward genomic autoregressive direction can shape spatial differences.\n\n'
            'Sources: ClinVar GRCh38 fileDate=2026-09-05, MANE v1.5 NM_007294.4; splice saturation ±20 bp. Known BRCA1 variants are not exhaustively covered. Model=evo2_7b, length=32768, norm, BF16. No measured RNA or AlphaGenome effects are included.\n')
        atomic_json(out/'status.json',dict(status='complete',preview=bool(args.preview),variants=len(rows)))
        print(f'Analysis saved to {out}',flush=True)
    except Exception as e:
        atomic_json(out/'status.json',dict(status='failed',error=repr(e)));raise


if __name__=='__main__':main()
