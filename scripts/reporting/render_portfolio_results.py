"""Regenerate portfolio figures from checked-in aggregates, without GPU inference."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'docs/results'


def rows(name):
    with (DATA/name).open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def metric(data,model,cell='MDA_MB_231'):
    hits=[r for r in data if r['cell_line']==cell and r['model']==model
          and r['layer']=='norm' and r['region']=='assayed_exon']
    assert len(hits)==1,(model,cell,len(hits))
    return float(hits[0]['macro_exon_MAE_pp'])


def render(output):
    output.mkdir(parents=True,exist_ok=True)
    qc=json.loads((DATA/'numerical_qc.json').read_text())
    effect=json.loads((DATA/'signed_primary_effect.json').read_text())
    signed=rows('signed_metrics.csv')
    purged=rows('context_purged_metrics.csv')
    evo=rows('evo2_only_metrics.csv')
    evo_purged=rows('evo2_only_context_purged_metrics.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                        'axes.spines.top':False,'axes.spines.right':False,
                        'axes.edgecolor':'#cdd9e2','text.color':'#21374b',
                        'axes.labelcolor':'#40576a','xtick.color':'#40576a','ytick.color':'#40576a'})
    fig,axs=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    fig.set_facecolor('#f3f6f8')
    fig.suptitle('Evo2 / BRCA1 | Numerical validity and held-out functional readout',fontsize=18,fontweight='bold')
    ax=axs[0,0]
    descriptive=qc['native_vs_new_descriptive']
    values=[descriptive[k]['mean'] for k in ['native_pre_variant_mean_relative_l2','new_pre_variant_mean_relative_l2']]
    bars=ax.bar(['Native inference','Numerically checked'],values,color=['#ad9b83','#3682a5'],width=.5)
    ax.bar_label(bars,labels=[f'{v:.6f}' for v in values],padding=6)
    ax.set(ylabel='Mean relative L2 (ratio)',ylim=(0,.0047),title='A  Identical input prefix: paired 183-variant comparison')
    ax.text(0,-.25,'Prefix invariance checked for 193 SNVs x 2 orientations x 6 selected layers.\nDoes not establish numerical correction of the full 13,455-SNV native screen.',transform=ax.transAxes,fontsize=8)
    ax=axs[0,1]
    x=effect['delta_MAE_pp']
    ax.axvline(0,color='#99aabd',linestyle='--',linewidth=1)
    ax.errorbar([x],[0],xerr=[[x-effect['ci_low']],[effect['ci_high']-x]],fmt='o',color='#b27545',capsize=7,markersize=9)
    ax.set(xlim=(-.35,1.4),ylim=(-.7,.7),yticks=[],xlabel='MAE(combined) - MAE(matched scalar), percentage points',
           title='B  Primary signed-vector comparison: no improvement')
    ax.text(x,.20,f"+{x:.3f} pp\n95% CI [{effect['ci_low']:.3f}, {effect['ci_high']:.3f}]",ha='center',fontsize=12)
    ax.text(0,-.25,'MDA-MB-231; n=193; norm / assayed exon; macro-exon MAE.\n5,000 exon bootstrap draws; conditional on fixed OOF predictions and 12 exons.',transform=ax.transAxes,fontsize=8)
    ax=axs[1,0]
    models=['SpliceAI_only','scalar_matched7','scalar_plus_signed','scalar_plus_REF']
    labels=['SpliceAI','Matched scalar','Scalar + signed','Scalar + REF']
    pos=np.arange(len(models))
    ax.barh(pos-.18,[metric(signed,m) for m in models],height=.34,label='Ordinary LOEO',color='#afc4d2')
    ax.barh(pos+.18,[metric(purged,m) for m in models],height=.34,label='Context-purged outer LOEO',color='#367d9e')
    ax.set(yticks=pos,yticklabels=labels,xlabel='Macro-exon MAE (pp), lower is better',title='C  Overlapping-context sensitivity: MDA-MB-231')
    ax.invert_yaxis();ax.set_ylim(3.6,-1)
    ax.legend(frameon=False,fontsize=8,loc='upper left',ncol=2)
    ax.text(0,-.24,'Outer train/test input-window overlap is removed in the purged analysis.\nInner LOEO remains ordinary; single-gene descriptive stress test, not replication.',transform=ax.transAxes,fontsize=8)
    ax=axs[1,1]
    models=['evo2_sequence2','evo2_scalar6','signed_only','REF_only']
    labels=['Sequence scores','Evo2 scalar6','Signed only','REF only']
    ax.barh(pos-.18,[metric(evo,m) for m in models],height=.34,color='#afc4d2')
    ax.barh(pos+.18,[metric(evo_purged,m) for m in models],height=.34,color='#367d9e')
    ax.set(yticks=pos,yticklabels=labels,xlabel='Macro-exon MAE (pp), lower is better',title='D  Evo2-only controls: post-hoc sensitivity analysis')
    ax.invert_yaxis()
    ax.text(0,-.24,'Same color key as C. These controls do not include SpliceAI as a new feature.\nNo confidence interval or p-value was computed for these purged point estimates.',transform=ax.transAxes,fontsize=8)
    for ax in axs.flat:ax.set_facecolor('white')
    fig.savefig(output/'research-results.png',dpi=180,bbox_inches='tight')
    fig.savefig(output/'research-results.svg',bbox_inches='tight')
    svg=output/'research-results.svg'
    svg.write_bytes(('\n'.join(x.rstrip() for x in svg.read_text(encoding='utf-8').splitlines())+'\n').encode())
    plt.close(fig)
    summary={'native_screen_variants':13455,'corrected_snv_cohort':193,'corrected_variant_views':386,
             'primary_effect':effect,
             'primary_macro_MAE':{cell:{model:metric(signed,model,cell)
                                      for model in ['scalar_matched7','scalar_plus_signed']}
                                  for cell in ['MDA_MB_231','HS578T']},
             'context_purged_macro_MAE':{m:metric(purged,m) for m in ['SpliceAI_only','scalar_matched7','scalar_plus_signed','scalar_plus_REF']}}
    (output/'figure-data.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'docs/figures')
    render(p.parse_args().output)
