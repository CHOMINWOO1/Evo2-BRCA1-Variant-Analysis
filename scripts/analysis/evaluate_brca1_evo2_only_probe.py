#!/usr/bin/env python3
"""Post-hoc Evo2-only ablation of the frozen minigene FL-loss probe.

--prepare opens code and protocol metadata only. Actual fitting requires --run
after review; original analyses and all their inputs remain immutable.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'

ROOT=Path(__file__).resolve().parents[2]
EXT=ROOT/'results/brca1_grch38/external_functional_validation'
BENCH=EXT/'seqsplice_complete_benchmark_20260916'
ORIGINAL=EXT/'seqsplice_signed_probe_20260916'
OUT=EXT/'seqsplice_evo2_only_probe_20260916'
KERNEL=Path(__file__).with_name('brca1_signed_kernel.py')
READER=Path(__file__).with_name('evaluate_brca1_context_purged_probe.py')
GATE=Path(__file__).with_name('report_brca1_signed_probe.py')
PINNED={'kernel':'77eb2584a2d18c0cedb0be6966346cd118741d62ae0a028e96a2446ed2fee399',
        'reader':'c9e0d8b6457029902a976e37960f92d547541d0c1f51b113bc82bb06174f2736',
        'gate':'020ee04df2acfc4b8b0883b867dc361dd30ed575b7b4dd5522c3c9b1c1cf3ea3'}
CELLS={'MDA_MB_231':193,'HS578T':191}
MODELS=['evo2_scalar6','evo2_sequence2','signed_only','REF_only','evo2_scalar6_plus_signed',
        'evo2_scalar6_plus_REF','evo2_scalar6_plus_unit_direction','evo2_scalar6_exact_duplicate']
CONTRASTS=[('evo2_scalar6','evo2_scalar6_plus_signed'),('evo2_scalar6','evo2_scalar6_plus_REF'),
           ('evo2_scalar6','evo2_scalar6_plus_unit_direction'),('evo2_scalar6','evo2_scalar6_exact_duplicate'),
           ('evo2_sequence2','evo2_scalar6'),('REF_only','signed_only'),('signed_only','evo2_scalar6_plus_signed')]
SCALAR6=['negative_corrected_forward_delta_score','negative_corrected_rc_delta_score',
         'corrected_forward_norm_whole32k_mean_relative_l2','corrected_rc_norm_whole32k_mean_relative_l2',
         'forward_assayed_exon_pooled_delta_norm','rc_assayed_exon_pooled_delta_norm']
OUTCOME_COLUMNS=['Cell_line','run_id','HGVSc','position_grch38_1based','ref','alt','Exon_legacy','full_length_loss_pp']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def require(c,m):
    if not c:raise RuntimeError(m)


def prepare():
    prior=read(ORIGINAL/'protocol.json')['specification']
    for key,path in [('kernel',KERNEL),('reader',READER),('gate',GATE)]:require(sha(path)==PINNED[key],'Frozen dependency changed: '+key)
    files=[Path(__file__),KERNEL,READER,GATE,ORIGINAL/'protocol.json',ORIGINAL/'implementation_clarification.json',
           ORIGINAL/'validation.json',ORIGINAL/'independent_result_review/validation.json']
    spec={
        'status':'Post-hoc exploratory ablation designed AFTER observing the negative primary signed-plus-SpliceAI result. Does not replace or revise that original primary analysis.',
        'question':'Within Evo2-derived features, does a signed pooled representation improve held-out minigene FL-loss prediction relative to Evo2 scalar6?',
        'not_tested_directly':'RNA or function scores are not outcomes or inputs in this fit. RNA information requires a separately labelled cross-assay follow-up; FL loss is not equivalent to RNA representation.',
        'preparation_reads':'Code and frozen protocol/validation metadata only; no activation, phenotype table or OOF prediction contents.',
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},
        'original_input_sha256':prior['inputs'],'expected_execution_fingerprint':prior['expected_execution_fingerprint'],
        'cohort':CELLS,'measured_variant_cells':384,'model_cell_groups':16,'prediction_rows':3072,'outer_fits':192,
        'inner_folds':2112,'inner_lambda_rows':12672,'layer_region':['norm','assayed_exon'],
        'models':MODELS,'outcome':prior['outcome'],'primary_cell':'MDA_MB_231','supporting_cell':'HS578T',
        'primary_effect':'MDA macro-exon MAE(evo2_scalar6_plus_signed) minus macro-exon MAE(evo2_scalar6). Positive means larger error.',
        'scalar6':SCALAR6,'sequence2':SCALAR6[:2],
        'vector_only':'Use the pinned kernel with an exactly-zero n×1 scalar baseline. It is inactive; average_active renormalizes to the active vector block only. Each view still uses train-only centering/RMS normalization.',
        'no_SpliceAI_features':'Read measured outcomes via the eight-column whitelist only. Construct every new model from sequence2, globalL2_2 and norm/exon REF or signed vectors. SpliceAI-only is a stored original OOF reference, never a predictor column or tuning target.',
        'outcome_columns_whitelist':OUTCOME_COLUMNS,
        'original_SpliceAI_reference':'Copy the original norm/assayed_exon SpliceAI_only OOF predictions and compute its descriptive point metrics as a separately labelled reference. Do not refit it or add it to any new feature block. No new paired tests against this reference in this protocol.',
        'kernel':{k:prior[k] for k in ['baseline_kernel','vector_kernel','unit_direction','combined_kernel','solver','lambda_grid','lambda_selection','outer_split']},
        'frozen_helper':'Use original brca1_signed_kernel.raw_gram,nested_loeo,kernels and its six lambda values unchanged; no adaptive alpha, new hyperparameters, layer or region selection.',
        'metrics':['macro_exon_MAE_pp','MAE_pp','RMSE_pp','spearman'],
        'fixed_contrasts':[list(x) for x in CONTRASTS],
        'paired_comparison_rows':28,'comparison_weightings':['macro_exon','variant_weighted'],
        'bootstrap':'Reuse exact original 5000×12 author-exon draw matrix and order, shared across both cells/all fixed contrasts. Conditional percentile95% intervals on fixed OOF predictions; no retraining uncertainty or p-values.',
        'execution_gate':'Original full193×2 extraction, original producer and independent result audit PASS; current source/marker/NPZ/REF-control hashes via pinned report gate. New prepare/synthetic review must be completed before root starts --run.',
        'activation_reader':'Pinned context-purged module load_activation_features_readonly only, not its fitting routine or original evaluator load_complete. Returns norm/exon signed and REF vectors, sequence penalties and whole32k L2.',
        'limits':['Designed after negative original primary results; all eight models and all fixed contrasts must be reported.',
                  'Two cells share variants;12 exons and overlapping32k contexts limit independence.',
                  'Vector-only or Evo2-only improvements do not establish added value over SpliceAI or overturn the original primary result.',
                  'A fixed linear kernel/regularization comparison does not exhaust all representation information or identify a biological mechanism.']}
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'protocol.json'
    if p.exists():require(read(p)['specification']==spec,'Preparation changed; preserve revision explicitly')
    else:
        require(not (OUT/'execution_manifest.json').exists(),'New run already started')
        write(p,{'frozen_utc':datetime.now(timezone.utc).isoformat(),'specification':spec})
    return spec


def model_inputs(sequence,l2,delta,ref):
    import numpy as np
    from brca1_signed_kernel import raw_gram
    n=len(sequence)
    require(sequence.shape==l2.shape==(n,2),'Expected two orientations for sequence/L2')
    require(delta.shape==ref.shape and delta.ndim==3 and delta.shape[:2]==(n,2),'Vector shape differs')
    require(all(np.isfinite(x).all() for x in [sequence,l2,delta,ref]),'Nonfinite feature')
    scalar6=np.concatenate([sequence,l2,np.linalg.norm(delta,axis=-1)],axis=1)
    inactive=np.zeros((n,1),dtype=np.float64)
    grams={'signed':[raw_gram(delta[:,v]) for v in range(2)],'REF':[raw_gram(ref[:,v]) for v in range(2)],
           'unit_direction':[raw_gram(delta[:,v],unit_direction=True) for v in range(2)]}
    models=[('evo2_scalar6','scalar',scalar6),('evo2_sequence2','scalar',sequence),
            ('signed_only','signed',inactive),('REF_only','REF',inactive),
            ('evo2_scalar6_plus_signed','signed',scalar6),('evo2_scalar6_plus_REF','REF',scalar6),
            ('evo2_scalar6_plus_unit_direction','unit_direction',scalar6),('evo2_scalar6_exact_duplicate','duplicate_scalar',scalar6)]
    require([x[0] for x in models]==MODELS and scalar6.shape==(n,6),'Model scope differs')
    return grams,models,scalar6


def metrics(frame):
    import numpy as np
    from scipy.stats import spearmanr
    error=np.abs(frame.prediction_pp.to_numpy()-frame.full_length_loss_pp.to_numpy())
    x,y=frame.prediction_pp.to_numpy(),frame.full_length_loss_pp.to_numpy()
    rho=float(spearmanr(x,y).statistic) if np.ptp(x)>0 and np.ptp(y)>0 else float('nan')
    return {'n':len(frame),'MAE_pp':float(error.mean()),'macro_exon_MAE_pp':float(frame.assign(error=error).groupby('Exon_legacy').error.mean().mean()),
            'RMSE_pp':float(np.sqrt(np.mean(error**2))),'spearman':rho,'spearman_defined':bool(np.isfinite(rho))}


def summarize(predictions,exons,draws):
    import numpy as np
    import pandas as pd
    require(draws.shape==(5000,12) and len(exons)==12,'Bootstrap scope differs')
    score=[];per=[];comparisons=[]
    for (cell,model),frame in predictions.groupby(['cell_line','model']):
        require(len(frame)==CELLS[cell] and frame.Exon_legacy.nunique()==12,'Prediction cohort incomplete')
        meta={'cell_line':cell,'model':model,'layer':'norm','region':'assayed_exon'}
        score.append(meta|metrics(frame))
        for exon,sub in frame.groupby('Exon_legacy'):
            per.append(meta|{'exon':int(exon),'n':len(sub),'MAE_pp':float(np.abs(sub.prediction_pp-sub.full_length_loss_pp).mean())})
    for cell in CELLS:
        sub=predictions[predictions.cell_line==cell]
        for a,b in CONTRASTS:
            aa=sub[sub.model==a].set_index('run_id').sort_index()
            bb=sub[sub.model==b].set_index('run_id').sort_index()
            require(aa.index.equals(bb.index) and np.array_equal(aa.Exon_legacy,bb.Exon_legacy) and np.array_equal(aa.full_length_loss_pp,bb.full_length_loss_pp),'Paired allele/outcome differs')
            diff=np.abs(bb.prediction_pp.to_numpy()-bb.full_length_loss_pp.to_numpy())-np.abs(aa.prediction_pp.to_numpy()-aa.full_length_loss_pp.to_numpy())
            groups=aa.Exon_legacy.to_numpy()
            sums=np.array([diff[groups==e].sum() for e in exons]);counts=np.array([sum(groups==e) for e in exons])
            require((counts>0).all(),'Empty bootstrap exon')
            means=sums/counts
            for weighting,estimate,boot in [('macro_exon',means.mean(),means[draws].mean(1)),
                ('variant_weighted',sums.sum()/counts.sum(),sums[draws].sum(1)/counts[draws].sum(1))]:
                low,high=np.quantile(boot,[.025,.975])
                comparisons.append({'cell_line':cell,'layer':'norm','region':'assayed_exon','baseline':a,'combined':b,'weighting':weighting,
                    'delta_MAE_pp':float(estimate),'ci_low':float(low),'ci_high':float(high),'n':len(aa),'bootstrap_draws':5000})
    return pd.DataFrame(score),pd.DataFrame(per),pd.DataFrame(comparisons)


def run(spec):
    import numpy as np
    import pandas as pd
    require(not (OUT/'execution_manifest.json').exists() and not (OUT/'validation.json').exists(),'Previous run exists; preserve and review')
    synthetic=read(OUT/'synthetic_CPU_validation.json')
    require(synthetic['status']=='PASS_SYNTHETIC_AND_METADATA_ONLY' and synthetic['script_sha256']==sha(Path(__file__)),
            'Current implementation has not passed synthetic validation')
    checker=Path(__file__).with_name('check_brca1_evo2_only_probe.py')
    require(synthetic['test_script_sha256']==sha(checker),'Synthetic checker changed')
    for mapping in [spec['source_sha256'],spec['original_input_sha256']]:
        for p,digest in mapping.items():require(sha(ROOT/p)==digest,'Pinned source changed '+p)
    from report_brca1_signed_probe import gate
    from evaluate_brca1_context_purged_probe import load_activation_features_readonly
    from brca1_signed_kernel import nested_loeo,LAMBDAS
    require(LAMBDAS.tolist()==spec['kernel']['lambda_grid'],'Lambda grid changed')
    snapshot=gate()
    independent=read(ORIGINAL/'independent_result_review/validation.json')
    require(independent['status']=='PASS_INDEPENDENT_COMPLETE_SIGNED_PROBE_RESULTS','Original independent results audit missing')
    require(independent['producer_validation_sha256']==sha(ORIGINAL/'validation.json'),'Original independent audit stale')
    manifest=pd.read_csv(BENCH/'benchmark_manifest_193.csv')
    outcome=pd.read_csv(BENCH/'measured_outcomes/SeqSplice193_per_variant_cell_values.csv',usecols=OUTCOME_COLUMNS)
    require(set(outcome.columns)==set(OUTCOME_COLUMNS) and not any('spliceai' in c.lower() for c in outcome.columns),'SpliceAI leaked into new input table')
    require(len(manifest)==193 and manifest.run_id.is_unique and manifest.chromosome.eq('chr17').all(),'Manifest scope differs')
    require(outcome.groupby('Cell_line').size().to_dict()==CELLS and not outcome.duplicated(['Cell_line','run_id']).any(),'Outcome cohort differs')
    require(np.isfinite(outcome.full_length_loss_pp).all(),'Nonfinite outcome')
    refs,deltas,sequence,l2,provenance=load_activation_features_readonly(manifest,spec['expected_execution_fingerprint'])
    require(len(provenance)==386,'Activation provenance incomplete')
    # This reference is read independently, never passed to model_inputs/nested_loeo.
    original=pd.read_csv(ORIGINAL/'heldout_predictions.csv')
    reference=original[(original.layer=='norm')&(original.region=='assayed_exon')&(original.model=='SpliceAI_only')].copy()
    require(reference.groupby('cell_line').size().to_dict()==CELLS,'Original SpliceAI-only reference incomplete')
    with np.load(ORIGINAL/'shared_exon_bootstrap_draws.npz',allow_pickle=False) as z:exons,draws=z['exon_order'],z['draws']
    require(np.array_equal(exons,np.sort(outcome.Exon_legacy.unique())) and np.array_equal(draws,np.random.default_rng(20260916).integers(0,12,(5000,12))),'Shared original bootstrap differs')
    snapshot.update(spec['source_sha256'])
    for p in [OUT/'protocol.json',OUT/'synthetic_CPU_validation.json',checker,ORIGINAL/'independent_result_review/validation.json']:
        snapshot[str(p.relative_to(ROOT))]=sha(p)
    write(OUT/'execution_manifest.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':snapshot,
        'no_SpliceAI_in_new_feature_columns':True,'outcome_columns_read':list(outcome.columns),
        'original_primary_result_preserved':True,'numerical_and_original_analysis_QC_complete_before_any_new_fit':True})
    pd.DataFrame(provenance).to_csv(OUT/'activation_provenance.csv',index=False)
    by_id={r:i for i,r in enumerate(manifest.run_id)}
    predictions=[];tuning=[];selected=[];audits=[];features=[];reference_metrics=[]
    for cell,sub in outcome.groupby('Cell_line'):
        sub=sub.sort_values(['position_grch38_1based','ref','alt']).reset_index(drop=True)
        idx=np.array([by_id[r] for r in sub.run_id]);geom=manifest.iloc[idx]
        for k in ['position_grch38_1based','ref','alt']:require(np.array_equal(sub[k].to_numpy(),geom[k].to_numpy()),'Outcome/activation allele mismatch')
        require(np.array_equal(sub.Exon_legacy.to_numpy(),geom.source_legacy_exon.to_numpy()),'Outcome assay exon mismatch')
        require(sub.Exon_legacy.nunique()==12,'Missing assay exon')
        ref=reference[reference.cell_line==cell].set_index('run_id').loc[sub.run_id].reset_index()
        for k in ['position_grch38_1based','ref','alt','Exon_legacy']:require(np.array_equal(sub[k],ref[k]),'Original reference allele mismatch')
        require(np.allclose(sub.full_length_loss_pp,ref.full_length_loss_pp,atol=1e-12,rtol=0),'Original reference outcome mismatch')
        reference_metrics.append({'cell_line':cell,'model':'original_SpliceAI_only_reference','role':'stored_original_OOF_reference_only'}|metrics(ref))
        grams,models,scalar6=model_inputs(sequence[idx],l2[idx],deltas[idx],refs[idx])
        feature=sub[['run_id','position_grch38_1based','ref','alt','Exon_legacy']].copy();feature['cell_line']=cell
        for j,name in enumerate(SCALAR6):feature[name]=scalar6[:,j]
        features.append(feature)
        duplicate={}
        for label,kind,baseline in models:
            require(baseline.shape[1] in [1,2,6],'Unexpected feature dimension')
            if label in ['signed_only','REF_only']:require(np.count_nonzero(baseline)==0,'Vector-only scalar block active')
            pred,inner,chosen,outer=nested_loeo(sub.full_length_loss_pp.to_numpy(),sub.Exon_legacy.to_numpy(),sub.position_grch38_1based.to_numpy(),baseline,grams,kind)
            metadata={'cell_line':cell,'layer':'norm','region':'assayed_exon','model':label,'role':'post_hoc_Evo2_only_ablation'}
            rows=sub[['run_id','HGVSc','position_grch38_1based','ref','alt','Exon_legacy','full_length_loss_pp']].copy()
            for k,v in metadata.items():rows[k]=v
            rows['prediction_pp']=pred;predictions.append(rows)
            for accum,data in [(tuning,inner),(selected,chosen),(audits,outer)]:accum.extend([metadata|r for r in data])
            duplicate[label]=pred
            pd.concat(predictions,ignore_index=True).to_csv(OUT/'heldout_predictions.partial.csv',index=False)
            print(json.dumps({'event':'model_complete','cell':cell,'model':label,'n':len(pred)}),flush=True)
        np.testing.assert_array_equal(duplicate['evo2_scalar6'],duplicate['evo2_scalar6_exact_duplicate'])
    pred=pd.concat(predictions,ignore_index=True)
    require(len(pred)==3072 and len(audits)==len(selected)==192 and len(tuning)==12672,'Fit accounting differs')
    m,e,c=summarize(pred,exons,draws)
    require(len(m)==16 and len(e)==192 and len(c)==28,'Summary accounting differs')
    tables={'heldout_predictions.csv':pred,'inner_tuning_scores.csv':pd.DataFrame(tuning),'selected_lambdas.csv':pd.DataFrame(selected),
        'outer_fit_audit.csv':pd.DataFrame(audits),'metrics.csv':m,'per_exon_metrics.csv':e,'paired_comparisons.csv':c,
        'input_scalar_features.csv':pd.concat(features,ignore_index=True),'original_SpliceAI_reference_predictions.csv':reference,
        'original_SpliceAI_reference_metrics.csv':pd.DataFrame(reference_metrics)}
    for name,table in tables.items():table.to_csv(OUT/name,index=False)
    primary=c[(c.cell_line=='MDA_MB_231')&(c.baseline=='evo2_scalar6')&(c.combined=='evo2_scalar6_plus_signed')&(c.weighting=='macro_exon')]
    require(len(primary)==1,'Primary effect absent')
    write(OUT/'primary_effect.json',primary.iloc[0].to_dict())
    require(all(sha(ROOT/p)==h for p,h in snapshot.items()),'Original source or results changed during ablation')
    (OUT/'heldout_predictions.partial.csv').unlink()
    write(OUT/'validation.json',{'status':'PASS_POST_HOC_EVO2_ONLY_PROBE','completed_utc':datetime.now(timezone.utc).isoformat(),
        'variants':193,'variant_cells':384,'model_cell_groups':16,'prediction_rows':3072,'outer_fits':192,'inner_lambda_rows':12672,
        'paired_summaries':28,'no_SpliceAI_new_features':True,'vector_only_zero_baselines':True,
        'duplicate_predictions_exact':True,'original_sources_results_unchanged':True,'original_primary_not_replaced':True,
        'max_outer_direct_solve_difference':float(max(r['direct_solve_max_abs_difference'] for r in audits)),
        'producer_script_sha256':sha(Path(__file__)),'kernel_sha256':sha(KERNEL),'activation_reader_sha256':sha(READER),
        'output_sha256':{name:sha(OUT/name) for name in tables}})


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args()
    if a.prepare:
        prepare();print(json.dumps({'status':'PREPARED_NO_NEW_ACTIVATION_OR_OUTCOME_READS_NO_FITS','protocol':str(OUT/'protocol.json')}))
    else:run(read(OUT/'protocol.json')['specification'])


if __name__=='__main__':main()
