#!/usr/bin/env python3
"""Prepare/run a post-hoc context-purged Evo2-only descriptive stress test."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[2]
EXT=ROOT/'results/brca1_grch38/external_functional_validation'
BENCH=EXT/'seqsplice_complete_benchmark_20260916'
ORDINARY=EXT/'seqsplice_evo2_only_probe_20260916'
CONTEXT=EXT/'context_purged_probe_20260916'
GEOMETRY=EXT/'seqsplice_context_overlap_audit_20260916'
OUT=EXT/'seqsplice_evo2_only_context_purged_20260916'
DEPENDENCIES={
 'evaluate_brca1_context_purged_probe.py':'c9e0d8b6457029902a976e37960f92d547541d0c1f51b113bc82bb06174f2736',
 'evaluate_brca1_evo2_only_probe.py':'4969f296b254d776141b19b790068caba32a9ffde989b580d258e7eaf38963b9',
 'brca1_signed_kernel.py':'77eb2584a2d18c0cedb0be6966346cd118741d62ae0a028e96a2446ed2fee399',
 'report_brca1_signed_probe.py':'020ee04df2acfc4b8b0883b867dc361dd30ed575b7b4dd5522c3c9b1c1cf3ea3'}
CELLS={'MDA_MB_231':193,'HS578T':191}
METRICS=['macro_exon_MAE_pp','MAE_pp','RMSE_pp','spearman']
CHECKER=Path(__file__).with_name('check_brca1_evo2_only_context_purged.py')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def require(ok,message):
 if not ok:raise RuntimeError(message)
def helpers():
 for name,digest in DEPENDENCIES.items():require(sha(Path(__file__).with_name(name))==digest,'Frozen helper changed: '+name)
 from evaluate_brca1_evo2_only_probe import model_inputs,metrics,MODELS,CONTRASTS,SCALAR6,OUTCOME_COLUMNS
 from evaluate_brca1_context_purged_probe import purged_nested_loeo,load_activation_features_readonly
 return model_inputs,metrics,MODELS,CONTRASTS,SCALAR6,OUTCOME_COLUMNS,purged_nested_loeo,load_activation_features_readonly

def preparation_spec():
 _,_,models,contrasts,scalars,columns,_,_=helpers()
 prior=read(ORDINARY/'protocol.json')['specification']
 ov=read(ORDINARY/'validation.json');review=read(ORDINARY/'independent_result_review/validation.json')
 require(ov['status']=='PASS_POST_HOC_EVO2_ONLY_PROBE','Ordinary Evo2 analysis incomplete')
 require(review['status']=='PASS_INDEPENDENT_COMPLETE_EVO2_ONLY_RESULTS' and review['producer_validation_sha256']==sha(ORDINARY/'validation.json'),'Ordinary independent review incomplete/stale')
 cv=read(CONTEXT/'validation.json');cr=read(CONTEXT/'result_checks/validation.json')
 require(cv['status']=='PASS_DESCRIPTIVE_CONTEXT_PURGE' and cr['status']=='PASS_SEPARATE_ALGORITHM_POSTRUN_CHECK' and cr['producer_validation_sha256']==sha(CONTEXT/'validation.json'),'Original context reference check incomplete/stale')
 geo=read(GEOMETRY/'validation.json');require(geo['status']=='PASS_GEOMETRY_ONLY' and geo['counts']['exon_window_union_connected_components']==1,'Geometry scope changed')
 require(prior['models']==models and prior['fixed_contrasts']==[list(p) for p in contrasts] and prior['layer_region']==['norm','assayed_exon'],'Ordinary model scope changed')
 paths=[Path(__file__),*(Path(__file__).with_name(name) for name in DEPENDENCIES),ORDINARY/'protocol.json',ORDINARY/'validation.json',ORDINARY/'independent_result_review/validation.json',CONTEXT/'protocol.json',CONTEXT/'validation.json',CONTEXT/'result_checks/validation.json',GEOMETRY/'validation.json',BENCH/'benchmark_manifest_193.csv']
 return {
  'status':'Post-hoc descriptive stress test designed AFTER ordinary Evo2-only signed-plus-scalar improvement was observed. Not a new primary analysis and does not replace any frozen prior result.',
  'question':'Does the same Evo2-only signed addition pattern persist when all outer-train32k windows overlapping heldout test windows are excluded?',
  'preparation_reads':'Code, fixed protocol/validation metadata, coordinate manifest only; no actual outcome table, activation or OOF contents. Known ordinary results motivate this post-hoc question.',
  'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},
  'original_input_sha256':prior['original_input_sha256'],'expected_execution_fingerprint':prior['expected_execution_fingerprint'],
  'cohort':CELLS,'measured_variant_cells':384,'model_cell_groups':16,'prediction_rows':3072,'outer_fits':192,'inner_folds':992,'inner_lambda_rows':5952,
  'layer_region':['norm','assayed_exon'],'models':models,'outcome':prior['outcome'],'scalar6':scalars,'outcome_columns_whitelist':columns,
  'no_SpliceAI_features':'Exact same eight Evo2-only feature constructions; eight-column outcome whitelist. SpliceAI context-purged original OOF is a separate stored reference, never fitted or supplied to a new feature/tuning function.',
  'outer_split':'Hold whole assay exon, remove every other variant with1bp-or-more overlap between its GRCh38closed32768bp window and any test window. Keep all test variants; keep same-position ALT groups together.',
  'inner_split':'Ordinary inner LOEO only within remaining outer train. No further inner purge. All feature transforms and six-lambda choice use those training indices only.',
  'kernel_and_model_rules':prior['kernel'],'vector_only':prior['vector_only'],
  'frozen_calls':'Use pinned context module purged_nested_loeo and load_activation_features_readonly; pinned ordinary Evo2 model_inputs and metrics. Do not call any old run(), mutating loader, or old output writer.',
  'metrics':METRICS,'fixed_contrasts':[list(p) for p in contrasts],'paired_point_rows':14,'paired_point_values':56,
  'ordinary_comparison':'All16 same cell/model exact-allele groups; four point metrics purged-minus-ordinary, plus paired per-observation prediction/error differences. Effects combine training exclusion and changed tuning distribution, not leakage magnitude.',
  'reference':'Copy completed original context-purged SpliceAI_only384 OOF and2 metric rows as separately labelled reference. Recompute reference point metrics only; no extra fit, predictor feature or new comparison test.',
  'bootstrap_draws':0,'confidence_intervals':False,'p_values':False,
  'limits':['One connected genomic context component; no bootstrap CI or p-value.','Inner folds retain context overlap and differ in distribution from outer test.','Training sample sizes shrink to28–146 and vary by cell/fold.','Same alleles across cells are not independent replication.','Post-hoc follow-up of positive Evo2-only ablation; report every fixed model and contrast regardless of direction.','FL-loss prediction is not direct RNA abundance/function prediction or a mechanistic assay.'],
  'execution_gate':'Before root-authorized --run: current synthetic PASS and checker SHA; original ordinary Evo2 independent PASS linked to producer validation; original context reference post-run check; pinned numerical/source gate and complete snapshots unchanged. Prepare does not authorize fitting.'}

def prepare():
 spec=preparation_spec();OUT.mkdir(parents=True,exist_ok=True);p=OUT/'protocol.json'
 require(not (OUT/'execution_manifest.json').exists() and not (OUT/'validation.json').exists(),'Execution already started/completed')
 if p.exists():require(read(p)['specification']==spec,'Frozen preparation changed; preserve explicit revision')
 else:write(p,{'frozen_utc':datetime.now(timezone.utc).isoformat(),'specification':spec})
 return spec

def validate_frame(frame,cell,models):
 import numpy as np
 require(set(frame.model)==set(models),'Model group set differs')
 require(not frame.duplicated(['cell_line','model','run_id']).any(),'Duplicate prediction observation')
 for model,rows in frame.groupby('model'):
  require(len(rows)==CELLS[cell] and rows.Exon_legacy.nunique()==12,'Prediction cohort differs')
  require(np.isfinite(rows[['prediction_pp','full_length_loss_pp']].to_numpy()).all(),'Nonfinite prediction/outcome')
  require(rows.layer.eq('norm').all() and rows.region.eq('assayed_exon').all(),'Prediction region differs')

def summarize(pred,ordinary,audit):
 import numpy as np
 import pandas as pd
 _,metrics,models,contrasts,_,_,_,_=helpers()
 require(set(pred.cell_line)==set(ordinary.cell_line)==set(CELLS),'Cell groups differ')
 results=[];per=[];comp=[];diff=[];joined=[];lookup={}
 for cell in CELLS:
  validate_frame(pred[pred.cell_line==cell],cell,models);validate_frame(ordinary[ordinary.cell_line==cell],cell,models)
  for model in models:
   new=pred[(pred.cell_line==cell)&(pred.model==model)];old=ordinary[(ordinary.cell_line==cell)&(ordinary.model==model)]
   keys=['run_id','position_grch38_1based','ref','alt','Exon_legacy']
   pair=new.merge(old[keys+['full_length_loss_pp','prediction_pp']],on=keys,validate='one_to_one',suffixes=('','_ordinary'))
   require(len(pair)==len(new) and np.allclose(pair.full_length_loss_pp,pair.full_length_loss_pp_ordinary,atol=1e-12,rtol=0),'Ordinary exact allele/outcome changed')
   pair['prediction_difference_purged_minus_ordinary_pp']=pair.prediction_pp-pair.prediction_pp_ordinary
   pair['absolute_error_difference_purged_minus_ordinary_pp']=abs(pair.prediction_pp-pair.full_length_loss_pp)-abs(pair.prediction_pp_ordinary-pair.full_length_loss_pp_ordinary)
   joined.append(pair);n,o=metrics(new),metrics(old);meta={'cell_line':cell,'layer':'norm','region':'assayed_exon','model':model}
   results.append(meta|n);lookup[cell,model]=n;d=meta|{'n':len(new),'interpretation':'training exclusion and tuning change; not leakage magnitude'}
   for metric in METRICS:d.update({metric+'_purged':n[metric],metric+'_ordinary':o[metric],'delta_'+metric:n[metric]-o[metric]})
   diff.append(d)
   for exon,rows in new.groupby('Exon_legacy'):
    a=audit[(audit.cell_line==cell)&(audit.model==model)&(audit.outer_exon==exon)];require(len(a)==1,'Outer fold audit missing')
    per.append(meta|{'outer_exon':int(exon)}|metrics(rows)|a[['initial_train_n','excluded_overlap_n','retained_train_n','retained_train_positions','retained_train_exons','outer_test_train_window_overlap_pairs']].iloc[0].to_dict())
  for base,combined in contrasts:
   aa=pred[(pred.cell_line==cell)&(pred.model==base)].set_index('run_id').sort_index();bb=pred[(pred.cell_line==cell)&(pred.model==combined)].set_index('run_id').sort_index()
   require(aa.index.equals(bb.index) and np.array_equal(aa[['position_grch38_1based','ref','alt','Exon_legacy','full_length_loss_pp']].to_numpy(),bb[['position_grch38_1based','ref','alt','Exon_legacy','full_length_loss_pp']].to_numpy()),'Fixed contrast allele/outcome mismatch')
   comp.append({'cell_line':cell,'layer':'norm','region':'assayed_exon','baseline':base,'combined':combined,'n':CELLS[cell]}|{'delta_'+metric:lookup[cell,combined][metric]-lookup[cell,base][metric] for metric in METRICS})
 return {'metrics.csv':pd.DataFrame(results),'per_exon_metrics.csv':pd.DataFrame(per),'model_comparisons_points.csv':pd.DataFrame(comp),'ordinary_Evo2_only_point_differences.csv':pd.DataFrame(diff),'same_allele_ordinary_and_purged_predictions.csv':pd.concat(joined,ignore_index=True)}

def completed_sources(spec):
 """Read-only gate; called only in --run after current synthetic verification."""
 from report_brca1_signed_probe import gate
 snapshot=gate()
 for folder,expected_status,reviewpath,reviewstatus in [(ORDINARY,'PASS_POST_HOC_EVO2_ONLY_PROBE',ORDINARY/'independent_result_review/validation.json','PASS_INDEPENDENT_COMPLETE_EVO2_ONLY_RESULTS'),(CONTEXT,'PASS_DESCRIPTIVE_CONTEXT_PURGE',CONTEXT/'result_checks/validation.json','PASS_SEPARATE_ALGORITHM_POSTRUN_CHECK')]:
  v=read(folder/'validation.json');r=read(reviewpath)
  require(v['status']==expected_status and r['status']==reviewstatus and r['producer_validation_sha256']==sha(folder/'validation.json'),'Prior completed result/review stale')
  for p,h in read(folder/'execution_manifest.json')['source_sha256'].items():
   require(sha(ROOT/p)==h,'Prior execution source changed: '+p);snapshot[p]=h
  for name,h in v['output_sha256'].items():
   p=folder/name;require(sha(p)==h,'Prior result changed: '+name);snapshot[str(p.relative_to(ROOT))]=h
  for p in [folder/'validation.json',folder/'execution_manifest.json',reviewpath]:snapshot[str(p.relative_to(ROOT))]=sha(p)
 for p,h in spec['source_sha256'].items():require(sha(ROOT/p)==h,'Prepared source changed: '+p);snapshot[p]=h
 return snapshot

def run(spec):
 import numpy as np
 import pandas as pd
 require(not (OUT/'execution_manifest.json').exists() and not (OUT/'validation.json').exists(),'Previous execution exists')
 synthetic=read(OUT/'synthetic_CPU_validation.json')
 require(synthetic['status']=='PASS_SYNTHETIC_PREPARATION_ONLY' and synthetic['producer_sha256']==sha(Path(__file__)) and synthetic['checker_sha256']==sha(CHECKER),'Current producer/checker lacks synthetic PASS')
 for mapping in [spec['source_sha256'],spec['original_input_sha256']]:
  for p,h in mapping.items():require(sha(ROOT/p)==h,'Pinned source changed: '+p)
 model_inputs,metrics,models,contrasts,scalar_names,columns,purge,load=helpers()
 snapshot=completed_sources(spec)
 for p in [OUT/'protocol.json',OUT/'synthetic_CPU_validation.json',CHECKER]:snapshot[str(p.relative_to(ROOT))]=sha(p)
 manifest=pd.read_csv(BENCH/'benchmark_manifest_193.csv');outcome=pd.read_csv(BENCH/'measured_outcomes/SeqSplice193_per_variant_cell_values.csv',usecols=columns)
 require(len(manifest)==193 and manifest.run_id.is_unique and manifest.chromosome.eq('chr17').all(),'Manifest cohort mismatch')
 require(set(outcome.columns)==set(columns) and not any('spliceai' in c.lower() for c in outcome.columns),'Outcome whitelist differs')
 require(outcome.groupby('Cell_line').size().to_dict()==CELLS and not outcome.duplicated(['Cell_line','run_id']).any(),'Outcome cell/ID counts differ')
 require(np.isfinite(outcome.full_length_loss_pp).all(),'Nonfinite outcome')
 refs,deltas,sequence,l2,provenance=load(manifest,spec['expected_execution_fingerprint']);require(len(provenance)==386,'Activation views incomplete')
 ordinary=pd.read_csv(ORDINARY/'heldout_predictions.csv');require(len(ordinary)==3072,'Ordinary comparison count differs')
 stored=pd.read_csv(CONTEXT/'heldout_predictions.csv');reference=stored[(stored.model=='SpliceAI_only')&(stored.layer=='norm')&(stored.region=='assayed_exon')].copy()
 stored_reference_metrics=pd.read_csv(CONTEXT/'metrics.csv')
 stored_reference_metrics=stored_reference_metrics[(stored_reference_metrics.model=='SpliceAI_only')&(stored_reference_metrics.layer=='norm')&(stored_reference_metrics.region=='assayed_exon')]
 require(len(stored_reference_metrics)==2 and not stored_reference_metrics.cell_line.duplicated().any(),'Stored context reference metrics incomplete')
 require(reference.groupby('cell_line').size().to_dict()==CELLS and not reference.duplicated(['cell_line','run_id']).any(),'Context reference cohort differs')
 write(OUT/'execution_manifest.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':snapshot,'ordinary_independent_review_and_numeric_QC_complete_before_fit':True,'no_SpliceAI_new_features':True,'bootstrap_draws':0,'original_mutating_loaders_called':False})
 pd.DataFrame(provenance).to_csv(OUT/'activation_provenance.csv',index=False)
 predrows=[];tunes=[];selected=[];audits=[];inners=[];features=[];refmetrics=[];byid={r:i for i,r in enumerate(manifest.run_id)}
 for cell,sub in outcome.groupby('Cell_line'):
  sub=sub.sort_values(['position_grch38_1based','ref','alt']).reset_index(drop=True);idx=np.array([byid[r] for r in sub.run_id]);geom=manifest.iloc[idx]
  for k in ['position_grch38_1based','ref','alt']:require(np.array_equal(sub[k].to_numpy(),geom[k].to_numpy()),'Outcome/activation allele mismatch')
  require(np.array_equal(sub.Exon_legacy.to_numpy(),geom.source_legacy_exon.to_numpy()) and sub.Exon_legacy.nunique()==12,'Outcome assay exon differs')
  rr=reference[reference.cell_line==cell].set_index('run_id').loc[sub.run_id].reset_index()
  for k in ['position_grch38_1based','ref','alt','Exon_legacy']:require(np.array_equal(sub[k],rr[k]),'Context reference allele differs')
  require(np.allclose(sub.full_length_loss_pp,rr.full_length_loss_pp,atol=1e-12,rtol=0),'Context reference endpoint differs')
  rm=metrics(rr);record=stored_reference_metrics[stored_reference_metrics.cell_line==cell].iloc[0]
  require(all(np.isclose(rm[k],record[k],atol=1e-10,rtol=0,equal_nan=True) for k in METRICS),'Context reference metric changed')
  refmetrics.append({'cell_line':cell,'model':'stored_context_purged_SpliceAI_only','role':'reference_only_no_new_fit'}|rm)
  grams,inputs,scalar=model_inputs(sequence[idx],l2[idx],deltas[idx],refs[idx]);f=sub[['run_id','position_grch38_1based','ref','alt','Exon_legacy']].copy();f['cell_line']=cell
  for j,name in enumerate(scalar_names):f[name]=scalar[:,j]
  features.append(f);duplicate={}
  for label,kind,baseline in inputs:
   if label in ['signed_only','REF_only']:require(baseline.shape==(len(sub),1) and not np.count_nonzero(baseline),'Inactive vector-only scalar became active')
   p,t,s,a,ia=purge(sub.full_length_loss_pp.to_numpy(),sub.Exon_legacy.to_numpy(),sub.position_grch38_1based.to_numpy(),geom.window_start_grch38_1based.to_numpy(),geom.window_end_grch38_1based.to_numpy(),baseline,grams,kind,sub.run_id.to_numpy())
   meta={'cell_line':cell,'layer':'norm','region':'assayed_exon','model':label,'role':'post_hoc_Evo2_only_context_stress_test'}
   frame=sub[['run_id','HGVSc','position_grch38_1based','ref','alt','Exon_legacy','full_length_loss_pp']].copy()
   for k,v in meta.items():frame[k]=v
   frame['prediction_pp']=p;predrows.append(frame);duplicate[label]=p
   for accumulator,rows in [(tunes,t),(selected,s),(audits,a),(inners,ia)]:accumulator.extend([meta|row for row in rows])
   pd.concat(predrows,ignore_index=True).to_csv(OUT/'heldout_predictions.partial.csv',index=False)
   print(json.dumps({'event':'model_complete','cell':cell,'model':label,'n':len(p)}),flush=True)
  np.testing.assert_array_equal(duplicate['evo2_scalar6'],duplicate['evo2_scalar6_exact_duplicate'])
 pred=pd.concat(predrows,ignore_index=True);audit=pd.DataFrame(audits)
 require(len(pred)==3072 and len(audits)==len(selected)==192 and len(inners)==992 and len(tunes)==5952,'Fit count mismatch')
 tables=summarize(pred,ordinary,audit)
 require(len(tables['metrics.csv'])==16 and len(tables['per_exon_metrics.csv'])==192 and len(tables['model_comparisons_points.csv'])==14 and len(tables['ordinary_Evo2_only_point_differences.csv'])==16,'Summary count mismatch')
 require(audit.outer_test_train_window_overlap_pairs.eq(0).all() and np.isfinite(audit.direct_solve_max_abs_difference).all() and audit.direct_solve_max_abs_difference.ge(0).all() and audit.direct_solve_max_abs_difference.lt(1e-7).all(),'Outer overlap/direct-solve failed')
 tables.update({'heldout_predictions.csv':pred,'inner_tuning_scores.csv':pd.DataFrame(tunes),'selected_lambdas.csv':pd.DataFrame(selected),'outer_fit_audit.csv':audit,'inner_fit_audit.csv':pd.DataFrame(inners),'input_scalar_features.csv':pd.concat(features,ignore_index=True),'context_SpliceAI_reference_predictions.csv':reference,'context_SpliceAI_reference_metrics.csv':pd.DataFrame(refmetrics),'activation_provenance.csv':pd.DataFrame(provenance)})
 for name,frame in tables.items():frame.to_csv(OUT/name,index=False)
 require(all(sha(ROOT/p)==h for p,h in snapshot.items()),'Frozen original source/result changed during new run')
 (OUT/'heldout_predictions.partial.csv').unlink()
 write(OUT/'validation.json',{'status':'PASS_POST_HOC_EVO2_ONLY_CONTEXT_PURGED','completed_utc':datetime.now(timezone.utc).isoformat(),'variants':193,'variant_cells':384,'model_cell_groups':16,'prediction_rows':3072,'outer_fits':192,'inner_folds':992,'inner_lambda_rows':5952,'paired_point_rows':14,'ordinary_point_difference_rows':16,'outer_overlap_pairs':0,'bootstrap_draws':0,'confidence_intervals':False,'p_values':False,'duplicate_predictions_exact':True,'no_SpliceAI_new_features':True,'original_sources_unchanged':True,'max_direct_solve_difference':float(audit.direct_solve_max_abs_difference.max()),'producer_sha256':sha(Path(__file__)),'pinned_helpers':DEPENDENCIES,'output_sha256':{name:sha(OUT/name) for name in tables}})

def main():
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args()
 if a.prepare:prepare();print(json.dumps({'status':'PREPARED_ONLY_NO_ACTUAL_FITS','protocol':str(OUT/'protocol.json')}))
 else:run(read(OUT/'protocol.json')['specification'])
if __name__=='__main__':main()
