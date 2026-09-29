#!/usr/bin/env python3
"""Synthetic integration and metadata-only preparation checks; no actual fits."""
import sys,os
sys.dont_write_bytecode=True
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import importlib.util,json,tempfile,math
import numpy as np
import pandas as pd
CODE=Path(__file__).with_name('evaluate_brca1_evo2_only_context_purged.py')
spec=importlib.util.spec_from_file_location('evo2_context_checked',CODE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
model_inputs,metrics,MODELS,CONTRASTS,SCALAR6,COLUMNS,purge,_=m.helpers()
# Block actual phenotype, activation and OOF reads for the whole checker.
forbidden=[m.BENCH/'measured_outcomes',m.ROOT/'results/brca1_grch38/seqsplice_signed_layers_20260916/variants',m.ROOT/'results/brca1_grch38/seqsplice_signed_layers_20260916/controls']
for folder in [m.ORDINARY,m.CONTEXT,m.OUT,m.EXT/'seqsplice_signed_probe_20260916']:
 forbidden += [folder/'heldout_predictions.csv',folder/'heldout_predictions.partial.csv',folder/'input_scalar_features.csv']
denied=[]
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).resolve()
  if any(p==q or q in p.parents for q in forbidden):denied.append(str(p));raise AssertionError('Forbidden actual-data read '+str(p))
sys.addaudithook(guard)

def expect_error(fn,label):
 try:fn()
 except (RuntimeError,AssertionError,ValueError,FloatingPointError):return label
 raise AssertionError('Negative fixture accepted: '+label)

def main():
 paths=[CODE,Path(__file__),*(CODE.with_name(name) for name in m.DEPENDENCIES),m.ORDINARY/'protocol.json',m.ORDINARY/'validation.json',m.ORDINARY/'independent_result_review/validation.json',m.CONTEXT/'protocol.json',m.CONTEXT/'validation.json',m.CONTEXT/'result_checks/validation.json']
 snapshots={str(p):m.sha(p) for p in paths}
 geom=pd.read_csv(m.BENCH/'benchmark_manifest_193.csv',usecols=['run_id','chromosome','position_grch38_1based','ref','alt','source_legacy_exon','window_start_grch38_1based','window_end_grch38_1based'])
 assert len(geom)==193 and geom.run_id.is_unique
 rng=np.random.default_rng(260916991);allpred=[];allaudit=[];counts={'outer':0,'inner':0,'tuning':0,'pred':0};maxgeometrydiff=0;lambda_choices=0;negative=[];basecase=None
 for cell,n in m.CELLS.items():
  # Synthetic HS missingness: first2 coordinate rows, not actual missing labels.
  g=geom.copy() if n==193 else geom.iloc[2:].copy();g=g.reset_index(drop=True)
  seq=rng.normal(size=(n,2));l2=rng.uniform(size=(n,2));delta=rng.normal(size=(n,2,6));ref=1e7+rng.normal(0,.1,size=(n,2,6));y=rng.normal(5,20,n)
  positions=g.position_grch38_1based.to_numpy();groups=g.source_legacy_exon.to_numpy();starts=g.window_start_grch38_1based.to_numpy();ends=g.window_end_grch38_1based.to_numpy();ids=np.array(['synthetic_'+r for r in g.run_id])
  grams,models,scalar=model_inputs(seq,l2,delta,ref);assert scalar.shape==(n,6)
  same={};heldout=np.flatnonzero(groups==3)
  intersects=np.array([any(abs(int(p)-int(positions[t]))<32768 for t in heldout) for p in positions]);removed=np.flatnonzero((groups!=3)&intersects)
  for label,kind,baseline in models:
   if label in ['signed_only','REF_only']:assert baseline.shape==(n,1) and not baseline.any()
   pred,tuning,selected,audit,inner=purge(y,groups,positions,starts,ends,baseline,grams,kind,ids)
   assert len(audit)==len(selected)==12 and len(inner)==62 and len(tuning)==372
   counts['outer']+=len(audit);counts['inner']+=len(inner);counts['tuning']+=len(tuning);counts['pred']+=len(pred)
   for a in audit:
    ex=a['outer_exon'];test=set(ids[groups==ex]);initial=set(ids[groups!=ex]);train={ids[i] for i,p in enumerate(positions) if groups[i]!=ex and all(abs(int(p)-int(positions[j]))>=32768 for j in np.flatnonzero(groups==ex))}
    assert set(json.loads(a['train_ids_json']))==train and set(json.loads(a['test_ids_json']))==test and set(json.loads(a['removed_ids_json']))==initial-train
    assert a['outer_test_train_window_overlap_pairs']==0 and a['direct_solve_max_abs_difference']<1e-7
    selected_row=next(r for r in selected if r['outer_exon']==ex);loss={lam:[] for lam in [.0001,.001,.01,.1,1.,10.]}
    for r in tuning:
     if r['outer_exon']==ex:loss[r['lambda']].append(r['validation_MAE_pp'])
    av={lam:math.fsum(v)/len(v) for lam,v in loss.items()};best=min(av.values());win=max(lam for lam,v in av.items() if v<=best+1e-12)
    assert win==selected_row['lambda'] and abs(av[win]-selected_row['inner_macro_exon_MAE_pp'])<1e-12;lambda_choices+=1
   # Both heldout and purged-away outcomes are unavailable to own-fold fit/tuning.
   changed=y.copy();changed[intersects]+=100000.
   alternate=purge(changed,groups,positions,starts,ends,baseline,grams,kind,ids)
   np.testing.assert_array_equal(pred[heldout],alternate[0][heldout]);assert selected[0]['lambda']==alternate[2][0]['lambda']
   assert [r for r in tuning if r['outer_exon']==3]==[r for r in alternate[1] if r['outer_exon']==3]
   # Purged-away features cannot change own-fold kernels or predictions.
   s2,l2b,d2,r2=[a.copy() for a in [seq,l2,delta,ref]]
   for a in [s2,l2b,d2,r2]:a[removed]+=1000.
   gg,mm,_=model_inputs(s2,l2b,d2,r2);bb=next(b for z,k,b in mm if z==label)
   altered=purge(y,groups,positions,starts,ends,bb,gg,kind,ids)
   np.testing.assert_array_equal(pred[heldout],altered[0][heldout]);assert [r for r in tuning if r['outer_exon']==3]==[r for r in altered[1] if r['outer_exon']==3]
   same[label]=pred
   frame=pd.DataFrame({'run_id':ids,'HGVSc':['synthetic_only']*n,'position_grch38_1based':positions,'ref':g.ref,'alt':g.alt,'Exon_legacy':groups,'full_length_loss_pp':y,'cell_line':cell,'layer':'norm','region':'assayed_exon','model':label,'prediction_pp':pred})
   allpred.append(frame);allaudit.extend([{'cell_line':cell,'layer':'norm','region':'assayed_exon','model':label}|a for a in audit])
  np.testing.assert_array_equal(same['evo2_scalar6'],same['evo2_scalar6_exact_duplicate'])
  if basecase is None:basecase=(y,groups,positions,starts,ends,models[0][2],grams,ids,seq,l2,delta,ref)
 assert counts=={'outer':192,'inner':992,'tuning':5952,'pred':3072} and lambda_choices==192
 pred=pd.concat(allpred,ignore_index=True);ordinary=pred.copy();ordinary['prediction_pp']+=.75;audit=pd.DataFrame(allaudit)
 tables=m.summarize(pred,ordinary,audit)
 assert {k:len(v) for k,v in tables.items()}=={'metrics.csv':16,'per_exon_metrics.csv':192,'model_comparisons_points.csv':14,'ordinary_Evo2_only_point_differences.csv':16,'same_allele_ordinary_and_purged_predictions.csv':3072}
 for _,r in tables['ordinary_Evo2_only_point_differences.csv'].iterrows():
  a=pred[(pred.cell_line==r.cell_line)&(pred.model==r.model)];b=ordinary[(ordinary.cell_line==r.cell_line)&(ordinary.model==r.model)]
  ae=np.abs(a.prediction_pp-a.full_length_loss_pp);be=np.abs(b.prediction_pp-b.full_length_loss_pp)
  assert abs(r.delta_MAE_pp-(ae.mean()-be.mean()))<1e-12
  am=a.assign(e=ae).groupby('Exon_legacy').e.mean().mean();bm=b.assign(e=be).groupby('Exon_legacy').e.mean().mean()
  assert abs(r.delta_macro_exon_MAE_pp-(am-bm))<1e-12
 assert np.allclose(tables['same_allele_ordinary_and_purged_predictions.csv'].prediction_difference_purged_minus_ordinary_pp,-.75,rtol=0,atol=1e-12)
 for _,r in tables['model_comparisons_points.csv'].iterrows():
  a=pred[(pred.cell_line==r.cell_line)&(pred.model==r.baseline)];b=pred[(pred.cell_line==r.cell_line)&(pred.model==r.combined)]
  for field in m.METRICS:assert abs(r['delta_'+field]-(metrics(b)[field]-metrics(a)[field]))<1e-12
 y,groups,pos,starts,ends,base,grams,ids,seq,l2,delta,ref=basecase
 for i,name in enumerate(['sequence','globalL2','delta','REF']):
  args=[a.copy() for a in [seq,l2,delta,ref]];args[i].flat[0]=np.nan
  negative.append(expect_error(lambda:model_inputs(*args),'nonfinite '+name))
 bady=y.copy();bady[0]=np.inf;negative.append(expect_error(lambda:purge(bady,groups,pos,starts,ends,base,grams,'scalar',ids),'nonfinite outcome'))
 badend=ends.copy();badend[0]+=1;negative.append(expect_error(lambda:purge(y,groups,pos,starts,badend,base,grams,'scalar',ids),'wrong window length'))
 badids=ids.copy();badids[1]=badids[0];negative.append(expect_error(lambda:purge(y,groups,pos,starts,ends,base,grams,'scalar',badids),'duplicate run ID'))
 closepos=np.arange(len(y))+100000;negative.append(expect_error(lambda:purge(y,groups,closepos,closepos-16384,closepos+16383,base,grams,'scalar',ids),'empty purged train'))
 badpred=pred.copy();badpred.loc[0,'prediction_pp']=np.nan;negative.append(expect_error(lambda:m.summarize(badpred,ordinary,audit),'nonfinite summary prediction'))
 badold=ordinary.copy();badold.loc[0,'alt']='Z';negative.append(expect_error(lambda:m.summarize(pred,badold,audit),'ordinary allele mismatch'))
 # All-inactive vector kernel gives exact train mean, and all six lambda losses tie.
 gg,mods,_=model_inputs(seq,l2,np.ones_like(delta),ref);bb=next(b for name,kind,b in mods if name=='signed_only')
 pp,_,cc,aa,_=purge(y,groups,pos,starts,ends,bb,gg,'signed',ids)
 assert all(c['lambda']==10. for c in cc)
 for a in aa:
  train=[list(ids).index(r) for r in json.loads(a['train_ids_json'])];test=[list(ids).index(r) for r in json.loads(a['test_ids_json'])]
  np.testing.assert_array_equal(pp[test],np.full(len(test),y[train].mean()))
 # Closed-window boundary: one shared bp is removed, immediately adjacent is not.
 p=100000
 def overlap(q):return max(0,min(p+16383,q+16383)-max(p-16384,q-16384)+1)
 assert overlap(p+32767)==1 and overlap(p+32768)==0
 realout=m.OUT
 with tempfile.TemporaryDirectory(prefix='evo2-context-check-') as temp:
  m.OUT=Path(temp);prepared=m.prepare();assert prepared['inner_folds']==992 and prepared['inner_lambda_rows']==5952 and prepared['bootstrap_draws']==0
  # Failed synthetic gate must stop before any actual source gate/data read.
  m.write(m.OUT/'synthetic_CPU_validation.json',{'status':'NOT_PASS','producer_sha256':m.sha(CODE),'checker_sha256':m.sha(Path(__file__))})
  negative.append(expect_error(lambda:m.run(prepared),'synthetic status gate'))
 m.OUT=realout
 assert not denied and all(m.sha(p)==h for p,h in snapshots.items())
 realout.mkdir(parents=True,exist_ok=True)
 result={'status':'PASS_SYNTHETIC_PREPARATION_ONLY','producer_sha256':m.sha(CODE),'checker_sha256':m.sha(Path(__file__)),'actual_outcome_activation_OOF_reads':0,'actual_data_fits':0,'new_GPU_API':0,'synthetic_coordinate_manifest_variants':193,'synthetic_HS_missingness':'First2coordinate rows only, not actual biological missing outcomes.','synthetic_counts':counts,'all192_lambda_choices_recomputed':True,'outer_geometry_memberships_recomputed':True,'heldout_and_removed_labels_no_own_prediction_or_tuning_effect':True,'removed_features_no_own_prediction_or_tuning_effect':True,'vector_only_inactive_and_allconstant_train_mean':True,'lambda_tie_chooses10':True,'duplicates_exact':True,'ordinary_pair_and_four_metric_differences_checked':True,'summary_counts':{k:len(v) for k,v in tables.items()},'negative_fixtures_rejected':negative,'closed1bp_boundary_checked':True,'forbidden_read_attempts':denied,'original_sources_unchanged':True,'immutable_snapshot_sha256':snapshots,'not_actual_fit_authorization':True}
 m.write(realout/'synthetic_CPU_validation.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
