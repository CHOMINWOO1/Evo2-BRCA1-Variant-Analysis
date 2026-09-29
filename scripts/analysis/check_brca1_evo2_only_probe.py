#!/usr/bin/env python3
"""Synthetic/metadata-only tests of the Evo2-only ablation. No actual fits."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import importlib.util
import tempfile
import json
import inspect
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).with_name('evaluate_brca1_evo2_only_probe.py')
spec=importlib.util.spec_from_file_location('evo2_only_checked',CODE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from brca1_signed_kernel import kernels,nested_loeo


def explicit_vector(x,train,test,unit=False):
    blocks=[]
    for v in range(2):
        a=x[:,v].copy()
        if unit:
            norm=np.sqrt((a*a).sum(1));a=np.divide(a,norm[:,None],out=np.zeros_like(a),where=norm[:,None]!=0)
        if not np.ptp(a[train],axis=0).any():continue
        mean=a[train].mean(0);ta,tb=a[train]-mean,a[test]-mean
        rms=np.sqrt((ta*ta).sum(1).mean());blocks.append((ta/rms,tb/rms))
    if not blocks:return np.zeros((len(train),1)),np.zeros((len(test),1))
    return np.concatenate([a for a,b in blocks],axis=1)/np.sqrt(len(blocks)),np.concatenate([b for a,b in blocks],axis=1)/np.sqrt(len(blocks))


def main():
    rng=np.random.default_rng(8731193);n=36
    sequence=rng.normal(size=(n,2));l2=rng.uniform(size=(n,2))
    delta=rng.normal(size=(n,2,7));ref=1e8+rng.normal(0,.01,size=(n,2,7));delta[0]=0
    grams,models,scalar=m.model_inputs(sequence,l2,delta,ref)
    assert np.array_equal(scalar,np.column_stack([sequence,l2,np.sqrt((delta*delta).sum(-1))]))
    assert [x[0] for x in models]==m.MODELS
    assert list(inspect.signature(m.model_inputs).parameters)==['sequence','l2','delta','ref']
    assert len(m.OUTCOME_COLUMNS)==8 and not any('spliceai' in c.lower() for c in m.OUTCOME_COLUMNS+m.SCALAR6)
    train,test=np.arange(25),np.arange(25,n)
    errors={}
    for label,kind,baseline in models:
        if label not in ['signed_only','REF_only']:continue
        assert baseline.shape==(n,1) and np.count_nonzero(baseline)==0
        tt,xt=kernels(baseline,grams,kind,train,test)
        a,b=explicit_vector(delta if kind=='signed' else ref,train,test)
        error=max(np.max(abs(tt-a@a.T)),np.max(abs(xt-b@a.T)));errors[label]=float(error)
        assert error<1e-12
        y=rng.normal(size=len(train))
        dual=xt@np.linalg.solve(tt+len(train)*.1*np.eye(len(train)),y-y.mean())+y.mean()
        primal=b@np.linalg.solve(a.T@a+len(train)*.1*np.eye(a.shape[1]),a.T@(y-y.mean()))+y.mean()
        assert np.max(abs(primal-dual))<1e-11
    # One inactive orientation must renormalize to the surviving view, not /2.
    d2=delta.copy();d2[:,1]=3.
    gg,mm,_=m.model_inputs(sequence,l2,d2,ref)
    tt,xt=kernels(mm[2][2],gg,'signed',train,test);a,b=explicit_vector(d2,train,test)
    assert np.allclose(tt,a@a.T,rtol=0,atol=1e-12) and np.allclose(xt,b@a.T,rtol=0,atol=1e-12)
    # Both inactive vector views and the inactive scalar block give a zero kernel.
    gg,mm,_=m.model_inputs(sequence,l2,np.ones_like(delta),ref)
    tt,xt=kernels(mm[2][2],gg,'signed',train,test);assert not tt.any() and not xt.any()
    groups=np.repeat(np.arange(1,13),3);positions=np.repeat(np.arange(12)*40000+100000,3)
    yy=rng.normal(5,15,n);records=[];counts=[];predictions={}
    for label,kind,baseline in models:
        pred,inner,chosen,audit=nested_loeo(yy,groups,positions,baseline,grams,kind)
        assert len(inner)==792 and len(chosen)==len(audit)==12
        changed=yy.copy();changed[groups==1]+=1000
        altered=nested_loeo(changed,groups,positions,baseline,grams,kind)[0]
        np.testing.assert_array_equal(pred[groups==1],altered[groups==1])
        predictions[label]=pred;counts.append((len(pred),len(chosen),len(inner)))
        records.append(pd.DataFrame({'run_id':[f'synthetic_{i}' for i in range(n)],'prediction_pp':pred,
            'full_length_loss_pp':yy,'Exon_legacy':groups,'cell_line':'synthetic','model':label}))
    np.testing.assert_array_equal(predictions['evo2_scalar6'],predictions['evo2_scalar6_exact_duplicate'])
    # Summaries tested with a temporary synthetic cell count, no biological rows.
    cells=m.CELLS;m.CELLS={'synthetic':n}
    draws=np.random.default_rng(20260916).integers(0,12,(5000,12))
    scores,exon,cmp=m.summarize(pd.concat(records,ignore_index=True),np.arange(1,13),draws)
    m.CELLS=cells
    assert len(scores)==8 and len(exon)==96 and len(cmp)==14
    for row in cmp.itertuples(index=False):
        diff=np.abs(predictions[row.combined]-yy)-np.abs(predictions[row.baseline]-yy)
        by_exon=np.array([diff[groups==e].mean() for e in range(1,13)])
        assert abs(row.delta_MAE_pp-by_exon.mean())<1e-12
        boot=np.array([np.mean([by_exon[i] for i in draw]) for draw in draws])
        assert np.allclose([row.ci_low,row.ci_high],np.percentile(boot,[2.5,97.5]),atol=1e-12,rtol=0)
    json.dumps(cmp.iloc[0].to_dict(),allow_nan=False)
    # Guard prepare against phenotype/activation/OOF reads, including content hashes.
    forbidden=[m.BENCH/'measured_outcomes',m.ROOT/'results/brca1_grch38/seqsplice_signed_layers_20260916/variants',
        m.ROOT/'results/brca1_grch38/seqsplice_signed_layers_20260916/controls',m.ORIGINAL/'heldout_predictions.csv',m.ORIGINAL/'heldout_predictions.partial.csv']
    guard={'active':False,'denied':[]}
    def audit(event,args):
        if not guard['active'] or event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if any(p==q or q in p.parents for q in forbidden):
            guard['denied'].append(str(p));raise AssertionError('Forbidden prepare read '+str(p))
    sys.addaudithook(audit)
    realout=m.OUT
    with tempfile.TemporaryDirectory(prefix='evo2-only-probe-synthetic-') as tmp:
        m.OUT=Path(tmp)
        guard['active']=True
        try:prepared=m.prepare()
        finally:guard['active']=False;m.OUT=realout
    assert not guard['denied'] and prepared['prediction_rows']==3072 and prepared['outer_fits']==192 and prepared['inner_lambda_rows']==12672
    realout.mkdir(parents=True,exist_ok=True)
    result={'status':'PASS_SYNTHETIC_AND_METADATA_ONLY','actual_outcome_activation_OOF_reads':False,'actual_data_fits':0,'GPU_API_calls':0,
        'models':8,'synthetic_exons':12,'synthetic_variant_count':n,'synthetic_outer_fits':96,'synthetic_inner_lambda_rows':6336,
        'vector_only_explicit_kernel_max_errors':errors,'vector_only_dual_primal_agree':True,
        'inactive_orientation_renormalizes':True,'all_constant_vector_only_kernel_zero':True,
        'heldout_labels_cannot_change_own_predictions_all8models':True,'exact_duplicate_predictions_equal':True,
        'fixed_contrast_summaries_independently_checked':14,'SpliceAI_not_in_feature_arguments_or_column_whitelists':True,
        'prepare_forbidden_file_attempts':guard['denied'],'script_sha256':m.sha(CODE),'test_script_sha256':m.sha(Path(__file__)),
        'pinned_dependencies':m.PINNED,'no_actual_fit_authorization_from_this_test':True}
    m.write(realout/'synthetic_CPU_validation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
