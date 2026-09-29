#!/usr/bin/env python3
"""Read-only CPU audit of committed SeqSplice signed-layer artifacts.

Writes only its own audit directory. A partially completed inference run receives
PARTIAL_PASS, never PASS. This audits numerical/structural consistency and does
not test phenotype associations or establish biological validity.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_INPUT=ROOT/'results/brca1_grch38/seqsplice_signed_layers_20260916'
BENCHMARK=ROOT/'results/brca1_grch38/external_functional_validation/seqsplice_complete_benchmark_20260916'
PILOT=ROOT/'results/brca1_grch38/numeric_validation_20260916/HCM_direct_HCL_FP64_pilot'
LAYERS=['blocks.0','blocks.7','blocks.14','blocks.21','blocks.28','norm']
REGIONS=['variant_pm20','assayed_exon','donor_pm20','acceptor_pm20','whole_32k']
METRICS=['relative_l2','rms_difference','cosine_distance','max_abs_difference','changed_dimension_fraction','reference_l2']
VIEWS=['forward','rc']
SENTINEL='cv_868688_43063900_C_A'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def write_csv(path,rows):
    if not rows:return
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def prefix_checks(checks):
    assert [x['layer'] for x in checks]==LAYERS
    for x in checks:
        assert x['exactly_equal'] is True and x['changed_elements']==0 and x['changed_positions']==0
        assert x['max_abs_difference']==0 and x['first_difference'] is None


def coverage(run):
    x=run['coverage']
    assert x['HCM']=={str(i):1 for i in [1,5,8,12,15,19,22,26,29]}
    assert x['HCL']=={str(i):1 for i in [2,6,9,13,16,20,23,27,30]}
    assert x['fft_calls']=={'fft':288,'rfft':288,'irfft':288}
    assert run['seconds']>0 and np.isfinite(run['score'])


def audit_one(folder,row,view,fingerprint,region_map,old_hashes):
    marker=folder/'variants'/row['run_id']/(view+'.complete.json')
    m=json.loads(marker.read_text());index=16384 if view=='forward' else 16383
    assert m['status']=='passed' and m['execution_fingerprint']==fingerprint
    assert m['run_id']==row['run_id'] and m['orientation']==view and m['input_manifest_row']==row
    assert m['layers']==LAYERS and m['regions']==REGIONS and m['variant_index_0based']==index
    assert m['region_coordinates']==region_map[row['run_id']]
    assert m['REF_sequence_sha256']==row[view+'_ref_sha256'] and m['ALT_sequence_sha256']==row[view+'_alt_sha256']
    assert m['delta_score']==m['alt_score']-m['ref_score']
    q=m['QC'];assert q['reference_control_passed'] is True and q['all_selected_layer_prefixes_exactly_equal'] is True and q['all_arrays_finite'] is True
    prefix_checks(q['layer_prefix_checks'])
    control_path=Path(q['reference_control_file']);assert sha(control_path)==q['reference_control_sha256']
    control=json.loads(control_path.read_text())
    assert control['passed'] is True and control['execution_fingerprint']==fingerprint
    assert control['position']==int(row['position_grch38_1based']) and control['orientation']==view
    assert control['reference_sequence_sha256']==row[view+'_ref_sha256']
    assert control['pooled_means_exact'] is True and control['score_exact'] is True and control['max_norm_relative_l2']==0
    prefix_checks(control['layer_prefix_checks'])
    assert control['reference_run']==m['reference_run']
    for run in [m['reference_run'],m['alternate_run'],control['repeat_run']]:coverage(run)
    assert m['reference_run']['score']==m['ref_score'] and m['alternate_run']['score']==m['alt_score']
    assert control['repeat_run']['score']==m['ref_score']
    path=marker.parent/m['npz_filename'];assert sha(path)==m['npz_sha256']
    expected={'REF_mean_float64':((6,5,4096),'float64'),'ALT_mean_float64':((6,5,4096),'float64'),
              'delta_float32':((6,5,4096),'float32'),'prefix_mismatch_counts':((6,index),'int32')}
    expected.update({k:((32768,),'float32') for k in METRICS})
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)==set(expected) and set(m['array_schema'])==set(expected)
        arrays={k:z[k] for k in expected}
    for key,(shape,dtype) in expected.items():
        array=arrays[key]
        assert array.shape==shape and str(array.dtype)==dtype and np.isfinite(array).all(),key
        assert m['array_schema'][key]=={'shape':list(shape),'dtype':dtype},key
    assert np.array_equal(arrays['delta_float32'],(arrays['ALT_mean_float64']-arrays['REF_mean_float64']).astype(np.float32))
    assert not arrays['prefix_mismatch_counts'].any()
    for key in ['relative_l2','rms_difference','max_abs_difference','changed_dimension_fraction']:
        assert not arrays[key][:index].any(),key
    for key in METRICS:assert (arrays[key]>=0).all(),key
    assert (arrays['cosine_distance']<=2).all() and (arrays['changed_dimension_fraction']<=1).all()
    assert (arrays['reference_l2']>0).all()
    l2=arrays['relative_l2'];assert float(l2.mean())==m['mean_norm_relative_l2']
    pooled=[]
    for i,layer in enumerate(LAYERS):
        for j,region in enumerate(REGIONS):
            d=arrays['delta_float32'][i,j].astype(np.float64)
            start,end=(region_map[row['run_id']][region][view+k] for k in ['_start_0based','_end_0based_exclusive'])
            if end<=index:
                assert not d.any(),'Entirely unchanged-prefix region has a nonzero pooled difference.'
                assert np.array_equal(arrays['REF_mean_float64'][i,j],arrays['ALT_mean_float64'][i,j])
            pooled.append({'run_id':row['run_id'],'orientation':view,'layer':layer,'region':region,
                'length_bp':end-start,'region_entirely_before_variant':end<=index,
                'signed_delta_l2':float(np.linalg.norm(d)),'signed_delta_nonzero_dimensions':int(np.count_nonzero(d)),
                'REF_mean_l2':float(np.linalg.norm(arrays['REF_mean_float64'][i,j]))})
    record={'run_id':row['run_id'],'orientation':view,'position_grch38_1based':int(row['position_grch38_1based']),
        'ref':row['ref'],'alt':row['alt'],'source_legacy_exon':row['source_legacy_exon'],
        'variant_index_0based':index,'all_prefix_elements_equal':True,
        'whole_32k_mean_relative_l2':float(l2.mean()),'pre_variant_mean_relative_l2':float(l2[:index].mean()),
        'variant_relative_l2':float(l2[index]),'post_variant_mean_relative_l2':float(l2[index+1:].mean()),
        'norm_peak_input_index_0based':int(np.argmax(l2)),'norm_peak_input_offset_from_variant':int(np.argmax(l2))-index,
        'delta_score':m['delta_score'],'npz_bytes':path.stat().st_size,'npz_sha256':m['npz_sha256'],
        'marker_sha256':sha(marker),'reference_control_sha256':sha(control_path)}
    native=None
    if view=='forward' and row['in_previous_Evo2_183']=='True':
        prior=Path(row['prior_metadata_path']);assert sha(prior)==old_hashes[str(prior)]
        p=m['prior_183_comparison'];assert p['comparison_only_not_reused'] is True and p['old_metadata_sha256']==sha(prior)
        assert p['old_delta_score']==json.loads(prior.read_text())['delta_score'] and p['new_delta_score']==m['delta_score']
        assert p['old_metrics_sha256']==sha(prior.with_suffix('.npz'))
        with np.load(prior.with_suffix('.npz'),allow_pickle=False) as z:old=z['relative_l2']
        assert old.shape==l2.shape and np.isfinite(old).all()
        assert p['old_mean_relative_l2']==float(old.mean()) and p['new_mean_relative_l2']==float(l2.mean())
        assert p['old_pre_variant_mean_relative_l2']==float(old[:index].mean()) and p['new_pre_variant_mean_relative_l2']==0
        native={'run_id':row['run_id'],'position_grch38_1based':int(row['position_grch38_1based']),
            'source_legacy_exon':row['source_legacy_exon'],'native_mean_relative_l2':float(old.mean()),
            'new_mean_relative_l2':float(l2.mean()),'native_pre_variant_mean_relative_l2':float(old[:index].mean()),
            'new_pre_variant_mean_relative_l2':0.,'native_post_variant_mean_relative_l2':float(old[index+1:].mean()),
            'new_post_variant_mean_relative_l2':float(l2[index+1:].mean()),
            'new_minus_native_whole_mean':float(l2.mean())-float(old.mean()),
            'new_minus_native_post_mean':float(l2[index+1:].mean())-float(old[index+1:].mean()),
            'native_delta_score':p['old_delta_score'],'new_delta_score':m['delta_score']}
    sentinel=None
    if row['run_id']==SENTINEL:
        pilot_file=PILOT/('forward_ref_alt_metrics.npz' if view=='forward' else 'rc_ref_alt_metrics_input_order.npz')
        pilot_summary=json.loads((PILOT/'summary.json').read_text())
        with np.load(pilot_file,allow_pickle=False) as z:
            sentinel={'orientation':view,'pilot_metrics_sha256':sha(pilot_file),
                'per_metric_exact_equality':{key:bool(np.array_equal(arrays[key],z[key])) for key in METRICS},
                'per_metric_max_absolute_difference':{key:float(np.max(np.abs(arrays[key].astype(np.float64)-z[key].astype(np.float64)))) for key in METRICS},
                'pilot_summary_sha256':sha(PILOT/'summary.json'),
                'REF_score_exact_equality':m['ref_score']==pilot_summary['current_scores'][view+'_ref'],
                'ALT_score_exact_equality':m['alt_score']==pilot_summary['current_scores'][view+'_alt']}
        sentinel['reproduced_exactly']=(all(sentinel['per_metric_exact_equality'].values())
            and sentinel['REF_score_exact_equality'] and sentinel['ALT_score_exact_equality'])
    return record,pooled,native,sentinel


def describe(rows,key):
    a=np.array([r[key] for r in rows],dtype=np.float64)
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'min':float(a.min()),'max':float(a.max())} if len(a) else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--benchmark',type=Path,default=BENCHMARK)
    parser.add_argument('--cohort',choices=['panel','all'],default='panel')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();out=args.output or args.input/('audit_'+args.cohort);out.mkdir(parents=True,exist_ok=True)
    config=json.loads((args.input/'execution_config.json').read_text())
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    assert config['layers']==LAYERS and config['regions']==REGIONS and config['orientations']==VIEWS
    assert config['REF_repeat_every_coordinate_group']==1 and config['patch_class']=='HCMDirectPatch'
    assert sha(ROOT/'scripts/inference/run_brca1_seqsplice_layers.py')==config['runner_sha256']
    for key in ['benchmark_sources_sha256','patch_dependencies_sha256','implementation_sources_sha256']:
        for path,expected in config[key].items():assert sha(path)==expected,f'Source changed: {path}'
    rows=read_csv(args.benchmark/'benchmark_manifest_193.csv');assert len(rows)==193
    panel_ids={r['run_id'] for r in read_csv(args.benchmark/'numerical_panel.csv')};assert len(panel_ids)==25
    if args.cohort=='panel':rows=[r for r in rows if r['run_id'] in panel_ids]
    regions={r['run_id']:{} for r in read_csv(args.benchmark/'benchmark_manifest_193.csv')}
    for r in read_csv(args.benchmark/'regions_forward_and_rc.csv'):
        name,region=r.pop('run_id'),r.pop('region');regions[name][region]={k:int(v) for k,v in r.items()}
    old_hashes=json.loads((args.benchmark/'source_metadata_sha256.json').read_text())
    snapshot=[(r,v) for r in rows for v in VIEWS if (args.input/'variants'/r['run_id']/(v+'.complete.json')).exists()]
    completed=[];pool=[];native=[];sentinel=[];errors=[]
    for row,view in snapshot:
        try:
            a,b,c,d=audit_one(args.input,row,view,fingerprint,regions,old_hashes)
            completed.append(a);pool.extend(b)
            if c:native.append(c)
            if d:
                sentinel.append(d)
                if not d['reproduced_exactly']:
                    errors.append({'run_id':row['run_id'],'orientation':view,'error':'Sentinel differs from the audited pilot; investigate before cohort acceptance.'})
        except Exception as exc:errors.append({'run_id':row['run_id'],'orientation':view,'error':type(exc).__name__+': '+str(exc),
                                             'traceback':traceback.format_exc()})
    quarantine=[]
    for path in (args.input/'quarantine').glob('**/*.failure.json'):
        x=json.loads(path.read_text())
        if x.get('execution_fingerprint')==fingerprint:quarantine.append(str(path))
    missing=[r['run_id']+'/'+v for r in rows for v in VIEWS if (r,v) not in snapshot]
    complete=len(completed)==2*len(rows) and not errors and not quarantine
    status='FAIL' if errors or quarantine else 'PASS' if complete else 'PARTIAL_PASS'
    summary={'status':status,'cohort':args.cohort,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'execution_fingerprint':fingerprint,'audit_script_sha256':sha(Path(__file__)),'expected_variant_views':2*len(rows),
        'committed_at_snapshot':len(snapshot),'verified_variant_views':len(completed),'cohort_complete':complete,
        'missing_at_snapshot':missing,'errors':errors,'quarantined_failures':quarantine,
        'selected_layer_full_prefix_checks_passed':6*len(completed),
        'verified_distinct_REF_repeat_controls':len({r['reference_control_sha256'] for r in completed}),
        'REF_repeat_selected_layer_full_prefix_checks_passed':6*len({r['reference_control_sha256'] for r in completed}),
        'signed_delta_equals_FP64_ALT_minus_REF_then_FP32':not errors and bool(completed),
        'verified_npz_MiB':sum(r['npz_bytes'] for r in completed)/2**20,
        'native_forward_comparison_n':len(native),
        'native_vs_new_descriptive':{key:describe(native,key) for key in ['native_mean_relative_l2','new_mean_relative_l2',
            'native_pre_variant_mean_relative_l2','new_pre_variant_mean_relative_l2','new_minus_native_post_mean']},
        'sentinel_pilot_comparison':sentinel,
        'interpretation':'Numerical and artifact consistency only; prefix checks verify stored mismatch counts and frozen audited producer logic, not an independent raw-activation recomputation. No phenotype association or biological validity tested.',
        'torch_imported':'torch' in sys.modules}
    assert not summary['torch_imported']
    write_json(out/'validation.json',summary)
    write_csv(out/'per_variant_view.csv',completed);write_csv(out/'signed_pool_descriptive.csv',pool)
    write_csv(out/'native_vs_new_forward.csv',native)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if errors or quarantine or (args.require_complete and not complete):sys.exit(1)


if __name__=='__main__':main()
