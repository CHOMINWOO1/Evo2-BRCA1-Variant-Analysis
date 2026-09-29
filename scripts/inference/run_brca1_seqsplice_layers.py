#!/usr/bin/env python3
"""Resumable signed pooled Evo2 layers for the frozen BRCA1 SeqSplice cohort.

--dry-run uses only the standard library and performs no model/GPU imports.
Execution is gated by a matching successful numerical pilot; --all additionally
requires the complete 25-variant, two-orientation panel under this fingerprint.
"""
import argparse
import ast
import csv
import datetime
import fcntl
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT/'results/brca1_grch38/external_functional_validation/seqsplice_complete_benchmark_20260916'
NUMERIC = ROOT/'results/brca1_grch38/numeric_validation_20260916'
DEFAULT_OUTPUT = ROOT/'results/brca1_grch38/seqsplice_signed_layers_20260916'
LAYERS = ['blocks.0','blocks.7','blocks.14','blocks.21','blocks.28','norm']
REGIONS = ['variant_pm20','assayed_exon','donor_pm20','acceptor_pm20','whole_32k']
METRICS = ['relative_l2','rms_difference','cosine_distance','max_abs_difference','changed_dimension_fraction','reference_l2']
VIEWS = ['forward','rc']
LENGTH, HIDDEN, CHUNK = 32768, 4096, 256
AUDITED_PATCH_SHA256 = 'ccfeedd19b368fbb09776e167243bcd4ad1e2e0bfb971f4919cf7d5aab641ee3'
AUDITED_PATCH_CLASS = 'HCMDirectPatch'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)


def atomic_npz(path,values):
    import numpy as np
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,**values);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def csv_rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def rc(sequence):
    return sequence.translate(str.maketrans('ACGTN','TGCAN'))[::-1]


def module_dependencies(path,seen=None):
    """Pin local Python imports recursively without importing the patch."""
    seen={} if seen is None else seen
    path=Path(path).resolve()
    if str(path) in seen:return seen
    seen[str(path)]=sha(path)
    for node in ast.walk(ast.parse(path.read_text())):
        names=[node.module] if isinstance(node,ast.ImportFrom) and node.module else []
        if isinstance(node,ast.Import):names += [alias.name for alias in node.names]
        for name in names:
            candidate=path.parent/(name.split('.')[0]+'.py')
            if candidate.exists():module_dependencies(candidate,seen)
    return seen


def region_signature(row,regions):
    return [regions[row['run_id']][region] for region in REGIONS]


def prepare(args):
    assert args.patch_class==AUDITED_PATCH_CLASS,'Only the independently audited HCMDirectPatch backend is supported.'
    assert args.ref_repeat_every==1,'Every reference coordinate/orientation must have its own identical REF repeat.'
    planning=json.loads((args.benchmark/'protocol.json').read_text())
    assert planning['planned_layers']==LAYERS and planning['planned_regions']==REGIONS
    names=['protocol.json','benchmark_manifest_193.csv','numerical_panel.csv','regions_forward_and_rc.csv','layer_manifest.csv','source_metadata_sha256.json']
    sources={str(args.benchmark/name):sha(args.benchmark/name) for name in names}
    for name,expected in planning['input_sha256'].items():
        path=ROOT/name
        assert sha(path)==expected,f'Frozen planning dependency changed: {path}'
        sources[str(path)]=expected
    rows=csv_rows(args.benchmark/'benchmark_manifest_193.csv')
    panel=csv_rows(args.benchmark/'numerical_panel.csv')
    assert len(rows)==193 and len(panel)==25
    by_id={r['run_id']:r for r in rows};assert len(by_id)==193
    old_sources=json.loads((args.benchmark/'source_metadata_sha256.json').read_text())
    assert len(old_sources)==183
    for path,expected in old_sources.items():
        assert sha(path)==expected,f'Frozen prior metadata changed: {path}'
    panel_ids={r['run_id'] for r in panel};assert len(panel_ids)==25 and panel_ids<=by_id.keys()
    for row in panel:
        assert all(row[k]==v for k,v in by_id[row['run_id']].items())
    regions={rid:{} for rid in by_id}
    for reg in csv_rows(args.benchmark/'regions_forward_and_rc.csv'):
        rid,name=reg.pop('run_id'),reg.pop('region')
        assert rid in by_id and name in REGIONS and name not in regions[rid]
        reg={k:int(v) for k,v in reg.items()}
        w=int(by_id[rid]['window_start_grch38_1based'])
        assert reg['forward_start_0based']==reg['start_grch38_1based']-w
        assert reg['forward_end_0based_exclusive']==reg['end_grch38_1based']-w+1
        assert reg['rc_start_0based']==LENGTH-reg['forward_end_0based_exclusive']
        assert reg['rc_end_0based_exclusive']==LENGTH-reg['forward_start_0based']
        for view in VIEWS:
            start,end=reg[view+'_start_0based'],reg[view+'_end_0based_exclusive']
            assert 0<=start<end<=LENGTH and end-start==reg['length_bp']
        regions[rid][name]=reg
    assert all(set(v)==set(REGIONS) for v in regions.values())
    reference=Path(planning['reference_fasta'])
    assert sha(reference)==planning['expected_reference_fasta_sha256']
    with gzip.open(reference,'rt') as f:
        assert next(f).startswith('>chr17')
        genome=''.join(x.strip().upper() for x in f)
    groups={}
    for r in rows:
        pos=int(r['position_grch38_1based']);start=int(r['window_start_grch38_1based'])-1
        assert int(r['length_bp'])==LENGTH and start==pos-1-16384
        assert int(r['window_end_grch38_1based'])==start+LENGTH
        assert int(r['forward_variant_index_0based'])==16384 and int(r['rc_variant_index_0based'])==16383
        ref=genome[start:start+LENGTH];assert len(ref)==LENGTH and ref[16384]==r['ref']
        alt=ref[:16384]+r['alt']+ref[16385:]
        assert r['ref']!=r['alt'] and r['ref'] in 'ACGT' and r['alt'] in 'ACGT'
        for view,sequence in [('forward',ref),('rc',rc(ref))]:
            assert hashlib.sha256(sequence.encode()).hexdigest()==r[view+'_ref_sha256']
            other=alt if view=='forward' else rc(alt)
            assert hashlib.sha256(other.encode()).hexdigest()==r[view+'_alt_sha256']
        groups.setdefault(pos,[]).append(r)
    assert len(groups)==136
    for group in groups.values():
        first=group[0]
        for r in group:
            for key in ['ref','window_start_grch38_1based','window_end_grch38_1based','forward_ref_sha256','rc_ref_sha256','source_legacy_exon','MANE_exon_number']:
                assert r[key]==first[key],f'Unsafe reference reuse at {r["run_id"]}: {key}'
            assert region_signature(r,regions)==region_signature(first,regions),'Different per-ALT regions; do not reuse reference pools.'
    numeric_protocol=json.loads((NUMERIC/'protocol.json').read_text())
    checkpoint=Path(planning['checkpoint'])
    assert str(checkpoint)==numeric_protocol['checkpoint']
    # Full content hashing is deliberate; dry-run remains entirely CPU/read-only.
    assert sha(checkpoint)==numeric_protocol['checkpoint_sha256']
    pilot=json.loads(args.pilot_validation.read_text()) if args.pilot_validation.exists() else None
    patch_hash=sha(args.patch_module)
    assert patch_hash==AUDITED_PATCH_SHA256,'Patch source differs from the independently audited source.'
    pilot_ok=bool(pilot and pilot.get('status')=='PASSED' and pilot.get('script_sha256')==patch_hash
                  and pilot.get('REF_repeat_relative_l2_exactly_zero') is True
                  and pilot.get('forward_unchanged_prefix_exactly_zero') is True
                  and pilot.get('RC_unchanged_prefix_exactly_zero') is True
                  and pilot.get('all_observed_whole_prefixes_exactly_equal') is True
                  and pilot.get('all_HCM_grouped_and_HCL_FFT_paths_covered') is True
                  and pilot.get('whole_prefix_all4096channels_all34layers_compared') is True
                  and pilot.get('no_untracked_FP32_torch_FFT_calls') is True)
    installed=ROOT/'.venv/lib/python3.12/site-packages'
    implementation_sources={str(installed/'vortex/model'/name):sha(installed/'vortex/model'/name)
                            for name in ['engine.py','model.py','layers.py','attention.py']}
    implementation_sources[str(ROOT/'scripts/inference/run_brca1_grch38_screen.py')]=sha(ROOT/'scripts/inference/run_brca1_grch38_screen.py')
    config={'version':1,'benchmark_sources_sha256':sources,'checkpoint':str(checkpoint),
        'checkpoint_sha256':numeric_protocol['checkpoint_sha256'],'checkpoint_revision':planning['checkpoint_revision'],
        'reference_fasta':str(reference),'reference_fasta_sha256':planning['expected_reference_fasta_sha256'],
        'patch_module':str(args.patch_module),'patch_class':args.patch_class,'patch_dependencies_sha256':module_dependencies(args.patch_module),
        'pilot_validation':str(args.pilot_validation),'pilot_validation_sha256':sha(args.pilot_validation) if pilot else None,
        'implementation_sources_sha256':implementation_sources,'runner_sha256':sha(Path(__file__)),
        'model':'evo2_7b','length_bp':LENGTH,'hidden_size':HIDDEN,'layers':LAYERS,'regions':REGIONS,'orientations':VIEWS,
        'variant_indices':{'forward':16384,'rc':16383},'pool_chunk_positions':CHUNK,
        'pooling':'CPU FP64 sums of exact converted BF16 activations in fixed256bp chunks; ALTmean64−REFmean64 before delta32 storage.',
        'stored_mean_dtype':'float64','stored_delta_dtype':'float32','norm_per_position_metrics_dtype':'float32',
        'storage_precision_revision':'Execution stores REF/ALT means in float64, superseding the preparation-only float32 estimate without overwriting it.',
        'common_prefix_acceptance':'Every position and dimension in each of six selected layers must be exactly equal; failure quarantines the entire variant/view and stops.',
        'REF_repeat_every_coordinate_group':args.ref_repeat_every,
        'REF_repeat_acceptance':'Exact pooled means, exact selected-layer full prefix, exact norm per-base difference metrics, and exact sequence score.',
        'same_position_reference_reuse':'Only identical window, orientation, reference hash, assay exon, all region boundaries and execution fingerprint.',
        'full_cohort_gate':'All25 frozen panel variants in both orientations must have valid completion markers under this same execution fingerprint.',
        'runtime_package_versions':{p:importlib.metadata.version(p) for p in ['torch','evo2','vtx','numpy']}}
    fingerprint=hashlib.sha256(canonical(config)).hexdigest()
    selected=[by_id[r['run_id']] for r in panel] if args.panel else rows
    if args.limit:selected=selected[:args.limit]
    variant_elements=2*len(LAYERS)*len(REGIONS)*HIDDEN
    estimates=[]
    for label,cohort in [('panel25',[by_id[r] for r in panel_ids]),('complete193',rows),('selected',selected)]:
        n=len(cohort);positions=len({r['position_grch38_1based'] for r in cohort})
        mean_bytes=n*variant_elements*16;delta_bytes=n*variant_elements*4
        metric_bytes=n*2*len(METRICS)*LENGTH*4
        repeats=2*math.ceil(positions/args.ref_repeat_every)
        estimates.append({'population':label,'variants':n,'positions':positions,
            'REF_ALT_means_MiB':mean_bytes/2**20,'signed_delta_MiB':delta_bytes/2**20,'norm_metrics_MiB':metric_bytes/2**20,
            'total_MiB_uncompressed_before_QC_metadata':(mean_bytes+delta_bytes+metric_bytes)/2**20,
            'reference_plus_ALT_forwards':2*(positions+n),'additional_REF_repeat_forwards':repeats,
            'temporary_reference_BF16_cache_MiB':((len(LAYERS)-1)*16384+LENGTH)*HIDDEN*2/2**20})
    plan={'status':'dry_run_prepared_no_GPU','created_utc':now(),'execution_fingerprint':fingerprint,
          'pilot_gate_passed':pilot_ok,'mode':'panel' if args.panel else 'all','limit':args.limit,
          'selected_variants':len(selected),'storage_and_forward_estimates':estimates,
          'source_and_sequence_QC_passed':True,'dry_run_imported_torch':'torch' in sys.modules,
          'output':str(args.output),'execution_config':config}
    assert 'torch' not in sys.modules,'Planning unexpectedly imported torch.'
    return rows,panel_ids,selected,regions,genome,config,fingerprint,plan


def marker_path(output,row,view):
    return output/'variants'/row['run_id']/(view+'.complete.json')


def valid_marker(output,row,view,fingerprint):
    path=marker_path(output,row,view)
    if not path.exists():return False
    value=json.loads(path.read_text())
    assert value['execution_fingerprint']==fingerprint,f'Incompatible completed output: {path}'
    assert value['status']=='passed' and value['run_id']==row['run_id'] and value['orientation']==view
    assert value['REF_sequence_sha256']==row[view+'_ref_sha256'] and value['ALT_sequence_sha256']==row[view+'_alt_sha256']
    assert value['QC']['all_selected_layer_prefixes_exactly_equal'] and value['QC']['reference_control_passed']
    control_path=Path(value['QC']['reference_control_file'])
    assert control_path.is_file() and sha(control_path)==value['QC']['reference_control_sha256'],f'Missing/corrupt REF control: {control_path}'
    control=json.loads(control_path.read_text())
    assert control['execution_fingerprint']==fingerprint and control['passed'] is True
    assert control['position']==int(row['position_grch38_1based']) and control['orientation']==view
    assert control['reference_sequence_sha256']==row[view+'_ref_sha256']
    assert control['pooled_means_exact'] is True and control['score_exact'] is True
    assert all(q['exactly_equal'] for q in control['layer_prefix_checks'])
    artifact=path.parent/value['npz_filename']
    assert artifact.exists() and sha(artifact)==value['npz_sha256'],f'Missing/corrupt committed artifact: {artifact}'
    return True


def commit(output,row,view,arrays,metadata,fingerprint,passed):
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    folder=output/'variants'/row['run_id'] if passed else output/'quarantine'/row['run_id']/stamp
    artifact=folder/(view+'.npz');atomic_npz(artifact,arrays)
    record={**metadata,'run_id':row['run_id'],'orientation':view,'status':'passed' if passed else 'quarantined',
            'execution_fingerprint':fingerprint,'npz_filename':artifact.name,'npz_sha256':sha(artifact),'completed_utc':now()}
    target=folder/(view+'.complete.json' if passed else view+'.failure.json')
    atomic_json(target,record)
    return record


class Collector:
    """One forward streamed on CPU; only REF prefix/norm are cached temporarily."""
    def __init__(self,layers,regions,index,metrics_function,reference=None,length=LENGTH,hidden=HIDDEN):
        import numpy as np
        self.layers,self.regions,self.index=layers,regions,index
        self.metrics_function,self.reference=metrics_function,reference
        self.length,self.hidden=length,hidden
        self.means=np.empty((len(layers),len(regions),hidden),dtype=np.float64)
        self.prefix={};self.norm=None;self.seen=[];self.qc=[]
        self.counts=np.zeros((len(layers),index),dtype=np.int32)
        self.metrics={k:np.empty(length,dtype=np.float32) for k in METRICS} if reference else None

    def hook(self,name):
        def collect(module,inputs,output):
            import numpy as np
            import torch
            if isinstance(output,tuple):output=output[0]
            assert tuple(output.shape)==(1,self.length,self.hidden) and output.dtype==torch.bfloat16
            assert name not in self.seen;self.seen.append(name)
            layer=self.layers.index(name);sums=np.zeros((len(self.regions),self.hidden),dtype=np.float64)
            if self.reference is None:
                if name=='norm':
                    self.norm=torch.empty((1,self.length,self.hidden),dtype=output.dtype,device='cpu')
                    self.prefix[name]=self.norm[:,:self.index]
                else:self.prefix[name]=torch.empty((1,self.index,self.hidden),dtype=output.dtype,device='cpu')
            first=None;maximum=0.;elements=0
            for start in range(0,self.length,CHUNK):
                end=min(start+CHUNK,self.length)
                value=output[:,start:end].detach().cpu()
                assert bool(torch.isfinite(value).all()),f'Nonfinite activations in {name}'
                converted=value[0].double().numpy()
                for r,(begin,stop) in enumerate(self.regions):
                    left,right=max(begin,start),min(stop,end)
                    if left<right:sums[r]+=converted[left-start:right-start].sum(axis=0,dtype=np.float64)
                stop=min(end,self.index)
                if self.reference is None:
                    if name=='norm':self.norm[:,start:end].copy_(value)
                    elif start<stop:self.prefix[name][:,start:stop].copy_(value[:,:stop-start])
                elif start<stop:
                    before=self.reference.prefix[name][:,start:stop]
                    current=value[:,:stop-start];changed=before!=current
                    n=changed.sum(dim=-1)[0].numpy().astype(np.int32)
                    self.counts[layer,start:stop]=n;elements+=int(n.sum())
                    if n.any():
                        maximum=max(maximum,float((current.float()-before.float()).abs().max()))
                        if first is None:
                            p=int(np.flatnonzero(n)[0]);c=int(torch.nonzero(changed[0,p])[0,0])
                            first={'input_position_0based':start+p,'channel':c,'REF':float(before[0,p,c]),'ALT':float(current[0,p,c])}
                if self.reference is not None and name=='norm':
                    values=self.metrics_function(self.reference.norm[:,start:end],value)
                    for key,array in values.items():self.metrics[key][start:end]=array
            self.means[layer]=sums/np.array([b-a for a,b in self.regions],dtype=np.float64)[:,None]
            assert np.isfinite(self.means[layer]).all()
            if self.reference is not None:
                self.qc.append({'layer':name,'exactly_equal':elements==0,'changed_elements':elements,
                    'changed_positions':int(np.count_nonzero(self.counts[layer])),'max_abs_difference':maximum,'first_difference':first})
        return collect

    def finish(self):
        assert self.seen==self.layers,f'Layer capture order mismatch: {self.seen}'


def old_comparison(row,view,metrics,score_delta):
    import numpy as np
    if view!='forward' or row['in_previous_Evo2_183']!='True':return None
    old_json=Path(row['prior_metadata_path']);old=json.loads(old_json.read_text());path=old_json.with_suffix('.npz')
    with np.load(path,allow_pickle=False) as archive:
        before=archive['relative_l2'];after=metrics['relative_l2']
        correlation=float(np.corrcoef(before,after)[0,1]) if np.std(before)>0 and np.std(after)>0 else None
    return {'comparison_only_not_reused':True,'old_metadata':str(old_json),'old_metadata_sha256':sha(old_json),
        'old_metrics_sha256':sha(path),'old_mean_relative_l2':float(before.mean()),'new_mean_relative_l2':float(after.mean()),
        'old_pre_variant_mean_relative_l2':float(before[:16384].mean()),'new_pre_variant_mean_relative_l2':float(after[:16384].mean()),
        'position_relative_l2_Pearson_descriptive':correlation,'old_delta_score':old['delta_score'],'new_delta_score':score_delta}


def execute(args,rows,panel_ids,selected,regions,genome,config,fingerprint,plan):
    assert plan['pilot_gate_passed'],'Numerical pilot is absent, unsuccessful or does not match the selected patch source.'
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config_path=out/'execution_config.json'
    if config_path.exists():assert json.loads(config_path.read_text())==config,'Execution fingerprint changed; use a new output directory.'
    else:atomic_json(config_path,config)
    if args.all:
        absent=[f'{r["run_id"]}/{view}' for r in rows if r['run_id'] in panel_ids for view in VIEWS
                if not valid_marker(out,r,view,fingerprint)]
        assert not absent,f'Full cohort requires the complete passing panel first: {len(absent)} missing variant/views.'
    pending=[(r,view) for r in selected for view in VIEWS if not valid_marker(out,r,view,fingerprint)]
    invocation=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    atomic_json(out/'invocations'/(invocation+'.json'),{**plan,'status':'authorized_execution_invocation','pending_variant_views':len(pending)})
    progress={'execution_fingerprint':fingerprint,'mode':plan['mode'],'selected_variants':len(selected),
              'selected_variant_views':2*len(selected),'pending_variant_views_at_start':len(pending),'completed_this_invocation':0,
              'pid':os.getpid(),'started_utc':now(),'status':'prepared'}
    atomic_json(out/'progress.json',progress)
    if not pending:
        all_done=all(valid_marker(out,r,v,fingerprint) for r in rows for v in VIEWS)
        panel_done=all(valid_marker(out,r,v,fingerprint) for r in rows if r['run_id'] in panel_ids for v in VIEWS)
        progress.update(status='complete_all193' if all_done else 'panel25_complete' if panel_done else 'selected_scope_complete',
                        complete_all193=all_done,complete_panel25=panel_done,finished_utc=now())
        atomic_json(out/'progress.json',progress);return
    # Imports below this line are never performed by --dry-run.
    os.environ['HF_HUB_OFFLINE']='1'
    import numpy as np
    import torch
    from evo2 import Evo2
    from evo2.scoring import logits_to_logprobs
    import vortex.model.engine as engine
    from run_brca1_grch38_screen import position_metrics
    assert torch.cuda.is_available(),'GPU unavailable.'
    sys.path.insert(0,str(args.patch_module.parent))
    spec=importlib.util.spec_from_file_location('seqsplice_precision_backend',args.patch_module)
    backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
    patch_class=getattr(backend,args.patch_class)
    torch.set_num_threads(4)
    started=time.monotonic();forward_seconds=[]
    progress.update(status='running',stage='model_load',GPU=torch.cuda.get_device_name(0));atomic_json(out/'progress.json',progress)
    model=Evo2('evo2_7b',local_path=config['checkpoint']);model.model.eval()
    assert not model.model.config.use_fp8_input_projections and not model.model.config.use_flashfft
    assert all(not m.training for m in model.model.modules())
    for i in model.model.config.hcm_layer_idxs+model.model.config.hcl_layer_idxs+model.model.config.hcs_layer_idxs:
        e=model.model.blocks[i].filter.engine
        assert not e.use_hcm_kernel and not e.use_hcl_kernel and not e.use_hcs_kernel
    assert all(i in model.model.config.hcs_layer_idxs for i in [0,7,14,21,28])
    parameter_versions={name:p._version for name,p in model.model.named_parameters()}
    groups={}
    for row,view in pending:groups.setdefault((int(row['position_grch38_1based']),view),[]).append(row)
    forwards_expected=len(pending)+2*len(groups)

    def forward(sequence,row,view,label,patch,reference=None):
        index=config['variant_indices'][view]
        bounds=[(regions[row['run_id']][r][view+'_start_0based'],regions[row['run_id']][r][view+'_end_0based_exclusive']) for r in REGIONS]
        collector=Collector(LAYERS,bounds,index,position_metrics,reference)
        hooks=[model.model.get_submodule(name).register_forward_hook(collector.hook(name)) for name in LAYERS]
        patch.begin(label);t=time.monotonic()
        ids=torch.tensor(model.tokenizer.tokenize(sequence),dtype=torch.long,device='cuda:0')[None]
        try:
            with torch.inference_mode():
                outputs,_=model(ids)
                score=float(logits_to_logprobs(outputs[0],ids).float().mean().item())
            torch.cuda.synchronize();collector.finish();patch.check(label)
        finally:
            for hook in hooks:hook.remove()
        assert math.isfinite(score)
        elapsed=time.monotonic()-t;forward_seconds.append(elapsed)
        coverage=patch.records.pop(label)
        return collector,score,{'label':label,'seconds':elapsed,'score':score,'coverage':coverage}

    try:
        with patch_class(engine,model.model,torch) as patch:
            for group_number,((position,view),group) in enumerate(groups.items()):
                row=group[0];start=int(row['window_start_grch38_1based'])-1
                genomic_ref=genome[start:start+LENGTH];sequence=genomic_ref if view=='forward' else rc(genomic_ref)
                prefix=f'{invocation}_{position}_{view}'
                progress.update(stage='reference_forward',position=position,orientation=view);atomic_json(out/'progress.json',progress)
                ref,ref_score,ref_run=forward(sequence,row,view,prefix+'_REF',patch)
                # A separate exact repeat is mandatory for every coordinate/orientation.
                if args.ref_repeat_every==1:
                    repeat,repeat_score,repeat_run=forward(sequence,row,view,prefix+'_REF_REPEAT',patch,ref)
                    control_ok=(all(q['exactly_equal'] for q in repeat.qc) and np.array_equal(ref.means,repeat.means)
                        and ref_score==repeat_score and all(np.count_nonzero(repeat.metrics[k])==0 for k in ['relative_l2','rms_difference','max_abs_difference','changed_dimension_fraction']))
                    control_path=out/'controls'/(prefix+'.json')
                    control={'passed':control_ok,'execution_fingerprint':fingerprint,'position':position,'orientation':view,
                        'reference_sequence_sha256':row[view+'_ref_sha256'],'layer_prefix_checks':repeat.qc,
                        'pooled_means_exact':bool(np.array_equal(ref.means,repeat.means)),'score_exact':ref_score==repeat_score,
                        'max_norm_relative_l2':float(repeat.metrics['relative_l2'].max()),'reference_run':ref_run,'repeat_run':repeat_run}
                    atomic_json(control_path,control)
                    if not control_ok:
                        arrays={'REF_mean_float64':ref.means,'ALT_mean_float64':repeat.means,'delta_float32':(repeat.means-ref.means).astype(np.float32),
                            'prefix_mismatch_counts':repeat.counts,**repeat.metrics}
                        commit(out,row,view,arrays,{'failure':'Identical REF repeat failed','QC':control},fingerprint,False)
                        raise RuntimeError('Identical REF repeat failed; quarantined; no valid variant marker.')
                    del repeat
                control_hash=sha(control_path)
                for row in group:
                    genomic_alt=genomic_ref[:16384]+row['alt']+genomic_ref[16385:]
                    sequence=genomic_alt if view=='forward' else rc(genomic_alt)
                    assert hashlib.sha256(sequence.encode()).hexdigest()==row[view+'_alt_sha256']
                    progress.update(stage='alternate_forward',run_id=row['run_id']);atomic_json(out/'progress.json',progress)
                    alt,alt_score,alt_run=forward(sequence,row,view,prefix+'_'+row['run_id']+'_ALT',patch,ref)
                    delta=(alt.means-ref.means).astype(np.float32)
                    arrays={'REF_mean_float64':ref.means,'ALT_mean_float64':alt.means,'delta_float32':delta,
                        'prefix_mismatch_counts':alt.counts,**alt.metrics}
                    good=all(q['exactly_equal'] for q in alt.qc)
                    assert all(np.isfinite(v).all() for v in arrays.values())
                    metadata={'input_manifest_row':row,'layers':LAYERS,'regions':REGIONS,
                        'region_coordinates':regions[row['run_id']],'variant_index_0based':config['variant_indices'][view],
                        'REF_sequence_sha256':row[view+'_ref_sha256'],'ALT_sequence_sha256':row[view+'_alt_sha256'],
                        'ref_score':ref_score,'alt_score':alt_score,'delta_score':alt_score-ref_score,
                        'QC':{'all_selected_layer_prefixes_exactly_equal':good,'reference_control_passed':True,
                              'reference_control_file':str(control_path),'reference_control_sha256':control_hash,
                              'layer_prefix_checks':alt.qc,'all_arrays_finite':True},
                        'reference_run':ref_run,'alternate_run':alt_run,
                        'mean_norm_relative_l2':float(alt.metrics['relative_l2'].mean()),
                        'prior_183_comparison':old_comparison(row,view,alt.metrics,alt_score-ref_score),
                        'array_schema':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()}}
                    commit(out,row,view,arrays,metadata,fingerprint,good)
                    if not good:raise RuntimeError(f'Nonzero shared prefix at {row["run_id"]}/{view}; quarantined and stopped.')
                    progress['completed_this_invocation']+=1
                    eta_seconds=max(0,forwards_expected-len(forward_seconds))*float(np.mean(forward_seconds))
                    progress.update(elapsed_seconds=time.monotonic()-started,mean_forward_seconds=float(np.mean(forward_seconds)),
                        forwards_completed=len(forward_seconds),forwards_expected_this_invocation=forwards_expected,
                        estimated_remaining_forward_seconds=eta_seconds,
                        estimated_finish_utc=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=eta_seconds)).isoformat())
                    atomic_json(out/'progress.json',progress)
                    print(json.dumps({'event':'completed_variant_view','run_id':row['run_id'],'orientation':view,
                        'completed_this_invocation':progress['completed_this_invocation'],'norm_mean_relative_l2':metadata['mean_norm_relative_l2']}),flush=True)
                    del alt,arrays
                del ref
        assert parameter_versions=={name:p._version for name,p in model.model.named_parameters()}
        assert all(sha(Path(p))==h for p,h in config['patch_dependencies_sha256'].items())
        assert all(sha(Path(p))==h for p,h in config['implementation_sources_sha256'].items())
        all_done=all(valid_marker(out,r,v,fingerprint) for r in rows for v in VIEWS)
        panel_done=all(valid_marker(out,r,v,fingerprint) for r in rows if r['run_id'] in panel_ids for v in VIEWS)
        progress.update(status='complete_all193' if all_done else 'panel25_complete' if panel_done else 'selected_scope_complete',
                        complete_all193=all_done,complete_panel25=panel_done,finished_utc=now())
        atomic_json(out/'progress.json',progress)
    except BaseException as exc:
        progress.update(status='stopped_incomplete',error=str(exc),traceback=traceback.format_exc(),stopped_utc=now())
        atomic_json(out/'progress.json',progress);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--panel',action='store_true');mode.add_argument('--all',action='store_true')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--benchmark',type=Path,default=BENCHMARK);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--patch-module',type=Path,default=ROOT/'scripts/inference/diagnose_brca1_HCM_direct.py')
    p.add_argument('--patch-class',default='HCMDirectPatch')
    p.add_argument('--pilot-validation',type=Path,default=NUMERIC/'HCM_direct_HCL_FP64_pilot/validation.json')
    p.add_argument('--ref-repeat-every',type=int,choices=[1],default=1,help='Fixed at 1: every coordinate/orientation has its own exact repeat.')
    args=p.parse_args()
    if args.limit<0 or args.ref_repeat_every<1:p.error('Invalid limit or reference repeat interval.')
    for key in ['benchmark','output','patch_module','pilot_validation']:setattr(args,key,getattr(args,key).resolve())
    data=prepare(args)
    if args.dry_run:
        args.output.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output/('dry_run_panel.json' if args.panel else 'dry_run_all.json'),data[-1])
        print(json.dumps({k:v for k,v in data[-1].items() if k!='execution_config'},ensure_ascii=False,indent=2));return
    execute(args,*data)


if __name__=='__main__':main()
