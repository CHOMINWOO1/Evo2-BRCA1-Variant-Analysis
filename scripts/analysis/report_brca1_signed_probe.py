#!/usr/bin/env python3
"""Render the completed, frozen signed-vector analysis; never fit a model.

Use .venv-alphagenome/bin/python --check-schema before results are ready.
--run writes four PDF pages and an Excel workbook only after all gates pass.
The workbook subprocess uses .venv/bin/python for the installed xlsxwriter.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / 'results/brca1_grch38/external_functional_validation'
BASE = EXT / 'seqsplice_signed_probe_20260916'
OUT = BASE / 'report'
INFERENCE = ROOT / 'results/brca1_grch38/seqsplice_signed_layers_20260916'
BENCH = EXT / 'seqsplice_complete_benchmark_20260916'
EVALUATOR = ROOT / 'scripts/analysis/evaluate_brca1_signed_probe.py'
KERNEL = ROOT / 'scripts/analysis/brca1_signed_kernel.py'
LAYERS = ['blocks.0', 'blocks.7', 'blocks.14', 'blocks.21', 'blocks.28', 'norm']
REGIONS = ['variant_pm20', 'assayed_exon', 'donor_pm20', 'acceptor_pm20', 'whole_32k']
PAIRS = [('norm', 'assayed_exon')] + [(l, 'assayed_exon') for l in LAYERS[:-1]] + [('norm', r) for r in REGIONS if r != 'assayed_exon']
MODELS = ['scalar_matched7', 'scalar_plus_signed', 'scalar_plus_REF', 'SpliceAI_only',
          'scalar_common5', 'scalar_plus_unit_direction', 'exact_duplicate_scalar_control']
LABELS = {'scalar_matched7':'Scalar 7', 'scalar_plus_signed':'Scalar 7 + signed Δ',
          'scalar_plus_REF':'Scalar 7 + REF', 'SpliceAI_only':'SpliceAI only',
          'scalar_common5':'Common scalar 5', 'scalar_plus_unit_direction':'Scalar 7 + unit Δ',
          'exact_duplicate_scalar_control':'Scalar 7 duplicate'}
CELL_N = {'MDA_MB_231':193, 'HS578T':191}
CELL_NAMES = {'MDA_MB_231':'MDA-MB-231 · 주 분석', 'HS578T':'HS578T · 지원 분석'}
COMPARISON_PAIRS = [('scalar_matched7','scalar_plus_signed'),('scalar_matched7','scalar_plus_REF'),
                    ('scalar_plus_REF','scalar_plus_signed'),('scalar_matched7','scalar_plus_unit_direction'),
                    ('scalar_matched7','exact_duplicate_scalar_control'),('SpliceAI_only','scalar_matched7'),
                    ('scalar_common5','scalar_matched7')]
BLUE, ORANGE, GRAY, DARK = '#286AA4', '#C57832', '#647485', '#243748'
SCHEMAS = {
    'metrics.csv': ['cell_line','layer','region','model','n','MAE_pp','macro_exon_MAE_pp','RMSE_pp','spearman'],
    'paired_comparisons.csv': ['cell_line','layer','region','baseline','combined','weighting','delta_MAE_pp','ci_low','ci_high','n','bootstrap_draws'],
    'heldout_predictions.csv': ['run_id','HGVSc','position_grch38_1based','ref','alt','Exon_legacy','full_length_loss_pp','cell_line','layer','region','model','role','prediction_pp'],
    'per_exon_metrics.csv': ['cell_line','layer','region','model','exon','n','MAE_pp'],
    'selected_lambdas.csv': ['cell_line','layer','region','model','role','outer_exon','lambda','inner_macro_exon_MAE_pp','n_train','n_test'],
    'inner_tuning_scores.csv': ['cell_line','layer','region','model','role','outer_exon','inner_exon','lambda','validation_MAE_pp','n_train','n_validation'],
    'outer_fit_audit.csv': ['cell_line','layer','region','model','role','outer_exon','direct_solve_max_abs_difference','heldout_position_overlap','transforms_fitted_only_on_train'],
    'activation_provenance.csv': ['run_id','orientation','complete_marker','marker_sha256','npz_sha256','REF_control_sha256'],
}
EXPECTED_ROWS = {'metrics.csv':68, 'paired_comparisons.csv':136, 'heldout_predictions.csv':13056,
                 'per_exon_metrics.csv':816, 'selected_lambdas.csv':816,
                 'inner_tuning_scores.csv':53856, 'outer_fit_audit.csv':816,
                 'activation_provenance.csv':386}
SHEETS = [('Metrics','metrics.csv'), ('Paired_comparisons','paired_comparisons.csv'),
          ('Heldout_predictions','heldout_predictions.csv'), ('Per_exon_metrics','per_exon_metrics.csv'),
          ('Selected_lambdas','selected_lambdas.csv'), ('Inner_tuning','inner_tuning_scores.csv'),
          ('Outer_fit_audit','outer_fit_audit.csv'), ('Activation_sources','activation_provenance.csv')]


class GateError(RuntimeError):
    pass


class NotReady(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GateError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def source_schema_check():
    """Read source syntax and frozen rules, without importing/running evaluator."""
    tree = ast.parse(EVALUATOR.read_text())
    constants = {n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    assignments = {t.id:n.value for n in tree.body if isinstance(n,ast.Assign)
                   for t in n.targets if isinstance(t,ast.Name)}
    require(ast.literal_eval(assignments['LAYERS'])==LAYERS,'Evaluator layers changed.')
    require(ast.literal_eval(assignments['REGIONS'])==REGIONS,'Evaluator regions changed.')
    require(set(SCHEMAS).issubset(constants),'Evaluator output filenames changed.')
    kernel_constants = {n.value for n in ast.walk(ast.parse(KERNEL.read_text()))
                        if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    require(set(MODELS).issubset(constants),'Evaluator model scope changed.')
    for name, columns in SCHEMAS.items():
        require(set(columns).issubset(constants | kernel_constants),f'Unknown source schema: {name}')
    protocol = read_json(BASE/'protocol.json')['specification']
    require(protocol['layer_region_pairs']==[list(p) for p in PAIRS],'Frozen pair order changed.')
    require(protocol['primary_pair']==['norm','assayed_exon'],'Primary pair changed.')
    require(protocol['primary_cell']=='MDA_MB_231' and protocol['supporting_cell']=='HS578T','Cell roles changed.')
    require(protocol['bootstrap']['draws']==5000 and protocol['bootstrap']['seed']==20260916,'Bootstrap definition changed.')
    # No evaluator/kernel imports, fitting calls, artificial predictions, or result files.
    return {'schema_status':'PASS','expected_pairs':10,'expected_model_cell_pairs':68,
            'expected_prediction_rows':13056,'expected_pages':4,
            'evaluator_sha256':sha(EVALUATOR),'report_script_sha256':sha(Path(__file__))}


def gate():
    """Return a complete provenance snapshot or fail before creating report files."""
    protocolp=BASE/'protocol.json'; protocol=read_json(protocolp)['specification']
    required=[INFERENCE/'progress.json',INFERENCE/'audit_all/validation.json',BASE/'validation.json',
              BASE/'analysis_execution_manifest.json',BASE/'primary_effect.json',
              BASE/'shared_exon_bootstrap_draws.npz',BASE/'implementation_clarification.json']
    required += [BASE/name for name in SCHEMAS]
    missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise NotReady('Required completed outputs absent: '+', '.join(missing))
    progress=read_json(INFERENCE/'progress.json')
    if progress.get('complete_all193') is not True or progress.get('status')!='complete_all193':
        raise NotReady('Full corrected193 extraction has not completed.')
    fingerprint=protocol['expected_execution_fingerprint']
    require(progress['execution_fingerprint']==fingerprint,'Progress fingerprint differs.')
    audit=read_json(INFERENCE/'audit_all/validation.json')
    require(audit['status']=='PASS' and audit['cohort']=='all' and audit['cohort_complete'] is True,'Independent all-cohort audit did not pass.')
    require(audit['execution_fingerprint']==fingerprint,'Audit fingerprint differs.')
    require(audit['expected_variant_views']==audit['verified_variant_views']==386,'Incomplete 386 variant/view audit.')
    require(not audit['errors'] and not audit['quarantined_failures'] and not audit['missing_at_snapshot'],'Audit has errors/quarantines/missing files.')
    audit_script=ROOT/'scripts/analysis/audit_brca1_signed_panel.py'
    require(audit['audit_script_sha256']==sha(audit_script),'Audit producer changed.')
    for p in (INFERENCE/'quarantine').glob('**/*.failure.json'):
        require(read_json(p).get('execution_fingerprint')!=fingerprint,'New same-protocol quarantined failure.')
    validation=read_json(BASE/'validation.json')
    require(validation['status']=='passed','Probe validation did not pass.')
    for k,v in {'variants':193,'measured_variant_cells':384,'layer_region_pairs':10,
                'model_cell_pair_combinations':68,'outer_fits':816}.items():
        require(validation[k]==v,f'Probe validation count: {k}')
    for k in ['all193_numerical_QC_required','duplicate_kernel_predictions_exactly_identical',
              'source_hashes_unchanged','implementation_hashes_unchanged']:
        require(validation[k] is True,f'Probe validation flag: {k}')
    require(validation['script_sha256']==sha(EVALUATOR) and validation['kernel_script_sha256']==sha(KERNEL),'Analysis implementation changed after fitting.')
    require(validation['max_outer_direct_solve_difference']<1e-7,'Probe solver audit failed.')
    execution=read_json(BASE/'analysis_execution_manifest.json')['specification']
    require(execution['execution_fingerprint']==fingerprint and execution['numerical_cohort_complete_before_any_fit'] is True,'Execution manifest failed.')
    expected={str(p.relative_to(ROOT)) for p in [EVALUATOR,KERNEL,protocolp,
              BASE/'implementation_clarification.json',INFERENCE/'audit_all/validation.json']}
    require(set(execution['input_code_and_rules_sha256'])==expected,'Execution manifest does not pin all required code/rules/audit.')
    snapshot={}
    for mapping in [protocol['inputs'],execution['input_code_and_rules_sha256']]:
        for relative,digest in mapping.items():
            p=ROOT/relative
            require(p.exists() and sha(p)==digest,f'Pinned input changed: {relative}')
            snapshot[relative]=digest
    # Recheck the artifacts used by the completed fit, not just the historical
    # PASS audit. A report must not silently retain a stale activation source.
    with (BENCH/'benchmark_manifest_193.csv').open(newline='') as f:
        run_ids={r['run_id'] for r in csv.DictReader(f)}
    with (BASE/'activation_provenance.csv').open(newline='') as f:
        activations=list(csv.DictReader(f))
    require(len(run_ids)==193 and len(activations)==386,'Activation provenance cohort size differs.')
    expected_views={(rid,view) for rid in run_ids for view in ['forward','rc']}
    require({(r['run_id'],r['orientation']) for r in activations}==expected_views,
            'Activation provenance has duplicate or missing variant/views.')
    for row in activations:
        marker=INFERENCE/'variants'/row['run_id']/(row['orientation']+'.complete.json')
        require(Path(row['complete_marker']).resolve()==marker.resolve(),'Activation marker path differs.')
        require(sha(marker)==row['marker_sha256'],'Activation marker changed since fitting.')
        meta=read_json(marker)
        require(meta['execution_fingerprint']==fingerprint and meta['status']=='passed','Activation marker failed.')
        data=marker.parent/meta['npz_filename']
        control=Path(meta['QC']['reference_control_file'])
        require(control.resolve().parent==(INFERENCE/'controls').resolve(),'REF-control path differs.')
        require(sha(data)==row['npz_sha256']==meta['npz_sha256'],'Activation NPZ changed since fitting.')
        require(sha(control)==row['REF_control_sha256']==meta['QC']['reference_control_sha256'],
                'REF control changed since fitting.')
        for p,digest in [(marker,row['marker_sha256']),(data,row['npz_sha256']),
                         (control,row['REF_control_sha256'])]:
            snapshot[str(p.relative_to(ROOT))]=digest
    measured_protocol=BENCH/'measured_outcomes/protocol.json'
    source_manifest=EXT/'rna_sources/source_manifest.json'
    for record in read_json(measured_protocol)['inputs'].values():
        require(sha(ROOT/record['path'])==record['sha256'],'Measured-input provenance changed.')
        snapshot[record['path']]=record['sha256']
    source_records=[r for r in read_json(source_manifest)['files']
                    if Path(r['file']).name in ['Supplemental_Table_S1.xlsx','Supplemental_Table_S2.xlsx']]
    require(len(source_records)==2,'Original S1/S2 provenance missing.')
    for record in source_records:
        require(sha(ROOT/record['file'])==record['sha256'],'Original source XLSX changed.')
        snapshot[record['file']]=record['sha256']
    require(read_json(BASE/'implementation_clarification.json')['protocol_sha256']==sha(protocolp),'Clarification references a different protocol.')
    for p in required+[Path(__file__),audit_script,measured_protocol,source_manifest]:
        snapshot[str(p.relative_to(ROOT))]=sha(p)
    require(not (BASE/'heldout_predictions.partial.csv').exists(),'Partial result file remains.')
    return snapshot


def load_and_validate_tables():
    """Check reported tables against fixed predictions, without refitting."""
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    tables={name:pd.read_csv(BASE/name) for name in SCHEMAS}
    for name,t in tables.items():
        require(set(SCHEMAS[name]).issubset(t.columns),f'Missing columns: {name}')
        require(len(t)==EXPECTED_ROWS[name],f'Unexpected row count: {name}')
    predictions=tables['heldout_predictions.csv']; metrics=tables['metrics.csv']
    keys=['cell_line','layer','region','model']
    require(not predictions.duplicated(keys+['run_id']).any(),'Repeated held-out prediction.')
    expected={(c,l,r,m) for c in CELL_N for l,r in PAIRS
              for m in (MODELS if (l,r)==PAIRS[0] else MODELS[:3])}
    require(set(map(tuple,metrics[keys].to_numpy()))==expected and not metrics.duplicated(keys).any(),'Missing or extra model/cell/pair.')
    require(set(map(tuple,predictions[keys].drop_duplicates().to_numpy()))==expected,'Prediction combinations differ.')
    outcomes=pd.read_csv(BENCH/'measured_outcomes/SeqSplice193_per_variant_cell_values.csv')
    source=outcomes.rename(columns={'Cell_line':'cell_line'})
    require(source.groupby('cell_line').size().to_dict()==CELL_N,'Measured cohort changed.')
    report_max_error=0.
    for group,frame in predictions.groupby(keys):
        cell,layer,region,model=group
        require(len(frame)==CELL_N[cell] and frame.Exon_legacy.nunique()==12,'Incomplete held-out cohort.')
        joined=frame.merge(source,on=['run_id','cell_line','position_grch38_1based','ref','alt','Exon_legacy'],suffixes=('_prediction','_source'),validate='one_to_one')
        require(len(joined)==len(frame),'Held-out/source exact allele mismatch.')
        require(np.allclose(joined.full_length_loss_pp_prediction,joined.full_length_loss_pp_source,rtol=0,atol=1e-12),'Observed outcomes differ from original source.')
        require(np.isfinite(frame[['prediction_pp','full_length_loss_pp']]).all().all(),'Nonfinite prediction/outcome.')
        error=(frame.prediction_pp-frame.full_length_loss_pp).abs()
        actual={'n':len(frame),'MAE_pp':error.mean(),'macro_exon_MAE_pp':error.groupby(frame.Exon_legacy).mean().mean(),
                'RMSE_pp':np.sqrt(np.mean(error**2)),'spearman':spearmanr(frame.prediction_pp,frame.full_length_loss_pp).statistic}
        stored=select(metrics,**dict(zip(keys,group)))
        for k,v in actual.items():
            require(np.isclose(stored[k],v,rtol=0,atol=1e-10,equal_nan=True),f'Metric mismatch: {group}/{k}')
            if np.isfinite(v):report_max_error=max(report_max_error,abs(stored[k]-v))
        per=tables['per_exon_metrics.csv']
        for exon,g in frame.groupby('Exon_legacy'):
            s=select(per,**dict(zip(keys,group)),exon=exon)
            require(s['n']==len(g) and abs(s.MAE_pp-(g.prediction_pp-g.full_length_loss_pp).abs().mean())<1e-10,'Per-exon metric mismatch.')
    with np.load(BASE/'shared_exon_bootstrap_draws.npz',allow_pickle=False) as z:
        exons,draws=z['exon_order'],z['draws']
    require(np.array_equal(exons,np.sort(predictions.Exon_legacy.unique())),'Bootstrap exon ordering changed.')
    require(np.array_equal(draws,np.random.default_rng(20260916).integers(0,12,size=(5000,12))),'Bootstrap draws changed.')
    comparisons=tables['paired_comparisons.csv']
    require(not comparisons.duplicated(['cell_line','layer','region','baseline','combined','weighting']).any(),'Duplicated comparison.')
    expected_comparisons={(c,l,r,a,b,w) for c in CELL_N for l,r in PAIRS
                          for a,b in (COMPARISON_PAIRS if (l,r)==PAIRS[0] else COMPARISON_PAIRS[:3])
                          for w in ['macro_exon','variant_weighted']}
    require(set(map(tuple,comparisons[['cell_line','layer','region','baseline','combined','weighting']].to_numpy()))==expected_comparisons,
            'Missing or extra fixed comparison.')
    for row in comparisons.itertuples(index=False):
        sub=predictions[(predictions.cell_line==row.cell_line)&(predictions.layer==row.layer)&(predictions.region==row.region)]
        a=sub[sub.model==row.baseline];b=sub[sub.model==row.combined]
        j=a.merge(b,on=['run_id','Exon_legacy','full_length_loss_pp'],suffixes=('_base','_added'),validate='one_to_one')
        require(len(j)==CELL_N[row.cell_line]==row.n,'Comparison has missing alleles.')
        diff=(j.prediction_pp_added-j.full_length_loss_pp).abs()-(j.prediction_pp_base-j.full_length_loss_pp).abs()
        sums=np.array([diff[j.Exon_legacy==e].sum() for e in exons]);counts=np.array([(j.Exon_legacy==e).sum() for e in exons])
        require((counts>0).all() and row.bootstrap_draws==5000,'Incomplete exon bootstrap.')
        if row.weighting=='macro_exon':
            mean=sums/counts;estimate=mean.mean();boot=mean[draws].mean(axis=1)
        else:
            require(row.weighting=='variant_weighted','Unknown comparison weighting.')
            estimate=sums.sum()/counts.sum();boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
        lo,hi=np.quantile(boot,[.025,.975])
        require(np.allclose([row.delta_MAE_pp,row.ci_low,row.ci_high],[estimate,lo,hi],rtol=0,atol=1e-10),'Paired estimate or conditional CI mismatch.')
    for cell in CELL_N:
        sub=predictions[(predictions.cell_line==cell)&(predictions.layer=='norm')&(predictions.region=='assayed_exon')]
        a=sub[sub.model=='scalar_matched7'].set_index('run_id').prediction_pp.sort_index()
        b=sub[sub.model=='exact_duplicate_scalar_control'].set_index('run_id').prediction_pp.sort_index()
        require(a.index.equals(b.index) and np.array_equal(a.to_numpy(),b.to_numpy()),'Exact-duplicate control differs.')
    primary=read_json(BASE/'primary_effect.json')
    stored=select(comparisons,cell_line='MDA_MB_231',layer='norm',region='assayed_exon',
                  baseline='scalar_matched7',combined='scalar_plus_signed',weighting='macro_exon')
    for k,v in primary.items():
        require(abs(stored[k]-v)<1e-10 if isinstance(v,(int,float)) else stored[k]==v,f'Primary effect disagrees: {k}')
    audits=tables['outer_fit_audit.csv']
    require((audits.heldout_position_overlap==0).all() and audits.transforms_fitted_only_on_train.eq(True).all(),'Outer-fit leakage audit failed.')
    require((audits.direct_solve_max_abs_difference<1e-7).all(),'Outer-fit numeric audit failed.')
    for name in ['selected_lambdas.csv','outer_fit_audit.csv']:
        table=tables[name]
        require(not table.duplicated(keys+['outer_exon']).any(),'Repeated outer-fold audit.')
        require(set(map(tuple,table[keys].drop_duplicates().to_numpy()))==expected,'Outer-fold model scope differs.')
        require(table.groupby(keys).outer_exon.nunique().eq(12).all(),'Outer-fold audit incomplete.')
    require(tables['selected_lambdas.csv']['lambda'].isin([.0001,.001,.01,.1,1.,10.]).all(),'Selected lambda outside frozen grid.')
    return tables,{'all_stored_metrics_recomputed':True,'all_136_paired_estimates_and_CIs_recomputed':True,
                   'exact_duplicate_predictions_equal':True,'max_metric_absolute_difference':report_max_error,
                   'new_model_fits':0,'GPU_inference':0}


def select(frame, **conditions):
    for key,value in conditions.items():frame=frame[frame[key].eq(value)]
    require(len(frame)==1,f'Ambiguous row: {conditions}')
    return frame.iloc[0]


def primary_conclusion():
    r=read_json(BASE/'primary_effect.json')
    if r['ci_high']<0:
        conclusion='지정한 선형 scalar 기준보다 오차가 감소했고, 조건부 95% 구간도 0 아래입니다.'
    elif r['ci_low']>0:
        conclusion='지정한 선형 scalar 기준보다 오차가 증가했고, 조건부 95% 구간도 0 위입니다.'
    else:
        conclusion='조건부 95% 구간이 0을 포함하여 추가 개선 여부가 불확실합니다.'
    return f"주 분석 ΔMAE {r['delta_MAE_pp']:+.3f} pp [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]. {conclusion}"


def readme_rows():
    return [
        ('문서','BRCA1 signed representation probe · GRCh38 · 32,768 bp · Evo2 7B'),
        ('주 결과',primary_conclusion()),
        ('먼저 볼 시트','Paired_comparisons의 MDA_MB_231 / norm / assayed_exon / scalar_matched7 → scalar_plus_signed / macro_exon.'),
        ('실제 정답','SeqSplice BRCA1 minigene의 WT 정상 RNA 비율 − ALT 정상 RNA 비율(pp). 저자가 제공한 짝별 차이를 반복 측정에서 평균했습니다. 음수도 유지합니다.'),
        ('자료 출처','Genome Research SeqSplice; https://doi.org/10.1101/gr.279557.124 . 원문 S1·S2와 measured_outcomes 검증 파일은 Source_hashes에 포함합니다.'),
        ('표본','193개 고유 변이 / 12개 저자 assay exon. MDA-MB-231 193개, HS578T 191개. 384개는 변이–세포주 관측값이며 독립 변이 수가 아닙니다.'),
        ('주 분석','norm × assayed_exon, MDA-MB-231, 엑손별 MAE를 동일 가중 평균한 macro-exon MAE. HS578T는 같은 변이의 지원 분석입니다.'),
        ('부호','ΔMAE = combined − baseline. 음수이면 combined 모델의 예측 오차 감소; 양수이면 악화. pp는 percentage points이며 정확도 %가 아닙니다.'),
        ('Common scalar 5','원본 S1 SpliceAI max delta + 교정된 forward/RC sequence penalty 2개 + 교정된 norm whole32k mean relative L2 2개.'),
        ('Scalar 7','Common 5 + 해당 layer/region의 pooled signed Δmean Euclidean norm 2개. 각 pair마다 다른 matched baseline입니다.'),
        ('Signed Δ','각 영역에서 ALT−REF 평균 임베딩 벡터. forward/RC를 각각 학습 데이터 평균으로 중심화하고 RMS 크기로 정규화한 선형 커널을 결합합니다.'),
        ('REF 대조군','같은 scalar7에 REF 평균 벡터를 추가. 문맥 정보 대조군이며 모든 문맥 교란을 제거했다는 뜻이 아닙니다.'),
        ('Unit Δ 대조군','변이별·방향별 Δ벡터를 자기 길이로 나눈 뒤 학습 중심화/RMS 정규화합니다. 정확히 0인 벡터는 0으로 유지합니다.'),
        ('Duplicate 대조군','동일 scalar 커널을 두 번 평균해 원래 scalar와 예측이 정확히 같은지 확인합니다. 추가 정보가 없습니다.'),
        ('주 pair 대조군 범위','Scalar7, +signed, +REF, +unit direction, exact duplicate, SpliceAI only, Common5: 총7개. 다른9개 pair는 Scalar7,+signed,+REF만 평가합니다.'),
        ('고정 비교','assayed_exon에서6개 layer, norm에서5개 region, 주 pair를 공유해 총10개. 모두 보고하며 전체 최적 layer/region을 사후 선정하지 않습니다.'),
        ('분할','12개 assay exon 중 하나 전체를 제외하는 outer LOEO. 동일 좌표의 다른 ALT는 같은 엑손에 속합니다. 모델이 BRCA1 밖 새 유전자에서 검증된 것은 아닙니다.'),
        ('학습','훈련은 변이 균일 가중 ridge. λ=[0.0001,0.001,0.01,0.1,1,10], inner 11-exon macro MAE로 선택, 동률은 더 큰 λ. 예측값 clipping 없음.'),
        ('커널','훈련자료에서만 표준화/중심화/크기 조절. scalar와 vector 커널을 동등 평균; 변화 없는 블록은 제외하고 가중치를 다시 정규화합니다.'),
        ('구간','고정된 outer OOF 예측값에서 12개 엑손 전체를 5,000회 재표집한 조건부 percentile95% 구간. 세포·모델·pair 간 같은 draw 사용. 재학습·재튜닝 불확실성을 포함하지 않습니다.'),
        ('한계','12개 엑손, 겹치는 32k 문맥, 단일 유전자·선정된 minigene 자료. 다중비교 보정된 발견 주장이나 외부 블라인드 사전등록이 아닙니다.'),
        ('해석 범위','지정한 선형 기준 대비 추가 정보만 평가합니다. 모든 비선형 scalar 관계를 넘어선 정보, splice 기전, 임상 병원성은 입증하지 않습니다.'),
        ('수치 교정','HCM grouped FP32 convolution + HCL FP64 FFT. 두 방향 전체 cohort에서 선택 layer prefix와 REF 반복 exact QC 및 독립 audit PASS를 요구했습니다. prefix를 강제로 0으로 바꾸지 않았습니다.'),
        ('보고서 gate','full193 완료, independent audit_all PASS/386, probe validation, 최초 fit execution manifest·protocol·입력 SHA 현재 일치, 모든 표/CI 재계산을 확인한 뒤 생성합니다.'),
        ('보고서 연산','새 fitting·GPU 추론 없음. 기존 OOF 예측값/동결된 결과를 검증하고 표시합니다.'),
        ('출처 기록','Source_hashes와 Report_validation 시트를 참조하세요. Report code 자체의 SHA도 포함합니다.'),
    ]


def workbook(snapshot):
    import xlsxwriter
    target=OUT/'BRCA1_signed_probe.xlsx'
    with xlsxwriter.Workbook(target,{'strings_to_formulas':False,'strings_to_urls':False}) as book:
        header=book.add_format({'bold':True,'bg_color':'#E5EDF5','text_wrap':True})
        wrap=book.add_format({'text_wrap':True,'valign':'top'})
        ws=book.add_worksheet('README');ws.write_row(0,0,['항목','설명'],header)
        ws.set_column(0,0,25);ws.set_column(1,1,125);ws.freeze_panes(1,1)
        for i,row in enumerate(readme_rows(),1):ws.write_row(i,0,row,wrap);ws.set_row(i,47 if len(row[1])>90 else 32)
        for sheet,filename in SHEETS:
            with (BASE/filename).open(newline='') as f:
                reader=csv.reader(f);columns=next(reader)
                ws=book.add_worksheet(sheet);ws.write_row(0,0,columns,header);ws.set_row(0,42)
                ws.set_column(0,len(columns)-1,22);ws.freeze_panes(1,1)
                n=0
                for n,row in enumerate(reader,1):
                    for j,value in enumerate(row):
                        if not value:continue
                        try:
                            number=float(value)
                            if number!=number or abs(number)==float('inf'):ws.write_string(n,j,value)
                            else:ws.write_number(n,j,number)
                        except ValueError:ws.write_string(n,j,value)
                ws.autofilter(0,0,n,len(columns)-1)
        for sheet,path in [('Protocol',BASE/'protocol.json'),('Clarification',BASE/'implementation_clarification.json'),
                           ('Probe_validation',BASE/'validation.json'),('Execution_manifest',BASE/'analysis_execution_manifest.json'),
                           ('Report_validation',OUT/'table_validation.json')]:
            ws=book.add_worksheet(sheet);ws.write_row(0,0,['key','value'],header);ws.set_column(0,0,35);ws.set_column(1,1,140)
            for i,(key,value) in enumerate(read_json(path).items(),1):
                ws.write_row(i,0,[key,json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else str(value)],wrap);ws.set_row(i,60)
        ws=book.add_worksheet('Source_hashes');ws.write_row(0,0,['path','sha256'],header);ws.set_column(0,0,100);ws.set_column(1,1,70)
        for i,row in enumerate(sorted(snapshot.items()),1):ws.write_row(i,0,row)
        ws.freeze_panes(1,1)
    return target


def setup_plotting():
    global np, plt, PdfPages
    os.environ.setdefault('MPLCONFIGDIR','/tmp/brca1-signed-probe-report-mpl')
    os.environ.setdefault('XDG_CACHE_HOME','/tmp/brca1-signed-probe-report-cache')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    require(font.exists(),'Required Korean report font is unavailable.')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
        'font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'text.color':DARK,
        'axes.labelcolor':DARK,'figure.facecolor':'white','savefig.facecolor':'white'})


def text(fig,x,y,message,size=10,color=DARK,weight='normal'):
    fig.text(x,y,message,ha='left',va='top',fontsize=size,color=color,weight=weight,linespacing=1.4)


def page(title,subtitle,number):
    fig=plt.figure(figsize=(12.8,9.2))
    text(fig,.055,.963,title,20,weight='bold');text(fig,.055,.912,subtitle,10,GRAY)
    text(fig,.055,.023,'BRCA1 · GRCh38 · Evo2 7B · 32k · nested exon-held-out linear probe · conditional 95% intervals',8,GRAY)
    fig.text(.945,.023,f'{number}/4',ha='right',va='top',fontsize=8,color=GRAY)
    return fig


def forest(ax,rows,labels,colors=None):
    colors=colors or [BLUE]*len(rows)
    for y,(row,color) in enumerate(zip(rows,colors)):
        ax.plot([row.ci_low,row.ci_high],[y,y],color=color,lw=2)
        ax.plot(row.delta_MAE_pp,y,'o',color=color,ms=5)
    ax.axvline(0,color=GRAY,lw=1,ls='--');ax.set_yticks(range(len(labels)),labels,fontsize=9)
    ax.invert_yaxis();ax.set_xlabel('ΔMAE (pp) · 음수 = added 모델의 오차 감소',fontsize=9)
    ax.grid(axis='x',alpha=.16);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)


def figures(tables):
    metrics=tables['metrics.csv']; comparisons=tables['paired_comparisons.csv'];per=tables['per_exon_metrics.csv']
    def effect(cell,layer='norm',region='assayed_exon',baseline='scalar_matched7',combined='scalar_plus_signed'):
        return select(comparisons,cell_line=cell,layer=layer,region=region,baseline=baseline,combined=combined,weighting='macro_exon')
    fig=page('방향을 가진 임베딩 변화가 실제 RNA 감소 예측을 돕는가?',
             '주 분석: norm × assayed exon · 측정값 = WT 정상 RNA 비율 − ALT 정상 RNA 비율 · 단위 pp',1)
    text(fig,.06,.855,primary_conclusion().replace(']. ', '].\n', 1),11,BLUE,'bold')
    ax=fig.add_axes([.29,.65,.57,.13])
    forest(ax,[effect(c) for c in CELL_N],[CELL_NAMES[c]+f' (n={CELL_N[c]})' for c in CELL_N],[BLUE,ORANGE])
    for left,cell,color in [(.20,'MDA_MB_231',BLUE),(.67,'HS578T',ORANGE)]:
        ax=fig.add_axes([left,.20,.265,.31])
        order=['SpliceAI_only','scalar_common5','scalar_matched7','scalar_plus_signed','scalar_plus_REF','scalar_plus_unit_direction','exact_duplicate_scalar_control']
        values=[select(metrics,cell_line=cell,layer='norm',region='assayed_exon',model=m).macro_exon_MAE_pp for m in order]
        ax.barh(range(7),values,color=[color if m=='scalar_plus_signed' else '#CFDAE4' for m in order],height=.65)
        ax.set_yticks(range(7),[LABELS[m] for m in order],fontsize=8);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.25)
        for i,v in enumerate(values):ax.text(v+max(values)*.015,i,f'{v:.2f}',va='center',fontsize=8)
        ax.set_title(CELL_NAMES[cell],fontsize=11);ax.set_xlabel('Macro-exon MAE (pp) · 작을수록 좋음',fontsize=9)
    text(fig,.06,.105,'193개 변이, 12개 assay exon. HS578T 191개는 같은 변이를 공유하는 지원 분석입니다.\n'
         '막대는 엑손별 평균 절대오차의 평균입니다. 20 pp 오차를 정확도 80%로 해석하지 않습니다.',9,GRAY)
    yield fig
    fig=page('문맥·변화 크기·커널 복제 대조군을 함께 비교합니다',
             '아래 모든 비교는 고정 주 pair(norm × assayed exon)입니다. 각 비교의 baseline과 added를 표시했습니다.',2)
    pairs=[('scalar_matched7','scalar_plus_signed','Scalar 7 → +signed Δ'),
           ('scalar_matched7','scalar_plus_REF','Scalar 7 → +REF'),
           ('scalar_plus_REF','scalar_plus_signed','Scalar 7+REF → Scalar 7+signed Δ'),
           ('scalar_matched7','scalar_plus_unit_direction','Scalar 7 → +unit Δ'),
           ('scalar_matched7','exact_duplicate_scalar_control','Scalar 7 → duplicate'),
           ('SpliceAI_only','scalar_matched7','SpliceAI only → Scalar 7'),
           ('scalar_common5','scalar_matched7','Common 5 → Scalar 7')]
    for left,cell,color in [(.31,'MDA_MB_231',BLUE),(.71,'HS578T',ORANGE)]:
        ax=fig.add_axes([left,.48,.235,.33])
        rows=[effect(cell,baseline=a,combined=b) for a,b,_ in pairs]
        forest(ax,rows,[label if cell=='MDA_MB_231' else '' for _,_,label in pairs],[color]*7)
        ax.set_title(CELL_NAMES[cell],fontsize=11)
    exons=sorted(per.exon.unique());values=[]
    for cell in CELL_N:
        values.append([select(per,cell_line=cell,layer='norm',region='assayed_exon',model='scalar_plus_signed',exon=e).MAE_pp-
                       select(per,cell_line=cell,layer='norm',region='assayed_exon',model='scalar_matched7',exon=e).MAE_pp for e in exons])
    ax=fig.add_axes([.20,.22,.68,.115]);lim=max(float(np.abs(values).max()),1e-10)
    im=ax.imshow(values,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
    ax.set_xticks(range(12),[str(e) for e in exons]);ax.set_yticks([0,1],['MDA-MB-231','HS578T'])
    ax.set_xlabel('논문상 assay exon 번호 · MANE 번호와 다를 수 있음',fontsize=9)
    for i in range(2):
        for j in range(12):ax.text(j,i,f'{values[i][j]:+.1f}',ha='center',va='center',fontsize=8,color='white' if abs(values[i][j])>.6*lim else DARK)
    cb=fig.colorbar(im,cax=fig.add_axes([.90,.22,.012,.115]));cb.set_label('ΔMAE pp',fontsize=8)
    text(fig,.06,.39,'엑손별 signed Δ 추가 효과 · 파랑은 오차 감소, 빨강은 증가',11,weight='bold')
    text(fig,.06,.10,'REF는 문맥 정보 대조군이며 모든 문맥 교란을 제거하지 않습니다. Unit Δ는 벡터 방향 대조군입니다.\n'
         'Duplicate는 동일 커널 복제로, 예측값 exact equality를 요구했습니다. 엑손별 표시는 보조 기술통계입니다.',9,GRAY)
    yield fig
    fig=page('지정한 6개 레이어와 5개 영역을 모두 보여줍니다',
             '각 점은 해당 pair의 matched Scalar 7에 signed Δ를 더한 변화입니다. 파랑 MDA-MB-231, 주황 HS578T.',3)
    for rect,pairs_here,labels,title in [([.19,.27,.32,.53],[(l,'assayed_exon') for l in LAYERS],LAYERS,'Assayed exon · 고정 6개 레이어'),
        ([.69,.27,.25,.53],[('norm',r) for r in REGIONS],['변이 ±20 bp','Assayed exon','Donor ±20 bp','Acceptor ±20 bp','전체 32k'],'Norm · 고정 5개 영역')]:
        ax=fig.add_axes(rect)
        for j,(layer,region) in enumerate(pairs_here):
            for offset,cell,color in [(-.13,'MDA_MB_231',BLUE),(.13,'HS578T',ORANGE)]:
                row=effect(cell,layer,region);y=j+offset
                ax.plot([row.ci_low,row.ci_high],[y,y],color=color,lw=1.7);ax.plot(row.delta_MAE_pp,y,'o',color=color,ms=4.5)
        ax.axvline(0,color=GRAY,ls='--',lw=1);ax.set_yticks(range(len(labels)),labels,fontsize=9);ax.invert_yaxis()
        ax.set_xlabel('Δ macro-exon MAE (pp)',fontsize=9);ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.16)
    text(fig,.06,.17,'주의: 각 pair의 Scalar 7에는 그 pair의 forward/RC Δnorm이 포함됩니다.',11,BLUE,'bold')
    text(fig,.06,.125,'따라서 서로 다른 baseline의 절대 MAE로 “최고 레이어”를 선정하지 않습니다. 주 pair는 두 패널에 공통으로 포함됩니다.\n'
         '총 10개 고정 pair를 모두 보고하며, 95% 구간은 다중비교를 보정한 발견 검정이 아닙니다.',9,GRAY)
    yield fig
    fig=page('검증 절차와 결론의 범위',
             '이 분석은 지정한 선형 기준 대비 추가 예측 정보를 평가합니다. splice 기전 또는 임상 병원성 판정이 아닙니다.',4)
    sections=[('실제 측정값과 동일 allele',
               '193개 minigene 변이의 측정값과 정확한 GRCh38 REF/ALT를 연결했습니다.\n서로 다른 위치를 임의로 짝짓지 않았고, 음수 RNA 감소량과 예측값을 자르지 않았습니다.'),
              ('학습과 평가를 엑손 단위로 분리',
               'Outer 12-exon LOEO / inner 11-exon LOEO. 같은 좌표의 모든 ALT는 같은 fold입니다.\n훈련 구획에서만 정규화하며 λ 6개 중 inner macro MAE로 선택합니다. 훈련 손실은 변이 균일 가중입니다.'),
              ('오차 구간이 포함하는 것',
               '고정 OOF 예측에서 12개 엑손을 5,000회 재표집한 조건부 95% 구간입니다.\n재학습·재튜닝 불확실성은 포함하지 않습니다. 겹치는 32k 문맥과 같은 변이의 두 세포주도 독립 반복이 아닙니다.'),
              ('수치 검증을 통과한 193개 전체',
               'HCM FP32 grouped convolution + HCL FP64 FFT. 두 방향에서 선택 레이어의 전체 prefix와 REF 반복을 검증했습니다.\n386개 variant/view의 독립 audit PASS, fit 이전 manifest와 현재 코드·규칙·원자료 SHA 일치를 요구합니다.'),
              ('해석의 경계',
               'Scalar 7 대비 개선이 있어도 모든 비선형 scalar 관계를 넘어선 정보라는 뜻은 아닙니다. REF 대조군도 모든 교란을 제거하지 않습니다.\n단일 유전자·12개 선정 엑손의 결과입니다. 과거 scalar 결과를 본 뒤 정한 탐색적 계획이며 외부 사전등록이 아닙니다.')]
    for i,(title,body) in enumerate(sections):
        y=.85-i*.145;text(fig,.06,y,title,12,BLUE,'bold');text(fig,.06,y-.043,body,9.4)
    text(fig,.06,.085,'원문: SeqSplice · Genome Research · doi:10.1101/gr.279557.124\n'
         'Excel: README, 모든 raw 결과표, source SHA, probe/report validation을 포함합니다. 보고서에서 새 모델을 fit하지 않았습니다.',8.5,GRAY)
    yield fig


def build(snapshot):
    tables,validation=load_and_validate_tables()
    require(gate()==snapshot,'Sources changed during report validation.')
    setup_plotting()
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'table_validation.json',{'status':'PASS',**validation})
    write_json(OUT/'source_hashes.json',snapshot)
    pdfpath=OUT/'BRCA1_signed_probe.pdf'
    with PdfPages(pdfpath,metadata={'Title':'BRCA1 signed representation: fixed nested exon probe','Author':'BRCA1 Evo2 analysis'}) as pdf:
        for i,fig in enumerate(figures(tables),1):
            pdf.savefig(fig);fig.savefig(OUT/f'page_{i:02d}.png',dpi=130);plt.close(fig)
    subprocess.run([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--workbook-only'],check=True,cwd=ROOT)
    require(gate()==snapshot,'Sources changed while rendering report.')
    (OUT/'README.md').write_text('# BRCA1 signed probe\n\n'+primary_conclusion()+'\n\n'
        '[4쪽 PDF](BRCA1_signed_probe.pdf) · [전체 결과 Excel](BRCA1_signed_probe.xlsx)\n\n'
        '원자료·코드·검증 hash는 source_hashes.json, 표 재계산 검증은 table_validation.json에 있습니다. '
        '조건부 OOF bootstrap이며 새 fitting 또는 GPU 추론은 없습니다.\n')
    artifacts={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='report_validation.json'}
    result={'status':'PASS','pages':4,'completed_utc':datetime.now(timezone.utc).isoformat(),
            'source_snapshot_unchanged':True,'new_model_fits':0,'GPU_inference':0,
            'table_validation':validation,'artifacts':artifacts}
    write_json(OUT/'report_validation.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-schema',action='store_true')
    group.add_argument('--run',action='store_true')
    group.add_argument('--workbook-only',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    schema=source_schema_check()
    try:snapshot=gate()
    except NotReady as error:
        print(json.dumps({**schema,'status':'READY_WAITING_FOR_COMPLETE_ANALYSIS','report_created':False,'reason':str(error)},ensure_ascii=False))
        return
    if args.check_schema:
        print(json.dumps({**schema,'status':'READY_TO_RENDER','report_created':False},ensure_ascii=False));return
    if args.workbook_only:
        require(read_json(OUT/'source_hashes.json')==snapshot,'Workbook sources differ from validated report snapshot.')
        require(read_json(OUT/'table_validation.json')['status']=='PASS','Table validation absent.')
        print(workbook(snapshot));return
    print(json.dumps(build(snapshot),ensure_ascii=False))


if __name__=='__main__':
    main()
