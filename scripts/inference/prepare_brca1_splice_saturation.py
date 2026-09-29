"""Build all single substitutions within 20 bp of MANE splice boundaries."""
import csv
import gzip
import json
import shutil
from pathlib import Path
from run_brca1_grch38_screen import ROOT, read_csv, key, sha, atomic_json
from build_brca_grch38_catalog import site_matches, write_rows


def main():
    parent = ROOT / 'results/brca1_grch38/screen_32768'
    out = ROOT / 'results/brca1_grch38/splice_saturation_20bp'
    out.mkdir(exist_ok=True)
    assert not (out / 'manifest.json').exists(), 'Catalog already prepared; do not overwrite'
    source = parent / 'dataset'
    original = read_csv(source / 'experiment_ready_snvs.csv')
    existing = {key(r): r for r in read_csv(parent / 'run_plan.csv')}
    junctions = read_csv(source / 'mane_select_junctions.csv')
    for j in junctions:
        for field in j:
            if field.endswith('_1based'):
                j[field] = int(j[field])
    config = json.loads((parent / 'config.json').read_text())
    fasta = Path(config['reference_fasta'])
    assert sha(fasta) == config['reference_sha256']
    with gzip.open(fasta, 'rt') as f:
        next(f)
        genome = ''.join(line.strip().upper() for line in f)
    positions = sorted({p for j in junctions for c in
                        ('donor_exonic_base_1based', 'acceptor_exonic_base_1based')
                        for p in range(j[c]-20, j[c]+21)})
    fields = list(original[0]) + ['synthetic_variant_id', 'evidence_status']
    extra = ['junction_id', 'site_type', 'site_exonic_base_1based',
             'offset_from_site_in_transcript_direction', 'site_category']
    all_rows, pending, nearby, essential, reuse = [], [], [], [], []
    for pos in positions:
        ref = genome[pos-1]
        assert ref in 'ACGT'
        sites = [s for s in site_matches(pos, junctions) if s[0] <= 20]
        for alt in 'ACGT':
            if alt == ref:
                continue
            r = dict.fromkeys(fields, '')
            r.update(gene='BRCA1', transcript='NM_007294.4', chromosome='chr17',
                     position_grch38_1based=pos, ref=ref, alt=alt, variant_type='SNV',
                     ref_matches_hg38='True', source='in_silico_saturation_MANEv1.5',
                     synthetic_variant_id=f'sat_{pos}_{ref}_{alt}',
                     evidence_status='computationally_generated_not_observation_evidence')
            k = key(r)
            r['nearest_junction_id'], r['nearest_site_type'] = sites[0][1:3]
            r['nearest_site_exonic_base_1based'] = sites[0][3]
            r['offset_from_site_in_transcript_direction'] = sites[0][4]
            r['nearest_site_category'] = sites[0][5]
            if k in existing:
                old = existing[k]
                for field in original[0]:
                    r[field] = old[field]
                r['evidence_status'] = 'overlap_with_original_ClinVar_catalog'
                reuse.append(dict(synthetic_variant_id=r['synthetic_variant_id'],
                                  source_run_id=old['run_id'],
                                  source_json=str(parent/'variants'/f"{old['run_id']}.json"),
                                  source_npz=str(parent/'variants'/f"{old['run_id']}.npz")))
            else:
                pending.append(r)
                for _, jid, kind, boundary, offset, category in sites:
                    a = dict(r, junction_id=jid, site_type=kind, site_exonic_base_1based=boundary,
                             offset_from_site_in_transcript_direction=offset, site_category=category)
                    nearby.append(a)
                    if category == 'canonical_intronic':
                        essential.append(a)
            all_rows.append(r)
    assert len(all_rows) == len(positions)*3 == len({key(r) for r in all_rows})
    assert not ({key(r) for r in pending} & set(existing))
    core = {int(j[c]) for j in junctions for c in
            ('donor_plus1_1based','donor_plus2_1based','acceptor_minus2_1based','acceptor_minus1_1based')}
    assert sum(int(r['position_grch38_1based']) in core for r in all_rows) == len(core)*3
    write_rows(out/'all_saturation_snvs.csv', fields, all_rows)
    write_rows(out/'experiment_ready_snvs.csv', fields, pending)
    write_rows(out/'splice_nearby_snvs_20bp.csv', fields+extra, nearby)
    write_rows(out/'splice_dinucleotide_snvs.csv', fields+extra, essential)
    write_rows(out/'reuse_existing_results.csv', list(reuse[0]), reuse)
    shutil.copy2(source/'mane_select_junctions.csv', out/'mane_select_junctions.csv')
    # Freeze the existing numerical implementation without modifying the active runner.
    runner_source = ROOT/'scripts/inference/run_brca1_grch38_screen.py'
    code = runner_source.read_text()
    replacements = {
        "CAT = ROOT / 'results/brca1_grch38'": "CAT = ROOT / 'results/brca1_grch38/splice_saturation_20bp'",
        "default=ROOT / 'results/brca1_grch38/screen_32768'": "default=ROOT / 'results/brca1_grch38/splice_saturation_20bp/screen_32768'",
        "default=10)": "default=0)",
        "checkpoint = Path(json.loads((ROOT / 'results/embedding_comparison/summary.json').read_text())['checkpoint'])":
        "checkpoint = Path(json.loads((ROOT / 'results/brca1_grch38/screen_32768/config.json').read_text())['checkpoint'])",
        "scope='ClinVar VCF reference-validated BRCA1 SNVs only; excludes indels, large/complex/imprecise variants and other databases'":
        "scope='In silico saturation SNVs at MANE splice boundaries +/-20bp; only combinations absent from original ClinVar screen; not observed-variant evidence'",
        '''r['run_id'] = f"cv_{r['clinvar_variation_id']}_{r['position_grch38_1based']}_{r['ref']}_{r['alt']}"''':
        "r['run_id'] = r['synthetic_variant_id']",
    }
    for before, after in replacements.items():
        assert code.count(before) == 1, before
        code = code.replace(before, after)
    code = code.replace('Resumable BRCA1 ClinVar SNV screen: 32k Evo2 scores and position metrics.',
                        'Frozen numerical runner for synthetic BRCA1 splice saturation SNVs.')
    runner = ROOT/'scripts/inference/run_brca1_splice_saturation.py'
    assert not runner.exists()
    runner.write_text(code)
    manifest = dict(scope='All three alternative bases at each unique boundary +/-20bp position',
                    unique_positions=len(positions), total_snvs=len(all_rows),
                    reused_snvs=len(reuse), additional_snvs=len(pending),
                    total_core_snvs=len(core)*3, additional_core_snvs=len({key(r) for r in essential}),
                    additional_unique_positions=len({key(r)[:2] for r in pending}),
                    original_config_sha256=sha(parent/'config.json'),
                    original_runner_sha256=sha(runner_source), runner_sha256=sha(runner),
                    files={p.name:sha(p) for p in out.glob('*.csv')})
    atomic_json(out/'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
