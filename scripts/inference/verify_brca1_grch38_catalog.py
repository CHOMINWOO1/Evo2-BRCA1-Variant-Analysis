"""Independently audit BRCA1 catalog against raw ClinVar, MANE, and hg38 FASTA."""

import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "results/brca1_grch38"
ORIGINAL = ROOT / "results/brca_grch38"
SOURCES = ORIGINAL / "sources"


def table(name):
    with (CATALOG / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    source_manifest = json.loads((ORIGINAL / "manifest.json").read_text())
    selected_manifest = json.loads((CATALOG / "manifest.json").read_text())
    gene = source_manifest["genes"]["BRCA1"]
    assert selected_manifest["gene"] == "BRCA1"
    for filename in ("clinvar.vcf.gz", "MANE.GRCh38.v1.5.refseq_genomic.gtf.gz", "chr17.fa.gz"):
        assert sha256(SOURCES / filename) == source_manifest["sources"][filename]["sha256"]

    variants = table("clinvar_gene_overlap.csv")
    variant_keys = {(row["chromosome"].removeprefix("chr"), int(row["position_grch38_1based"]),
                     row["clinvar_variation_id"], row["ref"], row["alt"]) for row in variants}
    assert len(variant_keys) == len(variants)
    raw_keys = set()
    vcf_file_date = None
    vcf_reference = None
    with gzip.open(SOURCES / "clinvar.vcf.gz", "rt") as stream:
        for line in stream:
            if line.startswith("##fileDate="):
                vcf_file_date = line.strip().partition("=")[2]
            elif line.startswith("##reference="):
                vcf_reference = line.strip().partition("=")[2]
            if line.startswith("#"):
                continue
            chrom, position, variation_id, ref, alts, *_ = line.rstrip("\n").split("\t")
            if chrom.removeprefix("chr") != "17":
                continue
            position = int(position)
            if position > gene["gene_end"] or position + len(ref) - 1 < gene["gene_start"]:
                continue
            raw_keys.update(("17", position, variation_id, ref, alt) for alt in alts.split(","))
    assert vcf_reference == "GRCh38" and vcf_file_date == source_manifest["clinvar_file_date"]
    assert variant_keys == raw_keys

    with gzip.open(SOURCES / "chr17.fa.gz", "rt") as stream:
        assert stream.readline().startswith(">chr17")
        chromosome = "".join(line.strip().upper() for line in stream)
    for row in variants:
        start = int(row["position_grch38_1based"]) - 1
        assert chromosome[start:start + len(row["ref"])] == row["ref"]
        assert row["ref_matches_hg38"] == "True" and row["gene"] == "BRCA1"

    exons = table("mane_select_exons.csv")
    raw_exons = []
    with gzip.open(SOURCES / "MANE.GRCh38.v1.5.refseq_genomic.gtf.gz", "rt") as stream:
        for line in stream:
            if ("\texon\t" not in line or 'transcript_id "NM_007294.4"' not in line
                    or 'tag "MANE Select"' not in line):
                continue
            fields = line.rstrip("\n").split("\t")
            exon_number = int(re.search(r'exon_number "(\d+)"', fields[8]).group(1))
            raw_exons.append((exon_number, int(fields[3]), int(fields[4])))
    raw_exons.sort()
    derived_exons = sorted((int(row["exon_number"]), int(row["exon_start_1based"]),
                            int(row["exon_end_1based"])) for row in exons)
    assert raw_exons == derived_exons and len(raw_exons) == 23

    junctions = table("mane_select_junctions.csv")
    assert len(junctions) == len(raw_exons) - 1 == 22
    for junction, (upstream, downstream) in zip(junctions, zip(raw_exons, raw_exons[1:])):
        assert junction["gene"] == "BRCA1" and junction["strand"] == "-"
        assert int(junction["upstream_exon"]) == upstream[0]
        assert int(junction["downstream_exon"]) == downstream[0]
        assert int(junction["donor_exonic_base_1based"]) == upstream[1]
        assert int(junction["acceptor_exonic_base_1based"]) == downstream[2]
        assert int(junction["intron_start_1based"]) == downstream[2] + 1
        assert int(junction["intron_end_1based"]) == upstream[1] - 1
        donor = upstream[1]
        acceptor = downstream[2]
        donor_motif = chromosome[donor - 3:donor - 1][::-1].translate(str.maketrans("ACGT", "TGCA"))
        acceptor_motif = chromosome[acceptor:acceptor + 2][::-1].translate(str.maketrans("ACGT", "TGCA"))
        assert junction["donor_dinucleotide"] == donor_motif
        assert junction["acceptor_dinucleotide"] == acceptor_motif

    snvs = table("experiment_ready_snvs.csv")
    nearby = table("splice_nearby_snvs_20bp.csv")
    dinucleotides = table("splice_dinucleotide_snvs.csv")
    snv_keys = {(r["position_grch38_1based"], r["ref"], r["alt"]) for r in snvs}
    assert len(snvs) == 11632 and len(nearby) == 3589 and len(dinucleotides) == 233
    assert all((r["position_grch38_1based"], r["ref"], r["alt"]) in snv_keys for r in nearby)
    assert all(abs(int(r["offset_from_site_in_transcript_direction"])) <= 20 for r in nearby)
    for row in dinucleotides:
        offset = int(row["offset_from_site_in_transcript_direction"])
        assert (row["site_type"] == "donor" and offset in (1, 2)) or \
               (row["site_type"] == "acceptor" and offset in (-1, -2))

    report = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "assembly": "GRCh38", "clinvar_file_date": vcf_file_date,
        "mane_release": source_manifest["mane_release"],
        "clinvar_variants_exact_raw_vcf_match": len(variants),
        "variants_ref_match_hg38": len(variants),
        "variants_with_germline_classification": sum(bool(r["clinvar_germline_classification"]) for r in variants),
        "variants_without_germline_classification": sum(not r["clinvar_germline_classification"] for r in variants),
        "variants_within_mane_select_transcript": sum(r["within_mane_transcript"] == "True" for r in variants),
        "mane_exons_exact_raw_gtf_match": len(exons),
        "mane_junctions_reconstructed_from_exons": len(junctions),
        "junction_motifs": dict(Counter(f"{r['donor_dinucleotide']}-{r['acceptor_dinucleotide']}" for r in junctions)),
        "experiment_ready_snvs": len(snvs),
        "splice_nearby_variant_site_pairs": len(nearby),
        "splice_dinucleotide_variant_site_pairs": len(dinucleotides),
        "scope": "Database record and reference-coordinate verification only; not patient-observation or RNA splicing-effect verification",
    }
    (CATALOG / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
