"""Build BRCA1/2 GRCh38 ClinVar and MANE Select splice-site tables.

Inputs are the unmodified ClinVar VCF, MANE v1.5 RefSeq GTF, and hg38
chromosome FASTA files described in results/brca_grch38/README.md.
"""

import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/brca_grch38"
SOURCE = OUT / "sources"
TRANSCRIPTS = {"BRCA1": "NM_007294.4", "BRCA2": "NM_000059.4"}
GENE_IDS = {"BRCA1": "672", "BRCA2": "675"}
NEARBY_BP = 20


def attributes(raw):
    return {key: value for key, value in re.findall(r'(\w+) "([^"]*)"', raw)}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_mane():
    genes = {gene: {"exons": []} for gene in TRANSCRIPTS}
    path = SOURCE / "MANE.GRCh38.v1.5.refseq_genomic.gtf.gz"
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            chrom, _, feature, start, end, _, strand, _, raw = fields
            attr = attributes(raw)
            gene = attr.get("gene_id")
            if gene not in genes:
                continue
            record = genes[gene]
            if feature == "gene":
                record.update(chrom=chrom, gene_start=int(start), gene_end=int(end), strand=strand)
            if attr.get("transcript_id") != TRANSCRIPTS[gene] or "MANE Select" not in raw:
                continue
            if feature == "transcript":
                record.update(transcript=attr["transcript_id"], transcript_start=int(start),
                              transcript_end=int(end), ensembl_transcript=re.search(r'Ensembl:([^"; ]+)', raw).group(1))
            elif feature == "exon":
                record["exons"].append({"exon_number": int(attr["exon_number"]),
                                        "start": int(start), "end": int(end)})
    for gene, record in genes.items():
        record["exons"].sort(key=lambda exon: exon["exon_number"])
        assert record["transcript"] == TRANSCRIPTS[gene] and len(record["exons"]) > 1
        assert all(exon["start"] < exon["end"] for exon in record["exons"])
        assert len({exon["exon_number"] for exon in record["exons"]}) == len(record["exons"])
    return genes


def load_reference(chrom):
    path = SOURCE / f"{chrom}.fa.gz"
    with gzip.open(path, "rt") as stream:
        header = stream.readline().strip()
        assert header.startswith(f">{chrom} ") or header == f">{chrom}", header
        sequence = "".join(line.strip().upper() for line in stream)
    return sequence


def oriented_bases(sequence, positions, strand):
    bases = "".join(sequence[position - 1] for position in positions)
    if strand == "-":
        return bases.translate(str.maketrans("ACGT", "TGCA"))
    return bases


def make_junctions(genes, reference):
    junctions, exons = [], []
    for gene, record in genes.items():
        chrom, strand = record["chrom"], record["strand"]
        sequence = reference[chrom]
        for exon in record["exons"]:
            exons.append(dict(gene=gene, transcript=record["transcript"], chromosome=chrom,
                              strand=strand, exon_number=exon["exon_number"],
                              exon_start_1based=exon["start"], exon_end_1based=exon["end"]))
        for first, second in zip(record["exons"], record["exons"][1:]):
            assert second["exon_number"] == first["exon_number"] + 1
            if strand == "+":
                assert first["end"] < second["start"]
                donor, acceptor = first["end"], second["start"]
                donor_intronic = [donor + 1, donor + 2]
                acceptor_intronic = [acceptor - 2, acceptor - 1]
            else:
                assert second["end"] < first["start"]
                donor, acceptor = first["start"], second["end"]
                donor_intronic = [donor - 1, donor - 2]
                acceptor_intronic = [acceptor + 2, acceptor + 1]
            intron_start = min(first["end"], second["end"]) + 1 if strand == "-" else first["end"] + 1
            intron_end = max(first["start"], second["start"]) - 1 if strand == "-" else second["start"] - 1
            assert intron_start <= intron_end
            junctions.append(dict(gene=gene, transcript=record["transcript"], chromosome=chrom,
                                  strand=strand, junction_id=f"{gene}_J{first['exon_number']:02d}",
                                  upstream_exon=first["exon_number"], downstream_exon=second["exon_number"],
                                  intron_start_1based=intron_start, intron_end_1based=intron_end,
                                  donor_exonic_base_1based=donor, acceptor_exonic_base_1based=acceptor,
                                  donor_plus1_1based=donor_intronic[0], donor_plus2_1based=donor_intronic[1],
                                  acceptor_minus2_1based=acceptor_intronic[0],
                                  acceptor_minus1_1based=acceptor_intronic[1],
                                  donor_dinucleotide=oriented_bases(sequence, donor_intronic, strand),
                                  acceptor_dinucleotide=oriented_bases(sequence, acceptor_intronic, strand),
                                  evidence="MANE_Select_annotation"))
    return junctions, exons


def parse_info(field):
    result = {}
    for part in field.split(";"):
        key, sep, value = part.partition("=")
        result[key] = value if sep else True
    return result


def site_matches(position, junctions):
    matches = []
    for junction in junctions:
        orientation = 1 if junction["strand"] == "+" else -1
        for site_type, column in (("donor", "donor_exonic_base_1based"),
                                  ("acceptor", "acceptor_exonic_base_1based")):
            boundary = junction[column]
            offset = (position - boundary) * orientation
            category = ("canonical_intronic" if (site_type == "donor" and offset in (1, 2))
                        or (site_type == "acceptor" and offset in (-1, -2)) else
                        "exonic_boundary" if offset == 0 else
                        "nearby" if abs(offset) <= NEARBY_BP else "distant")
            matches.append((abs(offset), junction["junction_id"], site_type, boundary, offset, category))
    return sorted(matches)


def write_rows(path, fieldnames, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    genes = read_mane()
    reference = {genes[gene]["chrom"]: load_reference(genes[gene]["chrom"])
                 for gene in TRANSCRIPTS}
    junctions, exons = make_junctions(genes, reference)
    write_rows(OUT / "mane_select_exons.csv", list(exons[0]), exons)
    write_rows(OUT / "mane_select_junctions.csv", list(junctions[0]), junctions)
    junctions_by_gene = {gene: [j for j in junctions if j["gene"] == gene] for gene in genes}

    variant_columns = ["gene", "transcript", "chromosome", "position_grch38_1based", "ref", "alt",
                       "variant_type", "ref_matches_hg38", "clinvar_variation_id", "clinvar_allele_id",
                       "clinvar_germline_classification", "clinvar_review_status", "clinvar_variant_type",
                       "clinvar_conflicting_classifications", "clinvar_condition", "clinvar_condition_ids",
                       "molecular_consequence", "clinvar_hgvs", "clinvar_geneinfo", "geneinfo_matches_gene",
                       "within_mane_transcript", "nearest_junction_id", "nearest_site_type",
                       "nearest_site_exonic_base_1based", "offset_from_site_in_transcript_direction",
                       "nearest_site_category", "clinvar_url", "source"]
    nearby_columns = variant_columns + ["junction_id", "site_type", "site_exonic_base_1based",
                                        "offset_from_site_in_transcript_direction", "site_category"]
    counts = Counter()
    vcf_metadata = {}
    vcf_path = SOURCE / "clinvar.vcf.gz"
    with (OUT / "clinvar_gene_overlap.csv").open("w", newline="") as all_stream, \
            (OUT / "experiment_ready_snvs.csv").open("w", newline="") as snv_stream, \
            (OUT / "splice_nearby_snvs_20bp.csv").open("w", newline="") as nearby_stream, \
            (OUT / "splice_dinucleotide_snvs.csv").open("w", newline="") as dinucleotide_stream:
        all_writer = csv.DictWriter(all_stream, fieldnames=variant_columns)
        snv_writer = csv.DictWriter(snv_stream, fieldnames=variant_columns)
        nearby_writer = csv.DictWriter(nearby_stream, fieldnames=nearby_columns)
        dinucleotide_writer = csv.DictWriter(dinucleotide_stream, fieldnames=nearby_columns)
        for writer in (all_writer, snv_writer, nearby_writer, dinucleotide_writer):
            writer.writeheader()
        with gzip.open(vcf_path, "rt") as vcf:
            for line in vcf:
                if line.startswith("##fileDate="):
                    vcf_metadata["file_date"] = line.strip().partition("=")[2]
                if line.startswith("##reference="):
                    vcf_metadata["reference"] = line.strip().partition("=")[2]
                if line.startswith("#"):
                    continue
                chrom, pos_raw, variation_id, ref, alts, _, _, info_raw = line.rstrip("\n").split("\t")[:8]
                chrom = f"chr{chrom}" if not chrom.startswith("chr") else chrom
                if chrom not in reference:
                    continue
                pos = int(pos_raw)
                info = parse_info(info_raw)
                for gene, record in genes.items():
                    if record["chrom"] != chrom or pos > record["gene_end"] or pos + len(ref) - 1 < record["gene_start"]:
                        continue
                    geneinfo = info.get("GENEINFO", "")
                    geneinfo_match = f"{gene}:{GENE_IDS[gene]}" in geneinfo.split("|")
                    for alt in alts.split(","):
                        snv = len(ref) == len(alt) == 1 and ref in "ACGT" and alt in "ACGT" and ref != alt
                        variant_type = ("SNV" if snv else "MNV" if len(ref) == len(alt) and ref.isalpha() and alt.isalpha()
                                        else "insertion" if len(ref) < len(alt) else "deletion" if len(ref) > len(alt)
                                        else "other")
                        matches_ref = reference[chrom][pos - 1:pos - 1 + len(ref)] == ref
                        sites = site_matches(pos, junctions_by_gene[gene]) if snv else []
                        nearest = sites[0] if sites else None
                        row = dict(gene=gene, transcript=record["transcript"], chromosome=chrom,
                                   position_grch38_1based=pos, ref=ref, alt=alt,
                                   variant_type=variant_type, ref_matches_hg38=matches_ref,
                                   clinvar_variation_id=variation_id, clinvar_allele_id=info.get("ALLELEID", ""),
                                   clinvar_germline_classification=info.get("CLNSIG", ""),
                                   clinvar_review_status=info.get("CLNREVSTAT", ""),
                                   clinvar_variant_type=info.get("CLNVC", ""),
                                   clinvar_conflicting_classifications=info.get("CLNSIGCONF", ""),
                                   clinvar_condition=info.get("CLNDN", ""),
                                   clinvar_condition_ids=info.get("CLNDISDB", ""),
                                   molecular_consequence=info.get("MC", ""),
                                   clinvar_hgvs=info.get("CLNHGVS", ""), clinvar_geneinfo=geneinfo,
                                   geneinfo_matches_gene=geneinfo_match,
                                   within_mane_transcript=record["transcript_start"] <= pos <= record["transcript_end"],
                                   nearest_junction_id=nearest[1] if nearest else "",
                                   nearest_site_type=nearest[2] if nearest else "",
                                   nearest_site_exonic_base_1based=nearest[3] if nearest else "",
                                   offset_from_site_in_transcript_direction=nearest[4] if nearest else "",
                                   nearest_site_category=nearest[5] if nearest else "",
                                   clinvar_url=f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{variation_id}/",
                                   source="ClinVar_GRCh38_VCF")
                        all_writer.writerow(row)
                        counts[f"{gene}_all"] += 1
                        if not matches_ref:
                            counts[f"{gene}_ref_mismatch"] += 1
                        if snv and matches_ref and geneinfo_match:
                            snv_writer.writerow(row)
                            counts[f"{gene}_experiment_ready_snvs"] += 1
                            for distance, junction_id, site_type, boundary, offset, category in sites:
                                if distance > NEARBY_BP:
                                    break
                                nearby_writer.writerow(dict(row, junction_id=junction_id, site_type=site_type,
                                                            site_exonic_base_1based=boundary,
                                                            offset_from_site_in_transcript_direction=offset,
                                                            site_category=category))
                                counts[f"{gene}_nearby_variant_site_pairs"] += 1
                                if category == "canonical_intronic":
                                    dinucleotide_writer.writerow(dict(row, junction_id=junction_id, site_type=site_type,
                                                                       site_exonic_base_1based=boundary,
                                                                       offset_from_site_in_transcript_direction=offset,
                                                                       site_category=category))
                                    counts[f"{gene}_canonical_variant_site_pairs"] += 1
    assert vcf_metadata.get("reference") == "GRCh38", vcf_metadata
    sources = [vcf_path, SOURCE / "MANE.GRCh38.v1.5.refseq_genomic.gtf.gz",
               SOURCE / "chr13.fa.gz", SOURCE / "chr17.fa.gz"]
    manifest = {"assembly": "GRCh38/hg38", "created_utc": datetime.now(timezone.utc).isoformat(),
                "clinvar_file_date": vcf_metadata["file_date"], "mane_release": "v1.5",
                "coordinate_system": "1-based inclusive; VCF POS is 1-based",
                "splice_nearby_radius_bp": NEARBY_BP,
                "junction_evidence": "MANE Select transcript annotation; not RNA-seq observed junctions",
                "genes": {gene: {key: value for key, value in record.items() if key != "exons"}
                          | {"exon_count": len(record["exons"])} for gene, record in genes.items()},
                "counts": dict(counts), "sources": {path.name: {"bytes": path.stat().st_size,
                                                        "sha256": digest(path)} for path in sources},
                "source_urls": {
                    "clinvar.vcf.gz": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz",
                    "MANE.GRCh38.v1.5.refseq_genomic.gtf.gz": "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/MANE.GRCh38.v1.5.refseq_genomic.gtf.gz",
                    "chr13.fa.gz": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr13.fa.gz",
                    "chr17.fa.gz": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr17.fa.gz"}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"counts": manifest["counts"], "junctions": len(junctions),
                      "exons": len(exons)}, indent=2))


if __name__ == "__main__":
    main()
