"""Prepare a small GRCh38 BRCA SNV batch for the existing Evo2 extractor."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from build_brca_grch38_catalog import OUT, load_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=OUT / "splice_dinucleotide_snvs.csv")
    parser.add_argument("--gene", choices=("BRCA1", "BRCA2"),
                        help="Restrict the prepared batch to one gene")
    parser.add_argument("--per-gene", type=int, default=5,
                        help="Take at most this many junctions per gene")
    parser.add_argument("--classification", action="append", default=[],
                        help="Exact ClinVar CLNSIG value; repeat to accept multiple values")
    parser.add_argument("--all-rows", action="store_true",
                        help="Select consecutive distinct SNVs instead of one per junction")
    parser.add_argument("--offset", type=int, default=0,
                        help="Number of eligible distinct SNVs to skip in --all-rows mode")
    parser.add_argument("--limit", type=int, default=10,
                        help="Maximum variants in --all-rows mode")
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.per_gene < 1 or args.offset < 0 or args.limit < 1 or args.length < 2 or args.length % 2:
        parser.error("counts must be nonnegative/positive and --length must be even and >= 2")
    output = args.output or OUT / (f"batch_{args.offset:05d}_{args.length}" if args.all_rows
                                   else f"pilot_{args.length}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {output}")

    grouped = {gene: {} for gene in (args.gene,) if gene} if args.gene else {"BRCA1": {}, "BRCA2": {}}
    consecutive = []
    seen = set()

    def priority(row):
        classification = row["clinvar_germline_classification"]
        is_pathogenic = classification in {"Pathogenic", "Likely_pathogenic",
                                           "Pathogenic/Likely_pathogenic"}
        review = row["clinvar_review_status"]
        review_rank = (3 if review == "reviewed_by_expert_panel" else
                       2 if review == "criteria_provided,_multiple_submitters,_no_conflicts" else
                       1 if review == "criteria_provided,_single_submitter" else 0)
        return (is_pathogenic, review_rank)

    with args.catalog.open(newline="") as stream:
        for row in csv.DictReader(stream):
            gene = row["gene"]
            if gene not in grouped:
                continue
            if args.classification and row["clinvar_germline_classification"] not in args.classification:
                continue
            if row["variant_type"] != "SNV" or row["ref_matches_hg38"] != "True":
                continue
            key = (gene, row["chromosome"], row["position_grch38_1based"], row["ref"], row["alt"])
            if args.all_rows:
                if key not in seen:
                    seen.add(key)
                    if len(seen) > args.offset:
                        consecutive.append(row)
                        if len(consecutive) >= args.limit:
                            break
                continue
            junction = row.get("junction_id") or row["nearest_junction_id"]
            if not junction:
                continue
            previous = grouped[gene].get(junction)
            if previous is None or priority(row) > priority(previous):
                grouped[gene][junction] = row
    selected = ({gene: [row for row in consecutive if row["gene"] == gene] for gene in grouped}
                if args.all_rows else
                {gene: [grouped[gene][junction] for junction in sorted(grouped[gene],
                 key=lambda value: int(value.rsplit("J", 1)[1]))[:args.per_gene]] for gene in grouped})
    if not any(selected.values()):
        raise ValueError("No matching, reference-validated SNVs in catalog")

    references = {}
    sequences = {}
    variants = []
    for gene in selected:
        for row in selected[gene]:
            chrom = row["chromosome"]
            if chrom not in references:
                references[chrom] = load_reference(chrom)
            genome = references[chrom]
            position = int(row["position_grch38_1based"])
            start0 = position - 1 - args.length // 2
            ref_seq = genome[start0:start0 + args.length]
            assert start0 >= 0 and len(ref_seq) == args.length
            assert ref_seq[args.length // 2] == row["ref"]
            alt_seq = ref_seq[:args.length // 2] + row["alt"] + ref_seq[args.length // 2 + 1:]
            assert sum(a != b for a, b in zip(ref_seq, alt_seq)) == 1
            variant_id = f"v{len(variants) + 1:04d}"
            ref_id = f"ref_{chrom}_{position}"
            sequences[ref_id] = ref_seq
            sequences[variant_id] = alt_seq
            variants.append(dict(row, id=variant_id, ref_id=ref_id,
                                 window_start_grch38_1based=start0 + 1,
                                 window_end_grch38_1based=start0 + args.length,
                                 variant_index_0based=args.length // 2))

    output.mkdir(parents=True, exist_ok=True)
    with (output / "inputs.fasta").open("w") as stream:
        for name, sequence in sequences.items():
            stream.write(f">{name}\n{sequence}\n")
    with (output / "variants.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(variants[0]))
        writer.writeheader()
        writer.writerows(variants)
    metadata = {"assembly": "GRCh38/hg38", "orientation": "forward genomic",
                "gene": args.gene or "BRCA1,BRCA2",
                "length_bp": args.length, "variant_count": len(variants),
                "unique_reference_count": len({row["ref_id"] for row in variants}),
                "selection_mode": "consecutive_rows" if args.all_rows else "one_per_junction",
                "eligible_row_offset": args.offset if args.all_rows else None,
                "catalog": str(args.catalog.resolve()), "variants": variants,
                "sequence_sha256": {name: hashlib.sha256(seq.encode()).hexdigest()
                                    for name, seq in sequences.items()}}
    (output / "inputs.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {len(variants)} variants and {len(sequences)} FASTA records in {output}")


if __name__ == "__main__":
    main()
