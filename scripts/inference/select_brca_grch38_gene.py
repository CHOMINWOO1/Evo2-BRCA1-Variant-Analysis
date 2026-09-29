"""Make a single-gene GRCh38 catalog from the validated BRCA1/2 tables."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/brca_grch38"
TABLES = ("clinvar_gene_overlap.csv", "experiment_ready_snvs.csv",
          "splice_nearby_snvs_20bp.csv", "splice_dinucleotide_snvs.csv",
          "mane_select_junctions.csv", "mane_select_exons.csv")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", choices=("BRCA1", "BRCA2"), default="BRCA1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f"results/{args.gene.lower()}_grch38"
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    counts = {}
    checksums = {}
    for filename in TABLES:
        source_path, output_path = SOURCE / filename, output / filename
        with source_path.open(newline="") as source_stream, output_path.open("w", newline="") as output_stream:
            reader = csv.DictReader(source_stream)
            assert reader.fieldnames and "gene" in reader.fieldnames
            writer = csv.DictWriter(output_stream, fieldnames=reader.fieldnames)
            writer.writeheader()
            count = 0
            for row in reader:
                if row["gene"] == args.gene:
                    writer.writerow(row)
                    count += 1
            counts[filename] = count
        checksums[filename] = sha256(output_path)
    manifest = {"assembly": "GRCh38/hg38", "gene": args.gene,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "clinvar_file_date": source_manifest["clinvar_file_date"],
                "mane_release": source_manifest["mane_release"],
                "transcript": source_manifest["genes"][args.gene]["transcript"],
                "source_catalog": str(SOURCE.resolve()),
                "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
                "counts": counts, "table_sha256": checksums}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
