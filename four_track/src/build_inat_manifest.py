"""Build species manifest for iNatSounds 2024 → BC2026 transfer-learning pretrain.

Outputs (to data/external/inat_sounds_2024/):
- inat_manifest.csv: one row per audio clip
    columns: split, file_path, inat_cat_id, scientific_name, common_name,
             tax_class, kingdom, phylum, order, family, genus,
             bc2026_primary_label, bc2026_class_name, audio_dir_name
- inat_species_summary.csv: one row per iNat species
    columns: inat_cat_id, scientific_name, tax_class, audio_dir_name,
             n_clips_train, n_clips_val, n_clips_total,
             bc2026_primary_label, bc2026_class_name
- inat_bc2026_coverage.json: summary stats (overlap counts, etc.)

Usage on deepthought (where data lives):
    python3 build_inat_manifest.py \\
        --inat-root /home/swatson/work/kaggle/BirdCLEF/four_track/data/external/inat_sounds_2024 \\
        --bc-taxonomy /tmp/bc2026_taxonomy.csv

Per Bundle 2 plan §14.22.9.5 step 3.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_inat_split(split_json_path: Path) -> dict:
    with open(split_json_path) as f:
        return json.load(f)


def load_bc2026_taxonomy(csv_path: Path) -> dict:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    by_sci = {r["scientific_name"]: r for r in rows}
    return by_sci


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inat-root", required=True, type=Path)
    p.add_argument("--bc-taxonomy", required=True, type=Path)
    args = p.parse_args()

    train = load_inat_split(args.inat_root / "schema_peek" / "train.json")
    val = load_inat_split(args.inat_root / "schema_peek" / "val.json")
    bc_by_sci = load_bc2026_taxonomy(args.bc_taxonomy)

    cats = train["categories"]
    assert len(cats) == len(val["categories"]), "category list should match across splits"
    cat_by_id = {c["id"]: c for c in cats}

    train_clip_count = Counter()
    val_clip_count = Counter()
    for a in train["annotations"]:
        train_clip_count[a["category_id"]] += 1
    for a in val["annotations"]:
        val_clip_count[a["category_id"]] += 1

    # Map audio_id → file_name for both splits.
    train_audio_lookup = {a["id"]: a for a in train["audio"]}
    val_audio_lookup = {a["id"]: a for a in val["audio"]}

    out_dir = args.inat_root
    manifest_path = out_dir / "inat_manifest.csv"
    summary_path = out_dir / "inat_species_summary.csv"
    coverage_path = out_dir / "inat_bc2026_coverage.json"

    rows_written = 0
    with open(manifest_path, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow([
            "split", "file_path", "inat_cat_id", "scientific_name", "common_name",
            "tax_class", "kingdom", "phylum", "order", "family", "genus",
            "bc2026_primary_label", "bc2026_class_name", "audio_dir_name",
        ])
        for split_name, audio_lookup, ann_list in [
            ("train", train_audio_lookup, train["annotations"]),
            ("val", val_audio_lookup, val["annotations"]),
        ]:
            for ann in ann_list:
                cat = cat_by_id[ann["category_id"]]
                audio = audio_lookup[ann["audio_id"]]
                bc_match = bc_by_sci.get(cat["name"], {})
                w.writerow([
                    split_name,
                    audio["file_name"],
                    cat["id"],
                    cat["name"],
                    cat.get("common_name", ""),
                    cat["class"],
                    cat["kingdom"],
                    cat["phylum"],
                    cat["order"],
                    cat["family"],
                    cat["genus"],
                    bc_match.get("primary_label", ""),
                    bc_match.get("class_name", ""),
                    cat["audio_dir_name"],
                ])
                rows_written += 1

    # Per-species summary.
    with open(summary_path, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow([
            "inat_cat_id", "scientific_name", "tax_class", "audio_dir_name",
            "n_clips_train", "n_clips_val", "n_clips_total",
            "bc2026_primary_label", "bc2026_class_name",
        ])
        for cat in cats:
            n_t = train_clip_count[cat["id"]]
            n_v = val_clip_count[cat["id"]]
            bc_match = bc_by_sci.get(cat["name"], {})
            w.writerow([
                cat["id"], cat["name"], cat["class"], cat["audio_dir_name"],
                n_t, n_v, n_t + n_v,
                bc_match.get("primary_label", ""),
                bc_match.get("class_name", ""),
            ])

    # BC2026 coverage summary.
    coverage_by_class: dict = defaultdict(lambda: {
        "bc2026_n_classes": 0,
        "bc2026_matched": 0,
        "bc2026_absent": 0,
        "inat_n_species_for_matched": 0,
        "inat_n_clips_for_matched_train": 0,
        "inat_n_clips_for_matched_val": 0,
        "absent_examples": [],
    })
    inat_by_sci = {c["name"]: c for c in cats}
    for r in csv.DictReader(open(args.bc_taxonomy)):
        cls = r["class_name"]
        sci = r["scientific_name"]
        bucket = coverage_by_class[cls]
        bucket["bc2026_n_classes"] += 1
        if sci in inat_by_sci:
            bucket["bc2026_matched"] += 1
            cat = inat_by_sci[sci]
            bucket["inat_n_species_for_matched"] += 1
            bucket["inat_n_clips_for_matched_train"] += train_clip_count[cat["id"]]
            bucket["inat_n_clips_for_matched_val"] += val_clip_count[cat["id"]]
        else:
            bucket["bc2026_absent"] += 1
            if len(bucket["absent_examples"]) < 5:
                bucket["absent_examples"].append(sci)

    overall = {
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "n_inat_species": len(cats),
        "n_inat_clips_train": len(train["annotations"]),
        "n_inat_clips_val": len(val["annotations"]),
        "n_inat_clips_total": rows_written,
        "n_manifest_rows": rows_written,
        "bc2026_coverage_by_class": dict(coverage_by_class),
    }
    with open(coverage_path, "w") as fout:
        json.dump(overall, fout, indent=2)

    # Pretty-print summary to stdout.
    print(f"manifest:  {manifest_path}  ({rows_written} rows)")
    print(f"summary:   {summary_path}   ({len(cats)} species)")
    print(f"coverage:  {coverage_path}")
    print()
    print(f"iNat species:  {len(cats):,}")
    print(f"iNat clips:    {rows_written:,} (train={len(train['annotations']):,} val={len(val['annotations']):,})")
    print()
    print(f"{'class':12s} {'BC26':>5s} {'matched':>8s} {'inat_sp':>8s} {'train_cl':>9s} {'val_cl':>7s}")
    for cls in ["Aves", "Insecta", "Amphibia", "Mammalia", "Reptilia"]:
        b = coverage_by_class[cls]
        print(
            f"{cls:12s} "
            f"{b['bc2026_n_classes']:>5d} "
            f"{b['bc2026_matched']:>8d} "
            f"{b['inat_n_species_for_matched']:>8d} "
            f"{b['inat_n_clips_for_matched_train']:>9d} "
            f"{b['inat_n_clips_for_matched_val']:>7d}"
        )


if __name__ == "__main__":
    main()
