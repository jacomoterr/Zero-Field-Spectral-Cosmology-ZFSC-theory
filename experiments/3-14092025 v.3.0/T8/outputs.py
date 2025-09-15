import os, csv, json

def save_meta(meta: dict, root: str):
    path = os.path.join(root, "meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def save_summary(summary_rows, root: str):
    """Сохраняем сырые summary.csv и summary.txt."""
    csv_path = os.path.join(root, "summary.csv")
    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=[
                "run_id", "N", "seed", "point", "num_plateaus",
                "gap_ratio", "edge", "alpha", "kappa", "gamma", "m", "eps"
            ])
            writer.writeheader()

    txt_path = os.path.join(root, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as ftxt:
        for row in summary_rows:
            ftxt.write(str(row) + "\n")
