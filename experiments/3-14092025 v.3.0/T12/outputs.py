# outputs.py
# Zero-Field Spectral Cosmology (ZFSC) v3.1.6
# Сохранение meta, summary, чекпоинтов, текстовых отчётов

import os
import csv
import json


def save_meta(meta: dict, root: str):
    """Сохраняет мета-информацию о прогоне"""
    path = os.path.join(root, "meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def save_summary(summary_rows, root: str):
    """
    Сохраняет полный summary.csv и summary.txt (одноразово).
    Для больших прогонов рекомендуется append_summary_rows().
    """
    csv_path = os.path.join(root, "summary.csv")
    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=["run_id", "N", "seed", "point"])
            writer.writeheader()

    txt_path = os.path.join(root, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as ftxt:
        for row in summary_rows:
            ftxt.write(str(row) + "\n")


def append_summary_rows(rows, root: str):
    """
    Добавляет строки в summary.csv и summary.txt (чекпоинтный режим).
    Автоматически создаёт header при первом вызове.
    """
    if not rows:
        return

    csv_path = os.path.join(root, "summary.csv")
    new_file = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)

    txt_path = os.path.join(root, "summary.txt")
    with open(txt_path, "a", encoding="utf-8") as ftxt:
        for row in rows:
            ftxt.write(str(row) + "\n")


def update_meta_progress(root: str, completed: int, total: int):
    """
    Записывает прогресс в meta_progress.json
    """
    path = os.path.join(root, "meta_progress.json")
    data = {"completed": completed, "total": total}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_report_text(lines, root: str):
    """
    Сохраняет читаемый отчёт по прогону
    """
    path = os.path.join(root, "report.txt")
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(lines, str):
            f.write(lines + "\n")
        else:
            for line in lines:
                f.write(str(line) + "\n")
    print(f"✅ Report сохранён: {path}")
