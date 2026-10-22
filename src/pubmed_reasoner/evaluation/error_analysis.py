def summarize_errors(rows):
    return [r for r in rows if r.get("gold") is not None and r.get("prediction") is not None and r["gold"]!=r["prediction"]]

