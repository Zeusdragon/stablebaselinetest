import os
import glob
import pandas as pd
import sys

LOG_DIR = os.path.join(".", "tb_logs")
ALL_CSV = os.path.join(LOG_DIR, "actions_all.csv")

# If aggregated file is missing, try to run aggregate_actions.py
if not os.path.exists(ALL_CSV):
    agg_path = os.path.join(os.path.dirname(__file__), "aggregate_actions.py")
    if os.path.exists(agg_path):
        print(
            "Aggregated actions_all.csv nicht gefunden — versuche, aggregate_actions.py auszuführen..."
        )
        rc = os.system(f'"{sys.executable}" "{agg_path}"')
        if rc != 0:
            print("Fehler beim Ausführen von aggregate_actions.py")
    else:
        print("Keine aggregate_actions.py gefunden. Bitte erst die Logs erzeugen.")

if not os.path.exists(ALL_CSV):
    print(f"Keine Aktions-Logs gefunden unter {ALL_CSV}.")
    print(
        "Falls du TensorBoard-Events prüfen willst, liste bitte den Inhalt von ./tb_logs mit:"
    )
    print(
        "  uv run python -c \"import glob; print(glob.glob('./tb_logs/**', recursive=True))\""
    )
    sys.exit(1)

print(f"Lese {ALL_CSV} ...")
df = pd.read_csv(ALL_CSV, parse_dates=["time"])

# Anzahl Action==1 pro Episode (episode_start)
counts = df.groupby("episode_start").size().rename("action1_count")
counts = counts.sort_values(ascending=False)

summary_csv = os.path.join(LOG_DIR, "actions_summary_by_episode.csv")
counts.to_csv(summary_csv, header=True)

print("Top 20 Episoden nach Anzahl Action==1:")
print(counts.head(20).to_string())
print("\nAggregierte Statistik gespeichert in:", summary_csv)

print("\nErste 100 Action-Ereignisse:")
print(
    df.sort_values(["episode_start", "step_in_episode"])
    .head(100)
    .to_string(index=False)
)

# Simple timeline: counts per 10000 global_idx window
if not df.empty:
    bins = (df["global_idx"] // 10000) * 10000
    time_counts = df.groupby(bins).size().rename("events_in_window")
    print("\nEvents per global_idx window (size=10000):")
    print(time_counts.to_string())

print("\nFertig.")
