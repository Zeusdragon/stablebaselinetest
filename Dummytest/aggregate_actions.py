import glob
import os
import pandas as pd

# Aggregiert alle action logs in ./tb_logs/actions_*.csv zu einer Datei actions_all.csv


def aggregate_logs(out_path="./tb_logs/actions_all.csv"):
    os.makedirs("./tb_logs/", exist_ok=True)
    files = glob.glob("./tb_logs/actions_*.csv")
    rows = []
    for f in files:
        with open(f, "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()

        if first_line.startswith("global_idx,"):
            df = pd.read_csv(f)
        else:
            df = pd.read_csv(
                f,
                header=None,
                names=[
                    "global_idx",
                    "episode_start",
                    "step_in_episode",
                    "action",
                    "reward",
                    "pid",
                    "time",
                ],
            )

        if "action" not in df.columns:
            df["action"] = 1
        if "reward" not in df.columns:
            df["reward"] = pd.NA
        df["source_file"] = os.path.basename(f)
        rows.append(df)
    if not rows:
        print("Keine Action-Logs gefunden in ./tb_logs/")
        return
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(out_path, index=False)
    print(f"Aggregierte {len(rows)} Dateien, {len(all_df)} Ereignisse -> {out_path}")


if __name__ == "__main__":
    aggregate_logs()
