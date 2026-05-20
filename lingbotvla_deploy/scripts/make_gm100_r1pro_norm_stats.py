#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
import pandas as pd


def stat(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return {
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": (arr.std(axis=0) + 1e-6).astype(float).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).astype(float).tolist(),
        "q02": np.quantile(arr, 0.02, axis=0).astype(float).tolist(),
        "q98": np.quantile(arr, 0.98, axis=0).astype(float).tolist(),
    }


def v(x):
    return np.asarray(x, dtype=np.float32).reshape(-1)


def main():
    root = Path.home() / "LingBotVLA/gm100_r1pro_sample/extracted_BM001"
    out_path = Path.home() / "LingBotVLA/lingbotvla_deploy/assets/norm_stats/galaxea_r1pro_bm001_3view.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found under {root}")

    state_arm = []
    state_eff = []
    action_arm = []
    action_eff = []

    count = 0

    for p in files:
        df = pd.read_parquet(p)
        for _, row in df.iterrows():
            s_left_arm = v(row["observation.state.left_arm"])
            s_left_g = v(row["observation.state.left_gripper"])
            s_right_arm = v(row["observation.state.right_arm"])
            s_right_g = v(row["observation.state.right_gripper"])

            a_left_arm = v(row["action.left_arm"])
            a_left_g = v(row["action.left_gripper"])
            a_right_arm = v(row["action.right_arm"])
            a_right_g = v(row["action.right_gripper"])

            state_arm.append(np.concatenate([s_left_arm, s_right_arm]))
            state_eff.append(np.concatenate([s_left_g, s_right_g]))
            action_arm.append(np.concatenate([a_left_arm, a_right_arm]))
            action_eff.append(np.concatenate([a_left_g, a_right_g]))
            count += 1

    data = {
        "norm_stats": {
            "observation.state.arm.position": stat(np.vstack(state_arm)),
            "observation.state.effector.position": stat(np.vstack(state_eff)),
            "action.arm.position": stat(np.vstack(action_arm)),
            "action.effector.position": stat(np.vstack(action_eff)),
        },
        "count": count,
    }

    out_path.write_text(json.dumps(data, indent=2))
    print("wrote:", out_path)
    print("count:", count)


if __name__ == "__main__":
    main()
