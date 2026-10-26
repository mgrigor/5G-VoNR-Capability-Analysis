"""
Normalize three vendor PM exports into one schema and classify each gNB's
5G voice capability.

Classification (over the evaluation window, using full-day totals):
    Active VoNR            -> setups > 0
    VoNR via Handover Only -> setups == 0 and media evidence > 0
    EPS Fallback           -> setups == 0 and media evidence == 0

Usage:
    python scripts/normalize_and_classify.py --data data --window 7
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

UNIFIED = ["vonr_setup_att", "vonr_setup_fail", "vonr_media_evidence", "ims_signaling",
           "vonr_drops", "peak_vonr_ue", "radio_quality", "media_gap"]

VENDOR_SPEC = {
    "Samsung": {
        "file": "samsung_pm_daily.csv",
        "date": ("DATE_ID", "%Y-%m-%d"), "id": "GNB_ID", "market": "MARKET_NAME",
    },
    "Ericsson": {
        "file": "ericsson_pm_daily.csv",
        "date": ("Date", "%m/%d/%Y"), "id": "NodeName", "market": "Market",
        "id_clean": lambda s: s.str.replace("^GNB", "", regex=True),
    },
    "Nokia": {
        "file": "nokia_pm_daily.csv",
        "date": ("PERIOD_START_TIME", "%Y%m%d"), "id": "MRBTS_ID", "market": "REGION",
    },
}


def normalize(data_dir: Path) -> pd.DataFrame:
    mapping = pd.read_csv(data_dir / "counter_mapping.csv")
    frames = []
    for vendor, spec in VENDOR_SPEC.items():
        raw = pd.read_csv(data_dir / spec["file"], dtype={spec["id"]: str, spec["date"][0]: str})
        rename = dict(mapping.loc[mapping.vendor == vendor, ["vendor_counter", "unified_name"]].values)
        df = raw.rename(columns=rename)
        ids = df[spec["id"]]
        df["gnb_id"] = spec["id_clean"](ids) if "id_clean" in spec else ids
        df["date"] = pd.to_datetime(df[spec["date"][0]], format=spec["date"][1])
        df["market"] = df[spec["market"]]
        df["vendor"] = vendor
        for col in UNIFIED:
            if col not in df:
                df[col] = np.nan
        frames.append(df[["date", "vendor", "market", "gnb_id"] + UNIFIED])
    fact = pd.concat(frames, ignore_index=True).sort_values(["vendor", "gnb_id", "date"])
    fact["has_vonr_traffic"] = (fact["vonr_setup_att"] > 0) | (fact["vonr_media_evidence"] > 0)
    return fact


def classify(fact: pd.DataFrame, window_days: int) -> pd.DataFrame:
    end = fact["date"].max()
    win = fact[fact["date"] > end - pd.Timedelta(days=window_days)]
    g = win.groupby(["vendor", "market", "gnb_id"], as_index=False).agg(
        setups=("vonr_setup_att", "sum"),
        setup_fails=("vonr_setup_fail", "sum"),
        media=("vonr_media_evidence", "sum"),
        days_reported=("date", "nunique"),
    )
    g["vonr_status"] = np.select(
        [g["setups"] > 0, g["media"] > 0],
        ["Active VoNR", "VoNR via Handover Only"],
        default="EPS Fallback",
    )
    g["setup_success_rate"] = np.where(g["setups"] > 0, 1 - g["setup_fails"] / g["setups"], np.nan)

    first = fact[fact["vonr_setup_att"] > 0].groupby("gnb_id")["date"].min().rename("first_vonr_setup_date")
    return g.merge(first, on="gnb_id", how="left")


def validate(status: pd.DataFrame, data_dir: Path):
    truth_path = data_dir / "ground_truth_profiles.csv"
    if not truth_path.exists():
        return
    truth = pd.read_csv(truth_path, dtype={"gnb_id": str})
    expected = {"active": "Active VoNR", "rollout": "Active VoNR",
                "handover_only": "VoNR via Handover Only", "eps_fallback": "EPS Fallback"}
    m = status.merge(truth[["gnb_id", "profile"]], on="gnb_id")
    m["expected"] = m["profile"].map(expected)
    print("\nDual rule (setups OR media) vs ground truth:")
    print(pd.crosstab(m["profile"], m["vonr_status"]))
    print(f"Accuracy: {(m['expected'] == m['vonr_status']).mean():.2%}")

    setup_only = np.where(m["setups"] > 0, "VoNR", "EPS Fallback")
    missed = ((m["profile"] == "handover_only") & (setup_only == "EPS Fallback")).sum()
    print(f"A setups-only rule would mislabel {missed} handover-only nodes as EPS Fallback.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--window", type=int, default=7, help="evaluation window in days")
    args = ap.parse_args()
    data_dir = Path(args.data)

    fact = normalize(data_dir)
    status = classify(fact, args.window)
    fact.to_csv(data_dir / "fact_vonr_daily.csv", index=False)
    status.to_csv(data_dir / "gnb_vonr_status.csv", index=False)

    print(f"fact_vonr_daily.csv  {len(fact):,} rows")
    print(f"gnb_vonr_status.csv  {len(status):,} gNBs (last {args.window} days)\n")
    print(status.pivot_table(index="vendor", columns="vonr_status", values="gnb_id",
                             aggfunc="count", fill_value=0))
    validate(status, data_dir)


if __name__ == "__main__":
    main()
