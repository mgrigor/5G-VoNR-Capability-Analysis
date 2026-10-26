"""
Synthetic multi-vendor 5G VoNR performance dataset.

Produces one row per gNB per day (full-day totals) for three RAN vendors,
each exported with its own counter-naming style, so the project has a real
normalization problem to solve.

Everything here is synthetic: gNB IDs, markets, and values are generated.
Counter names are vendor-flavored but generic, not any operator's internal KPI names.

Usage:
    python generator/generate_vonr_dataset.py --out data --days 60 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Network layout (fictional markets, one vendor per market)
# --------------------------------------------------------------------------
MARKETS = {
    "Ericsson": {"Coastal West": 280, "Desert Valley": 240, "Bay North": 230},
    "Samsung":  {"Central Plains": 210, "Hill Country": 190},
    "Nokia":    {"Harbor Metro": 160, "Lake Region": 140},
}
ID_PREFIX = {"Ericsson": 62, "Samsung": 81, "Nokia": 43}

# Share of nodes in each behavior profile (ground truth, hidden from the dashboard)
PROFILE_MIX = {
    "active": 0.80,         # native VoNR, carries calls every day
    "eps_fallback": 0.12,   # voice steered to LTE; all VoNR counters zero
    "handover_only": 0.03,  # no call setups, but VoNR media arrives via handover
    "rollout": 0.05,        # starts as fallback, VoNR enabled mid-period
}
OUTAGE_SHARE = 0.03          # share of active nodes with a multi-day outage
MISSING_ROW_SHARE = 0.005    # rows dropped to simulate export gaps


def assign_nodes(rng):
    rows = []
    for vendor, markets in MARKETS.items():
        serial = 0
        for market, n in markets.items():
            profiles = rng.choice(list(PROFILE_MIX), size=n, p=list(PROFILE_MIX.values()))
            for p in profiles:
                serial += 1
                rows.append({
                    "gnb_id": f"{ID_PREFIX[vendor]}{rng.integers(10, 99)}{serial:03d}",
                    "vendor": vendor,
                    "market": market,
                    "profile": p,
                    # node size: busier urban nodes vs quiet rural ones
                    "base_setups": float(rng.lognormal(mean=np.log(1500), sigma=0.8)),
                    "fail_rate": float(rng.beta(2, 300) if rng.random() > 0.05 else rng.beta(6, 150)),
                })
    nodes = pd.DataFrame(rows).drop_duplicates("gnb_id").reset_index(drop=True)
    return nodes


def daily_setups(rng, base, dates, start_idx=0, outage=None):
    """Daily 5QI=1 setup attempts with weekly pattern, slow growth, and noise."""
    n = len(dates)
    weekday_factor = np.where(dates.dayofweek >= 5, 0.80, 1.0)
    growth = np.linspace(1.0, 1.06, n)
    noise = rng.normal(1.0, 0.12, n).clip(0.6, 1.4)
    s = np.round(base * weekday_factor * growth * noise).astype(int)
    s[:start_idx] = 0
    if outage is not None:
        s[outage[0]:outage[1]] = 0
    return s


def build_daily(rng, nodes, dates):
    n_days = len(dates)
    out = []
    for node in nodes.itertuples():
        start_idx, outage = 0, None
        if node.profile == "rollout":
            start_idx = int(rng.integers(n_days // 4, 3 * n_days // 4))
        if node.profile == "active" and rng.random() < OUTAGE_SHARE / PROFILE_MIX["active"]:
            o_start = int(rng.integers(5, n_days - 6))
            outage = (o_start, o_start + int(rng.integers(2, 6)))

        if node.profile in ("eps_fallback", "handover_only"):
            setups = np.zeros(n_days, dtype=int)
        else:
            setups = daily_setups(rng, node.base_setups, dates, start_idx, outage)

        # handover-only nodes: tiny media volume, no setups
        ho_media = np.zeros(n_days)
        if node.profile == "handover_only":
            ho_media = rng.uniform(0.002, 0.01, n_days) * (rng.random(n_days) > 0.15)

        out.append(pd.DataFrame({
            "date": dates, "gnb_id": node.gnb_id, "vendor": node.vendor, "market": node.market,
            "setups": setups, "fail_rate": node.fail_rate, "ho_media": ho_media,
        }))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# Vendor-specific exports
# --------------------------------------------------------------------------
def samsung_export(rng, d):
    s = d["setups"].to_numpy()
    n = len(d)
    mb = s * rng.uniform(0.12, 0.25, n) + d["ho_media"].to_numpy() * 500
    cqi_reports = np.round(s * rng.uniform(15, 25, n) + (mb > 0) * (s == 0) * rng.integers(20, 200, n))
    return pd.DataFrame({
        "DATE_ID": d["date"].dt.strftime("%Y-%m-%d"),
        "GNB_ID": d["gnb_id"],
        "MARKET_NAME": d["market"],
        "SS_VoNR_ERABSetupAtt_5QI1": s,
        "SS_VoNR_ERABSetupFail_5QI1": rng.binomial(s, d["fail_rate"]),
        "SS_VoNR_DLDataVol_5QI1_MB": mb.round(2),
        "SS_VoNR_CQIReports_5QI1": cqi_reports.astype(int),
        "SS_VoNR_AvgCQI_5QI1": np.where(cqi_reports > 0, rng.normal(10.5, 0.8, n).clip(6, 14).round(2), np.nan),
    })


def ericsson_export(rng, d):
    s = d["setups"].to_numpy()
    n = len(d)
    fails = rng.binomial(s, d["fail_rate"])
    ho_calls = np.round(d["ho_media"].to_numpy() * 2000).astype(int)
    active = (s - fails) + ho_calls
    return pd.DataFrame({
        "Date": d["date"].dt.strftime("%m/%d/%Y"),
        "NodeName": "GNB" + d["gnb_id"],
        "Market": d["market"],
        "pmVoNrBearerSetupAtt5qi1": s,
        "pmVoNrBearerSetupFail5qi1": fails,
        "pmVoNrBearerActive5qi1": active,
        "pmVoNrBearerDrop5qi1": rng.binomial(active, rng.uniform(0.002, 0.006, n)),
        "pmVoNrPeakUe5qi1": np.ceil(active / rng.uniform(25, 40, n)).astype(int),
        "pmImsBearerSetupAtt5qi5": np.round(s * rng.uniform(1.2, 1.4, n) + ho_calls).astype(int),
    })


def nokia_export(rng, d):
    s = d["setups"].to_numpy()
    n = len(d)
    rtp_rx = np.round(s * rng.uniform(3500, 4500, n) + d["ho_media"].to_numpy() * 1.5e6).astype(np.int64)
    return pd.DataFrame({
        "PERIOD_START_TIME": d["date"].dt.strftime("%Y%m%d"),
        "MRBTS_ID": d["gnb_id"],
        "REGION": d["market"],
        "VONR_SESSION_SETUP_ATT": s,
        "VONR_SESSION_SETUP_FAIL": rng.binomial(s, d["fail_rate"]),
        "VONR_SIP_DIALOG_SETUPS": np.round(s * rng.uniform(0.62, 0.75, n)).astype(int),
        "VONR_RTP_PKT_RECEIVED": rtp_rx,
        "VONR_RTP_PKT_SENT": np.round(rtp_rx * rng.uniform(0.9, 1.1, n)).astype(np.int64),
        "VONR_RTP_END_GAP_COUNT": rng.binomial(s, 0.002) + (rtp_rx > 0) * (s == 0),
    })


COUNTER_MAP = [
    # unified_name, vendor, vendor_counter, description
    ("vonr_setup_att", "Samsung", "SS_VoNR_ERABSetupAtt_5QI1", "5QI=1 voice bearer setup attempts"),
    ("vonr_setup_att", "Ericsson", "pmVoNrBearerSetupAtt5qi1", "5QI=1 voice bearer setup attempts"),
    ("vonr_setup_att", "Nokia", "VONR_SESSION_SETUP_ATT", "Native 5G voice session setup attempts"),
    ("vonr_setup_fail", "Samsung", "SS_VoNR_ERABSetupFail_5QI1", "Failed 5QI=1 setups"),
    ("vonr_setup_fail", "Ericsson", "pmVoNrBearerSetupFail5qi1", "Failed 5QI=1 setups"),
    ("vonr_setup_fail", "Nokia", "VONR_SESSION_SETUP_FAIL", "Failed voice session setups"),
    ("vonr_media_evidence", "Samsung", "SS_VoNR_DLDataVol_5QI1_MB", "Downlink voice payload on 5QI=1 (MB)"),
    ("vonr_media_evidence", "Ericsson", "pmVoNrBearerActive5qi1", "5QI=1 calls in active conversational state"),
    ("vonr_media_evidence", "Nokia", "VONR_RTP_PKT_RECEIVED", "RTP voice packets received over 5G"),
    ("ims_signaling", "Ericsson", "pmImsBearerSetupAtt5qi5", "5QI=5 IMS signaling bearer setups"),
    ("ims_signaling", "Nokia", "VONR_SIP_DIALOG_SETUPS", "IMS SIP dialog setups on 5G SA"),
    ("vonr_drops", "Ericsson", "pmVoNrBearerDrop5qi1", "Dropped 5QI=1 calls"),
    ("peak_vonr_ue", "Ericsson", "pmVoNrPeakUe5qi1", "Peak concurrent VoNR users"),
    ("radio_quality", "Samsung", "SS_VoNR_AvgCQI_5QI1", "Average CQI reported by 5QI=1 UEs"),
    ("media_gap", "Nokia", "VONR_RTP_END_GAP_COUNT", "Calls where 5G RTP stopped abruptly at end of call"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--start", default="2026-08-01")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(args.start, periods=args.days, freq="D")
    nodes = assign_nodes(rng)
    daily = build_daily(rng, nodes, dates)
    daily = daily.sample(frac=1 - MISSING_ROW_SHARE, random_state=args.seed).sort_values(["gnb_id", "date"])

    exports = {
        "Samsung": ("samsung_pm_daily.csv", samsung_export),
        "Ericsson": ("ericsson_pm_daily.csv", ericsson_export),
        "Nokia": ("nokia_pm_daily.csv", nokia_export),
    }
    for vendor, (fname, fn) in exports.items():
        df = fn(rng, daily[daily["vendor"] == vendor].reset_index(drop=True))
        df.to_csv(out / fname, index=False)
        print(f"{fname:<26} {len(df):>7,} rows  {df.shape[1]} columns")

    nodes[["gnb_id", "vendor", "market"]].to_csv(out / "dim_gnb.csv", index=False)
    nodes[["gnb_id", "vendor", "profile"]].to_csv(out / "ground_truth_profiles.csv", index=False)
    pd.DataFrame(COUNTER_MAP, columns=["unified_name", "vendor", "vendor_counter", "description"]) \
        .to_csv(out / "counter_mapping.csv", index=False)

    print(f"\n{len(nodes):,} gNBs over {args.days} days")
    print(nodes.groupby(["vendor", "profile"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
