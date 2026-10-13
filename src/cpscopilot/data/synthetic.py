"""Physics-inspired synthetic water-treatment CPS with labelled attack campaigns.

The generator exists for smoke tests, CI and controlled ablations; it is not a
substitute for SWaT/WADI/TON_IoT. It simulates three process stages under
closed-loop hysteresis control plus control-network telemetry:

* P1 raw-water intake: FIT101 (inflow), MV101 (inlet valve), LIT101 (tank
  level), P101 (transfer pump)
* P2 chemical dosing: FIT201 (flow), AIT201 (conductivity), AIT202 (pH),
  P201 (dosing pump)
* P3 ultrafiltration: LIT301 (level), FIT301 (outflow), P301 (pump), DPIT301
  (differential pressure)
* control network: NET_PKTS, NET_UNIQ_DST, NET_WRITES, NET_AUTH_FAIL
* environment: AMB_TEMP

Attack campaigns follow a simplified kill chain: an optional reconnaissance
*precursor* (unlabelled, label 0) precedes the labelled attack phase, which makes
multi-horizon prediction learnable. Benign IT scans produce look-alike
precursors that are not followed by an attack, so prediction is not trivial.

Attack types: ``sensor_spoofing`` (LIT101 reports a replayed value and the
controller acts on it), ``actuator_manipulation`` (forced MV101/P101 commands),
``network_dos`` (flooding; sensor values go stale) and ``setpoint_tampering``
(pH dosing set-point changed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cpscopilot.config import SyntheticConfig
from cpscopilot.data.schema import ATTACK_TYPE, BENIGN_TYPE, LABEL, TIMESTAMP
from cpscopilot.utils.seed import make_rng

__all__ = ["SENSOR_COLS", "NETWORK_COLS", "AttackCampaign", "generate_synthetic_cps"]

SENSOR_COLS = [
    "FIT101", "MV101", "LIT101", "P101",
    "FIT201", "AIT201", "AIT202", "P201",
    "LIT301", "FIT301", "P301", "DPIT301",
]
NETWORK_COLS = ["NET_PKTS", "NET_UNIQ_DST", "NET_WRITES", "NET_AUTH_FAIL"]
ENV_COLS = ["AMB_TEMP"]

_MODE = {"none": 0, "sensor_spoofing": 1, "actuator_manipulation": 2,
         "network_dos": 3, "setpoint_tampering": 4}


@dataclass(frozen=True)
class AttackCampaign:
    """Ground truth of one injected campaign (indices are row positions)."""

    campaign_id: int
    attack_type: str
    precursor_start: int | None
    onset: int
    end: int  # exclusive


def _plan_campaigns(cfg: SyntheticConfig, n: int, rng: np.random.Generator) -> list[AttackCampaign]:
    dt = cfg.sampling_seconds
    dur_lo, dur_hi = (int(m * 60 / dt) for m in cfg.attack_duration_minutes)
    lead_lo, lead_hi = (int(m * 60 / dt) for m in cfg.precursor_lead_minutes)
    margin = int(10 * 60 / dt)
    start_offset = int(0.03 * n)
    slot = (n - start_offset) // cfg.n_attacks
    if slot < lead_hi + dur_hi + 2 * margin:
        raise ValueError(
            f"timeline too short for {cfg.n_attacks} attacks: slot of {slot} rows < "
            f"{lead_hi + dur_hi + 2 * margin} required; increase days or reduce n_attacks"
        )
    types = [cfg.attack_types[i % len(cfg.attack_types)] for i in range(cfg.n_attacks)]
    types = [types[i] for i in rng.permutation(len(types))]
    campaigns = []
    for k in range(cfg.n_attacks):
        lo = start_offset + k * slot + lead_hi + margin
        hi = start_offset + (k + 1) * slot - dur_hi - margin
        onset = int(rng.integers(lo, hi + 1))
        end = onset + int(rng.integers(dur_lo, dur_hi + 1))
        precursor = None
        if rng.random() < cfg.precursor_probability:
            precursor = onset - int(rng.integers(lead_lo, lead_hi + 1))
        campaigns.append(AttackCampaign(k, types[k], precursor, onset, end))
    return campaigns


def _benign_scans(cfg: SyntheticConfig, n: int, rng: np.random.Generator,
                  campaigns: list[AttackCampaign]) -> np.ndarray:
    """Mask of benign IT-inventory scans (look like reconnaissance, no attack follows)."""
    dt = cfg.sampling_seconds
    mask = np.zeros(n, dtype=bool)
    n_scans = rng.poisson(cfg.false_precursor_rate_per_day * cfg.days)
    busy = np.zeros(n, dtype=bool)
    guard = int(45 * 60 / dt)
    for c in campaigns:
        start = c.precursor_start if c.precursor_start is not None else c.onset
        busy[max(0, start - guard): min(n, c.end + guard)] = True
    for _ in range(n_scans):
        length = int(rng.integers(int(5 * 60 / dt), int(20 * 60 / dt) + 1))
        start = int(rng.integers(0, max(1, n - length)))
        if not busy[start: start + length].any():
            mask[start: start + length] = True
    return mask


def generate_synthetic_cps(cfg: SyntheticConfig,
                           seed: int) -> tuple[pd.DataFrame, list[AttackCampaign]]:
    """Simulate the plant and return ``(frame, campaigns)``.

    The frame contains ``timestamp``, the 17 telemetry columns, ``label``,
    ``attack_type``, ``phase`` and ``campaign_id``. Everything after the
    telemetry columns is ground truth/metadata and must never be used as input.
    """
    rng = make_rng(seed, "synthetic-cps")
    dt = cfg.sampling_seconds
    n = int(round(cfg.days * 86400 / dt))
    campaigns = _plan_campaigns(cfg, n, rng)
    scans = _benign_scans(cfg, n, rng, campaigns)

    mode = np.zeros(n, dtype=np.int8)
    phase = np.full(n, "normal", dtype=object)
    campaign_id = np.full(n, -1, dtype=np.int32)
    precursor = np.zeros(n, dtype=bool)
    phase[scans] = "benign_scan"
    for c in campaigns:
        if c.precursor_start is not None:
            precursor[c.precursor_start: c.onset] = True
            phase[c.precursor_start: c.onset] = "precursor"
            campaign_id[c.precursor_start: c.onset] = c.campaign_id
        mode[c.onset: c.end] = _MODE[c.attack_type]
        phase[c.onset: c.end] = "attack"
        campaign_id[c.onset: c.end] = c.campaign_id

    # Exogenous context: diurnal demand and ambient temperature.
    t = np.arange(n)
    hours = (t * dt / 3600.0) % 24.0
    demand = 0.75 + 0.3 * np.sin(2 * np.pi * (hours - 7.0) / 24.0)
    amb = 23.0 + 4.0 * np.sin(2 * np.pi * (hours - 9.0) / 24.0) + rng.normal(0, 0.3, n)
    drift = cfg.drift_strength * t / max(n - 1, 1)

    # Pre-drawn noise keeps the simulation loop cheap and seed-deterministic.
    e_lvl, e_flow, e_ph, e_cond, e_dp = (rng.normal(0, s, n) for s in (2.0, 0.05, 0.02, 1.5, 0.3))
    ph_attack_target = np.where(rng.random(len(campaigns)) < 0.5, 8.6, 5.9)

    out = {c: np.empty(n) for c in SENSOR_COLS}
    l1, l3, ph = 650.0, 700.0, 7.0
    mv, p101, p301 = 1, 0, 1
    spoof_hold = 0.0
    stale: dict[str, float] | None = None
    prev_mode = 0
    for i in range(n):
        m = int(mode[i])
        cid = campaign_id[i]
        # -- reported (possibly spoofed) level used by the controller
        if m == 1:
            if prev_mode != 1:
                spoof_hold = l1
            l1_rep = spoof_hold + e_lvl[i]
        else:
            l1_rep = l1 + e_lvl[i]
        l3_rep = l3 + e_lvl[(i * 7) % n]
        # -- closed-loop hysteresis control
        if l1_rep < 500:
            mv = 1
        elif l1_rep > 800:
            mv = 0
        if l3_rep < 500 and l1_rep > 250:
            p101 = 1
        elif l3_rep > 900 or l1_rep < 250:
            p101 = 0
        if l3_rep < 250:
            p301 = 0
        elif l3_rep > 400:
            p301 = 1
        if m == 2:  # forced actuator commands
            mv, p101 = 0, 1
        # -- physics
        fit101 = mv * (2.6 + e_flow[i])
        fit201 = p101 * (2.4 + e_flow[(i * 3) % n]) * (l1 > 50.0)
        fit301 = p301 * (2.2 * demand[i] + e_flow[(i * 5) % n]) * (l3 > 20.0)
        l1 = float(np.clip(l1 + fit101 - fit201, 0.0, 1100.0))
        l3 = float(np.clip(l3 + fit201 - fit301, 0.0, 1200.0))
        target = ph_attack_target[cid] if (m == 4 and cid >= 0) else 7.0
        ph_eq = target if p101 else 7.4
        ph = ph + 0.01 * (ph_eq - ph) + 0.002 * e_ph[i]
        values = {
            "FIT101": fit101, "MV101": float(mv), "LIT101": l1_rep, "P101": float(p101),
            "FIT201": fit201, "AIT201": 250.0 + 2.0 * (amb[i] - 23.0) + e_cond[i] + 5 * drift[i],
            "AIT202": ph + e_ph[i], "P201": float(p101),
            "LIT301": l3_rep, "FIT301": fit301, "P301": float(p301),
            "DPIT301": 18.0 + 6.0 * p301 + 3.0 * ((i % 720) / 720.0) + e_dp[i] + 2 * drift[i],
        }
        if m == 3:  # DoS: polling fails, historian repeats the last good values
            if stale is None:
                stale = values
            values = stale
        else:
            stale = None
        for c in SENSOR_COLS:
            out[c][i] = values[c]
        prev_mode = m

    # Control-network telemetry (vectorised).
    pk_rate = 400.0 * (1.0 + 0.1 * np.sin(2 * np.pi * hours / 24.0))
    pk_rate = pk_rate * np.where(precursor, 1.25, 1.0) * np.where(mode == 1, 1.05, 1.0)
    pk_rate = pk_rate * np.where(mode == 3, rng.uniform(8.0, 15.0, n), 1.0)
    net_pkts = rng.poisson(pk_rate).astype(float)
    net_dst = 6.0 + rng.poisson(0.5, n) + np.where(precursor, rng.poisson(25.0, n), 0) \
        + np.where(scans, rng.poisson(20.0, n), 0)
    writes = rng.poisson(1.5, n) + np.where(mode == 2, rng.poisson(20.0, n), 0)
    onset_burst = np.zeros(n, dtype=bool)
    for c in campaigns:
        if c.attack_type == "setpoint_tampering":
            onset_burst[c.onset: c.onset + 6] = True
    writes = writes + np.where(onset_burst, rng.poisson(8.0, n), 0)
    auth = rng.poisson(0.02, n) + np.where(precursor, rng.poisson(1.2, n), 0) \
        + np.where(scans, rng.poisson(0.3, n), 0)

    frame = pd.DataFrame(out)
    frame["NET_PKTS"] = net_pkts
    frame["NET_UNIQ_DST"] = net_dst.astype(float)
    frame["NET_WRITES"] = writes.astype(float)
    frame["NET_AUTH_FAIL"] = auth.astype(float)
    frame["AMB_TEMP"] = amb

    if cfg.missing_rate > 0:
        drop = rng.random((n, len(SENSOR_COLS))) < cfg.missing_rate
        sensor_values = frame[SENSOR_COLS].to_numpy(copy=True)
        sensor_values[drop] = np.nan
        frame[SENSOR_COLS] = sensor_values

    start = pd.Timestamp(cfg.start)
    frame.insert(0, TIMESTAMP, start + pd.to_timedelta(t * dt, unit="s"))
    attack_names = {v: k for k, v in _MODE.items()}
    frame[LABEL] = (mode > 0).astype(np.int8)
    frame[ATTACK_TYPE] = [BENIGN_TYPE if m == 0 else attack_names[int(m)] for m in mode]
    frame["phase"] = phase.astype(str)
    frame["campaign_id"] = campaign_id
    return frame, campaigns
