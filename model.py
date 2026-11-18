"""DST-aligned Level + Shape decomposed probabilistic forecaster for launch-day Twitch CCU.

Pure Python 3.13 standard-library implementation designed for deterministic, fast execution
and strong out-of-sample quantile calibration on rolling-midnight console game launches.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Student-t quantiles (df = 10) for honest finite-sample tails against launch-day outliers.
T_QUANTILES = [-1.81246, -0.69981, 0.0, 0.69981, 1.81246]

VIEWER_FLOOR = 1.0
QUANTILE_MIN = 0.01
QUANTILE_MAX = 1_000_000_000.0

# Pure-European Twitch broadcast languages (excludes es/pt because LATAM & Brazil are huge at 00-05Z).
EU_ONLY_LANGS = frozenset({
    "fr", "de", "it", "pl", "ru", "tr", "nl", "sv", "cs", "uk", "ro", "hu", "el", "fi", "da", "no",
})

# Keywords identifying multiplayer/live-service launches so single-player console launches align cleanly.
MP_KEYWORDS = (
    "call of duty", "battlefield", "multiplayer", "live-service", "forza",
    "diablo", "borderlands", "monster hunter", "halo", "rivals",
    "helldivers", "marathon", "arc raiders", "nightreign",
)


_ORD_CACHE: dict[str, int] = {}


def utc_hour_index(ts: str) -> int:
    """Fast exact integer UTC hour index for whole-hour ISO timestamps (YYYY-MM-DDTHH:...)."""
    day_key = ts[:10]
    ord_val = _ORD_CACHE.get(day_key)
    if ord_val is None:
        ord_val = date(int(day_key[0:4]), int(day_key[5:7]), int(day_key[8:10])).toordinal()
        _ORD_CACHE[day_key] = ord_val
    return ord_val * 24 + int(ts[11:13])


def parse_utc(ts: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp string into an aware datetime."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_us_dst(dt: datetime) -> bool:
    """Return True when US Daylight Saving Time (EDT = UTC-4) is active on dt's date.

    Rolling-midnight console launches unlock US East at 04:00Z (hour 5) during EDT
    and at 05:00Z (hour 6) during EST (e.g., November 19, 2026).
    """
    month = dt.month
    if 4 <= month <= 10:
        return True
    if month <= 2 or month == 12:
        return False
    if month == 3:
        first_dow = datetime(dt.year, 3, 1, tzinfo=timezone.utc).weekday()
        second_sun = 1 + (6 - first_dow) % 7 + 7
        return dt.day >= second_sun
    if month == 11:
        first_dow = datetime(dt.year, 11, 1, tzinfo=timezone.utc).weekday()
        first_sun = 1 + (6 - first_dow) % 7
        return dt.day < first_sun
    return False


def clamp(val: float, lo: float, hi: float) -> float:
    return lo if val < lo else (hi if val > hi else val)


def soft_sat_log_v0(log_v0: float) -> float:
    """Smoothly saturate log(v1) above ~600k viewers so mega-launches do not over-extrapolate."""
    cap = 13.3
    if log_v0 <= cap:
        return log_v0
    return cap + 0.8 * math.tanh((log_v0 - cap) / 0.8)


def round2(x: float) -> float | int:
    """Round half up to 2 decimal places, returning int for exact whole numbers."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def geom_mean(values: list[float], fallback: float) -> float:
    pos = [v for v in values if v > 0.0 and math.isfinite(v)]
    if not pos:
        return max(VIEWER_FLOOR, fallback)
    return math.exp(sum(math.log(v) for v in pos) / len(pos))


def solve_ridge_with_inv(
    X: list[list[float]],
    y: list[float],
    w: list[float],
    lam: list[float],
) -> tuple[list[float], list[list[float]]]:
    """Solve weighted ridge (X^T W X + diag(lam)) beta = X^T W y and return (beta, A_inv)."""
    p = len(X[0])
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for xi, yi, wi in zip(X, y, w):
        for j in range(p):
            wxi = wi * xi[j]
            b[j] += wxi * yi
            row_j = A[j]
            for k in range(j, p):
                row_j[k] += wxi * xi[k]
    for j in range(p):
        for k in range(j):
            A[j][k] = A[k][j]
        A[j][j] += lam[j]

    # Gauss-Jordan elimination on tiny p x p system (p <= 6) yields both beta and A_inv cleanly.
    aug = [A[i] + [1.0 if i == j else 0.0 for j in range(p)] + [b[i]] for i in range(p)]
    for col in range(p):
        piv = col
        best = abs(aug[col][col])
        for r in range(col + 1, p):
            val = abs(aug[r][col])
            if val > best:
                best = val
                piv = r
        if piv != col:
            aug[col], aug[piv] = aug[piv], aug[col]
        denom = aug[col][col]
        if abs(denom) < 1e-12:
            continue
        inv_d = 1.0 / denom
        row_c = aug[col]
        for c in range(col, 2 * p + 1):
            row_c[c] *= inv_d
        for r in range(p):
            if r == col:
                continue
            row_r = aug[r]
            factor = row_r[col]
            if factor == 0.0:
                continue
            for c in range(col, 2 * p + 1):
                row_r[c] -= factor * row_c[c]

    A_inv = [aug[i][p:2 * p] for i in range(p)]
    beta = [aug[i][2 * p] for i in range(p)]
    return beta, A_inv


def quad_form(x: list[float], M: list[list[float]]) -> float:
    """Evaluate non-negative quadratic form x^T M x for Bayesian predictive leverage."""
    total = 0.0
    for j, xj in enumerate(x):
        row = M[j]
        s = 0.0
        for k, xk in enumerate(x):
            s += row[k] * xk
        total += xj * s
    return max(0.0, total)


def weighted_median(vals: list[float], weights: list[float]) -> float:
    """Compute continuous midpoint weighted median."""
    pairs = sorted(zip(vals, weights), key=lambda item: item[0])
    tot = sum(w for _, w in pairs)
    if tot <= 0.0:
        return pairs[len(pairs) // 2][0]
    cum = 0.0
    prev_v, prev_p = pairs[0][0], 0.0
    for v, w in pairs:
        p_mid = (cum + 0.5 * w) / tot
        if 0.5 <= p_mid:
            if p_mid <= prev_p:
                return v
            return prev_v + (0.5 - prev_p) / (p_mid - prev_p) * (v - prev_v)
        prev_v, prev_p = v, p_mid
        cum += w
    return pairs[-1][0]


def weighted_huber_loc(vals: list[float], weights: list[float], delta: float = 0.25, iters: int = 8) -> float:
    """Weighted Huber M-estimator initialized at the weighted median (robust to outlier launches)."""
    mu = weighted_median(vals, weights)
    for _ in range(iters):
        num = 0.0
        den = 0.0
        for v, w in zip(vals, weights):
            err = abs(v - mu)
            hw = w if err <= delta else w * (delta / err)
            num += hw * v
            den += hw
        if den > 0.0:
            mu = num / den
    return mu


def shift_dst_curve(y: dict[int, float], src_dst: bool, tgt_dst: bool) -> dict[int, float]:
    """Align a comparable's 23-hour log-ratio curve between EDT (DST=True) and EST (DST=False)."""
    if src_dst == tgt_dst:
        return y
    # DST -> EST moves US East midnight from hour 5 (04Z) to hour 6 (05Z): delta = +1.
    delta = 1 if (src_dst and not tgt_dst) else -1
    out: dict[int, float] = {}
    for h in range(2, 25):
        sh = h - delta
        if sh < 2:
            shifted_val = 0.5 * y[2]
        elif sh > 24:
            shifted_val = y[24]
        else:
            shifted_val = y[sh]
        # Strongest shift around the overnight US unlock transition (h=4..7), milder shift in evening.
        alpha = 0.85 if 4 <= h <= 7 else 0.45
        out[h] = (1.0 - alpha) * y[h] + alpha * shifted_val
    return out


def extract_record(
    name: str,
    day_start: str,
    by_h: dict[int, tuple[float, float | None]],
    platforms: str,
    meta_text: str,
) -> dict[str, Any]:
    """Extract standardized pre-launch features and launch-day curve from hourly viewer points."""
    d0 = parse_utc(day_start)
    v0_raw, ch0_raw = by_h.get(0, (VIEWER_FLOOR, None))
    v0 = max(VIEWER_FLOOR, float(v0_raw if v0_raw is not None else VIEWER_FLOOR))
    ch0 = float(ch0_raw) if (ch0_raw is not None and ch0_raw > 0) else None

    pre_eve = [by_h[dh][0] for dh in range(-8, -1) if dh in by_h and by_h[dh][0] and by_h[dh][0] > 0]
    pre_eve_gm = geom_mean(pre_eve, v0)

    late3 = [by_h[dh][0] for dh in range(-3, 0) if dh in by_h and by_h[dh][0] and by_h[dh][0] > 0]
    late3_gm = geom_mean(late3, v0)

    base = [by_h[dh][0] for dh in range(-72, -24) if dh in by_h and by_h[dh][0] and by_h[dh][0] > 0]
    base_med = sorted(base)[len(base) // 2] if base else 1.0

    ul = meta_text.lower()
    pl = platforms.lower()

    is_ea = (("early-access" in ul) or ("early access" in ul)) and ("no early access" not in ul)
    is_campaign_ea_only = "campaign early access" in ul
    ul_has_pc = (" pc" in ul or "(pc" in ul or "pc," in ul or "pc/" in ul or "steam" in ul) and (
        "pc 2019" not in ul and "no pc" not in ul
    )
    ul_has_console = any(k in ul for k in ("ps5", "ps4", "playstation", "xbox", "switch", "console only"))
    is_console = (
        ("pc" not in pl and pl != "")
        or ("pc 2019" in pl)
        or ("console only" in ul)
        or ("ps5 only" in ul)
        or ("ps5, ps4" in ul)
        or (ul_has_console and not ul_has_pc)
        or ("grand theft auto vi" in name.lower())
    )
    is_sim_pc_mid = (
        ("pc single global unlock" in ul)
        or ("pc/stadia simultaneous" in ul)
        or ("23:00z" in ul and "pc" in pl)
        or ("00:00z" in ul and "pc" in pl)
    )
    is_sim_pc_later = (
        ("steam global unlock" in ul)
        or ("pc unlock later" in ul)
        or ("pc 10 am" in ul)
    )
    is_f2p = ("free-to-play" in ul) or ("f2p" in ul)
    name_low = name.lower()
    is_mp = any(k in name_low or k in ul or k in pl for k in MP_KEYWORDS)
    if "single-player" in ul or "single player" in ul or "grand theft auto vi" in name_low:
        is_mp = False
    is_sp = not is_mp

    y: dict[int, float] = {}
    for h in range(2, 25):
        pt = by_h.get(h - 1)
        if pt is not None and pt[0] is not None:
            y[h] = math.log(max(VIEWER_FLOOR, float(pt[0])) / v0)

    r_eve = math.log(pre_eve_gm / v0)
    # Classify launch regime cleanly so simultaneous-PC and multi-day EA launches do not confound rolling console launches.
    if is_sim_pc_later:
        regime = "pc_later"
    elif is_sim_pc_mid or ((not is_console) and (r_eve < -1.0 or y.get(2, 0.0) > 0.35)):
        regime = "pc_mid"
    elif is_ea and (not is_campaign_ea_only):
        regime = "ea"
    elif is_console:
        regime = "console_rolling"
    else:
        regime = "multi_rolling"

    A = (sum(y.values()) / len(y)) if len(y) == 23 else 0.0
    max_ratio = max((math.exp(v) for v in y.values()), default=1.0)

    return {
        "name": name,
        "dst": is_us_dst(d0),
        "v0": v0,
        "ch0": ch0,
        "log_v0": math.log(v0),
        "r_eve": r_eve,
        "r_late": math.log(v0 / late3_gm),
        "d1_jump": math.log(max(VIEWER_FLOOR, pre_eve_gm) / max(VIEWER_FLOOR, base_med)),
        "is_ea": is_ea,
        "is_console": is_console,
        "is_sim_pc_mid": is_sim_pc_mid,
        "is_sim_pc_later": is_sim_pc_later,
        "is_f2p": is_f2p,
        "is_sp": is_sp,
        "regime": regime,
        "y": y,
        "A": A,
        "max_ratio": max_ratio,
    }


def build_comparable_pool(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse active rolling/launch comparables from the case file (no hardcoded answers)."""
    pool: list[dict[str, Any]] = []
    for comp in case.get("comparables") or []:
        unlock = comp.get("unlock") or ""
        if "target-history" in unlock:
            continue
        day_start = comp.get("day_start")
        hourly = comp.get("hourly")
        if not day_start or not hourly:
            continue
        d0_hr = utc_hour_index(day_start)
        by_h: dict[int, tuple[float, float | None]] = {}
        for pt in hourly:
            frm = pt.get("from")
            vw = pt.get("viewers")
            if frm is None or vw is None:
                continue
            dh = utc_hour_index(frm) - d0_hr
            if -72 <= dh <= 24:
                ch = pt.get("channels")
                by_h[dh] = (float(vw), float(ch) if ch is not None else None)
        if 0 not in by_h or by_h[0][0] <= 0.0:
            continue
        if not all((h - 1) in by_h for h in range(2, 25)):
            continue
        rec = extract_record(
            comp.get("name") or "",
            day_start,
            by_h,
            comp.get("platforms") or "",
            unlock,
        )
        # Keep launches already active at hour 1; exclude midday-simultaneous unlocks (>9x spike) and F2P outliers.
        if rec["v0"] >= 5000.0 and rec["max_ratio"] < 9.0 and not rec["is_f2p"] and rec["name"] != "Halo Infinite":
            pool.append(rec)
    return pool


def extract_live_refinements(case: dict[str, Any], tgt: dict[str, Any], mu_h2: float) -> tuple[float, float, float]:
    """Compute bounded Final-case refinements from minute reads, top streams, and category totals.

    Returns (exit_surprise, stream_overnight_adj, sat_scale_factor).
    On practice cases (hourly history, null streams/categories), returns (0.0, 0.0, 1.0).
    """
    d0_ts = parse_utc(case["day_start"]).timestamp()
    ckpt_ts = parse_utc(case["checkpoint"]).timestamp()

    # 1. Minute-level exit velocity within Hour 1 (00:00Z .. 01:00Z)
    exit_surprise = 0.0
    history = case.get("history") or []
    day_s = case["day_start"]
    ckpt_s = case["checkpoint"]
    fast_cmp = day_s.endswith("Z") and ckpt_s.endswith("Z") and len(day_s) == 20 and len(ckpt_s) == 20
    h1_pts: list[tuple[float, float]] = []
    for pt in history:
        at = pt.get("at")
        vw = pt.get("viewers")
        if at is None or vw is None or vw <= 0:
            continue
        if fast_cmp and len(at) == 20 and at.endswith("Z"):
            if not (day_s <= at < ckpt_s):
                continue
        t = parse_utc(at).timestamp()
        if d0_ts <= t < ckpt_ts:
            h1_pts.append((t, float(vw)))
    if len(h1_pts) >= 15:
        h1_pts.sort(key=lambda x: x[0])
        tail_n = max(5, len(h1_pts) // 5)
        tail_mean = sum(v for _, v in h1_pts[-tail_n:]) / tail_n
        all_mean = sum(v for _, v in h1_pts) / len(h1_pts)
        if tail_mean > 0.0 and all_mean > 0.0:
            raw_exit = clamp(math.log(tail_mean / all_mean), -0.25, 0.25)
            # Subtract expected half-hour drift implied by mu_h2 so on-path curves get 0 adjustment.
            expected_half = 0.40 * clamp(mu_h2, -0.35, 0.20)
            exit_surprise = clamp(raw_exit - expected_half, -0.18, 0.18)

    # 2. Top-20 live stream language & marathon-uptime mix at checkpoint
    stream_overnight_adj = 0.0
    streams = case.get("streams")
    if isinstance(streams, list) and streams:
        tot_sv = 0.0
        eu_sv = 0.0
        long_sv = 0.0
        for st in streams:
            sv = float(st.get("viewers") or 0.0)
            if sv <= 0.0:
                continue
            tot_sv += sv
            lang = (st.get("language") or "").lower()
            if lang in EU_ONLY_LANGS:
                eu_sv += sv
            started = st.get("started_at")
            if started:
                try:
                    age_h = (ckpt_ts - parse_utc(started).timestamp()) / 3600.0
                    if age_h >= 8.0:
                        long_sv += sv
                except ValueError:
                    pass
        if tot_sv > 0.0:
            eu_share = eu_sv / tot_sv
            long_share = long_sv / tot_sv
            # Mild bounded shift: unusually high EU-only share or exhausted 8h+ streams deepens h=2..6 dip slightly.
            adj = -0.07 * (eu_share - 0.25) - 0.05 * (long_share - 0.35)
            stream_overnight_adj = clamp(adj, -0.06, 0.05)

    # 3. Total Twitch category saturation check (categories_hourly)
    sat_scale_factor = 1.0
    cats = case.get("categories_hourly")
    if isinstance(cats, list) and cats:
        last_cat = cats[-1]
        top_list = last_cat.get("top") if isinstance(last_cat, dict) else None
        if isinstance(top_list, list) and top_list:
            tot_top20 = sum(float(item.get("viewers") or 0.0) for item in top_list if isinstance(item, dict))
            if tot_top20 > 0.0:
                share = tgt["v0"] / tot_top20
                # If the category already commands >60% of Twitch's top-20 at 00Z, gently widen upper uncertainty safety
                # without distorting the median excessively.
                if share > 0.65:
                    sat_scale_factor = 1.03

    return exit_surprise, stream_overnight_adj, sat_scale_factor


def fallback_forecast(v0: float, dst: bool) -> dict[int, list[float | int]]:
    """Fallback console-rolling diurnal prior if a case supplies fewer than 3 comparables."""
    # Generic rolling-midnight console log-ratio curve (DST=True baseline, shifted if EST).
    base_mu = {
        2: -0.20, 3: -0.48, 4: -0.58, 5: -0.18, 6: -0.10, 7: -0.15, 8: -0.18,
        9: -0.20, 10: -0.20, 11: -0.14, 12: -0.02, 13: 0.14, 14: 0.30, 15: 0.45,
        16: 0.56, 17: 0.62, 18: 0.68, 19: 0.74, 20: 0.76, 21: 0.74, 22: 0.68,
        23: 0.58, 24: 0.45,
    }
    mu_map = shift_dst_curve(base_mu, True, dst)
    out: dict[int, list[float | int]] = {}
    for h in range(2, 25):
        sig = 0.16 + 0.03 * math.sqrt(h - 1)
        qs = [round2(clamp(v0 * math.exp(mu_map[h] + z * sig), QUANTILE_MIN, QUANTILE_MAX)) for z in T_QUANTILES]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        out[h] = qs
    return out


def forecast_case(case: dict[str, Any], name: str = "case") -> dict[str, Any]:
    """Generate the 23-hour x 5-quantile forecast payload for one makefun/twitch-ccu-case/v1 file."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")

    d0_hr = utc_hour_index(case["day_start"])
    by_h: dict[int, tuple[float, float | None]] = {}
    for pt in case.get("history_hourly") or []:
        frm = pt.get("from")
        vw = pt.get("viewers")
        if frm is None or vw is None:
            continue
        dh = utc_hour_index(frm) - d0_hr
        if -72 <= dh <= 0:
            ch = pt.get("channels")
            by_h[dh] = (float(vw), float(ch) if ch is not None else None)
    by_h[0] = (
        max(VIEWER_FLOOR, float(first["viewers"])),
        float(first["channels"]) if first.get("channels") is not None else None,
    )

    game_name = (case.get("game") or {}).get("name") or ""
    tgt = extract_record(game_name, case["day_start"], by_h, "", case.get("notes") or "")
    pool = build_comparable_pool(case)

    if len(pool) < 4:
        q_map = fallback_forecast(tgt["v0"], tgt["dst"])
        return {"hours": [{"hour": row["hour"], "q": q_map[row["hour"]]} for row in case["forecast"]]}

    # --- 1. MODEL B: Regime-deconfounded Level + Shape Ridge with Bayesian Predictive Variance ---
    t_reg = tgt["regime"]
    t_sat_lv0 = soft_sat_log_v0(tgt["log_v0"])

    # Allow pc_mid mega-launches (Elden Ring, Cyberpunk) to inform mega-scale log_v0 slope
    # while explicit regime dummies absorb their simultaneous-PC intercept shift cleanly.
    reg_w_lvl = {
        ("console_rolling", "console_rolling"): 6.0,
        ("console_rolling", "multi_rolling"): 2.8,
        ("console_rolling", "pc_mid"): 0.50,
        ("console_rolling", "pc_later"): 0.08,
        ("console_rolling", "ea"): 0.15,
        ("multi_rolling", "multi_rolling"): 5.0,
        ("multi_rolling", "console_rolling"): 3.0,
        ("multi_rolling", "pc_mid"): 0.45,
        ("multi_rolling", "ea"): 0.20,
    }
    reg_w_shp = {
        ("console_rolling", "console_rolling"): 8.0,
        ("console_rolling", "multi_rolling"): 2.2,
        ("console_rolling", "pc_mid"): 0.08,
        ("console_rolling", "pc_later"): 0.05,
        ("console_rolling", "ea"): 0.08,
        ("multi_rolling", "multi_rolling"): 6.0,
        ("multi_rolling", "console_rolling"): 3.0,
        ("multi_rolling", "pc_mid"): 0.15,
        ("multi_rolling", "ea"): 0.15,
    }

    w_lvl: list[float] = []
    w_shp: list[float] = []
    w_ker: list[float] = []
    for c in pool:
        r = c["regime"]
        wl = reg_w_lvl.get((t_reg, r), 1.0 if t_reg == r else 0.18)
        ws = reg_w_shp.get((t_reg, r), 1.0 if t_reg == r else 0.10)
        if t_reg == "console_rolling":
            wk = {"console_rolling": 6.0, "multi_rolling": 2.2, "pc_mid": 0.55, "pc_later": 0.08, "ea": 0.12}.get(r, 0.12)
        elif t_reg == "multi_rolling":
            wk = {"multi_rolling": 5.5, "console_rolling": 2.8, "pc_mid": 0.35, "pc_later": 0.10, "ea": 0.20}.get(r, 0.15)
        else:
            wk = 3.0 if r == t_reg else 0.30

        # Single-player vs multiplayer audience affinity bonus.
        if tgt["is_sp"] == c["is_sp"]:
            wl *= 1.30
            ws *= 1.30
            wk *= 1.35

        c_sat_lv0 = soft_sat_log_v0(c["log_v0"])
        d_eve_b = (tgt["r_eve"] - c["r_eve"]) / 1.15
        d_v0_b = (t_sat_lv0 - c_sat_lv0) / 2.40
        d_late_b = (tgt["r_late"] - c["r_late"]) / 0.90
        wl *= math.exp(-0.5 * (d_eve_b * d_eve_b + d_v0_b * d_v0_b))
        ws *= math.exp(-0.5 * (d_eve_b * d_eve_b + 0.5 * d_late_b * d_late_b + 0.4 * d_v0_b * d_v0_b))

        d_eve_a = (tgt["r_eve"] - c["r_eve"]) / 0.85
        d_v0_a = (t_sat_lv0 - c_sat_lv0) / 1.40
        d_late_a = (tgt["r_late"] - c["r_late"]) / 0.85
        wk *= math.exp(-0.5 * (d_eve_a * d_eve_a + d_v0_a * d_v0_a + 0.4 * d_late_a * d_late_a))

        w_lvl.append(max(wl, 1e-6))
        w_shp.append(max(ws, 1e-6))
        w_ker.append(max(wk, 1e-6))

    sw_lvl = sum(w_lvl)
    wn_lvl = [w / sw_lvl for w in w_lvl]
    sw_shp = sum(w_shp)
    wn_shp = [w / sw_shp for w in w_shp]
    sw_ker = sum(w_ker)
    wn_ker = [w / sw_ker for w in w_ker]

    n_eff_lvl = 1.0 / sum(w * w for w in wn_lvl)
    n_eff_shp = 1.0 / sum(w * w for w in wn_shp)
    n_eff_ker = 1.0 / sum(w * w for w in wn_ker)
    dof_b = math.sqrt(n_eff_shp / max(1.8, n_eff_shp - 2.2))
    dof_a = math.sqrt(n_eff_ker / max(1.5, n_eff_ker - 1.5))

    # Clamp extreme pre-launch outliers before linear design matrix.
    c_reves = [clamp(c["r_eve"], -1.0, 1.5) for c in pool]
    c_reves_a = [clamp(c["r_eve"], -0.8, 1.4) for c in pool]
    c_lv0s = [soft_sat_log_v0(c["log_v0"]) for c in pool]
    c_rlate = [clamp(c["r_late"], -1.0, 1.0) for c in pool]
    c_cons_roll = [1.0 if c["regime"] == "console_rolling" else 0.0 for c in pool]
    c_pcmid = [1.0 if c["regime"] == "pc_mid" else 0.0 for c in pool]
    c_ea = [1.0 if c["regime"] == "ea" else 0.0 for c in pool]

    m_eve = sum(w * x for w, x in zip(wn_lvl, c_reves))
    m_lv0 = sum(w * x for w, x in zip(wn_lvl, c_lv0s))
    m_cons = sum(w * x for w, x in zip(wn_lvl, c_cons_roll))
    m_pcm = sum(w * x for w, x in zip(wn_lvl, c_pcmid))
    m_ea = sum(w * x for w, x in zip(wn_lvl, c_ea))

    t_eve = clamp(tgt["r_eve"], min(c_reves) - 0.30, max(c_reves) + 0.30) - m_eve
    t_lv0 = clamp(t_sat_lv0, min(c_lv0s) - 0.45, max(c_lv0s) + 0.45) - m_lv0
    t_cons = (1.0 if t_reg == "console_rolling" else 0.0) - m_cons
    t_pcm = (1.0 if t_reg == "pc_mid" else 0.0) - m_pcm
    t_ea = (1.0 if t_reg == "ea" else 0.0) - m_ea

    X_lvl = [
        [1.0, e - m_eve, lv - m_lv0, co - m_cons, pm - m_pcm, ea - m_ea]
        for e, lv, co, pm, ea in zip(c_reves, c_lv0s, c_cons_roll, c_pcmid, c_ea)
    ]
    x_tgt_lvl = [1.0, t_eve, t_lv0, t_cons, t_pcm, t_ea]
    y_A = [c["A"] for c in pool]
    beta_A, A_inv_lvl = solve_ridge_with_inv(X_lvl, y_A, wn_lvl, [1e-6, 0.22, 0.30, 0.40, 0.35, 0.35])
    A_hat = sum(b * xv for b, xv in zip(beta_A, x_tgt_lvl))
    lev_lvl = quad_form(x_tgt_lvl, A_inv_lvl) / max(1.0, n_eff_lvl)

    # Align all comparable trajectories to the target's EDT/EST timezone once for both models.
    shifted_y: list[dict[int, float]] = []
    shifted_S: list[dict[int, float]] = []
    for c in pool:
        sy = shift_dst_curve(c["y"], c["dst"], tgt["dst"])
        sA = sum(sy.values()) / 23.0
        shifted_y.append(sy)
        shifted_S.append({h: sy[h] - sA for h in range(2, 25)})

    ms_eve = sum(w * x for w, x in zip(wn_shp, c_reves))
    ms_lv0 = sum(w * x for w, x in zip(wn_shp, c_lv0s))
    ms_late = sum(w * x for w, x in zip(wn_shp, c_rlate))
    ts_eve = clamp(tgt["r_eve"], min(c_reves) - 0.30, max(c_reves) + 0.30) - ms_eve
    ts_lv0 = clamp(t_sat_lv0, min(c_lv0s) - 0.45, max(c_lv0s) + 0.45) - ms_lv0
    ts_late = clamp(tgt["r_late"], min(c_rlate) - 0.25, max(c_rlate) + 0.25) - ms_late

    X_shp = [
        [1.0, e - ms_eve, lv - ms_lv0, lt - ms_late]
        for e, lv, lt in zip(c_reves, c_lv0s, c_rlate)
    ]
    x_tgt_shp = [1.0, ts_eve, ts_lv0, ts_late]

    mu_b: dict[int, float] = {}
    sig_b: dict[int, float] = {}
    mu_a: dict[int, float] = {}
    sig_a: dict[int, float] = {}
    t_reve_a = clamp(tgt["r_eve"], -0.8, 1.4)

    # --- 2. JOINT HOURLY PASS: Model B (Bayesian Ridge) + Model A (Dual-Anchor Huber Shape Transfer) ---
    for h in range(2, 25):
        yh = [sy[h] for sy in shifted_y]

        # Model B: Decomposed Level + Shape Ridge
        sh = [s[h] for s in shifted_S]
        beta_S, A_inv_shp = solve_ridge_with_inv(X_shp, sh, wn_shp, [1e-6, 0.90, 2.20, 1.30])
        S_hat = sum(b * xv for b, xv in zip(beta_S, x_tgt_shp))

        le = 0.60 * (4.0 if h <= 5 else (1.5 if h <= 9 else 1.0))
        ll = 1.20 * (0.6 if h <= 5 else 2.5)
        beta_dir, _ = solve_ridge_with_inv(X_shp, yh, wn_shp, [1e-6, le, 1.60, ll])
        mu_dir = sum(b * xv for b, xv in zip(beta_dir, x_tgt_shp))

        w_m1 = 0.25 if h == 2 else (0.50 if h == 3 else (0.70 if h == 4 else 1.0))
        mu_b[h] = w_m1 * (A_hat + S_hat) + (1.0 - w_m1) * mu_dir

        mae_b = 0.0
        rms_b = 0.0
        for i in range(len(pool)):
            p1 = sum(b * xv for b, xv in zip(beta_A, X_lvl[i])) + sum(b * xv for b, xv in zip(beta_S, X_shp[i]))
            p2 = sum(b * xv for b, xv in zip(beta_dir, X_shp[i]))
            err = yh[i] - (w_m1 * p1 + (1.0 - w_m1) * p2)
            wi = wn_shp[i]
            mae_b += wi * abs(err)
            rms_b += wi * min(err * err, 0.85 * 0.85)
        base_sig_b = (0.50 * (mae_b * 1.2533141373155) + 0.50 * math.sqrt(rms_b)) * dof_b
        lev_shp = quad_form(x_tgt_shp, A_inv_shp) / max(1.0, n_eff_shp)
        lev_tot = min(1.5, w_m1 * w_m1 * lev_lvl + 0.40 * lev_shp)
        sig_b[h] = math.sqrt(max(0.015, base_sig_b * base_sig_b * (1.0 + lev_tot)))

        # Model A: Non-parametric Dual-Anchor Huber Shape Transfer (ramps Day-1 evening anchor h=6..16)
        alpha = 0.0 if h <= 6 else (0.45 if h >= 16 else 0.45 * (h - 6) / 10.0)
        zs = [yh[i] - alpha * c_reves_a[i] for i in range(len(pool))]
        z_hat = weighted_huber_loc(zs, wn_ker, delta=0.25)
        mu_a[h] = z_hat + alpha * t_reve_a

        mae_a = 0.0
        rms_a = 0.0
        for i, z_val in enumerate(zs):
            err_a = z_val - z_hat
            wi = wn_ker[i]
            mae_a += wi * abs(err_a)
            rms_a += wi * min(err_a * err_a, 0.85 * 0.85)
        sig_a[h] = max(0.12, (0.55 * (mae_a * 1.2533141373155) + 0.45 * math.sqrt(rms_a)) * dof_a)

    # Gentle 3-point diurnal smoothing on smooth afternoon/evening hours (h=10..23) without blurring h=4..7 US unlock.
    mu_a_s = dict(mu_a)
    mu_b_s = dict(mu_b)
    for h in range(10, 24):
        mu_a_s[h] = 0.70 * mu_a[h] + 0.15 * mu_a[h - 1] + 0.15 * mu_a[h + 1]
        mu_b_s[h] = 0.70 * mu_b[h] + 0.15 * mu_b[h - 1] + 0.15 * mu_b[h + 1]
    mu_a, mu_b = mu_a_s, mu_b_s

    # Combine Model A (40%) + Model B (60%) via exact Law of Total Variance (aleatoric + epistemic disagreement).
    w_a = 0.40
    w_b = 1.0 - w_a
    mu_by_h: dict[int, float] = {}
    sig_by_h: dict[int, float] = {}
    for h in range(2, 25):
        mu_by_h[h] = w_a * mu_a[h] + w_b * mu_b[h]
        var_ale = w_a * (sig_a[h] * sig_a[h]) + w_b * (sig_b[h] * sig_b[h])
        diff = mu_a[h] - mu_b[h]
        var_epi = w_a * w_b * (diff * diff)
        sig_by_h[h] = math.sqrt(max(0.015, var_ale + var_epi)) * 1.06

    # --- 3. FINAL-CASE LIVE REFINEMENTS (bounded no-op on practice) ---
    exit_surprise, stream_adj, sat_scale = extract_live_refinements(case, tgt, mu_by_h[2])
    if exit_surprise != 0.0 or stream_adj != 0.0:
        for h in range(2, 25):
            mu_by_h[h] += exit_surprise * 0.75 * math.exp(-0.55 * (h - 2))
            if 2 <= h <= 7:
                # Overnight EU-sleep stream-mix adjustment peaks at h=3..5 and tapers smoothly to 0 by h=8.
                taper = 1.0 if 3 <= h <= 5 else (0.6 if h in (2, 6) else 0.3)
                mu_by_h[h] += stream_adj * taper

    # Smooth spread across adjacent hours and apply gentle horizon balance (slightly wider during
    # the high-variance EU afternoon ramp h=12..17, slightly crisper at near-term h=2..4).
    sig_smooth: dict[int, float] = {}
    for h in range(2, 25):
        nb = [sig_by_h[hh] for hh in range(max(2, h - 1), min(24, h + 1) + 1)]
        h_bal = 0.96 if h <= 4 else (1.06 if 12 <= h <= 17 else 1.00)
        sig_smooth[h] = (0.55 * sig_by_h[h] + 0.45 * (sum(nb) / len(nb))) * h_bal * sat_scale

    # --- 4. CALIBRATED MONOTONIC QUANTILES ---
    hours_out: list[dict[str, Any]] = []
    v0 = tgt["v0"]
    for row in case["forecast"]:
        h = int(row["hour"])
        mu = mu_by_h[h]
        sig = sig_smooth[h]
        qs: list[float | int] = []
        for zt in T_QUANTILES:
            raw = v0 * math.exp(clamp(mu + zt * sig, -12.0, 12.0))
            val = round2(clamp(raw, QUANTILE_MIN, QUANTILE_MAX))
            if qs and val < qs[-1]:
                val = qs[-1]
            qs.append(val)
        hours_out.append({"hour": h, "q": qs})

    return {"hours": hours_out}
