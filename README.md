# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ricoxyz/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ricoxyz/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Probabilistic hour-by-hour Twitch concurrent viewer (CCU) forecaster for **Grand Theft Auto VI** launch day (`2026-11-19` UTC), built in Python 3.13 using only the standard library.

## Approach

1. **Dynamic Comparable Learning**: Reads all comparable launch trajectories provided inside each `<id>.in` case file (38 in practice, all 42 in the final) and filters active launches without hardcoded game answers.
2. **DST-Aligned Rolling-Midnight Physics**: Computes exact US Daylight Saving Time (EDT vs EST) from `day_start` to align the overnight EU-sleep trough and US East midnight unlock jump (`04:00Z` during EDT vs `05:00Z` during EST on November 19).
3. **Level + Zero-Mean Shape Decomposition**: Separates each launch's 23-hour log-ratio curve `ln(v(h) / v(1))` into a daily level shift `A` (modeled via similarity-weighted ridge regression on pre-launch evening amplitude, soft-saturated `ln(v1)` scale, platform, and unlock structure across all active launches) and a zero-mean diurnal/unlock shape `S(h)` (estimated from archetype-matched rolling launches).
4. **Calibrated Student-t Log Quantiles**: Produces monotonic 5%, 25%, 50%, 75%, and 95% quantiles using robust finite-sample MAE/RMS scale blending with temporal smoothing and extrapolation uncertainty.
5. **Final-Case Live Signals**: When minute-level Hour-1 reads, top-20 live stream metadata, or hourly Twitch top-20 category totals are present (final case `h00`), applies bounded end-of-hour exit velocity carry-over and stream/saturation refinements while remaining a deterministic no-op when absent.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
