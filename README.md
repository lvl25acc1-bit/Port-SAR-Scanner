

# Port Volume — SAR-Based Port Anchorage Monitoring

<img width="253" height="320" alt="Klein (santos_presentation)" src="https://github.com/user-attachments/assets/63c6b3b5-abdc-4659-80b5-e9d65ac84d9a" />

Sentinel-1 SAR pipeline for monitoring vessel traffic at commodity port anchorage zones. Detects vessels using CFAR, assigns them to spatial zones (anchorage/berth/channel), and correlates activity with economic indicators.

## What this project found

We tested whether satellite radar imagery could produce tradeable signals from port vessel counts. After analyzing 244 scenes at Santos (Brazil's largest grain port) and 349 scenes at Rotterdam across 6 years:

**What works:**
- SAR reliably detects vessels at 10m resolution using CFAR
- Anchorage zone isolation separates signal from noise — berth counts carry no economic information anywhere
- Seasonal port activity patterns are detectable (Santos harvest seasonality: Mann-Whitney p = 8.2e-06)

**What doesn't work:**
- Level correlations between vessel counts and commodity prices are spurious (driven by coincident trends, not causation)
- First-differencing kills all correlations (rho drops from +0.48 to +0.01)
- SAR's 12-day revisit is too slow to predict futures prices — markets already have the information through AIS and broker networks
- Wind explains ~43% of detection variance at Rotterdam

**Honest conclusion:** SAR can measure physical port activity and detect seasonal patterns, but cannot produce alpha on liquid commodity futures. The information asymmetry isn't there for legal trade at well-monitored ports. Potential value exists for sanctioned/dark vessel detection where AIS is unavailable.

## Architecture

```
src/portvolume/
    config.py              # Frozen dataclasses + YAML loader
    pipeline.py            # SARMonitoringPipeline orchestrator
    sites.py               # Site definitions with commodity metadata
    acquisition/           # Sentinel-1 STAC search + COG download
    detection/             # CA-CFAR, adaptive wind correction, water masking
    zones/                 # Anchorage/berth/channel zone assignment
    index/                 # Weekly aggregation + composite indices
    economic/              # FRED, yfinance, commodity-specific indicators
    analysis/              # Correlation, Granger causality, cointegration
    backtest/              # Rotterdam deep-dive validation
    validation/            # Multi-reference scorecard framework
```

## Setup

```bash
pip install -e ".[dev]"
```

### Environment variables

```bash
export FRED_API_KEY="your-fred-api-key"      # https://fred.stlouisfed.org/docs/api/api_key.html
export GFW_API_TOKEN="your-gfw-token"        # https://globalfishingwatch.org/our-apis/
```

## Usage

### Full pipeline

```python
from portvolume.config import load_config
from portvolume.pipeline import SARMonitoringPipeline

config = load_config()
pipeline = SARMonitoringPipeline(config)
pipeline.run_full_pipeline()
```

### Santos analysis

```bash
python scripts/acquire_santos_extended.py    # Download + preprocess + detect
python scripts/run_santos_analysis.py        # Zone metrics + economic correlation
python scripts/santos_extended_correlations.py  # Multi-indicator backtesting
python scripts/generate_paper.py             # IMRaD paper with figures
```

## Key scripts

| Script | Purpose |
|--------|---------|
| `acquire_santos_extended.py` | Download Santos SAR scenes (2020-2026), preprocess, run CFAR |
| `run_santos_analysis.py` | Zone assignment, soy futures correlation, seasonality tests |
| `santos_extended_correlations.py` | BRL/USD, soy meal, Baltic Dry correlations |
| `santos_soy_deepdive.py` | Permutation tests, bootstrap CIs (small sample) |
| `apply_wind_correction.py` | Post-hoc wind deconfounding on Rotterdam data |
| `apply_zone_metrics.py` | Compute congestion index from zone-tagged detections |
| `run_validation_scorecard.py` | Multi-signal validation with pass/fail thresholds |
| `generate_paper.py` | Clean IMRaD PDF with figures |

## Detection pipeline

1. **Acquisition** — STAC search on Planetary Computer, windowed COG download clipped to site bbox
2. **Preprocessing** — Linear power to sigma-naught dB, Lee speckle filter (3x3)
3. **Water masking** — Manual QGIS masks (priority) or automated JRC/EU-Hydro/OSM fallback
4. **CFAR detection** — 2D Cell-Averaging CFAR (Pfa=1e-8) with optional wave-age adaptive threshold
5. **Post-filtering** — Aspect ratio, brightness floor, dB contrast thresholds
6. **Zone assignment** — Spatial join to GeoJSON-defined anchorage/berth/channel polygons

## Statistical methodology (lessons learned)

The analysis pipeline includes robustness checks developed after discovering spurious correlations:

1. **First-difference test** — If differencing kills the correlation, it's trend-driven, not causal
2. **Detrended residuals** — Remove linear trends before correlating
3. **Within-year stability** — If the sign flips between years, the relationship isn't real
4. **Out-of-time validation** — Train on one period, predict another. Negative R-squared = no signal
5. **Benjamini-Hochberg correction** — Applied across all indicator tests
6. **Autocorrelation awareness** — Time series p-values overstate significance; effective n < nominal n

## Configured ports

13 commodity-specific ports in `config/sites_v2.yaml`:

| Port | Commodity | Region |
|------|-----------|--------|
| Ras Tanura | Crude oil | Middle East |
| Fujairah | Crude oil | Middle East |
| Corpus Christi | Crude oil | North America |
| Port Hedland | Iron ore | Asia Pacific |
| Tubarao | Iron ore | South America |
| Richards Bay | Coal | Africa |
| Santos | Grain | South America |
| Paranagua | Grain | South America |
| New Orleans | Grain | North America |
| Ras Laffan | LNG | Middle East |
| Sabine Pass | LNG | North America |
| Rotterdam | Containers (control) | Europe |
| Los Angeles | Containers (control) | North America |

## Tests

```bash
python -m pytest tests/ -v   # 101 tests
```

## Data

All data is excluded from version control (`data/` in `.gitignore`). The pipeline downloads Sentinel-1 scenes from Microsoft Planetary Computer (free, no account needed) and economic data from FRED/Yahoo Finance.

Typical data volume per port: ~40-50 MB per scene, ~2-6 scenes per month.

## License

Research use. Sentinel-1 data is provided by ESA under the Copernicus open access license.
