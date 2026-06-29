# BESS Arbitrage Dashboard — Spec / Save Point

## Current State (2026-06-29)

### Version: v17.3-aligned

App deployed on Streamlit Cloud from GitHub: `https://github.com/apiwatosanch-cyber/bess-dashboard.git`
Local file: `/Users/gaiich/Downloads/bess_dashboard/app.py`

---

## Architecture

7 tabs:
- Tab 1: Product Catalogue (BESS model selector)
- Tab 2: Technical Model (detailed calc_model engine, energy flow chart)
- Tab 3: Arbitrage Summary
- Tab 4: Sensitivity / Scenario
- Tab 5: ☀️ Solar+BESS Bundle (detailed Zero Export model via `calc_solar_bess()`)
- Tab 6: 📐 Bill Sizer (quick estimate via `recommend_sizing()`)
- Tab 7: 🤝 PPA / นักลงทุน (v16 investor model with IRR/NPV)

---

## Key Logic

### Zero Export (Tab 5 — `calc_solar_bess()`)
- Solar split: Direct Load% + BESS Charge% = 100% (no grid export)
- Solar Direct → saves at on-peak rate
- BESS Solar Cycle → solar_bess_in × RTE × deg × pk × realization
- BESS Grid Cycle → bess_kwh × RTE × grid_cycles_day × working_days × deg × arb_margin × realization
- arb_margin = pk − op/RTE

### Bill Sizer Quick Estimate (Tab 6 — `recommend_sizing()`)
- Solar save = kWh × (85% × on_peak + 15% × 2.2 THB/kWh) — v17.3 Excel B43 formula
- Conditional cycles: `2 if solar_mw > 0 else 1` (Thai TOU = 1 on-peak window)
- MEA/PEA toggle → demand rate 74.14 MEA / 132.93 PEA

### Rates
- Peak default: 5.7982 THB/kWh (MEA TOU, updated v17.3)
- Off-Peak default: 2.6107 THB/kWh
- RTE = ηc × ηd = 94% × 94% = 88.4%
- Realization Factor default: 70% (v17.3 aligned)
- Working Days: 247/ปี (จ-ศ ลบวันหยุดนักขัตฤกษ์)
- MEA Demand: 74.14 THB/kW/เดือน | PEA: 132.93 THB/kW/เดือน

### BOI ๓/๒๕๖๘
- Eligible = min(BESS CAPEX, 12M THB × Solar MW)
- CIT exempt 3Y, cap 50% of eligible

---

## Completed (v17.3 comparison done)

### Differences found vs Excel v17.3 — 10 items
| # | Item | Status |
|---|------|--------|
| 1 | MEA/PEA toggle (demand rate 74.14 vs 132.93) | ✅ Done (Tab 5 + Tab 6) |
| 2 | Bill Sizer solar save = 0.85×peak + 0.15×2.2 | ✅ Done |
| 3 | Realization Factor default = 70% | ✅ Done |
| 4 | Peak/Off-Peak updated to 5.7982/2.6107 | ✅ Done |
| 5 | IDC 3% added to CAPEX (Tab 5) | ✅ Done — sidebar slider 0-6%, applied to total_capex before OM |
| 6 | Solar PR = 0.82 (Tab 5) | ✅ Done — sidebar slider 0.70-1.00, applied to solar_kwh_y1 |
| 7 | BESS Availability = 0.96 (Tab 5) | ✅ Done — sidebar slider 0.85-1.00, applied to both BESS cycles |
| 8 | Insurance 0.4%/yr | Not implemented — use OM% slider as proxy |
| 9 | Augmentation Y11 = 30% (was 20%) | ✅ Done in PPA tab |
| 10 | I-REC revenue (100 THB/MWh × 90%) | ✅ Done — toggle OFF by default, Solar kWh × 90% × price/1000 |

---

## Engineering Assumptions v17.3 (new sidebar section in Tab 5)
- Solar PR: default 0.82 — reduces solar_kwh_y1
- BESS Availability: default 0.96 — reduces both solar_bess_out and bess_grid_out
- IDC: default 3% of (Solar+BESS CAPEX) — added to total_capex before OM calc
- I-REC: default OFF — revenue = solar_kwh × 0.90 × price(THB/MWh)/1000

## PPA Tab
- Augmentation Y11 = 30% of BESS CAPEX (was 20%) — conservative LFP assumption

---

## Files
- `app.py` — main Streamlit app (~1750+ lines)
- `BESS_Dashboard_UserGuide_EN.pdf` in Downloads — English PDF user guide
- `gunkul_logo_crop.png` in Downloads — Gunkul logo
