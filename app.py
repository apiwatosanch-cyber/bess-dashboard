import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import brentq

st.set_page_config(page_title="BESS Arbitrage Dashboard", layout="wide", page_icon="⚡")

# ── Loan helper ───────────────────────────────────────────────────────────────
def calc_loan_cashflows(investment, loan_pct, loan_rate, loan_years):
    """Returns (equity_outflow, annual_pmt, schedule_dict)"""
    financed  = investment * loan_pct
    equity    = investment * (1 - loan_pct)
    if loan_pct == 0 or loan_rate == 0:
        return equity, 0.0, {}
    r = loan_rate
    pmt = financed * r / (1 - (1 + r) ** (-loan_years))
    schedule = {}
    balance = financed
    for y in range(1, loan_years + 1):
        interest  = balance * r
        principal = pmt - interest
        balance  -= principal
        schedule[y] = {"pmt": pmt, "interest": interest, "principal": principal}
    return equity, pmt, schedule

def apply_loan_to_cf(df_net_col, investment, loan_pct, loan_rate, loan_years):
    """Overlay loan repayment onto operating net-CF list → return adjusted cum-CF series."""
    equity, pmt, schedule = calc_loan_cashflows(investment, loan_pct, loan_rate, loan_years)
    cum = -equity
    cum_series = []
    for i, net_op in enumerate(df_net_col, start=1):
        debt_service = schedule[i]["pmt"] if i in schedule else 0.0
        cum += (net_op - debt_service)
        cum_series.append(cum)
    return cum_series, equity, pmt

# ── Solar+BESS calculation engine ─────────────────────────────────────────────
def calc_solar_bess(
    solar_kwp, solar_capex_per_mw, solar_cuf, solar_deg_pct,
    bess_kwh, bess_capex_per_mwh,
    bess_deg_curve,          # list[30] — degradation from product, or flat if custom
    peak_rate_y1, off_peak_rate_y1, tou_esc,
    demand_cut_kw, demand_rate_per_kw_mo,
    rte, cycles_day, days_year,
    om_pct, bundled_om_years,
    cit, wacc,
    use_boi,                 # True = ใช้ BOI ๓/๒๕๖๘ (ต้องมี Solar)
    realization_factor,
    # ── v14 Solar split (Operation Ratio) — Zero Export ──────────────────────
    solar_direct_pct=0.60,   # Solar → Direct Load offset (ช่วง 09-17h)
    solar_bess_pct=0.40,     # Solar → BESS charge (ปล่อยช่วงเย็น 17-22h); direct+bess=100%
    grid_cycles_day=1.0,     # Grid OP-charged cycles/day (อาจเป็น 0 ถ้าไม่ charge จาก grid)
    working_days=247,        # วันที่ grid arb ทำได้ (จ-ศ − หยุดนักขัตฤกษ์)
):
    solar_mw  = solar_kwp / 1000
    bess_mwh  = bess_kwh  / 1000

    solar_capex = solar_mw  * solar_capex_per_mw
    bess_capex  = bess_mwh  * bess_capex_per_mwh
    total_capex = solar_capex + bess_capex

    annual_om        = total_capex * om_pct
    bundled_om_total = annual_om * bundled_om_years
    total_investment = total_capex + bundled_om_total

    # BOI ๓/๒๕๖๘ — BESS + Solar bundle
    # Eligible cap = min(BESS CAPEX, 12,000,000 THB per MW of Solar PV)
    # CIT exempt = 3 ปี, cap = 50% of eligible
    boi_eligible_cap_per_mw = 12_000_000   # 12M THB per MW of Solar PV
    if use_boi and solar_mw > 0:
        boi_eligible = min(bess_capex, boi_eligible_cap_per_mw * solar_mw)
        boi_cap = boi_eligible * 0.50       # 50% cap
        boi_years = 3
    else:
        boi_cap   = 0.0
        boi_years = 0

    rows = []
    cum_cf  = -total_investment
    cum_exempt = 0.0

    solar_kwh_y1 = solar_mw * 8760 * solar_cuf * 1000   # kWh/year at Y1

    for y in range(1, 31):
        sol_deg   = (1 - solar_deg_pct) ** (y - 1)
        bess_deg  = bess_deg_curve[y - 1]
        pk  = peak_rate_y1     * (1 + tou_esc) ** (y - 1)
        op  = off_peak_rate_y1 * (1 + tou_esc) ** (y - 1)
        avg = (pk + op) / 2

        # --- Revenue streams (v14 aligned) ---
        solar_kwh_y = solar_kwh_y1 * sol_deg

        # Solar revenue (Zero Export — ไฟทั้งหมดใช้ใน site):
        # (1) Direct Load offset — save on-peak rate ช่วง 09:00-17:00
        # (2) BESS Charge portion — counted in BESS Solar Cycle below
        # direct_pct + bess_pct = 100% (no grid export)
        solar_direct_kwh  = solar_kwh_y * solar_direct_pct
        solar_direct_save = solar_direct_kwh * pk * realization_factor
        solar_save        = solar_direct_save  # no export revenue

        # BESS revenue แยก 2 cycles (v14: Solar Cycle + Grid OP Cycle):
        # Solar Cycle — ชาร์จฟรีจาก Solar, ปล่อย 17-22h on-peak
        #   Input: solar_bess_pct × solar_kwh → RTE → to load
        solar_bess_in  = solar_kwh_y * solar_bess_pct          # kWh/yr into BESS from solar
        solar_bess_out = solar_bess_in * rte * bess_deg        # kWh/yr delivered (RTE + deg)
        bess_solar_rev = solar_bess_out * pk * realization_factor  # margin = pk (no charge cost)

        # Grid Cycle — ชาร์จ off-peak ราคาถูก, ปล่อย morning on-peak
        arb_margin      = pk - (op / rte)                      # spread
        bess_grid_out   = bess_kwh * (rte) * grid_cycles_day * working_days * bess_deg
        bess_grid_rev   = bess_grid_out * max(arb_margin, 0) * realization_factor

        bess_arb  = bess_solar_rev + bess_grid_rev
        bess_kwh_y = solar_bess_out + bess_grid_out  # total BESS output for display

        demand_save  = demand_cut_kw * demand_rate_per_kw_mo * 12

        om_cost = 0.0 if y <= bundled_om_years else annual_om
        pre_tax = solar_save + bess_arb + demand_save - om_cost

        full_cit = max(0.0, pre_tax * cit)
        if cum_exempt < boi_cap and y <= boi_years:
            exempt   = min(max(0.0, pre_tax * cit), boi_cap - cum_exempt)
            cum_exempt += exempt
            tax_paid = full_cit - exempt
        else:
            exempt   = 0.0
            tax_paid = full_cit

        net_cf  = pre_tax - tax_paid
        cum_cf += net_cf

        rows.append({
            "Year": y,
            "Sol Deg": sol_deg,
            "BESS Deg": bess_deg,
            "Solar kWh": solar_kwh_y,
            "Solar Direct kWh": solar_direct_kwh,
            "Solar BESS kWh": solar_bess_in,
            "Solar Direct Save (THB)": solar_direct_save,
            "Solar Save (THB)": solar_save,
            "BESS Solar Out kWh": solar_bess_out,
            "BESS Grid Out kWh": bess_grid_out,
            "BESS kWh Out": bess_kwh_y,
            "BESS Solar Rev (THB)": bess_solar_rev,
            "BESS Grid Rev (THB)": bess_grid_rev,
            "BESS Arb (THB)": bess_arb,
            "Demand Save (THB)": demand_save,
            "OM Cost (THB)": om_cost,
            "Pre-Tax CF (THB)": pre_tax,
            "Full CIT (THB)": full_cit,
            "BOI Exempt (THB)": exempt,
            "Tax Paid (THB)": tax_paid,
            "Net CF (THB)": net_cf,
            "Cum CF (THB)": cum_cf,
            "Cum Exempt (THB)": cum_exempt,
        })

    df = pd.DataFrame(rows)

    def payback_yr():
        idx = df[df["Cum CF (THB)"] >= 0].index
        if len(idx) == 0:
            return None
        y = int(df.loc[idx[0], "Year"])
        if y == 1:
            return 1
        prev = float(df.loc[idx[0]-1, "Cum CF (THB)"])
        curr = float(df.loc[idx[0],   "Cum CF (THB)"])
        return (y - 1) + (-prev / (curr - prev))

    def _irr(horizon):
        cfs = [-total_investment] + list(df[df["Year"] <= horizon]["Net CF (THB)"])
        return safe_irr(cfs)

    def _npv(horizon, r):
        cfs = [-total_investment] + list(df[df["Year"] <= horizon]["Net CF (THB)"])
        return sum(c / (1+r)**i for i, c in enumerate(cfs))

    metrics = {
        "payback":  payback_yr(),
        "irr_15":   _irr(15),
        "irr_20":   _irr(20),
        "irr_30":   _irr(30),
        "npv_30":   _npv(30, wacc),
        "cum_15":   float(df[df["Year"]==15]["Cum CF (THB)"].values[0]),
        "cum_30":   float(df[df["Year"]==30]["Cum CF (THB)"].values[0]),
        "total_investment": total_investment,
        "solar_capex": solar_capex,
        "bess_capex": bess_capex,
        "boi_cap": boi_cap,
        "annual_om": annual_om,
    }
    return df, metrics

# ── Product catalogue ─────────────────────────────────────────────────────────
PRODUCTS = {
    "SG257 (Sungrow 257 kWh)": {
        "unit_kwh": 257, "dealer_usd_kwh": 160, "duration_hr": 2,
        "deg": [0.9589,0.9352,0.9155,0.898,0.8819,0.867,0.8528,0.8394,0.8265,0.8141,
                0.8021,0.7905,0.7792,0.7682,0.7575,0.7465,0.7355,0.7245,0.7135,0.7025,
                0.6915,0.6805,0.6695,0.6585,0.6475,0.6365,0.6255,0.6145,0.6035,0.5925],
    },
    "SG514 (Sungrow 514 kWh)": {
        "unit_kwh": 514, "dealer_usd_kwh": 170, "duration_hr": 4,
        "deg": [0.9596,0.9364,0.917,0.8998,0.8841,0.8694,0.8555,0.8423,0.8296,0.8175,
                0.8057,0.7943,0.7833,0.7725,0.762,0.7512,0.7404,0.7296,0.7188,0.708,
                0.6972,0.6864,0.6756,0.6648,0.654,0.6432,0.6324,0.6216,0.6108,0.6],
    },
    "HW241 (Huawei 241 kWh)": {
        "unit_kwh": 241, "dealer_usd_kwh": 165, "duration_hr": 2,
        "deg": [0.9424,0.9156,0.895,0.877,0.8605,0.8449,0.83,0.8157,0.8018,0.7883,
                0.7751,0.7622,0.7496,0.7372,0.725,0.7126,0.7002,0.6878,0.6754,0.663,
                0.6506,0.6382,0.6258,0.6134,0.601,0.5886,0.5762,0.5638,0.5514,0.539],
    },
    "SE197 (SolarEdge 197 kWh)": {
        "unit_kwh": 197, "dealer_usd_kwh": 170, "duration_hr": 2,
        "deg": [0.9424,0.9156,0.895,0.877,0.8605,0.8449,0.83,0.8157,0.8018,0.7883,
                0.7751,0.7622,0.7496,0.7372,0.725,0.7126,0.7002,0.6878,0.6754,0.663,
                0.6506,0.6382,0.6258,0.6134,0.601,0.5886,0.5762,0.5638,0.5514,0.539],
    },
}

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DEFAULT_BILL = {
    "days":    [31,28,31,30,31,30,31,31,30,31,30,31],
    "on_peak": [20000,18000,20000,19000,20000,19000,20000,20000,19000,20000,19000,20000],
    "off_peak":[12927,13034,13034,15938,14749,15237,15938,14749,15237,15938,14749,15237],
    "holiday": [0]*12,
    "kw_peak": [100,110,110,115,117,107,115,117,107,115,117,107],
}

def fmt_thb(v):
    if abs(v) >= 1e6:
        return f"฿{v/1e6:,.2f}M"
    return f"฿{v:,.0f}"

def safe_irr(cashflows):
    try:
        cf = np.array(cashflows, dtype=float)
        if cf[0] >= 0:
            return None
        def npv_fn(r): return np.sum(cf / (1+r)**np.arange(len(cf)))
        if npv_fn(0) <= 0:
            return None
        return brentq(npv_fn, -0.5, 5.0, maxiter=200)
    except Exception:
        return None

def calc_model(p, ass, units):
    cap_kwh = p["unit_kwh"] * units
    equip_cost_thb_kwh = p["dealer_usd_kwh"] * ass["fx"]
    cust_price_thb_kwh = equip_cost_thb_kwh / (1 - ass["margin"])
    equip_capex = cust_price_thb_kwh * cap_kwh
    annual_om = equip_capex * ass["om_pct"]
    bundled_om_total = annual_om * ass["bundled_om_years"]
    customer_price = equip_capex + bundled_om_total
    boi_duty_saving = equip_capex * ass["boi_duty_pct"]
    net_capex = customer_price - boi_duty_saving

    # ── BOI ๓/๒๕๖๘ ─────────────────────────────────────────────────────────────
    # เงื่อนไข: BESS ต้องติดคู่กับ Solar PV เพื่อขอ BOI
    # Eligible = min(BESS CAPEX, 12M THB × MW ของ Solar PV)
    # CIT ยกเว้น 3 ปี, วงเงินสูงสุด 50% ของ eligible investment
    solar_ref_mw = ass.get("solar_ref_mw", 0.0)
    use_boi      = ass.get("use_boi", False) and solar_ref_mw > 0
    if use_boi:
        boi_eligible = min(equip_capex, 12_000_000 * solar_ref_mw)
        boi_cap      = boi_eligible * 0.50   # 50% cap
        boi_years    = 3
    else:
        boi_eligible = 0.0
        boi_cap      = 0.0
        boi_years    = 0

    # ── แยก charge / discharge efficiency ──────────────────────────────────────
    eta_c   = ass["charge_eff"]        # ηc: Grid → Battery
    eta_d   = ass["discharge_eff"]     # ηd: Battery → Load
    rte     = eta_c * eta_d            # RTE = ηc × ηd (auto)
    aux_kw  = ass["aux_kw"]            # Parasitic load (BMS, cooling) kW

    # ── TOU Calendar ────────────────────────────────────────────────────────────
    working_days = ass["working_days"]   # วัน จ-ศ ลบวันหยุดนักขัตฤกษ์

    # ── 2-Cycle Mode ────────────────────────────────────────────────────────────
    dual_cycle    = ass.get("dual_cycle", False)
    solar_days    = ass.get("solar_days", 310)       # วันที่มีแดดพอชาร์จ (Cycle 1)
    solar_real    = ass.get("solar_realization", 0.8) # realization factor ของ Solar cycle

    years = list(range(1, 31))
    rows = []
    cum_nb  = -customer_price
    cum_boi = -customer_price
    cum_exempt_boi = 0.0

    for y in years:
        deg = p["deg"][y-1]
        on_peak  = ass["on_peak_y1"]  * (1 + ass["tou_esc"]) ** (y-1)
        off_peak = ass["off_peak_y1"] * (1 + ass["tou_esc"]) ** (y-1)

        # ── Cycle 2: Grid Arb (จ-ศ เท่านั้น) ───────────────────────────────────
        # ชาร์จ off-peak ราคาถูก → ปล่อยช่วงเช้า on-peak
        # Margin = on_peak − off_peak/RTE  (spread ปกติ)
        arb_margin   = on_peak - (off_peak / rte)
        c2_energy_out = cap_kwh * eta_d * working_days * deg
        c2_revenue    = c2_energy_out * arb_margin

        # ── Cycle 1: Solar (วันที่มีแดด) ────────────────────────────────────────
        # ชาร์จฟรีจาก Solar → ปล่อย 18:00-22:00 on-peak
        # Margin = on_peak เต็ม (ไม่มีต้นทุนชาร์จ)
        if dual_cycle:
            c1_energy_out = cap_kwh * eta_d * solar_days * deg
            c1_revenue    = c1_energy_out * on_peak * solar_real
        else:
            c1_energy_out = 0.0
            c1_revenue    = 0.0

        total_energy_out = c2_energy_out + c1_energy_out
        gross_saving     = c2_revenue + c1_revenue

        # Aux load cost: kW × 8,760 h × THB/kWh = THB  (ไม่หาร 1000)
        aux_cost = aux_kw * 8760 * off_peak

        om_cost = (0.0 if y <= ass["bundled_om_years"] else annual_om) + aux_cost
        pre_tax = gross_saving - om_cost

        tax_nb = max(0.0, pre_tax * ass["cit"])

        # BOI ๓/๒๕๖๘ — single scheme, 3Y, 50% cap
        if use_boi and cum_exempt_boi < boi_cap and y <= boi_years:
            exempt_boi = min(max(0.0, pre_tax * ass["cit"]), boi_cap - cum_exempt_boi)
            cum_exempt_boi += exempt_boi
            tax_boi = max(0.0, pre_tax * ass["cit"] - exempt_boi)
        else:
            exempt_boi = 0.0
            tax_boi    = tax_nb

        net_nb  = pre_tax - tax_nb
        net_boi = pre_tax - tax_boi
        cum_nb  += net_nb
        cum_boi += net_boi

        # Energy waterfall (Cycle 2 grid-side only)
        e_in_grid   = cap_kwh / eta_c * working_days * deg
        e_chg_loss  = e_in_grid - (cap_kwh * working_days * deg)
        e_stored    = cap_kwh * working_days * deg
        e_dchg_loss = e_stored - c2_energy_out

        rows.append({
            "Year": y, "Degradation": deg,
            "Working Days (Arb)": working_days,
            "Solar Days (C1)": solar_days if dual_cycle else 0,
            "C1 Solar Energy Out (kWh)": c1_energy_out,
            "C1 Solar Revenue (THB)": c1_revenue,
            "C2 Grid Energy Out (kWh)": c2_energy_out,
            "C2 Arb Revenue (THB)": c2_revenue,
            "E In from Grid (kWh)": e_in_grid,
            "Charging Loss (kWh)": e_chg_loss,
            "E Stored (kWh)": e_stored,
            "Discharging Loss (kWh)": e_dchg_loss,
            "Energy Out (kWh)": total_energy_out,
            "RTE (ηc×ηd)": rte,
            "On-Peak (THB/kWh)": on_peak,
            "Off-Peak (THB/kWh)": off_peak,
            "Arb Margin C2 (THB/kWh)": arb_margin,
            "Gross Saving": gross_saving,
            "Aux Cost": aux_cost,
            "OM Cost": om_cost,
            "Pre-Tax CF": pre_tax,
            "Tax Non-BOI": tax_nb,
            "BOI Exempt": exempt_boi,
            "Tax BOI ๓/๒๕๖๘": tax_boi,
            "Net CF Non-BOI": net_nb,
            "Net CF BOI ๓/๒๕๖๘": net_boi,
            "Cum CF Non-BOI": cum_nb,
            "Cum CF BOI ๓/๒๕๖๘": cum_boi,
        })

    df = pd.DataFrame(rows)

    def payback(col):
        s = df[col]
        idx = s[s >= 0].index
        if len(idx) == 0:
            return None
        y = int(df.loc[idx[0], "Year"])
        if y == 1:
            return 1
        prev = float(df.loc[idx[0]-1, col])
        curr = float(df.loc[idx[0],   col])
        return (y - 1) + (-prev / (curr - prev))

    def irr_at(col, yr):
        cfs = [-customer_price] + list(df[df["Year"] <= yr][col])
        return safe_irr(cfs)

    def npv_at(col, yr, wacc):
        cfs = [-customer_price] + list(df[df["Year"] <= yr][col])
        return sum(c/(1+wacc)**i for i, c in enumerate(cfs))

    metrics = {}
    for tag, col in [("NB","Net CF Non-BOI"), ("BOI","Net CF BOI ๓/๒๕๖๘")]:
        cum_c = col.replace("Net CF", "Cum CF")
        metrics[tag] = {
            "payback": payback(cum_c),
            "irr_15":  irr_at(col, 15),
            "irr_20":  irr_at(col, 20),
            "irr_30":  irr_at(col, 30),
            "npv_30":  npv_at(col, 30, ass["wacc"]),
            "cum_15":  float(df[df["Year"]==15][cum_c].values[0]),
            "cum_30":  float(df[df["Year"]==30][cum_c].values[0]),
        }

    # Leasing
    n_months = ass["lease_months"]
    r_mo = ass["lease_rate"] / 12
    down = customer_price * ass["lease_down"]
    financed = customer_price - down
    pmt = financed * r_mo / (1 - (1+r_mo)**(-n_months))
    annual_pmt = pmt * 12

    lease = {
        "down": down, "financed": financed,
        "monthly_pmt": pmt, "annual_pmt": annual_pmt,
    }

    ppa_cust_cum30  = float(df["Cum CF BOI ๓/๒๕๖๘"].iloc[-1]) * ass["ppa_share"]
    ppa_owner_cum30 = float(df["Cum CF BOI ๓/๒๕๖๘"].iloc[-1]) * (1 - ass["ppa_share"])

    return {
        "df": df,
        "customer_price": customer_price,
        "net_capex": net_capex,
        "equip_capex": equip_capex,
        "annual_om": annual_om,
        "boi_duty_saving": boi_duty_saving,
        "boi_eligible": boi_eligible,
        "boi_cap": boi_cap,
        "metrics": metrics,
        "lease": lease,
        "ppa_cust_cum30": ppa_cust_cum30,
        "ppa_owner_cum30": ppa_owner_cum30,
        "cap_kwh": cap_kwh,
    }


# ── Sizing Recommendation Engine (v14 aligned) ────────────────────────────────
import math as _math

def recommend_sizing(
    annual_on_peak_kwh, annual_off_peak_kwh, peak_kw,
    on_peak_rate, off_peak_rate, demand_rate_per_kw_mo,
    cuf=0.16, working_days=247, rte=0.88,
    solar_capex_per_mw=20_000_000,
    bess_capex_per_mwh=11_000_000,   # Turnkey AC ~10-12M/MWh (v14 default)
    roof_area_m2=0,                   # 0 = no limit; 1 MW ≈ 6,500 m²
    self_consumption_target=0.90,     # Zero Export — Solar ทั้งหมดใช้ใน site
    demand_reduction_target=0.30,     # BESS shave 30% of peak demand
    cit=0.20,
):
    """
    v14 Bill Sizer logic:
    Section 5: Solar sizing from daytime load + self-consumption target + roof limit
    Section 6: BESS scoring (On/Off ratio + Peak/Avg ratio) → score-based sizing
    Section 7: Combined package with BOI tax shield
    """
    DAYTIME_FRAC = 0.60   # 60% ของ on-peak เป็นช่วง 09-17h (solar hours)
    SOLAR_AREA_PER_MW = 6500  # m² per MW

    # Annual bill estimate
    energy_cost = annual_on_peak_kwh * on_peak_rate + annual_off_peak_kwh * off_peak_rate
    demand_cost = peak_kw * demand_rate_per_kw_mo * 12
    ft_service  = energy_cost * 0.08
    annual_bill = energy_cost + demand_cost + ft_service

    # Avg kW during on-peak hours (working days × 13h)
    avg_kw = annual_on_peak_kwh / max(working_days * 13, 1)

    # ── SECTION 5: Solar sizing ──────────────────────────────────────────────
    annual_daytime_load = annual_on_peak_kwh * DAYTIME_FRAC        # kWh/yr
    solar_prod_required = annual_daytime_load / self_consumption_target
    solar_mw_from_load  = solar_prod_required / max(cuf * 8760 * 1000, 1)
    solar_mw_from_roof  = roof_area_m2 / SOLAR_AREA_PER_MW if roof_area_m2 > 0 else None
    solar_mw = min(solar_mw_from_load, solar_mw_from_roof) if solar_mw_from_roof else solar_mw_from_load
    solar_mw = max(solar_mw, 0.001)

    solar_kwh_yr = solar_mw * 1000 * cuf * 8760
    # Solar save: Zero Export — 100% offset ค่าไฟ on-peak rate ทั้งหมด
    solar_save = solar_kwh_yr * on_peak_rate
    solar_capex = solar_mw * solar_capex_per_mw
    solar_pb    = solar_capex / solar_save if solar_save > 0 else None

    # ── SECTION 6: BESS scoring ──────────────────────────────────────────────
    on_off_ratio  = annual_on_peak_kwh / max(annual_off_peak_kwh, 1)
    peak_avg_ratio = peak_kw / max(avg_kw, 1)

    score = 0.0
    if on_off_ratio >= 1.5:   score += 1.5
    elif on_off_ratio >= 1.0: score += 1.0
    if peak_avg_ratio >= 1.5:   score += 1.5
    elif peak_avg_ratio >= 1.3: score += 1.0

    if score >= 3:   bess_rec_label = "🟢 BESS แนะนำอย่างยิ่ง"
    elif score >= 2: bess_rec_label = "🔵 BESS แนะนำ"
    elif score >= 1.5: bess_rec_label = "🟡 BESS Optional"
    else:            bess_rec_label = "🔴 BESS ไม่ค่อยคุ้ม (load flat)"

    # BESS sizing
    bess_mw_from_peak  = peak_kw * demand_reduction_target / 1000
    bess_mw_from_solar = solar_mw * 0.40   # store 40% of solar for evening shift
    bess_mw = max(bess_mw_from_peak, bess_mw_from_solar)
    bess_mw = _math.ceil(bess_mw * 4) / 4  # round up to 0.25 MW

    bess_hr  = 3 if score >= 3 else 2       # 3h if strong, 2h otherwise
    bess_mwh = bess_mw * bess_hr

    bess_capex = bess_mwh * bess_capex_per_mwh

    # BESS savings (v14: 2 cycles/day × 247d × spread × 70% realization)
    spread      = on_peak_rate - off_peak_rate / rte
    bess_arb_save  = bess_mwh * max(spread, 0) * working_days * 2 * 0.70  # 2 cycles
    demand_kw_cut  = bess_mw * 1000 * demand_reduction_target
    demand_save    = demand_kw_cut * demand_rate_per_kw_mo * 12 * 0.70
    bess_total_save = bess_arb_save + demand_save
    bess_pb = bess_capex / bess_total_save if bess_total_save > 0 else None

    # ── SECTION 7: Combined ──────────────────────────────────────────────────
    total_capex = solar_capex + bess_capex
    total_save  = solar_save + bess_total_save
    pb_no_boi   = total_capex / total_save if total_save > 0 else None

    boi_eligible    = min(bess_capex, 12_000_000 * solar_mw)
    boi_exempt      = boi_eligible * 0.50
    boi_tax_shield  = boi_exempt * cit       # actual tax saved (3Y)
    effective_capex = total_capex - boi_tax_shield
    pb_with_boi     = effective_capex / total_save if total_save > 0 else None
    bill_reduction  = total_save / max(annual_bill, 1)

    # Pitch messages
    solar_pitch = (
        f"Solar {solar_mw:.2f} MW → ผลิต {solar_kwh_yr:,.0f} kWh/ปี "
        f"ประหยัด {fmt_thb(solar_save)}/ปี (Payback ~{solar_pb:.1f}Y)"
        if solar_pb else f"Solar {solar_mw:.2f} MW → {fmt_thb(solar_save)}/ปี"
    )
    bess_pitch = (
        f"BESS {bess_mw:.2f} MW / {bess_mwh:.1f} MWh ({bess_hr}h) → "
        f"{fmt_thb(bess_total_save)}/ปี — {bess_rec_label}"
    )
    combined_pitch = (
        f"Solar+BESS Bundle: ลงทุน {fmt_thb(total_capex)} | "
        f"ประหยัด {fmt_thb(total_save)}/ปี | "
        f"ลดบิล {bill_reduction*100:.0f}% | "
        f"Payback {f'{pb_with_boi:.1f}Y (BOI)' if pb_with_boi else '>30Y'}"
    )

    return {
        # Solar
        "solar_mw": solar_mw, "solar_mw_from_load": solar_mw_from_load,
        "solar_mw_from_roof": solar_mw_from_roof,
        "solar_kwh_yr": solar_kwh_yr, "solar_save": solar_save,
        "solar_capex": solar_capex, "solar_pb": solar_pb,
        # BESS
        "score": score, "bess_rec_label": bess_rec_label,
        "on_off_ratio": on_off_ratio, "peak_avg_ratio": peak_avg_ratio, "avg_kw": avg_kw,
        "bess_mw": bess_mw, "bess_mwh": bess_mwh, "bess_hr": bess_hr,
        "bess_mw_from_peak": bess_mw_from_peak, "bess_mw_from_solar": bess_mw_from_solar,
        "bess_capex": bess_capex, "bess_arb_save": bess_arb_save,
        "demand_kw_cut": demand_kw_cut, "demand_save": demand_save,
        "bess_total_save": bess_total_save, "bess_pb": bess_pb,
        # Combined
        "total_capex": total_capex, "total_save": total_save,
        "pb_no_boi": pb_no_boi, "pb_with_boi": pb_with_boi,
        "boi_eligible": boi_eligible, "boi_exempt": boi_exempt,
        "boi_tax_shield": boi_tax_shield, "annual_bill": annual_bill,
        "bill_reduction": bill_reduction,
        # Pitches
        "solar_pitch": solar_pitch, "bess_pitch": bess_pitch,
        "combined_pitch": combined_pitch,
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ BESS Arbitrage")
    st.caption("กรอกข้อมูลด้านล่าง — ผลคำนวณอัปเดตทันที")

    # ── Product ──
    st.subheader("🔋 สินค้า BESS")
    selected_product = st.selectbox("เลือก Product", list(PRODUCTS.keys()))
    n_units = st.number_input("จำนวน Unit", min_value=1, max_value=100, value=1)
    p = PRODUCTS[selected_product]
    cap_kwh_display = p["unit_kwh"] * n_units
    st.info(f"System Capacity: **{cap_kwh_display:,} kWh** ({p['duration_hr']}hr)")

    st.divider()

    # ── TOU Tariff ──
    st.subheader("⚡ TOU Tariff")
    on_peak_y1  = st.number_input("On-Peak Y1 (THB/kWh)", value=4.1839, step=0.01, format="%.4f")
    off_peak_y1 = st.number_input("Off-Peak Y1 (THB/kWh)", value=2.6037, step=0.01, format="%.4f")
    tou_esc     = st.slider("TOU Escalation (%/ปี)", 0.0, 5.0, 2.0, 0.1) / 100

    st.divider()

    # ── Operating ──
    st.subheader("⚙️ Operating Parameters")
    cycles_day  = st.slider("Cycles/Day", 0.5, 3.0, 1.0, 0.25)
    om_pct      = st.slider("OM % of CAPEX/ปี (Y11+)", 0.5, 3.0, 1.5, 0.1) / 100
    bundled_om_years = st.number_input("Bundled OM ปี (รวมในราคา)", min_value=1, max_value=15, value=10)

    st.markdown("**🗓️ TOU Calendar — วันที่ Arbitrage ได้จริง**")
    st.caption("ไทย: จันทร์–ศุกร์ 09:00–22:00 เท่านั้น เสาร์/อาทิตย์/หยุดนักขัตฤกษ์ = Off-Peak ทั้งวัน")
    weekdays_yr   = st.number_input("วันจันทร์–ศุกร์/ปี", min_value=240, max_value=262, value=261)
    pub_holidays  = st.number_input("วันหยุดนักขัตฤกษ์/ปี (ตกวันทำการ)", min_value=0, max_value=20, value=14)
    working_days  = int(weekdays_yr - pub_holidays)
    st.info(f"วันที่ Arbitrage ได้จริง: **{working_days} วัน/ปี** (vs 365 วัน = overestimate {365-working_days} วัน)")

    st.markdown("**⚡ Energy Loss (แยก Charge / Discharge)**")
    charge_eff    = st.slider("Charging Efficiency ηc (%)", 90, 99, 94, 1) / 100
    discharge_eff = st.slider("Discharging Efficiency ηd (%)", 90, 99, 94, 1) / 100
    rte_calc      = charge_eff * discharge_eff
    aux_kw        = st.number_input("Aux Load — BMS/Cooling (kW)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    st.info(f"RTE = ηc × ηd = {charge_eff*100:.0f}% × {discharge_eff*100:.0f}% = **{rte_calc*100:.1f}%** | Aux: {aux_kw} kW × 8,760 ชม./ปี")
    rte = rte_calc
    days_year = 365

    st.divider()
    st.subheader("☀️ 2-Cycle Mode")
    st.caption("Cycle 1: Solar → BESS → ปล่อย 18-22 น. (margin = on-peak เต็ม)\nCycle 2: Grid off-peak → BESS → ปล่อยเช้า (margin = spread)")
    dual_cycle = st.toggle("เปิด 2-Cycle (Solar + Grid Arb)", value=False)
    if dual_cycle:
        solar_days        = st.number_input("วันที่มีแดดพอชาร์จ BESS/ปี", min_value=200, max_value=365, value=310)
        solar_realization = st.slider("Realization Factor — Solar Cycle (%)", 50, 100, 80, 5) / 100
        c1_kwh_est = p["unit_kwh"] * n_units * charge_eff * discharge_eff * solar_days
        c2_kwh_est = p["unit_kwh"] * n_units * charge_eff * discharge_eff * working_days
        st.info(
            f"**C1 Solar**: {c1_kwh_est:,.0f} kWh/ปี × on-peak × {solar_realization*100:.0f}%\n\n"
            f"**C2 Grid Arb**: {c2_kwh_est:,.0f} kWh/ปี × spread\n\n"
            f"รวม {solar_days + working_days} cycle-days/ปี (ถ้า BESS ทำได้ 2 รอบ/วัน)"
        )
    else:
        solar_days        = 310
        solar_realization = 0.8

    st.divider()

    # ── BOI & Tax ──
    st.subheader("🏛️ BOI ๓/๒๕๖๘ & ภาษี")
    cit  = st.slider("CIT Rate (%)", 0, 30, 20) / 100
    wacc = st.slider("WACC / Discount Rate (%)", 3.0, 15.0, 7.0, 0.5) / 100

    st.markdown("**BOI ๓/๒๕๖๘ — Energy Storage**")
    st.caption(
        "• ยกเว้น CIT **3 ปี**, วงเงิน **50%** ของ eligible investment\n"
        "• Eligible cap = ≤ **12M THB per MW** ของ Solar PV ที่ติดคู่กัน\n"
        "• ⚠️ BESS อยู่เดี่ยวไม่ได้ขอ BOI — ต้องมี Solar PV"
    )
    use_boi = st.toggle("ขอ BOI ๓/๒๕๖๘ (มี Solar คู่กัน)", value=False)
    if use_boi:
        solar_ref_mw = st.number_input(
            "Solar PV Reference (MW) — สำหรับคำนวณ BOI Cap",
            min_value=0.01, max_value=100.0, value=0.25, step=0.05,
            format="%.2f",
            help="MW ของ Solar PV ที่ติดคู่กับ BESS (ใช้คำนวณ eligible = min(BESS CAPEX, 12M × MW))"
        )
        boi_eligible_preview = 12_000_000 * solar_ref_mw
        st.info(
            f"BOI eligible cap (Solar {solar_ref_mw:.2f} MW): **{fmt_thb(boi_eligible_preview)}**\n\n"
            f"วงเงิน CIT ยกเว้นสูงสุด (50%): **{fmt_thb(boi_eligible_preview * 0.5)}**"
        )
    else:
        solar_ref_mw = 0.0

    boi_duty_pct = st.slider("BOI Import Duty Saving (%)", 0.0, 10.0, 5.0, 0.5) / 100

    st.divider()

    # ── Commercial ──
    st.subheader("💼 Commercial Terms")
    fx          = st.number_input("FX (THB/USD)", value=35.0, step=0.5)
    margin      = st.slider("Dealer Margin GP%", 5, 40, 15) / 100
    dealer_usd  = st.number_input(f"Dealer Price (USD/kWh) — {selected_product.split('(')[0].strip()}", value=float(p["dealer_usd_kwh"]), step=5.0)
    lease_rate  = st.slider("Lease Interest Rate (%/ปี)", 3.0, 12.0, 6.0, 0.25) / 100
    lease_months= st.number_input("Lease Term (เดือน)", 12, 84, 60)
    lease_down  = st.slider("Down Payment (%)", 0, 30, 10) / 100
    ppa_share   = st.slider("PPA Customer Share (%)", 50, 95, 80) / 100

# ── Assemble assumptions dict ─────────────────────────────────────────────────
p_override = dict(p)
p_override["dealer_usd_kwh"] = dealer_usd

ass = {
    "on_peak_y1": on_peak_y1, "off_peak_y1": off_peak_y1, "tou_esc": tou_esc,
    "cycles_day": cycles_day, "rte": rte, "days_year": days_year,
    "charge_eff": charge_eff, "discharge_eff": discharge_eff, "aux_kw": aux_kw,
    "working_days": working_days,
    "dual_cycle": dual_cycle, "solar_days": solar_days, "solar_realization": solar_realization,
    "om_pct": om_pct, "bundled_om_years": bundled_om_years,
    "cit": cit, "wacc": wacc,
    "use_boi": use_boi, "solar_ref_mw": solar_ref_mw,
    "boi_duty_pct": boi_duty_pct,
    "fx": fx, "margin": margin,
    "lease_rate": lease_rate, "lease_months": lease_months,
    "lease_down": lease_down, "ppa_share": ppa_share,
}

result = calc_model(p_override, ass, n_units)
df = result["df"]
m  = result["metrics"]

# ── Bill Sidebar (collapsible) ────────────────────────────────────────────────
with st.sidebar:
    st.divider()

    # ── Loan / Financing ──
    st.subheader("🏦 Financing (กู้เงิน)")
    use_loan = st.toggle("เปิดใช้ Loan Model", value=False)
    if use_loan:
        loan_pct   = st.slider("% กู้ (ของ Investment)", 0, 90, 70) / 100
        loan_rate  = st.slider("ดอกเบี้ย (%/ปี)", 1.0, 12.0, 4.5, 0.25) / 100
        loan_years = st.number_input("ระยะกู้ (ปี)", min_value=1, max_value=20, value=7)
    else:
        loan_pct, loan_rate, loan_years = 0.0, 0.0, 0

    st.divider()
    with st.expander("📄 กรอก Bill ลูกค้า (รายเดือน)", expanded=False):
        bill_data = {"Month": MONTHS, "Days": [], "On-Peak kWh": [], "Off-Peak kWh": [], "Holiday kWh": [], "kW Peak": []}
        for i, mo in enumerate(MONTHS):
            cols = st.columns(2)
            with cols[0]:
                on_p = st.number_input(f"{mo} On-Peak kWh", value=DEFAULT_BILL["on_peak"][i], step=100, key=f"op_{i}")
            with cols[1]:
                kw_p = st.number_input(f"{mo} kW Peak", value=float(DEFAULT_BILL["kw_peak"][i]), step=1.0, key=f"kw_{i}")
            bill_data["On-Peak kWh"].append(on_p)
            bill_data["kW Peak"].append(kw_p)
            bill_data["Days"].append(DEFAULT_BILL["days"][i])
            bill_data["Off-Peak kWh"].append(DEFAULT_BILL["off_peak"][i])
            bill_data["Holiday kWh"].append(DEFAULT_BILL["holiday"][i])

        bill_df = pd.DataFrame(bill_data)
        annual_on_peak_kwh  = sum(bill_data["On-Peak kWh"])
        annual_off_peak_kwh = sum(bill_data["Off-Peak kWh"])
        annual_bill_thb = (annual_on_peak_kwh * on_peak_y1 + annual_off_peak_kwh * off_peak_y1)
        st.metric("ค่าไฟต่อปี (ประมาณ)", f"฿{annual_bill_thb:,.0f}")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("⚡ BESS Arbitrage Dashboard")
st.caption(f"**{selected_product}** · {result['cap_kwh']:,} kWh · {n_units} unit(s) · อัปเดตแบบ Real-time")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", "📋 Cash Flow ตาราง", "🔄 เปรียบเทียบ 4 Products",
    "📄 Bill Analysis", "☀️ Solar+BESS Bundle", "📐 ประเมินขนาดระบบ"
])

# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    # BOI Scenario selector
    scenario_opts = ["Non-BOI", "BOI ๓/๒๕๖๘ (3Y/50%)"]
    scenario_label = st.radio("BOI Scenario", scenario_opts, horizontal=True)
    scenario_key = {"Non-BOI": "NB", "BOI ๓/๒๕๖๘ (3Y/50%)": "BOI"}[scenario_label]
    sm = m[scenario_key]
    cum_col = {"NB": "Cum CF Non-BOI", "BOI": "Cum CF BOI ๓/๒๕๖๘"}[scenario_key]
    net_col = {"NB": "Net CF Non-BOI", "BOI": "Net CF BOI ๓/๒๕๖๘"}[scenario_key]

    st.subheader("📌 Key Metrics")
    c1,c2,c3,c4,c5 = st.columns(5)
    pb = sm["payback"]
    c1.metric("Payback Year", f"ปีที่ {pb:.1f}" if pb else "ไม่ถึง 30Y")
    irr15 = sm["irr_15"]
    c2.metric("IRR 15Y", f"{irr15*100:.1f}%" if irr15 else "N/A")
    irr30 = sm["irr_30"]
    c3.metric("IRR 30Y", f"{irr30*100:.1f}%" if irr30 else "N/A")
    c4.metric("NPV 30Y", fmt_thb(sm["npv_30"]))
    c5.metric("Cum CF 30Y", fmt_thb(sm["cum_30"]))

    st.divider()
    c6,c7,c8,c9 = st.columns(4)
    c6.metric("Customer Price (upfront)", fmt_thb(result["customer_price"]))
    c7.metric("NET CAPEX (หลัง BOI Duty)", fmt_thb(result["net_capex"]))
    c8.metric("BOI Duty Saving", fmt_thb(result["boi_duty_saving"]))
    if use_boi:
        c9.metric("BOI CIT Cap (50%)", fmt_thb(result["boi_cap"]))
    else:
        c9.metric("BOI CIT Cap", "ไม่ได้ขอ BOI")

    st.divider()

    col_l, col_r = st.columns(2)

    # Cumulative CF chart
    with col_l:
        st.subheader("📈 Cumulative Cash Flow (30Y)")
        fig_cum = go.Figure()
        fig_cum.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig_cum.add_trace(go.Scatter(
            x=[0]+list(df["Year"]), y=[-result["customer_price"]]+list(df[cum_col]),
            mode="lines+markers", name=scenario_label,
            line=dict(color="#00b4d8", width=2.5),
            marker=dict(size=5),
            fill="tozeroy", fillcolor="rgba(0,180,216,0.08)"
        ))
        # Payback marker
        if pb:
            pb_y = int(pb)
            if pb_y <= 30 and pb_y >= 1:
                pb_val = float(df[df["Year"]==pb_y][cum_col].values[0])
                fig_cum.add_trace(go.Scatter(
                    x=[pb], y=[pb_val], mode="markers+text",
                    marker=dict(color="gold", size=12, symbol="star"),
                    text=[f"Payback Y{pb:.1f}"], textposition="top center",
                    showlegend=False
                ))
        # Loan overlay
        if use_loan:
            net_col_vals = list(df[net_col])
            loan_cum, loan_equity, loan_pmt_val = apply_loan_to_cf(
                net_col_vals, result["customer_price"], loan_pct, loan_rate, int(loan_years))
            fig_cum.add_trace(go.Scatter(
                x=[0]+list(df["Year"]), y=[-loan_equity]+loan_cum,
                mode="lines", name=f"กู้ {loan_pct*100:.0f}% @ {loan_rate*100:.2f}%",
                line=dict(color="#ffd166", width=2, dash="dot"),
            ))
        fig_cum.update_layout(
            xaxis_title="ปี", yaxis_title="THB",
            height=350, margin=dict(l=10,r=10,t=30,b=30),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    # Annual net CF bar
    with col_r:
        st.subheader("📊 Annual Net Cash Flow")
        colors = ["#06d6a0" if v >= 0 else "#ef476f" for v in df[net_col]]
        fig_bar = go.Figure(go.Bar(
            x=df["Year"], y=df[net_col],
            marker_color=colors, name="Net CF"
        ))
        fig_bar.update_layout(
            xaxis_title="ปี", yaxis_title="THB",
            height=350, margin=dict(l=10,r=10,t=30,b=30)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # 2-Cycle Revenue breakdown
    if dual_cycle:
        st.subheader("☀️ 2-Cycle Revenue Breakdown")
        y1r = df.iloc[0]
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("C1 Solar Revenue Y1", fmt_thb(y1r["C1 Solar Revenue (THB)"]),
                   help=f"Solar {solar_days}วัน × on-peak เต็ม × {solar_realization*100:.0f}% realization")
        dc2.metric("C2 Grid Arb Revenue Y1", fmt_thb(y1r["C2 Arb Revenue (THB)"]),
                   help=f"Grid Arb {working_days}วัน × spread (on-peak − off-peak/RTE)")
        dc3.metric("รวม Gross Saving Y1", fmt_thb(y1r["Gross Saving"]))

        fig_dc = go.Figure()
        fig_dc.add_trace(go.Bar(x=df["Year"], y=df["C1 Solar Revenue (THB)"],
                                name="☀️ C1 Solar (free charge)", marker_color="#ffd166"))
        fig_dc.add_trace(go.Bar(x=df["Year"], y=df["C2 Arb Revenue (THB)"],
                                name="⚡ C2 Grid Arb (spread)", marker_color="#00b4d8"))
        fig_dc.update_layout(barmode="stack", xaxis_title="ปี", yaxis_title="THB",
                             height=300, margin=dict(l=10,r=10,t=10,b=30))
        st.plotly_chart(fig_dc, use_container_width=True)
        st.divider()

    # BOI scenario comparison
    st.subheader("🏛️ เปรียบเทียบ: Non-BOI vs BOI ๓/๒๕๖๘")
    boi_rows = []
    for label, key in [("Non-BOI","NB"), ("BOI ๓/๒๕๖๘ (3Y/50%)","BOI")]:
        sm2 = m[key]
        boi_rows.append({
            "Scenario": label,
            "Payback Year": f"ปีที่ {sm2['payback']:.1f}" if sm2["payback"] else ">30Y",
            "IRR 15Y": f"{sm2['irr_15']*100:.1f}%" if sm2["irr_15"] else "N/A",
            "IRR 20Y": f"{sm2['irr_20']*100:.1f}%" if sm2["irr_20"] else "N/A",
            "IRR 30Y": f"{sm2['irr_30']*100:.1f}%" if sm2["irr_30"] else "N/A",
            "NPV 30Y": fmt_thb(sm2["npv_30"]),
            "Cum CF 15Y": fmt_thb(sm2["cum_15"]),
            "Cum CF 30Y": fmt_thb(sm2["cum_30"]),
        })
    if use_boi:
        boi_cap_val = result["boi_cap"]
        st.caption(f"BOI ๓/๒๕๖๘: Solar {solar_ref_mw:.2f} MW → eligible cap {fmt_thb(result['boi_eligible'])} → CIT cap 50% = **{fmt_thb(boi_cap_val)}**")
    else:
        st.caption("⚠️ ยังไม่ได้เปิดใช้ BOI — ผลลัพธ์ BOI จะเท่ากับ Non-BOI (เปิดที่ Sidebar > BOI ๓/๒๕๖๘)")
    st.dataframe(pd.DataFrame(boi_rows).set_index("Scenario"), use_container_width=True)

    # Leasing & PPA summary
    st.divider()
    col_lea, col_ppa = st.columns(2)
    with col_lea:
        st.subheader("🏦 Leasing Model")
        lea = result["lease"]
        st.table(pd.DataFrame({
            "รายการ": ["Down Payment","Financed Amount","Monthly PMT","Annual PMT"],
            "มูลค่า (THB)": [fmt_thb(lea["down"]), fmt_thb(lea["financed"]),
                             fmt_thb(lea["monthly_pmt"]), fmt_thb(lea["annual_pmt"])]
        }).set_index("รายการ"))

    with col_ppa:
        st.subheader("🤝 PPA / ESCO Model")
        y1_gross = float(df["Gross Saving"].iloc[0])
        boi_label = "BOI ๓/๒๕๖๘" if use_boi else "Non-BOI"
        st.table(pd.DataFrame({
            "รายการ": [f"Customer Share ({ppa_share*100:.0f}%)","Owner Share",
                       "Y1 Customer Saving",f"Cum Customer Saving 30Y ({boi_label})",
                       f"Cum Owner Revenue 30Y ({boi_label})"],
            "มูลค่า": [f"{ppa_share*100:.0f}%", f"{(1-ppa_share)*100:.0f}%",
                       fmt_thb(y1_gross * ppa_share),
                       fmt_thb(result["ppa_cust_cum30"]),
                       fmt_thb(result["ppa_owner_cum30"])]
        }).set_index("รายการ"))

# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"📋 Year-by-Year Cash Flow — {selected_product}")
    scenario_label2 = st.radio("BOI Scenario", ["Non-BOI", "BOI ๓/๒๕๖๘ (3Y/50%)"], horizontal=True, key="t2")
    sk2 = {"Non-BOI": "NB", "BOI ๓/๒๕๖๘ (3Y/50%)": "BOI"}[scenario_label2]
    net_col2 = {"NB":"Net CF Non-BOI", "BOI":"Net CF BOI ๓/๒๕๖๘"}[sk2]
    cum_col2 = {"NB":"Cum CF Non-BOI", "BOI":"Cum CF BOI ๓/๒๕๖๘"}[sk2]

    # ── Energy Loss Waterfall (Y1) ──────────────────────────────────────────────
    y1r = df.iloc[0]
    st.subheader("⚡ Energy Flow — Y1 (Waterfall)")
    wf_labels = ["Grid Input","Charging Loss","Stored in Battery","Discharging Loss","Delivered to Load"]
    wf_values = [
        y1r["E In from Grid (kWh)"],
        -y1r["Charging Loss (kWh)"],
        y1r["E Stored (kWh)"],
        -y1r["Discharging Loss (kWh)"],
        y1r["Energy Out (kWh)"],
    ]
    wf_measure = ["absolute","relative","absolute","relative","absolute"]
    wf_fig = go.Figure(go.Waterfall(
        orientation="v", measure=wf_measure,
        x=wf_labels, y=wf_values,
        connector=dict(line=dict(color="gray", width=1, dash="dot")),
        increasing=dict(marker_color="#06d6a0"),
        decreasing=dict(marker_color="#ef476f"),
        totals=dict(marker_color="#00b4d8"),
        text=[f"{abs(v):,.0f} kWh" for v in wf_values],
        textposition="outside",
    ))
    wf_fig.update_layout(
        title=f"Energy Flow per ปี (working_days={int(y1r['Working Days (Arb)'])}วัน, ηc={charge_eff*100:.0f}%, ηd={discharge_eff*100:.0f}%, RTE={rte*100:.1f}%)",
        yaxis_title="kWh/ปี", height=380, margin=dict(l=10,r=10,t=50,b=10)
    )
    st.plotly_chart(wf_fig, use_container_width=True)

    wf_cols = st.columns(5)
    wf_cols[0].metric("Grid Input (kWh/ปี)",    f"{y1r['E In from Grid (kWh)']:,.0f}")
    wf_cols[1].metric("Charging Loss",           f"{y1r['Charging Loss (kWh)']:,.0f} kWh ({(1-charge_eff)*100:.1f}%)")
    wf_cols[2].metric("Stored",                  f"{y1r['E Stored (kWh)']:,.0f} kWh")
    wf_cols[3].metric("Discharging Loss",         f"{y1r['Discharging Loss (kWh)']:,.0f} kWh ({(1-discharge_eff)*100:.1f}%)")
    wf_cols[4].metric("Delivered to Load (kWh/ปี)", f"{y1r['Energy Out (kWh)']:,.0f}")
    st.caption(f"Aux Load: {aux_kw} kW × 8,760 ชม. = {aux_kw*8760:,.0f} kWh/ปี → ค่าใช้จ่ายเพิ่ม {fmt_thb(y1r['Aux Cost'])}/ปี")
    st.divider()

    base_cols = ["Year","C1 Solar Energy Out (kWh)","C1 Solar Revenue (THB)",
                 "C2 Grid Energy Out (kWh)","C2 Arb Revenue (THB)",
                 "E In from Grid (kWh)","Charging Loss (kWh)","Energy Out (kWh)",
                 "On-Peak (THB/kWh)","Off-Peak (THB/kWh)",
                 "Arb Margin C2 (THB/kWh)","Gross Saving","Aux Cost","OM Cost","Pre-Tax CF",
                 "BOI Exempt", net_col2, cum_col2]
    display_cols = [c for c in base_cols if c in df.columns]
    df_display = df[display_cols].copy()

    # Format numbers — only format columns that actually exist in display_cols
    fmt_int_cols = ["C1 Solar Energy Out (kWh)","C1 Solar Revenue (THB)",
                    "C2 Grid Energy Out (kWh)","C2 Arb Revenue (THB)",
                    "E In from Grid (kWh)","Charging Loss (kWh)","Energy Out (kWh)",
                    "Gross Saving","Aux Cost","OM Cost","Pre-Tax CF",
                    "BOI Exempt", net_col2, cum_col2]
    fmt_dec_cols = ["On-Peak (THB/kWh)","Off-Peak (THB/kWh)","Arb Margin C2 (THB/kWh)"]
    for col in fmt_int_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].map(lambda x: f"{x:,.0f}")
    for col in fmt_dec_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].map(lambda x: f"{x:.4f}")

    df_display["Year"] = df_display["Year"].map(lambda x: f"Y{x}")
    st.dataframe(df_display.set_index("Year"), use_container_width=True, height=600)

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "bess_cashflow.csv", "text/csv")

# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🔄 เปรียบเทียบทุก Product ด้วย Assumptions เดียวกัน")
    scenario_label3 = st.radio("BOI Scenario", ["Non-BOI", "BOI ๓/๒๕๖๘ (3Y/50%)"], horizontal=True, key="t3")
    sk3 = {"Non-BOI": "NB", "BOI ๓/๒๕๖๘ (3Y/50%)": "BOI"}[scenario_label3]
    cum_col3 = {"NB":"Cum CF Non-BOI", "BOI":"Cum CF BOI ๓/๒๕๖๘"}[sk3]

    comp_rows = []
    cum_fig = go.Figure()
    cum_fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    colors_prod = ["#00b4d8","#06d6a0","#ffd166","#ef476f"]

    for (prod_name, prod), color in zip(PRODUCTS.items(), colors_prod):
        r = calc_model(prod, ass, n_units)
        sm3 = r["metrics"][sk3]
        comp_rows.append({
            "Product": prod_name.split("(")[0].strip(),
            "Cap (kWh)": f"{r['cap_kwh']:,}",
            "Customer Price": fmt_thb(r["customer_price"]),
            "NET CAPEX": fmt_thb(r["net_capex"]),
            "Payback": f"ปีที่ {sm3['payback']:.1f}" if sm3["payback"] else ">30Y",
            "IRR 15Y": f"{sm3['irr_15']*100:.1f}%" if sm3["irr_15"] else "N/A",
            "IRR 30Y": f"{sm3['irr_30']*100:.1f}%" if sm3["irr_30"] else "N/A",
            "NPV 30Y": fmt_thb(sm3["npv_30"]),
            "Cum CF 15Y": fmt_thb(sm3["cum_15"]),
            "Cum CF 30Y": fmt_thb(sm3["cum_30"]),
        })
        cum_y = [-r["customer_price"]] + list(r["df"][cum_col3])
        cum_fig.add_trace(go.Scatter(
            x=list(range(31)), y=cum_y,
            mode="lines", name=prod_name.split("(")[0].strip(),
            line=dict(color=color, width=2)
        ))

    st.dataframe(pd.DataFrame(comp_rows).set_index("Product"), use_container_width=True)

    cum_fig.update_layout(
        title="Cumulative Cash Flow — ทุก Product",
        xaxis_title="ปี", yaxis_title="THB",
        height=420, margin=dict(l=10,r=10,t=50,b=30)
    )
    st.plotly_chart(cum_fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("📄 Bill Analysis — ข้อมูลการใช้ไฟลูกค้า")
    try:
        bill_df
    except NameError:
        bill_df = pd.DataFrame({
            "Month": MONTHS,
            "Days": DEFAULT_BILL["days"],
            "On-Peak kWh": DEFAULT_BILL["on_peak"],
            "Off-Peak kWh": DEFAULT_BILL["off_peak"],
            "Holiday kWh": DEFAULT_BILL["holiday"],
            "kW Peak": DEFAULT_BILL["kw_peak"],
        })

    bill_df["On-Peak Cost (THB)"]  = bill_df["On-Peak kWh"]  * on_peak_y1
    bill_df["Off-Peak Cost (THB)"] = bill_df["Off-Peak kWh"] * off_peak_y1
    bill_df["Total Cost (THB)"]    = bill_df["On-Peak Cost (THB)"] + bill_df["Off-Peak Cost (THB)"]
    st.dataframe(bill_df.set_index("Month"), use_container_width=True)

    # Monthly cost chart
    fig_bill = px.bar(bill_df, x="Month", y="Total Cost (THB)",
                      title="ค่าไฟรายเดือน (ประมาณ)", color_discrete_sequence=["#00b4d8"])
    st.plotly_chart(fig_bill, use_container_width=True)

    # BESS Y1 saving vs bill
    y1_arb = float(df["Gross Saving"].iloc[0])
    annual_bill = bill_df["Total Cost (THB)"].sum()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("ค่าไฟ/ปี (ประมาณ)", fmt_thb(annual_bill))
    col_b.metric("BESS Gross Saving Y1", fmt_thb(y1_arb))
    col_c.metric("สัดส่วนประหยัด Y1", f"{y1_arb/annual_bill*100:.1f}%" if annual_bill > 0 else "N/A")

# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("☀️ Solar+BESS Bundle — กรอกข้อมูล")
    st.caption(
        "Solar ชาร์จแบตฟรี (ต้นทุน 0) + BESS Arbitrage + Demand Charge Reduction\n\n"
        "**BOI ๓/๒๕๖๘**: BESS + Solar → 3Y CIT exempt, 50% cap, ≤ 12M THB/MW Solar PV"
    )

    col_s, col_b2 = st.columns(2)

    with col_s:
        st.markdown("**☀️ Solar**")
        sol_kwp       = st.number_input("Solar ขนาด (kWp)", min_value=10, max_value=50000, value=250, step=10)
        sol_capex_mw  = st.number_input("Solar CAPEX (THB/MW)", value=20_000_000, step=500_000, format="%d")
        sol_cuf       = st.slider("Solar CUF (%)", 10.0, 25.0, 16.0, 0.5) / 100
        sol_deg_pct   = st.slider("Solar Degradation (%/ปี)", 0.3, 1.0, 0.5, 0.1) / 100

        auto_ratio = st.checkbox("ล็อค Solar:BESS = 1:4 (kWp:kWh)", value=True)
        if auto_ratio:
            bess_kwh_sol = sol_kwp * 4
            st.info(f"BESS อัตโนมัติ: **{bess_kwh_sol:,} kWh** ({sol_kwp:,} kWp × 4)")

    with col_b2:
        st.markdown("**🔋 BESS**")
        if auto_ratio:
            bess_kwh_input = bess_kwh_sol
            st.metric("BESS Capacity (kWh)", f"{bess_kwh_input:,}")
        else:
            bess_kwh_input = st.number_input("BESS Capacity (kWh)", min_value=100, max_value=100000, value=1000, step=100)

        bess_capex_mwh = st.number_input("BESS CAPEX (THB/MWh)", value=6_000_000, step=100_000, format="%d")

        st.markdown("**เลือก Degradation Curve:**")
        bess_prod_choice = st.selectbox("ใช้ Degradation จาก Product", list(PRODUCTS.keys()), key="sb_prod")
        bess_deg_curve_sol = PRODUCTS[bess_prod_choice]["deg"]

    st.divider()

    col_t, col_d, col_r = st.columns(3)
    with col_t:
        st.markdown("**⚡ TOU Tariff (Solar+BESS)**")
        sb_peak_y1    = st.number_input("Peak Rate Y1 (THB/kWh)", value=5.7982, step=0.01, format="%.4f", key="sb_pk")
        sb_offpk_y1   = st.number_input("Off-Peak Rate Y1 (THB/kWh)", value=2.6107, step=0.01, format="%.4f", key="sb_op")
        sb_tou_esc    = st.slider("TOU Escalation (%/ปี)", 0.0, 5.0, 2.0, 0.1, key="sb_esc") / 100

    with col_d:
        st.markdown("**🔌 Demand Charge**")
        sb_demand_cut = st.number_input("Demand Reduction (kW)", min_value=0, max_value=5000, value=250, step=10)
        sb_demand_rate= st.number_input("Demand Rate (THB/kW/เดือน)", value=74.14, step=1.0, format="%.2f")

    with col_r:
        st.markdown("**🏛️ BOI ๓/๒๕๖๘ & ปัจจัยอื่น**")
        sb_use_boi = st.toggle("ขอ BOI ๓/๒๕๖๘ (BESS + Solar)", value=True, key="sb_boi")
        if sb_use_boi:
            sol_mw_preview = sol_kwp / 1000
            bess_capex_preview = (bess_kwh_input / 1000) * bess_capex_mwh
            eligible_preview = min(bess_capex_preview, 12_000_000 * sol_mw_preview)
            st.caption(
                f"Solar {sol_mw_preview:.3f} MW → eligible = min(BESS CAPEX, 12M×{sol_mw_preview:.3f}MW)\n\n"
                f"= min({fmt_thb(bess_capex_preview)}, {fmt_thb(12_000_000 * sol_mw_preview)})\n\n"
                f"= **{fmt_thb(eligible_preview)}** → CIT cap 50% = **{fmt_thb(eligible_preview * 0.5)}**"
            )
        sb_real_factor  = st.slider("Realization Factor (%)", 50, 100, 70, 5) / 100
        sb_om_pct       = st.slider("OM % of Total CAPEX (Y11+)", 0.3, 2.0, 0.5, 0.1, key="sb_om") / 100
        sb_bundled_om   = st.number_input("Bundled OM ปี", min_value=1, max_value=15, value=10, key="sb_bom")

    st.divider()

    # ── Solar Split + 2-Cycle (Zero Export) ──
    st.markdown("**☀️ Solar Operation Split — Zero Export**")
    st.caption(
        "ระบบ Zero Export: Solar output ทั้งหมดใช้ใน site เท่านั้น "
        "แบ่งเป็น Direct Load (offset ช่วงกลางวัน 09-17h) + BESS Charge (เก็บปล่อยช่วงเย็น 17-22h) "
        "รวมกัน = 100% ไม่มีขายคืนการไฟฟ้า"
    )
    sp_c1, sp_c2 = st.columns(2)
    with sp_c1:
        sb_solar_direct_pct = st.slider("Direct Load (%)", 30, 90, 60, 5, key="sb_sd") / 100
    with sp_c2:
        sb_solar_bess_pct = round(1.0 - sb_solar_direct_pct, 2)
        st.metric("→ BESS Charge (auto)", f"{sb_solar_bess_pct*100:.0f}%",
                  help="= 100% − Direct Load%  (Zero Export)")

    gc1, gc2 = st.columns(2)
    with gc1:
        sb_grid_cycles = st.slider("Grid OP-Charged Cycles/Day", 0.0, 2.0, 1.0, 0.25, key="sb_gc",
                                    help="ชาร์จจาก Grid off-peak → ปล่อยช่วงเช้า on-peak")
    with gc2:
        st.info(
            f"Solar Cycle: **{sb_solar_bess_pct*100:.0f}%** ของ Solar output → BESS → ปล่อย 17-22h (margin = Peak)\n\n"
            f"Grid Cycle: **{sb_grid_cycles}** cycle/day × 247 วัน → ปล่อยช้าก on-peak (margin = Spread)"
        )

    st.divider()
    col_loan = st.container()
    with col_loan:
        st.markdown("**🏦 Financing (กู้เงิน)**")
        st.caption("ถ้าไม่กู้ → ใช้ค่าจาก Sidebar / ถ้าจะตั้งใหม่เฉพาะ Solar+BESS ก็ปรับได้ที่นี่")
        sb_use_loan  = st.toggle("เปิด Loan สำหรับ Solar+BESS", value=use_loan, key="sb_loan")
        if sb_use_loan:
            sb_loan_pct   = st.slider("% กู้", 0, 90, int(loan_pct*100) if use_loan else 70, key="sb_lp") / 100
            sb_loan_rate  = st.slider("ดอกเบี้ย (%/ปี)", 1.0, 12.0, loan_rate*100 if use_loan else 4.5, 0.25, key="sb_lr") / 100
            sb_loan_years = st.number_input("ระยะกู้ (ปี)", 1, 20, int(loan_years) if use_loan else 7, key="sb_ly")
        else:
            sb_loan_pct, sb_loan_rate, sb_loan_years = 0.0, 0.0, 0

    st.divider()

    # ── Run calculation ──
    sb_df, sb_m = calc_solar_bess(
        solar_kwp             = sol_kwp,
        solar_capex_per_mw    = sol_capex_mw,
        solar_cuf             = sol_cuf,
        solar_deg_pct         = sol_deg_pct,
        bess_kwh              = bess_kwh_input,
        bess_capex_per_mwh    = bess_capex_mwh,
        bess_deg_curve        = bess_deg_curve_sol,
        peak_rate_y1          = sb_peak_y1,
        off_peak_rate_y1      = sb_offpk_y1,
        tou_esc               = sb_tou_esc,
        demand_cut_kw         = sb_demand_cut,
        demand_rate_per_kw_mo = sb_demand_rate,
        rte                   = rte,
        cycles_day            = cycles_day,
        days_year             = days_year,
        om_pct                = sb_om_pct,
        bundled_om_years      = sb_bundled_om,
        cit                   = cit,
        wacc                  = wacc,
        use_boi               = sb_use_boi,
        realization_factor    = sb_real_factor,
        # v14 solar split (Zero Export: direct + bess = 100%)
        solar_direct_pct      = sb_solar_direct_pct,
        solar_bess_pct        = sb_solar_bess_pct,
        grid_cycles_day       = sb_grid_cycles,
        working_days          = working_days,
    )

    # Loan overlay for Solar+BESS
    if sb_use_loan:
        sb_loan_cum, sb_loan_equity, sb_loan_pmt_val = apply_loan_to_cf(
            list(sb_df["Net CF (THB)"]), sb_m["total_investment"],
            sb_loan_pct, sb_loan_rate, int(sb_loan_years))
        sb_loan_irr15 = safe_irr([-sb_loan_equity] + [a-b for a,b in zip(
            list(sb_df["Net CF (THB)"][:15]),
            [sb_loan_pmt_val if i < sb_loan_years else 0 for i in range(15)])])
        sb_loan_irr30 = safe_irr([-sb_loan_equity] + [a-b for a,b in zip(
            list(sb_df["Net CF (THB)"]),
            [sb_loan_pmt_val if i < sb_loan_years else 0 for i in range(30)])])

    # ── KPI Cards ──
    st.subheader("📌 Key Metrics — Solar+BESS Bundle")
    k1, k2, k3, k4, k5 = st.columns(5)
    pb_sol = sb_m["payback"]
    k1.metric("Payback (ไม่กู้)", f"ปีที่ {pb_sol:.1f}" if pb_sol else ">30Y")
    k2.metric("IRR 15Y (ไม่กู้)", f"{sb_m['irr_15']*100:.1f}%" if sb_m["irr_15"] else "N/A")
    k3.metric("IRR 30Y (ไม่กู้)", f"{sb_m['irr_30']*100:.1f}%" if sb_m["irr_30"] else "N/A")
    k4.metric("NPV 30Y", fmt_thb(sb_m["npv_30"]))
    k5.metric("Cum CF 30Y", fmt_thb(sb_m["cum_30"]))

    if sb_use_loan:
        st.markdown(f"**🏦 กู้ {sb_loan_pct*100:.0f}% — ดอกเบี้ย {sb_loan_rate*100:.2f}%/ปี — {sb_loan_years} ปี**")
        lk1, lk2, lk3, lk4, lk5 = st.columns(5)
        lk1.metric("Equity ที่ใช้ (Y0)", fmt_thb(sb_loan_equity))
        lk2.metric("Annual PMT", fmt_thb(sb_loan_pmt_val))
        lk3.metric("IRR 15Y (กู้)", f"{sb_loan_irr15*100:.1f}%" if sb_loan_irr15 else "N/A")
        lk4.metric("IRR 30Y (กู้)", f"{sb_loan_irr30*100:.1f}%" if sb_loan_irr30 else "N/A")
        lb_payback = next((i+1 for i, v in enumerate(sb_loan_cum) if v >= 0), None)
        lk5.metric("Payback (กู้)", f"ปีที่ {lb_payback}" if lb_payback else ">30Y")

    k6, k7, k8, k9 = st.columns(4)
    k6.metric("Total Investment", fmt_thb(sb_m["total_investment"]))
    k7.metric("Solar CAPEX", fmt_thb(sb_m["solar_capex"]))
    k8.metric("BESS CAPEX", fmt_thb(sb_m["bess_capex"]))
    boi_label_sol = f"BOI Cap (50%) = {fmt_thb(sb_m['boi_cap'])}" if sb_use_boi else "ไม่ได้ขอ BOI"
    k9.metric("BOI ๓/๒๕๖๘ CIT Cap", fmt_thb(sb_m["boi_cap"]) if sb_use_boi else "—")

    st.divider()

    # ── Y1 Revenue Breakdown ──
    y1 = sb_df.iloc[0]
    st.subheader("💰 Y1 Revenue Breakdown (Zero Export)")
    rev_cols = st.columns(5)
    rev_cols[0].metric("☀️ Solar Direct Save", fmt_thb(y1["Solar Direct Save (THB)"]),
                        help=f"Solar {sb_solar_direct_pct*100:.0f}% → Direct Load × Peak Rate (ช่วง 09-17h)")
    rev_cols[1].metric("🔋 BESS Solar Cycle", fmt_thb(y1["BESS Solar Rev (THB)"]),
                        help=f"Solar {sb_solar_bess_pct*100:.0f}% → BESS → Evening Discharge × Peak Rate (17-22h)")
    rev_cols[2].metric("⚡ BESS Grid Cycle", fmt_thb(y1["BESS Grid Rev (THB)"]),
                        help="Grid Off-Peak → BESS → Morning On-Peak × Spread")
    rev_cols[3].metric("🔌 Demand Save", fmt_thb(y1["Demand Save (THB)"]))
    total_gross = y1["Solar Save (THB)"] + y1["BESS Arb (THB)"] + y1["Demand Save (THB)"]
    rev_cols[4].metric("✅ Total Gross Y1", fmt_thb(total_gross))

    # Revenue mix pie
    pie_fig = go.Figure(go.Pie(
        labels=["Solar Saving", "BESS Arbitrage", "Demand Saving"],
        values=[y1["Solar Save (THB)"], y1["BESS Arb (THB)"], y1["Demand Save (THB)"]],
        hole=0.4,
        marker_colors=["#ffd166","#00b4d8","#06d6a0"]
    ))
    pie_fig.update_layout(title="สัดส่วน Revenue Y1", height=320, margin=dict(l=0,r=0,t=40,b=0))

    # Cumulative CF chart
    cum_fig_sol = go.Figure()
    cum_fig_sol.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    cum_fig_sol.add_trace(go.Scatter(
        x=[0] + list(sb_df["Year"]),
        y=[-sb_m["total_investment"]] + list(sb_df["Cum CF (THB)"]),
        mode="lines+markers", name="Cum CF",
        line=dict(color="#ffd166", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,209,102,0.1)"
    ))
    if pb_sol:
        cum_fig_sol.add_trace(go.Scatter(
            x=[pb_sol], y=[0], mode="markers+text",
            marker=dict(color="gold", size=13, symbol="star"),
            text=[f"Payback Y{pb_sol:.1f}"], textposition="top center",
            showlegend=False
        ))
    if sb_use_loan:
        cum_fig_sol.add_trace(go.Scatter(
            x=[0]+list(sb_df["Year"]),
            y=[-sb_loan_equity]+sb_loan_cum,
            mode="lines", name=f"กู้ {sb_loan_pct*100:.0f}% @ {sb_loan_rate*100:.2f}%",
            line=dict(color="#ef476f", width=2, dash="dot")
        ))
    cum_fig_sol.update_layout(title="Cumulative Cash Flow 30Y", xaxis_title="ปี", yaxis_title="THB",
                               height=320, margin=dict(l=10,r=10,t=40,b=30))

    chart_l, chart_r = st.columns([1, 1])
    with chart_l:
        st.plotly_chart(pie_fig, use_container_width=True)
    with chart_r:
        st.plotly_chart(cum_fig_sol, use_container_width=True)

    # ── Stacked Bar — Revenue Streams ──
    st.subheader("📊 Revenue Streams รายปี (30Y)")
    fig_stack = go.Figure()
    fig_stack.add_trace(go.Bar(x=sb_df["Year"], y=sb_df["Solar Save (THB)"],    name="Solar Saving",  marker_color="#ffd166"))
    fig_stack.add_trace(go.Bar(x=sb_df["Year"], y=sb_df["BESS Arb (THB)"],     name="BESS Arbitrage",marker_color="#00b4d8"))
    fig_stack.add_trace(go.Bar(x=sb_df["Year"], y=sb_df["Demand Save (THB)"],  name="Demand Saving", marker_color="#06d6a0"))
    fig_stack.add_trace(go.Bar(x=sb_df["Year"], y=-sb_df["OM Cost (THB)"],      name="OM Cost (−)",   marker_color="#ef476f"))
    fig_stack.update_layout(barmode="relative", xaxis_title="ปี", yaxis_title="THB",
                             height=350, margin=dict(l=10,r=10,t=10,b=30))
    st.plotly_chart(fig_stack, use_container_width=True)

    # ── Comparison vs BESS-only ──
    st.subheader("🆚 เปรียบเทียบ: Solar+BESS vs BESS Only")
    r_bess_only = calc_model(p_override, ass, n_units)
    bess_cum_col = "Cum CF BOI ๓/๒๕๖๘" if use_boi else "Cum CF Non-BOI"
    bess_scenario_label = "BOI ๓/๒๕๖๘" if use_boi else "Non-BOI"

    comp_cum_fig = go.Figure()
    comp_cum_fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    comp_cum_fig.add_trace(go.Scatter(
        x=[0]+list(sb_df["Year"]),
        y=[-sb_m["total_investment"]]+list(sb_df["Cum CF (THB)"]),
        name=f"☀️ Solar+BESS Bundle ({'BOI ๓/๒๕๖๘' if sb_use_boi else 'Non-BOI'})",
        line=dict(color="#ffd166", width=2.5)
    ))
    comp_cum_fig.add_trace(go.Scatter(
        x=[0]+list(r_bess_only["df"]["Year"]),
        y=[-r_bess_only["customer_price"]]+list(r_bess_only["df"][bess_cum_col]),
        name=f"🔋 BESS Only — {selected_product.split('(')[0].strip()} ({bess_scenario_label})",
        line=dict(color="#00b4d8", width=2.5, dash="dash")
    ))
    comp_cum_fig.update_layout(xaxis_title="ปี", yaxis_title="THB (Cumulative CF)",
                                height=380, margin=dict(l=10,r=10,t=10,b=30))
    st.plotly_chart(comp_cum_fig, use_container_width=True)

    boi_key_bess = "BOI" if use_boi else "NB"
    comp_table = pd.DataFrame([
        {
            "รายการ": "☀️ Solar+BESS Bundle",
            "Investment": fmt_thb(sb_m["total_investment"]),
            "Payback": f"{pb_sol:.1f}Y" if pb_sol else ">30Y",
            "IRR 15Y": f"{sb_m['irr_15']*100:.1f}%" if sb_m["irr_15"] else "N/A",
            "IRR 30Y": f"{sb_m['irr_30']*100:.1f}%" if sb_m["irr_30"] else "N/A",
            "NPV 30Y": fmt_thb(sb_m["npv_30"]),
            "Cum CF 30Y": fmt_thb(sb_m["cum_30"]),
        },
        {
            "รายการ": f"🔋 BESS Only ({selected_product.split('(')[0].strip()})",
            "Investment": fmt_thb(r_bess_only["customer_price"]),
            "Payback": f"{r_bess_only['metrics'][boi_key_bess]['payback']:.1f}Y" if r_bess_only["metrics"][boi_key_bess]["payback"] else ">30Y",
            "IRR 15Y": f"{r_bess_only['metrics'][boi_key_bess]['irr_15']*100:.1f}%" if r_bess_only["metrics"][boi_key_bess]["irr_15"] else "N/A",
            "IRR 30Y": f"{r_bess_only['metrics'][boi_key_bess]['irr_30']*100:.1f}%" if r_bess_only["metrics"][boi_key_bess]["irr_30"] else "N/A",
            "NPV 30Y": fmt_thb(r_bess_only["metrics"][boi_key_bess]["npv_30"]),
            "Cum CF 30Y": fmt_thb(r_bess_only["metrics"][boi_key_bess]["cum_30"]),
        }
    ]).set_index("รายการ")
    st.dataframe(comp_table, use_container_width=True)

    # ── Full table ──
    with st.expander("📋 ดูตาราง Cash Flow ทั้ง 30 ปี"):
        fmt_cols = ["Solar kWh","Solar Save (THB)","BESS kWh Out","BESS Arb (THB)",
                    "Demand Save (THB)","OM Cost (THB)","Pre-Tax CF (THB)","Net CF (THB)","Cum CF (THB)"]
        sb_display = sb_df.copy()
        for c in fmt_cols:
            if c in sb_display.columns:
                sb_display[c] = sb_display[c].map(lambda x: f"{x:,.0f}")
        sb_display["Sol Deg"] = sb_display["Sol Deg"].map(lambda x: f"{x:.4f}")
        sb_display["BESS Deg"] = sb_display["BESS Deg"].map(lambda x: f"{x:.4f}")
        sb_display["Year"] = sb_display["Year"].map(lambda x: f"Y{x}")
        st.dataframe(sb_display.set_index("Year"), use_container_width=True, height=500)

        csv_sol = sb_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv_sol, "solar_bess_cashflow.csv", "text/csv")

# ──────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("📐 Bill Sizer — ประเมินขนาด Solar+BESS จากบิลค่าไฟ")
    st.caption(
        "**v14 logic** — กรอกบิลค่าไฟในแถบซ้าย (Sidebar → Bill ลูกค้า) "
        "ระบบจะวิเคราะห์ลักษณะโหลด + แนะนำขนาด Solar+BESS พร้อม Pitch message"
    )

    # ── SECTION 1: Bill Summary ───────────────────────────────────────────────
    st.subheader("📄 Section 1-2 — บิลและ Site Parameters")
    _on_peak_yr  = sum(bill_data["On-Peak kWh"])
    _off_peak_yr = sum(bill_data["Off-Peak kWh"])
    _avg_kw_bill = sum(bill_data["kW Peak"]) / 12
    _max_kw_bill = max(bill_data["kW Peak"])

    sz_c1, sz_c2 = st.columns([2, 1])
    with sz_c1:
        sz_a1, sz_a2, sz_a3, sz_a4 = st.columns(4)
        sz_a1.metric("On-Peak kWh/ปี",  f"{_on_peak_yr:,.0f}")
        sz_a2.metric("Off-Peak kWh/ปี", f"{_off_peak_yr:,.0f}")
        sz_a3.metric("Peak Demand (avg)", f"{_avg_kw_bill:.0f} kW")
        sz_a4.metric("Peak Demand (max)", f"{_max_kw_bill:.0f} kW")

    with sz_c2:
        sz_roof     = st.number_input("พื้นที่หลังคา (ม²) — 0=ไม่จำกัด", min_value=0,
                                       max_value=500000, value=0, step=100, key="sz_roof",
                                       help="1 MW Solar ≈ 6,500 ม²")
        sz_self_con = st.slider("Self-Consumption Target (%)", 70, 100, 90, 5, key="sz_sc_t") / 100
        sz_dem_red  = st.slider("Demand Reduction Target (%)", 10, 50, 30, 5, key="sz_dr") / 100

    sz_b1, sz_b2, sz_b3, sz_b4 = st.columns(4)
    with sz_b1: sz_cuf = st.slider("Solar CUF (%)", 12.0, 22.0, 16.0, 0.5, key="sz_cuf") / 100
    with sz_b2: sz_solar_capex = st.number_input("Solar CAPEX (THB/MW)", value=20_000_000,
                                                   step=500_000, format="%d", key="sz_sc2")
    with sz_b3: sz_bess_capex  = st.number_input("BESS CAPEX (THB/MWh) — Turnkey",
                                                   value=11_000_000, step=500_000,
                                                   format="%d", key="sz_bc",
                                                   help="Turnkey AC ~10-12M/MWh (v14 default)")
    with sz_b4: sz_working_days = st.number_input("Working Days/ปี", min_value=220, max_value=262,
                                                   value=working_days, key="sz_wd")

    # ── SECTION 3-4: Bill Analysis ────────────────────────────────────────────
    st.divider()
    sz_avg_kw_onpeak = _on_peak_yr / max(sz_working_days * 13, 1)
    sz_on_off_ratio  = _on_peak_yr / max(_off_peak_yr, 1)
    sz_peak_avg_ratio = _max_kw_bill / max(sz_avg_kw_onpeak, 1)
    sz_energy_cost    = _on_peak_yr * on_peak_y1 + _off_peak_yr * off_peak_y1
    sz_demand_cost_yr = _max_kw_bill * 74.14 * 12
    sz_ft_est         = sz_energy_cost * 0.08
    sz_annual_bill    = sz_energy_cost + sz_demand_cost_yr + sz_ft_est

    st.subheader("📊 Section 3-4 — Bill Analysis")
    ba1, ba2, ba3, ba4, ba5 = st.columns(5)
    ba1.metric("★ Total Annual Bill",     fmt_thb(sz_annual_bill))
    ba2.metric("Energy On-Peak/ปี",       fmt_thb(_on_peak_yr * on_peak_y1))
    ba3.metric("Energy Off-Peak/ปี",      fmt_thb(_off_peak_yr * off_peak_y1))
    ba4.metric("Demand Charge/ปี",        fmt_thb(sz_demand_cost_yr))
    ba5.metric("Ft+Service (est 8%)",     fmt_thb(sz_ft_est))

    # TOU profile chart
    fig_tou = go.Figure()
    fig_tou.add_trace(go.Bar(
        x=["☀️ 09-17h\n(Solar+On-Peak)", "🔋 17-22h\n(Evening On-Peak)", "⚡ 22-09h\n(Off-Peak)"],
        y=[_on_peak_yr * 0.60, _on_peak_yr * 0.40, _off_peak_yr],
        marker_color=["#ffd166", "#ef476f", "#00b4d8"],
        text=[f"{_on_peak_yr*0.60:,.0f} kWh\nSolar offset",
              f"{_on_peak_yr*0.40:,.0f} kWh\nBESS discharge",
              f"{_off_peak_yr:,.0f} kWh\nBESS charge"],
        textposition="inside",
    ))
    fig_tou.update_layout(
        title="โครงสร้างการใช้ไฟตาม TOU — วันทำการ (ประมาณ)",
        yaxis_title="kWh/ปี", height=260, margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )

    bpi, bpii = st.columns([2, 1])
    with bpi:
        st.plotly_chart(fig_tou, use_container_width=True)
    with bpii:
        sz_lf = sz_avg_kw_onpeak / max(_max_kw_bill, 1)
        st.metric("On/Off kWh Ratio", f"{sz_on_off_ratio:.2f}",
                   delta="> 1.5 = TOU arbitrage ดี" if sz_on_off_ratio >= 1.5 else "< 1.5 = spread แคบ")
        st.metric("Peak/Avg kW Ratio", f"{sz_peak_avg_ratio:.2f}",
                   delta="> 1.3 = Peak shaving ดี" if sz_peak_avg_ratio >= 1.3 else "< 1.3 = load flat")
        st.metric("Avg kW (on-peak)", f"{sz_avg_kw_onpeak:.0f} kW")
        st.metric("Load Factor", f"{sz_lf*100:.0f}%",
                   delta="Flat" if sz_lf > 0.7 else ("Peaky" if sz_lf < 0.5 else "Normal"))

    # ── Run v14 Sizing ────────────────────────────────────────────────────────
    sz_result = recommend_sizing(
        annual_on_peak_kwh      = _on_peak_yr,
        annual_off_peak_kwh     = _off_peak_yr,
        peak_kw                 = _max_kw_bill,
        on_peak_rate            = on_peak_y1,
        off_peak_rate           = off_peak_y1,
        demand_rate_per_kw_mo   = 74.14,
        cuf                     = sz_cuf,
        working_days            = sz_working_days,
        rte                     = rte,
        solar_capex_per_mw      = sz_solar_capex,
        bess_capex_per_mwh      = sz_bess_capex,
        roof_area_m2            = sz_roof,
        self_consumption_target = sz_self_con,
        demand_reduction_target = sz_dem_red,
    )
    r = sz_result  # shorthand

    # ── SECTION 5: Solar Recommendation ──────────────────────────────────────
    st.divider()
    st.subheader("☀️ Section 5 — Solar PV ที่แนะนำ")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("★ Recommended Solar", f"{r['solar_mw']:.3f} MW")
    s2.metric("Solar kWp", f"{r['solar_mw']*1000:.0f} kWp")
    s3.metric("Expected Y1 Production", f"{r['solar_kwh_yr']:,.0f} kWh")
    s4.metric("Solar Y1 Saving (est)", fmt_thb(r["solar_save"]))
    s5.metric("Solar Simple Payback", f"{r['solar_pb']:.1f} ปี" if r["solar_pb"] else ">30Y")

    sz_s1, sz_s2 = st.columns(2)
    with sz_s1:
        st.info(
            f"Solar MW จาก Load Analysis: **{r['solar_mw_from_load']:.3f} MW**\n\n"
            + (f"Solar MW จาก Roof Limit ({sz_roof:,} ม²): **{r['solar_mw_from_roof']:.3f} MW**\n\n"
               if r["solar_mw_from_roof"] else "Roof: ไม่จำกัด\n\n")
            + f"✅ ใช้ค่า: **{r['solar_mw']:.3f} MW** (เลือก min)\n\n"
            f"Solar CAPEX: **{fmt_thb(r['solar_capex'])}**"
        )
    with sz_s2:
        st.caption(
            f"สูตร: Annual Daytime Load = {_on_peak_yr:,.0f} × 60% = {_on_peak_yr*0.6:,.0f} kWh/ปี\n"
            f"Solar Production Required = Daytime Load ÷ {sz_self_con*100:.0f}% = "
            f"{_on_peak_yr*0.6/sz_self_con:,.0f} kWh\n"
            f"Solar MW = {_on_peak_yr*0.6/sz_self_con:,.0f} ÷ ({sz_cuf*100:.0f}%×8760×1000) = "
            f"{r['solar_mw_from_load']:.3f} MW"
        )

    # ── SECTION 6: BESS Recommendation ───────────────────────────────────────
    st.divider()
    st.subheader("🔋 Section 6 — BESS ที่แนะนำ")

    # Score gauge
    score_color = "#06d6a0" if r["score"] >= 3 else ("#00b4d8" if r["score"] >= 2 else
                  ("#ffd166" if r["score"] >= 1.5 else "#ef476f"))
    sc_a, sc_b = st.columns([1, 2])
    with sc_a:
        fig_score = go.Figure(go.Indicator(
            mode="gauge+number",
            value=r["score"],
            title={"text": "BESS Score"},
            gauge={
                "axis": {"range": [0, 4]},
                "bar": {"color": score_color},
                "steps": [
                    {"range": [0, 1.5], "color": "#fee8e8"},
                    {"range": [1.5, 2], "color": "#fff3cd"},
                    {"range": [2, 3],   "color": "#d1ecf1"},
                    {"range": [3, 4],   "color": "#d4edda"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "value": 2},
            }
        ))
        fig_score.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_score, use_container_width=True)
        st.markdown(f"**{r['bess_rec_label']}**")
        st.caption(
            f"On/Off Ratio: {r['on_off_ratio']:.2f} "
            f"({'✅ >1.5' if r['on_off_ratio']>=1.5 else '⚠️ <1.5'})\n\n"
            f"Peak/Avg Ratio: {r['peak_avg_ratio']:.2f} "
            f"({'✅ >1.3' if r['peak_avg_ratio']>=1.3 else '⚠️ <1.3'})"
        )

    with sc_b:
        b1, b2, b3 = st.columns(3)
        b1.metric("★ BESS MW", f"{r['bess_mw']:.2f} MW")
        b2.metric("★ BESS MWh", f"{r['bess_mwh']:.2f} MWh")
        b3.metric("Duration", f"{r['bess_hr']} hr")
        b4, b5, b6 = st.columns(3)
        b4.metric("BESS CAPEX", fmt_thb(r["bess_capex"]))
        b5.metric("Arb Save/ปี (est)", fmt_thb(r["bess_arb_save"]),
                   help=f"MWh × {sz_working_days}d × 2cycles × spread × 70% realization")
        b6.metric("Demand Save/ปี (est)", fmt_thb(r["demand_save"]),
                   help=f"{r['demand_kw_cut']:.0f} kW shaved × 74.14 × 12 × 70%")
        st.info(
            f"BESS MW from Peak Shaving: **{r['bess_mw_from_peak']:.3f} MW** "
            f"({_max_kw_bill:.0f} kW × {sz_dem_red*100:.0f}%)\n\n"
            f"BESS MW from Solar Shift (40%): **{r['bess_mw_from_solar']:.3f} MW** "
            f"({r['solar_mw']:.3f} MW Solar × 40%)\n\n"
            f"→ ใช้ค่ามากกว่า + Round ↑ 0.25 MW = **{r['bess_mw']:.2f} MW**"
        )
        st.metric("BESS Payback (est)", f"{r['bess_pb']:.1f} ปี" if r["bess_pb"] else ">30Y")

    # ── SECTION 7: Combined ───────────────────────────────────────────────────
    st.divider()
    st.subheader("🏆 Section 7 — Solar+BESS Combined Package")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total CAPEX",      fmt_thb(r["total_capex"]))
    c2.metric("Y1 Total Save",    fmt_thb(r["total_save"]))
    c3.metric("Payback (No BOI)", f"{r['pb_no_boi']:.1f} ปี" if r["pb_no_boi"] else ">30Y")
    c4.metric("BOI Tax Shield",   fmt_thb(r["boi_tax_shield"]),
               help=f"BOI Eligible {fmt_thb(r['boi_eligible'])} × 50% × 20% CIT")
    c5.metric("★ Payback (BOI)",  f"{r['pb_with_boi']:.1f} ปี" if r["pb_with_boi"] else ">30Y",
               delta=f"ลดลง {r['pb_no_boi']-r['pb_with_boi']:.1f}Y จาก BOI" if r["pb_no_boi"] and r["pb_with_boi"] else None)
    c6.metric("Bill Reduction",   f"{r['bill_reduction']*100:.0f}%")

    # CAPEX vs Saving chart
    fig_pkg = go.Figure()
    fig_pkg.add_trace(go.Bar(name="Solar CAPEX", x=["Package"], y=[r["solar_capex"]],
                              marker_color="#ffd166", text=[fmt_thb(r["solar_capex"])], textposition="inside"))
    fig_pkg.add_trace(go.Bar(name="BESS CAPEX", x=["Package"], y=[r["bess_capex"]],
                              marker_color="#00b4d8", text=[fmt_thb(r["bess_capex"])], textposition="inside"))
    fig_pkg.add_trace(go.Bar(name="Solar Save/Y", x=["Y1 Saving"], y=[r["solar_save"]],
                              marker_color="#ffd166", text=[fmt_thb(r["solar_save"])], textposition="inside"))
    fig_pkg.add_trace(go.Bar(name="BESS Save/Y", x=["Y1 Saving"], y=[r["bess_total_save"]],
                              marker_color="#06d6a0", text=[fmt_thb(r["bess_total_save"])], textposition="inside"))
    fig_pkg.update_layout(barmode="stack", height=280, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_pkg, use_container_width=True)

    # ── SECTION 8: Pitch ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("🎯 Section 8 — Sales Pitch (auto-generated)")
    p1, p2 = st.columns(2)
    with p1:
        st.success(f"☀️ **Solar:** {r['solar_pitch']}")
        st.info(f"🔋 **BESS:** {r['bess_pitch']}")
    with p2:
        st.warning(f"🏆 **Bundle Pitch:** {r['combined_pitch']}")
        st.caption(
            "📋 **Next Steps:**\n"
            "1. ขอ AMR 15-min data 1-3 เดือน → re-calc แม่นยำ\n"
            "2. โอนค่า Solar MW + BESS MWh ไปกรอกใน **☀️ Solar+BESS Bundle** tab\n"
            "3. ปรับ Realization Factor ตาม risk appetite ของลูกค้า\n"
            "4. ยื่น BOI ถ้า Solar MW ≥ 0.1 MW"
        )

    # ── Custom Sizing ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔧 Custom Sizing — ลองปรับขนาดเองแล้วดูผล")
    cz1, cz2 = st.columns(2)
    with cz1:
        custom_solar_kwp = st.number_input(
            "Solar PV (kWp)", min_value=10, max_value=50000,
            value=int(r["solar_mw"] * 1000), step=10, key="cz_solar"
        )
    with cz2:
        custom_bess_mwh = st.number_input(
            "BESS (MWh)", min_value=0.1, max_value=100.0,
            value=round(r["bess_mwh"], 1), step=0.25, format="%.2f", key="cz_bess"
        )

    cz_solar_mw   = custom_solar_kwp / 1000
    cz_solar_cap  = cz_solar_mw * sz_solar_capex
    cz_bess_cap   = custom_bess_mwh * sz_bess_capex
    cz_total_cap  = cz_solar_cap + cz_bess_cap
    cz_solar_kwh  = cz_solar_mw * 1000 * sz_cuf * 8760
    cz_solar_save = cz_solar_kwh * (0.85 * on_peak_y1 + 0.15 * 2.2)
    cz_spread     = max(on_peak_y1 - off_peak_y1 / rte, 0)
    cz_bess_save  = custom_bess_mwh * cz_spread * sz_working_days * 2 * 0.70
    cz_dem_save   = (cz_solar_mw * 1000 * sz_dem_red) * 74.14 * 12 * 0.70
    cz_total_save = cz_solar_save + cz_bess_save + cz_dem_save
    cz_pb_no_boi  = cz_total_cap / cz_total_save if cz_total_save > 0 else None
    cz_boi_elig   = min(cz_bess_cap, 12_000_000 * cz_solar_mw)
    cz_boi_shield = cz_boi_elig * 0.50 * 0.20
    cz_pb_boi     = (cz_total_cap - cz_boi_shield) / cz_total_save if cz_total_save > 0 else None

    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("Solar MW",         f"{cz_solar_mw:.3f}")
    r2.metric("BESS MWh",         f"{custom_bess_mwh:.2f}")
    r3.metric("Total CAPEX",      fmt_thb(cz_total_cap))
    r4.metric("Y1 Saving (est)",  fmt_thb(cz_total_save))
    r5.metric("Payback (No BOI)", f"{cz_pb_no_boi:.1f} ปี" if cz_pb_no_boi else ">30Y")
    r6.metric("★ Payback (BOI)",  f"{cz_pb_boi:.1f} ปี" if cz_pb_boi else ">30Y")

    st.caption(
        f"Bill Reduction: **{cz_total_save/max(sz_annual_bill,1)*100:.0f}%** | "
        f"BOI Tax Shield: **{fmt_thb(cz_boi_shield)}** | "
        f"Solar:BESS = 1:{custom_bess_mwh*1000/max(custom_solar_kwp,1):.1f} kWp:kWh"
    )

    with st.expander("📖 อ่านรายละเอียดสูตรคำนวณ"):
        arb_margin_val = on_peak_y1 - off_peak_y1 / rte
        st.markdown(f"""
**BESS Score (rule-based):**
| ตัวชี้วัด | ค่า | คะแนน |
|---|---|---|
| On/Off Ratio = {r['on_off_ratio']:.2f} | {'≥1.5' if r['on_off_ratio']>=1.5 else '≥1.0' if r['on_off_ratio']>=1.0 else '<1.0'} | +{'1.5' if r['on_off_ratio']>=1.5 else '1.0' if r['on_off_ratio']>=1.0 else '0'} |
| Peak/Avg Ratio = {r['peak_avg_ratio']:.2f} | {'≥1.5' if r['peak_avg_ratio']>=1.5 else '≥1.3' if r['peak_avg_ratio']>=1.3 else '<1.3'} | +{'1.5' if r['peak_avg_ratio']>=1.5 else '1.0' if r['peak_avg_ratio']>=1.3 else '0'} |
| **รวม** | | **{r['score']:.1f}** |

≥3 = Strong | ≥2 = Recommend | ≥1.5 = Optional | <1.5 = Not recommended

**Solar sizing:** MW = (On-Peak×60% ÷ Self-Con%) ÷ (CUF×8760×1000)

**BESS sizing:** max(Peak shaving={r['bess_mw_from_peak']:.3f}MW, Solar shift={r['bess_mw_from_solar']:.3f}MW) → round↑0.25

**BESS savings (2 cycles):** MWh × spread × {sz_working_days}d × 2 cycles × 70%
Arb Spread = {on_peak_y1:.4f} − {off_peak_y1:.4f}/{rte:.2f} = **{arb_margin_val:.4f} THB/kWh**

**BOI Tax Shield:** min(BESS CAPEX, 12M × Solar MW) × 50% × 20% CIT
        """)
