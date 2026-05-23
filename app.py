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
    boi_scheme,              # "A" = 3Y/50% cap | "B" = 8Y/100% cap
    realization_factor,
    solar_charge_ratio=1.0,  # 0–1 : สัดส่วน BESS ที่ชาร์จจาก Solar (ฟรี) vs Grid
):
    solar_mw  = solar_kwp / 1000
    bess_mwh  = bess_kwh  / 1000

    solar_capex = solar_mw  * solar_capex_per_mw
    bess_capex  = bess_mwh  * bess_capex_per_mwh
    total_capex = solar_capex + bess_capex

    annual_om        = total_capex * om_pct
    bundled_om_total = annual_om * bundled_om_years
    total_investment = total_capex + bundled_om_total

    # BOI — BESS only (Solar ไม่ได้ BOI ตั้งแต่ ก.ค. 2568)
    bess_eligible_per_mwh = 12_000_000   # ≤ 12M THB per MW (ตาม BOI)
    bess_eligible = min(bess_capex, bess_eligible_per_mwh * bess_mwh)
    if boi_scheme == "A (3Y/50%)":
        boi_years, boi_cap_ratio = 3, 0.50
    else:
        boi_years, boi_cap_ratio = 8, 1.00
    boi_cap = bess_eligible * boi_cap_ratio

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

        # --- Revenue streams ---
        solar_kwh_y  = solar_kwh_y1 * sol_deg
        solar_save   = solar_kwh_y  * avg * realization_factor

        # BESS effective margin ขึ้นกับ solar_charge_ratio:
        #   ส่วน Solar (ฟรี) → margin = on-peak (ไม่หัก charging cost)
        #   ส่วน Grid (off-peak) → margin = on-peak − off-peak/RTE (spread ปกติ)
        bess_kwh_y   = (bess_kwh * rte * cycles_day * days_year * bess_deg)
        margin_solar = pk                        # ชาร์จฟรี → เก็บ on-peak เต็ม
        margin_grid  = pk - (op / rte)           # ชาร์จ off-peak → spread
        eff_margin   = solar_charge_ratio * margin_solar + (1 - solar_charge_ratio) * margin_grid
        bess_arb     = bess_kwh_y * eff_margin * realization_factor

        demand_save  = demand_cut_kw * demand_rate_per_kw_mo * 12

        om_cost = 0.0 if y <= bundled_om_years else annual_om
        pre_tax = solar_save + bess_arb + demand_save - om_cost

        full_cit = max(0.0, pre_tax * cit)
        if cum_exempt < boi_cap and y <= boi_years:
            exempt   = min(full_cit, boi_cap - cum_exempt)
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
            "Solar Save (THB)": solar_save,
            "BESS kWh Out": bess_kwh_y,
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
    boi_cap = net_capex * ass["boi_cap_x"]

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
    cum_nb = cum_epc = cum_ppa = -customer_price
    cum_exempt_epc = cum_exempt_ppa = 0.0

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

        # Aux load cost
        aux_cost = aux_kw * 8760 * off_peak / 1000

        om_cost = (0.0 if y <= ass["bundled_om_years"] else annual_om) + aux_cost
        pre_tax = gross_saving - om_cost

        tax_nb = max(0.0, pre_tax * ass["cit"])

        if cum_exempt_epc < boi_cap and y <= ass["boi_epc_y"]:
            exempt_epc = min(pre_tax * ass["cit"], boi_cap - cum_exempt_epc)
            cum_exempt_epc += exempt_epc
            tax_epc = max(0.0, pre_tax * ass["cit"] - exempt_epc)
        else:
            tax_epc = tax_nb

        if cum_exempt_ppa < boi_cap and y <= ass["boi_ppa_y"]:
            exempt_ppa = min(pre_tax * ass["cit"], boi_cap - cum_exempt_ppa)
            cum_exempt_ppa += exempt_ppa
            tax_ppa = max(0.0, pre_tax * ass["cit"] - exempt_ppa)
        else:
            tax_ppa = tax_nb

        net_nb  = pre_tax - tax_nb
        net_epc = pre_tax - tax_epc
        net_ppa = pre_tax - tax_ppa
        cum_nb  += net_nb
        cum_epc += net_epc
        cum_ppa += net_ppa

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
            "Tax EPC": tax_epc,
            "Tax PPA": tax_ppa,
            "Net CF Non-BOI": net_nb,
            "Net CF EPC": net_epc,
            "Net CF PPA": net_ppa,
            "Cum CF Non-BOI": cum_nb,
            "Cum CF EPC": cum_epc,
            "Cum CF PPA": cum_ppa,
        })

    df = pd.DataFrame(rows)

    def payback(col):
        s = df[col]
        idx = s[s >= 0].index
        return int(df.loc[idx[0], "Year"]) if len(idx) else None

    def irr_at(col, yr):
        cfs = [-customer_price] + list(df[df["Year"] <= yr][col])
        return safe_irr(cfs)

    def npv_at(col, yr, wacc):
        cfs = [-customer_price] + list(df[df["Year"] <= yr][col])
        return sum(c/(1+wacc)**i for i, c in enumerate(cfs))

    metrics = {}
    for tag, col in [("NB","Net CF Non-BOI"),("EPC","Net CF EPC"),("PPA","Net CF PPA")]:
        metrics[tag] = {
            "payback": payback(f"Cum CF {col.replace('Net CF ','')}"),
            "irr_15": irr_at(col, 15),
            "irr_20": irr_at(col, 20),
            "irr_30": irr_at(col, 30),
            "npv_30": npv_at(col, 30, ass["wacc"]),
            "cum_15": df[df["Year"]==15][f"Cum CF {col.replace('Net CF ','')}"].values[0],
            "cum_30": df[df["Year"]==30][f"Cum CF {col.replace('Net CF ','')}"].values[0],
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

    ppa_cust_cum30 = float(df["Cum CF PPA"].iloc[-1]) * ass["ppa_share"]
    ppa_owner_cum30 = float(df["Cum CF PPA"].iloc[-1]) * (1 - ass["ppa_share"])

    return {
        "df": df,
        "customer_price": customer_price,
        "net_capex": net_capex,
        "equip_capex": equip_capex,
        "annual_om": annual_om,
        "boi_duty_saving": boi_duty_saving,
        "metrics": metrics,
        "lease": lease,
        "ppa_cust_cum30": ppa_cust_cum30,
        "ppa_owner_cum30": ppa_owner_cum30,
        "cap_kwh": cap_kwh,
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
    pub_holidays  = st.number_input("วันหยุดนักขัตฤกษ์/ปี (ตกวันทำการ)", min_value=0, max_value=20, value=13)
    working_days  = int(weekdays_yr - pub_holidays)
    st.info(f"วันที่ Arbitrage ได้จริง: **{working_days} วัน/ปี** (vs 365 วัน = overestimate {365-working_days} วัน)")

    st.markdown("**⚡ Energy Loss (แยก Charge / Discharge)**")
    charge_eff    = st.slider("Charging Efficiency ηc (%)", 90, 99, 97, 1) / 100
    discharge_eff = st.slider("Discharging Efficiency ηd (%)", 90, 99, 97, 1) / 100
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
    st.subheader("🏛️ BOI & ภาษี")
    cit         = st.slider("CIT Rate (%)", 0, 30, 20) / 100
    wacc        = st.slider("WACC / Discount Rate (%)", 3.0, 15.0, 7.0, 0.5) / 100
    boi_epc_y   = st.number_input("BOI EPC – Section 7 (ปีที่ยกเว้น CIT)", 1, 10, 3)
    boi_ppa_y   = st.number_input("BOI PPA – Section 7.1 (ปีที่ยกเว้น CIT)", 1, 15, 8)
    boi_cap_x   = st.slider("BOI Cap (× CAPEX)", 0.5, 2.0, 1.0, 0.1)
    boi_duty_pct= st.slider("BOI Import Duty Saving (%)", 0.0, 10.0, 5.0, 0.5) / 100

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
    "boi_epc_y": boi_epc_y, "boi_ppa_y": boi_ppa_y,
    "boi_cap_x": boi_cap_x, "boi_duty_pct": boi_duty_pct,
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📋 Cash Flow ตาราง", "🔄 เปรียบเทียบ 4 Products", "📄 Bill Analysis", "☀️ Solar+BESS Bundle"])

# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    # BOI Scenario selector
    scenario_label = st.radio("BOI Scenario", ["Non-BOI", "EPC (Section 7)", "PPA (Section 7.1)"], horizontal=True)
    scenario_key = {"Non-BOI": "NB", "EPC (Section 7)": "EPC", "PPA (Section 7.1)": "PPA"}[scenario_label]
    sm = m[scenario_key]
    cum_col = {"NB":"Cum CF Non-BOI","EPC":"Cum CF EPC","PPA":"Cum CF PPA"}[scenario_key]
    net_col = {"NB":"Net CF Non-BOI","EPC":"Net CF EPC","PPA":"Net CF PPA"}[scenario_key]

    st.subheader("📌 Key Metrics")
    c1,c2,c3,c4,c5 = st.columns(5)
    pb = sm["payback"]
    c1.metric("Payback Year", f"ปีที่ {pb}" if pb else "ไม่ถึง 30Y")
    irr15 = sm["irr_15"]
    c2.metric("IRR 15Y", f"{irr15*100:.1f}%" if irr15 else "N/A")
    irr30 = sm["irr_30"]
    c3.metric("IRR 30Y", f"{irr30*100:.1f}%" if irr30 else "N/A")
    c4.metric("NPV 30Y", fmt_thb(sm["npv_30"]))
    c5.metric("Cum CF 30Y", fmt_thb(sm["cum_30"]))

    st.divider()
    c6,c7,c8 = st.columns(3)
    c6.metric("Customer Price (upfront)", fmt_thb(result["customer_price"]))
    c7.metric("NET CAPEX (หลัง BOI Duty)", fmt_thb(result["net_capex"]))
    c8.metric("BOI Duty Saving", fmt_thb(result["boi_duty_saving"]))

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
            pb_val = float(df[df["Year"]==pb][cum_col].values[0])
            fig_cum.add_trace(go.Scatter(
                x=[pb], y=[pb_val], mode="markers+text",
                marker=dict(color="gold", size=12, symbol="star"),
                text=[f"Payback Y{pb}"], textposition="top center",
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
    st.subheader("🏛️ เปรียบเทียบ 3 BOI Scenarios")
    boi_rows = []
    for label, key in [("Non-BOI","NB"),("EPC (Section 7)","EPC"),("PPA (Section 7.1)","PPA")]:
        sm2 = m[key]
        boi_rows.append({
            "Scenario": label,
            "Payback Year": f"ปีที่ {sm2['payback']}" if sm2["payback"] else ">30Y",
            "IRR 15Y": f"{sm2['irr_15']*100:.1f}%" if sm2["irr_15"] else "N/A",
            "IRR 20Y": f"{sm2['irr_20']*100:.1f}%" if sm2["irr_20"] else "N/A",
            "IRR 30Y": f"{sm2['irr_30']*100:.1f}%" if sm2["irr_30"] else "N/A",
            "NPV 30Y": fmt_thb(sm2["npv_30"]),
            "Cum CF 15Y": fmt_thb(sm2["cum_15"]),
            "Cum CF 30Y": fmt_thb(sm2["cum_30"]),
        })
    st.dataframe(pd.DataFrame(boi_rows).set_index("Scenario"), use_container_width=True)

    # Leasing & PPA summary
    st.divider()
    col_lea, col_ppa = st.columns(2)
    with col_lea:
        st.subheader("🏦 Leasing Model (EPC 3Y BOI)")
        lea = result["lease"]
        st.table(pd.DataFrame({
            "รายการ": ["Down Payment","Financed Amount","Monthly PMT","Annual PMT"],
            "มูลค่า (THB)": [fmt_thb(lea["down"]), fmt_thb(lea["financed"]),
                             fmt_thb(lea["monthly_pmt"]), fmt_thb(lea["annual_pmt"])]
        }).set_index("รายการ"))

    with col_ppa:
        st.subheader("🤝 PPA / ESCO Model (PPA 8Y BOI)")
        y1_gross = float(df["Gross Saving"].iloc[0])
        st.table(pd.DataFrame({
            "รายการ": [f"Customer Share ({ppa_share*100:.0f}%)","Owner Share","Y1 Customer Saving","Cum Customer Saving 30Y","Cum Owner Revenue 30Y"],
            "มูลค่า": [f"{ppa_share*100:.0f}%", f"{(1-ppa_share)*100:.0f}%",
                       fmt_thb(y1_gross * ppa_share),
                       fmt_thb(result["ppa_cust_cum30"]),
                       fmt_thb(result["ppa_owner_cum30"])]
        }).set_index("รายการ"))

# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"📋 Year-by-Year Cash Flow — {selected_product}")
    scenario_label2 = st.radio("BOI Scenario", ["Non-BOI", "EPC (Section 7)", "PPA (Section 7.1)"], horizontal=True, key="t2")
    sk2 = {"Non-BOI": "NB", "EPC (Section 7)": "EPC", "PPA (Section 7.1)": "PPA"}[scenario_label2]
    net_col2 = {"NB":"Net CF Non-BOI","EPC":"Net CF EPC","PPA":"Net CF PPA"}[sk2]
    cum_col2 = {"NB":"Cum CF Non-BOI","EPC":"Cum CF EPC","PPA":"Cum CF PPA"}[sk2]

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
        title=f"Energy Flow per ปี (working_days={int(y1r['Working Days'])}วัน, ηc={charge_eff*100:.0f}%, ηd={discharge_eff*100:.0f}%, RTE={rte*100:.1f}%)",
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
                 "Arb Margin C2 (THB/kWh)","Gross Saving","Aux Cost","OM Cost","Pre-Tax CF",net_col2,cum_col2]
    display_cols = [c for c in base_cols if c in df.columns]
    df_display = df[display_cols].copy()

    # Format numbers
    for col in ["Energy Out (kWh)","Gross Saving","OM Cost","Pre-Tax CF",net_col2,cum_col2]:
        df_display[col] = df_display[col].map(lambda x: f"{x:,.0f}")
    for col in ["On-Peak (THB/kWh)","Off-Peak (THB/kWh)","Arb Margin (THB/kWh)"]:
        df_display[col] = df_display[col].map(lambda x: f"{x:.4f}")

    df_display["Year"] = df_display["Year"].map(lambda x: f"Y{x}")
    st.dataframe(df_display.set_index("Year"), use_container_width=True, height=600)

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "bess_cashflow.csv", "text/csv")

# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🔄 เปรียบเทียบทุก Product ด้วย Assumptions เดียวกัน")
    scenario_label3 = st.radio("BOI Scenario", ["Non-BOI", "EPC (Section 7)", "PPA (Section 7.1)"], horizontal=True, key="t3")
    sk3 = {"Non-BOI": "NB", "EPC (Section 7)": "EPC", "PPA (Section 7.1)": "PPA"}[scenario_label3]
    cum_col3 = {"NB":"Cum CF Non-BOI","EPC":"Cum CF EPC","PPA":"Cum CF PPA"}[sk3]

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
            "Payback": f"ปีที่ {sm3['payback']}" if sm3["payback"] else ">30Y",
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
    st.caption("Solar ชาร์จแบตฟรี (ต้นทุน 0) + BESS Arbitrage + Demand Charge Reduction · BOI เฉพาะ BESS")

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
        st.markdown("**🏛️ BOI & ปัจจัยอื่น**")
        sb_boi_scheme   = st.radio("BOI Scheme (BESS only)", ["A (3Y/50%)", "B (8Y/100%)"])
        sb_real_factor  = st.slider("Realization Factor (%)", 50, 100, 70, 5) / 100
        sb_om_pct       = st.slider("OM % of Total CAPEX (Y11+)", 0.3, 2.0, 0.5, 0.1, key="sb_om") / 100
        sb_bundled_om   = st.number_input("Bundled OM ปี", min_value=1, max_value=15, value=10, key="sb_bom")

    st.divider()
    col_ratio, col_loan = st.columns(2)
    with col_ratio:
        st.markdown("**☀️ Solar:Grid Operation Ratio**")
        st.caption("% ของพลังงานที่ BESS ชาร์จมาจาก Solar (ฟรี) vs ดึงจาก Grid (เสียค่า off-peak)")
        solar_charge_ratio = st.slider("BESS ชาร์จจาก Solar (%)", 0, 100, 100, 5, key="sb_scr") / 100
        grid_charge_ratio  = 1 - solar_charge_ratio
        sc1, sc2 = st.columns(2)
        sc1.metric("จาก Solar", f"{solar_charge_ratio*100:.0f}%")
        sc2.metric("จาก Grid (off-peak)", f"{grid_charge_ratio*100:.0f}%")

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
        solar_kwp          = sol_kwp,
        solar_capex_per_mw = sol_capex_mw,
        solar_cuf          = sol_cuf,
        solar_deg_pct      = sol_deg_pct,
        bess_kwh           = bess_kwh_input,
        bess_capex_per_mwh = bess_capex_mwh,
        bess_deg_curve     = bess_deg_curve_sol,
        peak_rate_y1       = sb_peak_y1,
        off_peak_rate_y1   = sb_offpk_y1,
        tou_esc            = sb_tou_esc,
        demand_cut_kw      = sb_demand_cut,
        demand_rate_per_kw_mo = sb_demand_rate,
        rte                = rte,
        cycles_day         = cycles_day,
        days_year          = days_year,
        om_pct             = sb_om_pct,
        bundled_om_years   = sb_bundled_om,
        cit                = cit,
        wacc               = wacc,
        boi_scheme         = sb_boi_scheme,
        realization_factor = sb_real_factor,
        solar_charge_ratio = solar_charge_ratio,
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
    k9.metric("BOI Cap (BESS)", fmt_thb(sb_m["boi_cap"]))

    st.divider()

    # ── Y1 Revenue Breakdown ──
    y1 = sb_df.iloc[0]
    st.subheader("💰 Y1 Revenue Breakdown")
    rev_cols = st.columns(4)
    rev_cols[0].metric("Solar Saving Y1", fmt_thb(y1["Solar Save (THB)"]))
    rev_cols[1].metric("BESS Arbitrage Y1", fmt_thb(y1["BESS Arb (THB)"]))
    rev_cols[2].metric("Demand Save Y1", fmt_thb(y1["Demand Save (THB)"]))
    rev_cols[3].metric("Total Gross Saving Y1", fmt_thb(y1["Solar Save (THB)"] + y1["BESS Arb (THB)"] + y1["Demand Save (THB)"]))

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
    bess_sk = "EPC" if "EPC" in sb_boi_scheme else "PPA"
    bess_cum_col = f"Cum CF {bess_sk}"

    comp_cum_fig = go.Figure()
    comp_cum_fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    comp_cum_fig.add_trace(go.Scatter(
        x=[0]+list(sb_df["Year"]),
        y=[-sb_m["total_investment"]]+list(sb_df["Cum CF (THB)"]),
        name=f"☀️ Solar+BESS Bundle ({sb_boi_scheme})",
        line=dict(color="#ffd166", width=2.5)
    ))
    comp_cum_fig.add_trace(go.Scatter(
        x=[0]+list(r_bess_only["df"]["Year"]),
        y=[-r_bess_only["customer_price"]]+list(r_bess_only["df"][bess_cum_col]),
        name=f"🔋 BESS Only — {selected_product.split('(')[0].strip()} (EPC BOI)",
        line=dict(color="#00b4d8", width=2.5, dash="dash")
    ))
    comp_cum_fig.update_layout(xaxis_title="ปี", yaxis_title="THB (Cumulative CF)",
                                height=380, margin=dict(l=10,r=10,t=10,b=30))
    st.plotly_chart(comp_cum_fig, use_container_width=True)

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
            "Payback": f"{r_bess_only['metrics']['EPC']['payback']}Y" if r_bess_only["metrics"]["EPC"]["payback"] else ">30Y",
            "IRR 15Y": f"{r_bess_only['metrics']['EPC']['irr_15']*100:.1f}%" if r_bess_only["metrics"]["EPC"]["irr_15"] else "N/A",
            "IRR 30Y": f"{r_bess_only['metrics']['EPC']['irr_30']*100:.1f}%" if r_bess_only["metrics"]["EPC"]["irr_30"] else "N/A",
            "NPV 30Y": fmt_thb(r_bess_only["metrics"]["EPC"]["npv_30"]),
            "Cum CF 30Y": fmt_thb(r_bess_only["metrics"]["EPC"]["cum_30"]),
        }
    ]).set_index("รายการ")
    st.dataframe(comp_table, use_container_width=True)

    # ── Full table ──
    with st.expander("📋 ดูตาราง Cash Flow ทั้ง 30 ปี"):
        fmt_cols = ["Solar kWh","Solar Save (THB)","BESS kWh Out","BESS Arb (THB)",
                    "Demand Save (THB)","OM Cost (THB)","Pre-Tax CF (THB)","Net CF (THB)","Cum CF (THB)"]
        sb_display = sb_df.copy()
        for c in fmt_cols:
            sb_display[c] = sb_display[c].map(lambda x: f"{x:,.0f}")
        sb_display["Sol Deg"] = sb_display["Sol Deg"].map(lambda x: f"{x:.4f}")
        sb_display["BESS Deg"] = sb_display["BESS Deg"].map(lambda x: f"{x:.4f}")
        sb_display["Year"] = sb_display["Year"].map(lambda x: f"Y{x}")
        st.dataframe(sb_display.set_index("Year"), use_container_width=True, height=500)

        csv_sol = sb_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv_sol, "solar_bess_cashflow.csv", "text/csv")
