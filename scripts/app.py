import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model_hospitalization import run_model

# --------------------------
# 1. LOAD + ALIGN DATA
# --------------------------
#region
df_hosp = pd.read_csv("data_processed/hospitalizations_clean.csv")
df_cases = pd.read_csv("data_processed/cases_clean.csv")

df_hosp["DATE"] = pd.to_datetime(df_hosp["DATE"])
df_cases["DATE"] = pd.to_datetime(df_cases["DATE"])

# Merge on DATE (inner join keeps only common dates)
df = pd.merge(df_hosp, df_cases, on="DATE", how="inner")

# Sort just in case
df = df.sort_values("DATE")

# Extract aligned series
dates = df["DATE"]
H = df["NEW_IN"].values
I = df["CASES"].values
#endregion

# --------------------------
# 2. BASIC PARAMETERS 
# --------------------------
#region
# Effectiveness (scenario)
#TODO: slider
E = 0.25  # 40% reduction in hospitalization risk? going with a bit less first

# Assumption - Average delay infection → hospitalization (approx)
delay_inf_to_hosp = 7  # days 

#TODO: find out
high_risk_fraction = 0.05  # e.g. 10% of infections are eligible (scenario)

## Supply constants ##
# From Elise: 3 donations × 500 mL = 1.5 L per donor
# Total doses = total plasma volume / dose volume
# current spray = 600 µL per nostril = 1.2 mL per dose 
# (600 is the value for the current safety study, but then they need to assess volumes)
# 1.5 L / 0.0012 L ≈ 1250 doses per donor
# treatment: 2 doses/day for 3 months (≈90 days) (so 180 doses per patient)
# patients_per_donor ≈ 1250 / 180 ≈ 7 people
donation_volume = 0.5 #500 mL
#TODO: slider with between 100 µL per nostril to 600 µL per nostril - simulation of their volume study
dose_volume = 0.0012 # 600 µL per nostril = 1.2 mL per dose 
#TODO: radio button of how many donations we consider (1 to 3)
donations_per_donor = 1
doses_per_patient_per_day = 2
treatment_duration = 90  # days

#TODO: slider
capacity_per_day = 200  # TEMP placeholder (maximum donations/day)

#TODO: find out
# this is taking into account new variants and that the antigens keeps changing, but maybe it is too conservative
ccp_lifetime = 90  # days plasma remains clinically relevant

# Assumption of donation window:
#  People can donate in a window between [window_start] and [window_end] days post-infection
#TODO: FIND OUT
window_start = 30
window_end = 50

# Donor rate (% of the recovered that actually donate)
#TODO: slider
potential_donor_rate = 0.1  # 10% of recovered donate (in Belgium - Elise was using Flanders)
over_titre_donor_rate = 0.2 # 20% of the donors have antibody titres above 20 µg/mL (the threshold used for CP in our hamster study; see the EBioMedicine paper). 
# This estimate is based on donor data from the Meuri & Confident studies in 2021 (approximately n = 70).

## Adoption constants
#TODO: slider
A_max = 1  # max 50% adoption
# Timing
t_start = 60   # days after start
T_rollout = 200  # days to reach max


# Peak hospitalizations (for normalization)
H_peak = np.max(H)
#endregion

# ------------------------------------------------------
st.title(
    "Intranasal CCP Prophylaxis Model"
)

st.sidebar.header(
    "Scenario Parameters"
)

E = st.sidebar.slider(
    "Effectiveness",
    min_value=0.05,
    max_value=0.70,
    value=0.25,
    step=0.01
)

high_risk_fraction = st.sidebar.slider(
    "High-risk fraction",
    min_value=0.01,
    max_value=0.50,
    value=0.05,
    step=0.01
)

dose_volume_ul = st.sidebar.slider(
    "Dose volume (µL)",
    min_value=100,
    max_value=600,
    value=600,
    step=50
)

dose_volume = dose_volume_ul / 500000

capacity_per_day = st.sidebar.slider(
    "Maximum donations/day",
    min_value=50,
    max_value=1000,
    value=900,
    step=50
)

potential_donor_rate = st.sidebar.slider(
    "Potential donor rate",
    min_value=0.01,
    max_value=1.00,
    value=0.10,
    step=0.01
)

A_max = st.sidebar.slider(
    "Maximum adoption",
    min_value=0.05,
    max_value=1.00,
    value=1.00,
    step=0.05
)

results = run_model(
    H=H,
    I=I,
    E=E,
    A_max=A_max,
    capacity_per_day=capacity_per_day,
    treatment_duration=treatment_duration,
    ccp_lifetime=ccp_lifetime,
    potential_donor_rate=potential_donor_rate,
    over_titre_donor_rate=over_titre_donor_rate,
    doses_per_patient_per_day=doses_per_patient_per_day,
    donation_volume=donation_volume,
    donations_per_donor=donations_per_donor,
    dose_volume=dose_volume,
    high_risk_fraction=high_risk_fraction,
    delay_inf_to_hosp=delay_inf_to_hosp,
    t_start=t_start,
    T_rollout=T_rollout,
    window_start=window_start,
    window_end=window_end,
    debug=False
)


st.header("Key Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Prevented hospitalizations",
        f"{np.sum(results['H_prevented']):,.0f}"
    )

with col2:
    st.metric(
        "Average coverage",
        f"{100*results['average_coverage']:.1f}%"
    )

with col3:
    st.metric(
        "Discard rate",
        f"{100*results['total_discarded']/results['total_produced']:.1f}%"
    )


# PLOTS
fig, ax = plt.subplots(
    figsize=(10,5)
)

ax.plot(
    dates,
    H,
    label="Hospitalizations",
    color="black"
)

ax.plot(
    dates,
    results["H_ccp"],
    label="With CCP",
    color="blue"
)

ax.set_ylabel(
    "Hospital admissions/day"
)

ax.legend()

ax.grid()

ax2 = ax.twinx()

ax2.plot(
    dates,
    results["H_reduction_pct"],
    color="red",
    linestyle="--"
)

ax2.set_ylabel(
    "% reduction"
)

st.pyplot(fig)


doses_per_treatment = (
    doses_per_patient_per_day
    * treatment_duration
)

report = pd.DataFrame({
    "Metric": [
        "Total treatment courses produced",
        "Total treatment courses delivered",
        "Total treatment courses discarded",
        "Average inventory",
        "Peak inventory",
        "Average coverage",
        "Peak coverage",
        "Stockout days",
        "% days supply limited",
        "Peak daily demand",
        "Peak treatment starts",
        "Peak treatment production",
        "Hospitalizations prevented",
        "Maximum daily hospitalizations reduction",        
        "Total hospitalizations reduction"
    ],
    "Value": 
[
        f"{results['total_produced']/doses_per_treatment:,.0f}",
        f"{results['total_delivered']/doses_per_treatment:,.0f}",
        f"{results['total_discarded']/doses_per_treatment:,.0f}",
        f"{results['average_stock']/doses_per_treatment:,.0f}",
        f"{results['maximum_stock']/doses_per_treatment:,.0f}",
        f"{100*results['average_coverage']:.1f}%",
        f"{100*results['peak_coverage']:.1f}%",
        results["stockout_days"],
        f"{100*results['fraction_supply_limited']:.1f}%",
        f"{results['peak_daily_demand']:.0f} patients/day",
        f"{results['peak_treatment_starts']:.0f} patients/day",
        f"{results['peak_daily_production']/doses_per_treatment:.0f} treatments/day",
        f"{np.sum(results['H_prevented']):,.0f}",
        f"{np.max(results['H_reduction_pct']):.1f}%",
        f"{100*np.sum(results['H_prevented'])/np.sum(H):.1f}%"
    ]

})

st.dataframe(
    report,
    use_container_width=True
)