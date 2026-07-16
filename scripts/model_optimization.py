from scipy.optimize import differential_evolution
from model_hospitalization import run_model
import pandas as pd
import numpy as np

## LOAD + ALIGN DATA
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
# Peak hospitalizations (for normalization)
H_peak = np.max(H)
#endregion


def objective(x):
    
    fixed_params = {
    "E": 0.4,
    "ccp_lifetime": 90,
    "donations_per_donor": 1,
    "dose_volume": 100/500000,
    "treatment_duration": 90,
    "over_titre_donor_rate": 0.20,
    "doses_per_patient_per_day": 2,
    "donation_volume": 0.5,
    "delay_inf_to_hosp": 7,
    "window_start": 30,
    "window_end": 50,
    }   


    hr, donor, cap, adopt, start, rollout = x

    results = run_model(
        H=H,
        I=I,
        E=fixed_params["E"],
        A_max=adopt,
        capacity_per_day=cap,
        treatment_duration=fixed_params["treatment_duration"],
        ccp_lifetime=fixed_params["ccp_lifetime"],
        potential_donor_rate=donor,
        over_titre_donor_rate=fixed_params["over_titre_donor_rate"],
        doses_per_patient_per_day=fixed_params["doses_per_patient_per_day"],
        donation_volume=fixed_params["donation_volume"],
        donations_per_donor=fixed_params["donations_per_donor"],
        dose_volume=fixed_params["dose_volume"],
        delay_inf_to_hosp=fixed_params["delay_inf_to_hosp"],
        t_start=int(start),
        T_rollout=int(rollout),
        window_start=fixed_params["window_start"],
        window_end=fixed_params["window_end"],
        debug=False
    )

    prevented = np.sum(
        results["H_prevented"]
    )

    # optimizer minimizes
    return -prevented

bounds = [
    (0.01, 1.00),   # potential_donor_rate
    (100, 1500),    # capacity_per_day
    (0.01, 1.00),   # A_max
    (1, 200),       # t_start
    (10, 400)       # T_rollout
]

result = differential_evolution(
    objective,
    bounds,
    seed=42,
    polish=True
)

best_hr, best_donor, best_cap, best_adopt, best_start, best_rollout = result.x

best_prevented = -result.fun

print("### Optimal parameters")

print(
    {
        "High-risk fraction": round(best_hr, 3),
        "Potential donor rate": round(best_donor, 3),
        "Maximum donations/day": round(best_cap),
        "Maximum adoption": round(best_adopt, 3),
        "Start time": round(best_start),
        "Rollout time": round(best_rollout),
        "Hospitalizations prevented": round(best_prevented)
    }
)