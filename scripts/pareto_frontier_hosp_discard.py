import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_hospitalization import run_model

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


n_samples = 10000

results_list = []

for _ in range(n_samples):

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

    hr = np.random.uniform(0.01, 1.00)

    donor = np.random.uniform(0.01, 1.00)

    cap = np.random.uniform(100, 1500)

    adopt = np.random.uniform(0.01, 1.00)

    start = np.random.randint(0, 201)

    rollout = np.random.randint(10, 401)

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
        high_risk_fraction=hr,
        delay_inf_to_hosp=fixed_params["delay_inf_to_hosp"],
        t_start=int(start),
        T_rollout=int(rollout),
        window_start=fixed_params["window_start"],
        window_end=fixed_params["window_end"],
        debug=False
    )

    results_list.append({
        "prevented": np.sum(results["H_prevented"]),
        "discarded": results["total_discarded"],
        "high_risk_fraction": hr,
        "potential_donor_rate": donor,
        "capacity_per_day": cap,
        "A_max": adopt,
        "t_start": start,
        "T_rollout": rollout
    })

df_results = pd.DataFrame(results_list)

pareto_mask = np.ones(len(df_results), dtype=bool)

for i, row in df_results.iterrows():

    dominated = (
        (df_results["prevented"] >= row["prevented"])
        &
        (df_results["discarded"] <= row["discarded"])
        &
        (
            (df_results["prevented"] > row["prevented"])
            |
            (df_results["discarded"] < row["discarded"])
        )
    )

    dominated.iloc[i] = False

    if dominated.any():
        pareto_mask[i] = False

pareto_df = df_results[pareto_mask].copy()

pareto_df = pareto_df.sort_values(
    "discarded"
)

fig, ax = plt.subplots(figsize=(8,6))

ax.scatter(
    df_results["discarded"],
    df_results["prevented"],
    alpha=0.25,
    s=10,
    color="lightgray",
    label="Sampled policies"
)

ax.plot(
    pareto_df["discarded"],
    pareto_df["prevented"],
    color="red",
    linewidth=2,
    label="Pareto frontier"
)

ax.scatter(
    pareto_df["discarded"],
    pareto_df["prevented"],
    color="red",
    s=20
)

ax.set_xlabel("Total CCP discarded")

ax.set_ylabel(
    "Total hospitalizations prevented"
)

ax.set_title(
    "Pareto frontier: impact versus wastage"
)

ax.legend()

plt.tight_layout()

plt.show()

# lowest waste
print(
    pareto_df.iloc[0]
)

# highest impact
print(
    pareto_df.iloc[-1]
)

# best balance
pareto_df["norm_prevented"] = (
    pareto_df["prevented"]
    / pareto_df["prevented"].max()
)

pareto_df["norm_discarded"] = (
    pareto_df["discarded"]
    / pareto_df["discarded"].max()
)

pareto_df["score"] = (
    pareto_df["norm_prevented"]
    - pareto_df["norm_discarded"]
)

print(
    pareto_df.loc[
        pareto_df["score"].idxmax()
    ]
)

a=1
