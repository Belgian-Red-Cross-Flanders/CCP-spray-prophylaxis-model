import pandas as pd
import numpy as np

from model_hospitalization import run_model

def sample_activity_parameters(n):
    collected = []

    while len(collected) < n:
        initial = np.random.uniform(0.40, 0.80)
        high_risk = np.random.uniform(0.30, 0.50)
        minimum = np.random.uniform(0.01, 0.10)

        if minimum < high_risk < initial:
            collected.append(
                (initial, high_risk, minimum)
            )

    return pd.DataFrame(
        collected,
        columns=[
            "Initial CCP activity",
            "High-risk usage activity threshold",
            "Minimum usable activity"
        ]
    )

## Model parameters (fixed)
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

variant_changes = []

# date in which the variant became dominant (exceded 50% proportion) https://epidata.sciensano.be/epistat/dashboard/#covid_variants
variant_dates = {
    "Alpha": "2021-02-20",
    "Delta": "2021-06-29", # previous date I had was "2021-06-15". This new one is from https://epidata.sciensano.be/epistat/dashboard/#covid_variants
    "Omicron": "2021-12-31" # previous date I had was "2021-12-15" 
}

for variant, date_str in variant_dates.items():

    day = (
        pd.Timestamp(date_str)
        - pd.Timestamp(dates.iloc[0])
    ).days

    variant_changes.append({
        "day": day,
        "name": variant
    })

treatment_duration=90
over_titre_donor_rate=0.2
doses_per_patient_per_day=2
donation_volume=0.6
min_donation_interval=14
delay_inf_to_hosp=7 # 7 days according to cross-correlation between hospitalizations and infections
window_start=30
window_end=180

initial_ccp_activity = 0.7
high_risk_use_threshold = 0.4
minimum_usable_activity = 0.05


## Sensitivity analysis parameters (and the ones to vary)
n_sim = 10000

# ----------------------
# Activity parameters
# ----------------------

samples = sample_activity_parameters(n_sim)

# samples = pd.DataFrame()
# ----------------------
# Independent parameters
# ----------------------

samples["Potential donor rate"] = np.random.uniform(0.05, 0.5, n_sim)
samples["Max storage age"] = np.random.randint(
        30,
        1201,
        n_sim
    )
samples["Dose volume per nostril"] = np.random.uniform(
        100/ 500000,
        700/ 500000,
        n_sim
    )
samples["Donations per donor"] = np.random.randint(
        1,
        5,
        n_sim
    )
samples["Maximum donations/day"] = np.random.randint(
        10,
        450,
        n_sim
    )
samples["Release threshold"] = np.random.uniform(
    0.0, 0.1, n_sim
)
samples["Release fraction"] = np.random.uniform(
    0.0, 0.1, n_sim
)
samples["Maximum adoption"] = np.random.uniform(
        0.1,
        1.0,
        n_sim
    )
samples["Start time"] = np.random.randint(
        0,
        180,
        n_sim
    )
samples["Rollout time"] = np.random.randint(
        14,
        180,
        n_sim
    )

# Sample 10k combinations of these parameters and use them to run the model and get the outcomes
outcomes = []
for i, row in samples.iterrows():
    print(f"Iteration {i}")

    results = run_model(
        H,
        I,
        variant_changes,

        initial_ccp_activity=
            row["Initial CCP activity"],

        high_risk_use_threshold=
            row["High-risk usage activity threshold"],

        minimum_usable_activity=
            row["Minimum usable activity"],

        # initial_ccp_activity = initial_ccp_activity,
        # high_risk_use_threshold = high_risk_use_threshold,
        # minimum_usable_activity = minimum_usable_activity,

        max_storage_age=
            int(row["Max storage age"]),

        A_max=
            row["Maximum adoption"],

        capacity_per_day=
            int(row["Maximum donations/day"]),

        release_threshold =
            row["Release threshold"],

        release_fraction=
            row["Release fraction"],

        treatment_duration=
            int(treatment_duration),

        potential_donor_rate=
            row["Potential donor rate"],

        over_titre_donor_rate=
            over_titre_donor_rate,

        doses_per_patient_per_day=
            doses_per_patient_per_day,

        donation_volume=
            donation_volume,

        donations_per_donor=
            int(row["Donations per donor"]),

        min_donation_interval=
            int(min_donation_interval),

        dose_volume=
            row["Dose volume per nostril"],

        delay_inf_to_hosp=
            delay_inf_to_hosp,

        t_start=
            int(row["Start time"]),

        T_rollout=
            int(row["Rollout time"]),

        window_start=
            window_start,

        window_end=
            window_end,

        debug=False
    )

    outcomes.append({
        "Hospitalizations prevented":
            np.sum(
                results["H_prevented"]
            ),

        "General treated":
            np.sum(
                results["general_patients"]
            )
    })

global_sensitivity_results = pd.concat(
    [
        samples.reset_index(drop=True),
        pd.DataFrame(outcomes)
    ],
    axis=1
)

global_sensitivity_results.to_csv(
    "outputs/global_sensitivity_results.csv",
    index=False
)
