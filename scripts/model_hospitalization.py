import pandas as pd
import numpy as np
import model_methods

def run_model(
    H,
    I,
    variant_changes,
    initial_ccp_activity=0.70,
    high_risk_use_threshold=0.40,
    minimum_usable_activity = 0.05,
    max_storage_age = 365,
    A_max=1.0,
    capacity_per_day=100,
    treatment_duration=90,
    potential_donor_rate=0.2,
    over_titre_donor_rate=0.2,
    doses_per_patient_per_day=2,
    donation_volume=0.6,
    donations_per_donor=3,
    min_donation_interval=14, 
    dose_volume=0.0012,
    delay_inf_to_hosp=7,
    t_start=30,
    T_rollout=30,
    window_start=30,
    window_end=180,
    debug=False
):
    # Simple calculations:
    doses_per_donation = donation_volume/dose_volume
    doses_per_donor = donations_per_donor * doses_per_donation
    doses_per_treatment = doses_per_patient_per_day * treatment_duration
    treatments_per_donor = doses_per_donor/doses_per_treatment
    donor_rate = potential_donor_rate * over_titre_donor_rate


    n_days = len(H)

    adoption = model_methods.calculate_adoption(
        n_days,
        A_max,
        t_start,
        T_rollout,
        debug
        )   

    daily_batches, daily_potential_donations = model_methods.calculate_daily_batches(
    I,
    n_days,
    donor_rate,
    capacity_per_day,
    doses_per_donation,
    donations_per_donor,
    min_donation_interval,
    window_start,
    window_end,
    initial_ccp_activity,
    variant_changes,
    debug = False
    )

    daily_doses = np.zeros(n_days)
    daily_donations = np.zeros(n_days)
    for day in range(n_days):
        daily_doses[day] = sum(
            batch["doses"]
            for batch in daily_batches[day]
        )
        daily_donations[day] = sum(
                    batch["donors"]
                    for batch in daily_batches[day]
                )

    high_risk_population, general_population = model_methods.calculate_populations(
        H,
        I,
        delay_inf_to_hosp,
        debug
    )

    stock = np.zeros_like(H, dtype=float)

    # Track patients in active treatment (cohort accumulation) 
    active_patients = np.zeros_like(H, dtype=float) # high risk and general
    high_risk_patients_series = np.zeros_like(H, dtype=float)
    high_risk_demand_series = np.zeros_like(H, dtype=float)
    general_demand_series = np.zeros_like(H, dtype=float) 
    reserved_doses_series = np.zeros_like(H, dtype=float)
    discarded_doses = np.zeros_like(H, dtype=float)
    reserved_doses_series_high_risk = np.zeros_like(H, dtype=float)
    reserved_doses_series_general = np.zeros_like(H, dtype=float)
    high_risk_treatment_efficacy_series = np.zeros_like(H, dtype=float)
    general_treatment_efficacy_series = np.zeros_like(H, dtype=float)
    effective_coverage = np.zeros_like(H, dtype=float)
    stock_efficacy_series = np.zeros_like(H, dtype=float)
    general_patients_series = np.zeros_like(H, dtype=float)
    high_risk_stock_series = np.zeros_like(H)
    general_stock_series = np.zeros_like(H)

    variant_series = {
        "wuhan_high_risk": np.zeros_like(H, dtype=float),
        "wuhan_general": np.zeros_like(H, dtype=float),
        "alpha_high_risk": np.zeros_like(H, dtype=float),
        "alpha_general": np.zeros_like(H, dtype=float),
        "delta_high_risk": np.zeros_like(H, dtype=float),
        "delta_general": np.zeros_like(H, dtype=float),
        "omicron_high_risk": np.zeros_like(H, dtype=float),
        "omicron_general": np.zeros_like(H, dtype=float)
    }

    C = np.zeros_like(H, dtype=float)
    C_supply = np.zeros_like(H, dtype=float)

    supply_limited = np.zeros_like(H, dtype=int)


    inventory = []
    high_risk_ages = []
    general_ages = []
    for i in range(n_days):

        inventory = model_methods.update_inventory(
            inventory,
            daily_batches,
            i,
            debug=False
        )

        variant_today = model_methods.get_variant(variant_changes, i) # the COVID variant that the people treated today have

        inventory, expired_today = model_methods.classify_inventory(
            inventory,
            variant_today,
            high_risk_use_threshold,
            minimum_usable_activity,
            max_storage_age
        )

        discarded_doses[i] = expired_today
        
        # adoption determines the fraction of these eligible patients that seek treatment
        high_risk_requested = (
            adoption[i]
            * high_risk_population[i]
        )

        general_requested = (
            adoption[i]
            * general_population[i]
        )

        high_risk_stock, general_stock = (
                model_methods.summarize_inventory(inventory)
            )

        variant_counts = model_methods.variant_accounting(inventory)

        for name, value in variant_counts.items():
            variant_series[name][i] = value

    
        (   inventory,
            high_risk_patients, # actual patients that started (can be less than capacity)
            max_high_risk_patients, # capacity implied by inventory
            high_risk_reserved,
            high_risk_activity,
            doses_age_high_risk
        ) = model_methods.allocate_patients(
                inventory,
                high_risk_requested,
                doses_per_treatment,
                high_risk_stock,
                "high_risk",
                variant_today
            )

        (   inventory,
            general_patients,
            _, # not analyzing general population stock limits for now
            general_reserved,
            general_activity,
            doses_age_general
        ) = model_methods.allocate_patients(
                inventory,
                general_requested,
                doses_per_treatment,
                general_stock,
                "general",
                variant_today
            )
        
        high_risk_ages.append(doses_age_high_risk)
        general_ages.append(doses_age_general)

        high_risk_demand_series[i] = high_risk_requested
        general_demand_series[i] = general_requested
        general_patients_series[i] = general_patients
        high_risk_patients_series[i] = high_risk_patients
        high_risk_treatment_efficacy_series[i] = high_risk_activity
        general_treatment_efficacy_series[i] = general_activity
        high_risk_stock_series[i] = high_risk_stock
        general_stock_series[i] = general_stock

        inventory = model_methods.remove_empty_batches(inventory)

        # patients remain active for the entire treatment duration 
        # even though inventory was already reserved at treatment initiation
        active_patients[i] = (
            np.sum(
                high_risk_patients_series[
                    max(0, i - treatment_duration + 1): i + 1
                ]
            )
            +
            np.sum(
                general_patients_series[
                    max(0, i - treatment_duration + 1): i + 1
                ]
            )
        )

        stock[i], stock_efficacy_series[i] = (
            model_methods.calculate_stock_metrics(
                inventory
            )
        )

        # Coverage:
        C[i], effective_coverage[i] = (
            model_methods.calculate_hospitalization_reduction(
                high_risk_patients,
                high_risk_population[i],
                high_risk_activity
            )
        )

        # Supply coverage: fraction of the requested that was fulfilled
        C_supply[i] = model_methods.calculate_supply_coverage(
            high_risk_patients,
            high_risk_requested)
        
        
        # identify limiting factor 
        if max_high_risk_patients < high_risk_requested: # more requests than availability
            supply_limited[i] = 1


        # daily delivered doses (for reporting) - 
        # daily delivered today corrspond to the treatment courses started today
        reserved_doses_series[i] = (
                high_risk_reserved
                + general_reserved
            )

        reserved_doses_series_high_risk[i] = high_risk_reserved
        reserved_doses_series_general[i] = general_reserved

    # Shift coverage forward
    C_effective = np.roll(
        effective_coverage,
        delay_inf_to_hosp
    )
    C_effective[:delay_inf_to_hosp] = 0

    # --------------------------
    # APPLY MODEL
    # --------------------------
    # Assumption: The fraction of hospitalizations is reduced proportionally to the fraction of treated high‑risk infections
    # the model no longer needs the separate efficacy term in the final equation because efficacy is already embedded in coverage
    # A patient treated with 70%-effective plasma contributes more than one treated with 20%-effective plasma.
    H_ccp = H * (
        1 - C_effective
    )

    # Prevented hospitalizations
    H_prevented = H - H_ccp
    H_reduction_pct = (H - H_ccp) / np.maximum(H, 1) * 100

    # Doses remaining at the end of the simulation
    remaining_doses = sum(
    batch["doses"]
    for batch in inventory
    )

    return {
        # Epidemiological trajectories
        "H_ccp": H_ccp,
        "H_prevented": H_prevented,
        "H_reduction_pct": H_reduction_pct,

        "daily_potential_donations": daily_potential_donations, 
        "daily_donations": daily_donations,

        # Demand/adoption
        "adoption": adoption,
        "high_risk_population": high_risk_population,
        "general_population": general_population,
        "high_risk_demand": high_risk_demand_series,
        "general_demand": general_demand_series,

        # Treatment delivery
        "high_risk_patients": high_risk_patients_series,
        "general_patients": general_patients_series,
        "active_patients": active_patients,
        "high_risk_age_stock_delivered": [[item["age"] for item in day] for day in high_risk_ages],
        "high_risk_patient_variant_stock_delivered": [[item["patient_variant"] for item in day] for day in high_risk_ages],
        "high_risk_donor_variant_stock_delivered": [[item["donor_variant"] for item in day] for day in high_risk_ages],

        # Coverage
        "coverage": C,
        "coverage_effective": C_effective,
        "supply_limit": C_supply,

        # Inventory
        "stock": stock,
        "daily_production": daily_doses,
        "daily_reserved_doses": reserved_doses_series,
        "daily_reserved_doses_high_risk": reserved_doses_series_high_risk,
        "daily_reserved_doses_general": reserved_doses_series_general,
        "discarded_doses": discarded_doses,
        "remaining_doses": remaining_doses,
        "high_risk_stock": high_risk_stock_series,
        "general_stock": general_stock_series,
        **variant_series,


        # Activity
        "high_risk_treatment_efficacy": high_risk_treatment_efficacy_series,
        "general_treatment_efficacy": general_treatment_efficacy_series,
        "mean_stock_efficacy": stock_efficacy_series,

        # Capacity diagnostics
        "supply_limited": supply_limited,

        # Constants needed later
        "donations_per_donor": donations_per_donor,
        "doses_per_treatment": doses_per_treatment,
        "treatments_per_donor": treatments_per_donor
    }

def summarize_results(results):

    return {
        "total_donations": np.sum(
            results["daily_donations"]
        ),

        "total_donors": np.sum(
            results["daily_donations"]/results["donations_per_donor"]
        ),

        "total_produced": np.sum(
            results["daily_production"]
        ),

        "total_delivered": np.sum(
            results["daily_reserved_doses"]
        ),

        "total_delivered_high_risk": np.sum(
            results["daily_reserved_doses_high_risk"]
        ),

        "total_delivered_general": np.sum(
            results["daily_reserved_doses_general"]
        ),

        "total_discarded": np.sum(
            results["discarded_doses"]
        ),

        "average_stock": np.mean(
            results["stock"]
        ),

        "maximum_stock": np.max(
            results["stock"]
        ),

        "maximum_high_risk_stock": np.max(
            results["high_risk_stock"]
        ),

        "maximum_general_stock": np.max(
            results["general_stock"]
        ),

        "stockout_days": np.sum(
            results["stock"] <= 0
        ),

        "average_coverage": np.mean(
            results["coverage"]
        ),

        "average_effective_coverage": np.mean(
            results["coverage_effective"]
        ),

        "peak_coverage": np.max(
            results["coverage"]
        ),

        "peak_treatment_starts": np.max(
            results["high_risk_patients"]
        ),

        "peak_general_users": np.max(
            results["general_patients"]
        ),

        "peak_daily_production": np.max(
            results["daily_production"]
        ),

        "fraction_supply_limited": np.mean(
            results["supply_limited"]
        ),

        "peak_high_risk_demand": np.max(
            results["high_risk_demand"]
        ),

        "max_reduction_pct": np.max(
            results["H_reduction_pct"]
        )
    }


if __name__ == "__main__":     

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


    variant_changes = []

    # date in which the variant became dominant (exceded 50% proportion) https://epidata.sciensano.be/epistat/dashboard/#covid_variants
    variant_dates = {
        "Alpha": "2020-12-15",
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
    
    variant_changes = sorted(
        variant_changes,
        key=lambda x: x["day"]
    )

    results = run_model(
        H=H,
        I=I,
        variant_changes = variant_changes,
        debug=False
    )
    
    # donation_window_start = 30
    # donation_window_end = 180
    # min_donation_interval = 14
    # donations_per_donor=3
    # initial_ccp_activity=0.7
    # cross_neutralization = {
    #         "Wuhan": {
    #             "Wuhan": 1.00,
    #             "Alpha": 1/np.sqrt(2.3), # 0.66
    #             "Delta": 1/np.sqrt(1.6), # 0.79
    #             "Omicron": 1/np.sqrt(20) # 0.22
    #         },
    
    #         "Alpha": {
    #             "Alpha": 1.00,
    #             "Delta": 1/np.sqrt(2.2), # 0.67
    #             "Omicron": 1/np.sqrt(50) # 0.14
    #         },
    
    #         "Delta": {
    #             "Delta": 1.00,
    #             "Omicron": 1/np.sqrt(11) # 0.30
    #         },
    
    #         "Omicron": {
    #             "Omicron": 1.00
    #         }
    #     }

    # debug_donation_schedule(
    #     donation_window_start,
    #     donation_window_end,
    #     min_donation_interval,
    #     donations_per_donor,
    #     initial_ccp_activity,
    #     variant_changes,
    #     cross_neutralization
    #     )
    # summary = summarize_results(results)

    # print_report(results)
    # plot_results(results, summary)
