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
    max_storage_age = 730,
    A_max=1.0,
    capacity_per_day=100,
    release_threshold=0.05,
    release_fraction=0.05,
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
    """
    Orchestrator: Simulate the collection, storage, allocation, and delivery of intranasal
    CCP prophylaxis and estimate the resulting reduction in hospitalizations.
    """

    # Derive treatment and donor productivity metrics
    doses_per_donation = donation_volume/dose_volume
    doses_per_donor = donations_per_donor * doses_per_donation
    doses_per_treatment = doses_per_patient_per_day * treatment_duration
    treatments_per_donor = doses_per_donor/doses_per_treatment

    # Fraction of infected individuals that ultimately become eligible donors
    donor_rate = potential_donor_rate * over_titre_donor_rate

    n_days = len(H)

    # Generate the prophylaxis adoption trajectory
    adoption = model_methods.calculate_adoption(
        n_days,
        A_max,
        t_start,
        T_rollout,
        debug
        )   

    # Simulate donor recruitment and CCP collection
    daily_batches, daily_potential_donations, donor_statistics = model_methods.calculate_daily_batches_agent(
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
        variant_changes
        )

    # Aggregate donated doses and completed donations by day
    daily_doses = np.zeros(n_days)
    daily_donations = np.zeros(n_days)
    for day in range(n_days):
        daily_doses[day] = sum(
            batch["doses"]
            for batch in daily_batches[day]
        )
        daily_donations[day] = sum(
            batch["donations"]
            for batch in daily_batches[day]
        )

    # Split infections into high-risk and general-population groups
    high_risk_population, general_population = model_methods.calculate_populations(
        H,
        I,
        delay_inf_to_hosp,
        debug
    )

    # Storage arrays for model outputs
    stock = np.zeros_like(H, dtype=float)
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
    # Track inventory composition by donor variant and eligibility class
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
    hr_stock_request_rates = []
    no_stock_days = 0
    for i in range(n_days): # Simulate collection, allocation, and inventory dynamics day by day
        
        # Add newly collected CCP units to inventory
        inventory = model_methods.update_inventory(
            inventory,
            daily_batches,
            i
        )

        variant_today = model_methods.get_variant(variant_changes, i) # the COVID variant that the people treated today have

        # Reclassify inventory based on projected activity against
        # the currently circulating variant and remove expired stock
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

        if high_risk_stock == 0:
            no_stock_days += 1
            hr_stock_request_rate = 1 # Ratio of today's high-risk demand to available high-risk inventory
            # Used to determine whether inventory can be released to the general population
        else:
            hr_stock_request_rate = (
                high_risk_requested
                * doses_per_treatment
                / high_risk_stock
            )
        hr_stock_request_rates.append(hr_stock_request_rate)

        # Record inventory composition for reporting and visualization
        variant_counts = model_methods.variant_accounting(inventory)
        for name, value in variant_counts.items():
            variant_series[name][i] = value

        # Allocate CCP to high-risk patients first using match-first FIFO inventory matching
        # High-risk patients can use all high-risk stock that we have available
        (   inventory,
            high_risk_patients, # actual patients that started (can be less than capacity)
            max_high_risk_patients, # capacity implied by inventory
            high_risk_allocated,
            high_risk_activity,
            doses_age_high_risk
        ) = model_methods.match_fifo_allocate_high_risk(
                inventory,
                high_risk_requested,
                doses_per_treatment,
                high_risk_stock,
                variant_today,
            )

        # Allocate remaining eligible inventory to the general population (including any released high-risk stock)
        # General patients can use all general use stock and may also use a released fraction of high-risk stock
        (   inventory,
            general_patients,
            _, # not analyzing general population stock limits for now
            general_allocated,
            general_activity,
            doses_age_general
        ) = model_methods.match_fifo_allocate_general(
                inventory,
                general_requested,
                doses_per_treatment,
                high_risk_stock,
                general_stock,
                variant_today,
                hr_stock_request_rate,
                release_threshold,
                release_fraction
            )

        # Store characteristics of CCP units delivered today
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

        # Remove fully depleted inventory batches
        inventory = model_methods.remove_empty_batches(inventory)

        # Count patients currently receiving prophylaxis.
        # Inventory is reserved upfront, but patients remain active for the entire treatment duration (what I call treatment course)
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

        # Calculate total inventory and mean projected efficacy
        stock[i], stock_efficacy_series[i] = (
            model_methods.calculate_stock_metrics(
                inventory
            )
        )

        # Calculate treatment coverage and activity-adjusted coverage
        C[i], effective_coverage[i] = (
            model_methods.calculate_hospitalization_reduction(
                high_risk_patients,
                high_risk_population[i],
                high_risk_activity
            )
        )

        # Supply coverage: fraction of the requested high-risk treatments that was fulfilled
        C_supply[i] = model_methods.calculate_supply_coverage(
            high_risk_patients,
            high_risk_requested)
        
        
        # Flag days when available inventory cannot satisfy demand
        if max_high_risk_patients < high_risk_requested: # more requests than availability
            supply_limited[i] = 1


        # Record doses reserved for treatment courses initiated today 
        reserved_doses_series[i] = (
                high_risk_allocated
                + general_allocated
            )
        reserved_doses_series_high_risk[i] = high_risk_allocated
        reserved_doses_series_general[i] = general_allocated

    # Shift coverage forward (to account for the delay between infection and potential hospitalization)
    C_effective = np.roll(
        effective_coverage,
        delay_inf_to_hosp
    )
    C_effective[:delay_inf_to_hosp] = 0


    # APPLY MODEL (prophylaxis impact on hospitalization incidence)
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

        # Donations
        "daily_potential_donations": daily_potential_donations, 
        "daily_donations": daily_donations,
        "participating_donors": donor_statistics["unique_participating_donors"],
        "average_donations_per_participating_donor": donor_statistics["average_donations_per_participating_donor"],

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
        "high_risk_activity_stock_delivered": [[item["deploy_activity"] for item in day] for day in high_risk_ages],
        "high_risk_age_stock_delivered": [[item["age"] for item in day] for day in high_risk_ages],
        "high_risk_patient_variant_stock_delivered": [[item["patient_variant"] for item in day] for day in high_risk_ages],
        "high_risk_donor_variant_stock_delivered": [[item["donor_variant"] for item in day] for day in high_risk_ages],
        "general_activity_stock_delivered": [[item["deploy_activity"] for item in day] for day in general_ages],
        "general_age_stock_delivered": [[item["age"] for item in day] for day in general_ages],
        "general_patient_variant_stock_delivered": [[item["patient_variant"] for item in day] for day in general_ages],
        "general_donor_variant_stock_delivered": [[item["donor_variant"] for item in day] for day in general_ages],

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
    """
    Calculate summary metrics from a model simulation.
    """
    # Aggregate model outputs into summary statistics for reporting
    return {
        "total_donations": np.sum(
            results["daily_donations"]
        ),

        "total_donors": np.sum(
            results["participating_donors"]
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

    # Load and align hospitalization and case data
    #region
    df_hosp = pd.read_csv("data_processed/hospitalizations_clean.csv")
    df_cases = pd.read_csv("data_processed/cases_clean.csv")

    df_hosp["DATE"] = pd.to_datetime(df_hosp["DATE"])
    df_cases["DATE"] = pd.to_datetime(df_cases["DATE"])

    # Merge on DATE (inner join keeps only common dates)
    df = pd.merge(df_hosp, df_cases, on="DATE", how="inner")

    # Sort just in case
    df = df.sort_values("DATE")

    # Extract aligned time series
    dates = df["DATE"]
    H = df["NEW_IN"].values
    I = df["CASES"].values

    # Hospitalizations
    H_mean = np.mean(H)
    H_max = np.max(H)
    H_min = np.min(H)
    H_std = np.std(H)

    # Infections
    I_mean = np.mean(I)
    I_max = np.max(I)
    I_min = np.min(I)
    I_std = np.std(I)

    print(f"Hospitalizations/day")
    print(f"  Mean: {H_mean:.1f}")
    print(f"  Max : {H_max:.0f}")
    print(f"  Min : {H_min:.0f}")
    print(f"  Std : {H_std:.1f}")

    print(f"\nInfections/day")
    print(f"  Mean: {I_mean:.1f}")
    print(f"  Max : {I_max:.0f}")
    print(f"  Min : {I_min:.0f}")
    print(f"  Std : {I_std:.1f}")
    #endregion

    # Variant dominance transition dates
    variant_changes = []

    # date in which the variant became dominant (exceded 50% proportion) https://epidata.sciensano.be/epistat/dashboard/#covid_variants
    variant_dates = {
        "Alpha": "2021-02-20",
        "Delta": "2021-06-29", 
        "Omicron": "2021-12-31" 
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

    # Run the model with default parameters
    results = run_model(
        H=H,
        I=I,
        variant_changes = variant_changes,
        debug=False
    )
    
