import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def print_report(results):
    print("\nMODEL SUMMARY")
    print("-" * 40)

    print("\nEPIDEMIOLOGICAL OUTCOMES")
    print("-" * 40)

    print(
        f"Total doses produced: "
        f"{results['total_produced']:,.0f}"
    )

    print(
        f"Total doses delivered: "
        f"{results['total_delivered']:,.0f}"
    )

    print(
        f"Total doses discarded: "
        f"{results['total_discarded']:,.0f}"
    )

    print(
        f"Average stock: "
        f"{results['average_stock']:,.0f}"
    )

    print(
        f"Maximum stock: "
        f"{results['maximum_stock']:,.0f}"
    )

    print(
        f"Stockout days: "
        f"{results['stockout_days']}"
    )

    print(
        f"Average coverage: "
        f"{100*results['average_coverage']:.1f}%"
    )

    print(
        f"Peak coverage: "
        f"{100*results['peak_coverage']:.1f}%"
    )

    print(
        f"Total hospitalizations prevented: "
        f"{np.sum(results['H_prevented']):,.0f}"
    )

    print(
        f"Maximum daily reduction: "
        f"{np.max(results['H_reduction_pct']):.1f}%"
    )

    print(
        f"Total reduction: "
        f"{100*np.sum(results['H_prevented'])/np.sum(H):.1f}%"
    ) 


    print("\nOPERATIONAL METRICS")
    print("-" * 40)

    discard_rate = (
        100
        * results["total_discarded"]
        / max(results["total_produced"],1)
    )   

    print(
        f"Peak daily demand: "
        f"{results['peak_daily_demand']:.0f} patients/day"
    )

    print(
        f"Peak treatment starts: "
        f"{results['peak_treatment_starts']:.0f} patients/day"
    )

    print(
        f"Peak production: "
        f"{results['peak_daily_production']:,.0f} doses/day"
    )

    print(
        f"Peak inventory: "
        f"{results['peak_inventory']:,.0f} doses"
    )

    print(
        f"Discard rate: "
        f"{discard_rate:.1f}%"
    )

    print(
        f"Stockout days: "
        f"{results['stockout_days']} "
        f"({100*results['stockout_days']/len(H):.1f}%)"
    )


    print(
        f"Fraction supply limited: "
        f"{100*results['fraction_supply_limited']:.1f}%"
    )

    production_utilization = (
        results["total_delivered"]
        /
        results["total_produced"]
    )

    # how much of the produced plasma was used clinically
    print(
        f"Production utilization: "
        f"{100*production_utilization:.1f}%"
    )


def calculate_adoption(
    n_days,
    A_max,
    t_start,
    T_rollout
):
    # --------------------------
    # ADOPTION FUNCTION
    # Adoption(t) = A_max × ramp(t)
    # --------------------------
    adoption = np.zeros(n_days, dtype=float)

    for i in range(n_days):
        if i < t_start:
            adoption[i] = 0
        elif t_start <= i <= (t_start + T_rollout):
            adoption[i] = A_max * (i - t_start) / T_rollout
        else:
            adoption[i] = A_max
    return adoption


def calculate_daily_doses(
    I,
    n_days,
    donor_rate,
    capacity_per_day,
    doses_per_donor,
    window_start,
    window_end
):
    # --------------------------
    # SUPPLY → STOCK (daily production)
    # people who were infected in the past may become donors after recovery
    # donation capacity is limited
    # --------------------------

    daily_doses = np.zeros(n_days, dtype=float)
    
    window_length = window_end - window_start + 1
    
    # smoothed infection curve - infections generate potential donors 
    # but daily reports are noisy bc of reporting delays, weekends, short-term fluctuations
    I_smooth = pd.Series(I).rolling(14).mean().bfill().values
    for i in range(n_days):

        # distribute the potential donors through the window 
        potential_donors = 0
        for k in range(window_start, window_end+1):
            if i - k >= 0:
                potential_donors += (
                    donor_rate
                    * I_smooth[i - k]
                    / window_length
                )
                # /window_length means every recovered donor is equaly likely to donate on any daty between window_start and window_end

        # capacity constraint acts on DONORS (how many donors we can collect from in a day)
        actual_donors = min(potential_donors, capacity_per_day)

        # convert to doses (full donor yield). These are the CCP doses produced in day i
        daily_doses[i] = actual_donors * doses_per_donor
    return daily_doses


def calculate_populations(
    H,
    I,
    delay_inf_to_hosp
):
    # estimate high-risk infections from observed hospitalizations
    high_risk_population = np.roll(
        H,
        -delay_inf_to_hosp
    ).astype(float)

    # last days have no future hospitalization data
    high_risk_population[-delay_inf_to_hosp:] = 0

    general_population = np.maximum(
    I - high_risk_population,
    0
    )

    return (
        high_risk_population,
        general_population
    )

def update_inventory(
    inventory,
    daily_doses,
    day,
    daily_activity_decay,
    initial_ccp_activity,
    minimum_usable_activity,
    variant_changes
):

    # age the inventory (all plasma batch ages 1 day) and make it lose efficacy
    for batch in inventory:
        batch["age"] += 1
        batch["activity"] *= (
            1.0 - daily_activity_decay
        )
    
    # on the day of the variant change, the inventory gets more penalized
    current_variant = "Wuhan" #first variant
    if variant_changes:
        for event in variant_changes:
            if day >= event["day"]:
                current_variant = event["name"]
                if day == event["day"]:
                    for batch in inventory:
                        batch["activity"] *= (
                            1 - event["penalty"]
                        )

    # today's production (prepare to save the age of each stock addition)
    inventory.append(
        {
            "age": 0,
            "doses": daily_doses[day], 
            "activity": initial_ccp_activity,
            "variant": current_variant,
            "donation_day": day 
        }
    )

    # delete doses that are below the minimum usable efficacy
    expired_today = sum(
        batch["doses"]
        for batch in inventory
        if batch["activity"] < minimum_usable_activity
    )

    inventory = [
    batch
    for batch in inventory
    if batch["activity"] >= minimum_usable_activity
    ]


    return (inventory, expired_today)

def project_activity(
    current_activity,
    daily_activity_decay,
    days,
    decay_mode="exponential"
):  
    if decay_mode == "exponential":
        return (
            current_activity
            * (1.0 - daily_activity_decay) ** days
        )

    elif decay_mode == "linear":
        return max(
            0,
            current_activity
            - daily_activity_decay * days
        )

    else:
        raise ValueError(
            f"Unknown decay mode: {decay_mode}"
        )
    
def calculate_available_stock(
    inventory,
    treatment_duration,
    daily_activity_decay,
    minimum_usable_activity,
    high_risk_use_threshold,
    decay_mode = "exponential" # or "linear"
):
    high_risk_stock = 0 #doses suitable for high-risk patients
    general_stock = 0 #doses for the general population

    for batch in inventory:
        # project activity to the day of treatment completion
        future_activity = project_activity(
            batch["activity"],
            daily_activity_decay,
            treatment_duration,
            decay_mode
        )
        # stock high quality plasma
        if future_activity >= high_risk_use_threshold:
            high_risk_stock += batch["doses"]
        # stock moderate quality plasma
        elif future_activity >= minimum_usable_activity:
            general_stock += batch["doses"]

    return (
        high_risk_stock,
        general_stock
        )

def allocate_high_risk_patients(
        inventory,
        requested_patients,
        doses_per_treatment,
        treatment_duration,
        daily_activity_decay, 
        high_risk_use_threshold,
        available_stock,
        decay_mode = "exponential" # or "linear"
):
    # How many patients could start based on usable stock  (only doses that remain above activity threshold at treatment completion)
    max_new_patients = (
        available_stock
        / doses_per_treatment
    )
    # clinical demand, capped at maximum based on stock
    requested_starts = min(
        requested_patients,
        max_new_patients
    )
    # total number of doses that would be required to start all treatment courses requested today
    doses_needed = (
        requested_starts
        * doses_per_treatment
    )

    # Reserve stock (FIFO)
    remaining = doses_needed
    # "effective_treatments" accumulates the activity-
    # weighted dose volume and is later used to compute
    # mean treatment activity.
    effective_treatments = 0
    for batch in inventory:

        if remaining <= 0: # Stop if all required doses have been allocated.
            break

        # Activity expected at treatment completion
        future_activity = project_activity(
            current_activity=batch["activity"],
            daily_activity_decay=daily_activity_decay,
            days=treatment_duration,
            decay_mode=decay_mode
        )

        # Ignore stock that won't be used for high risk patients
        if future_activity < high_risk_use_threshold:
            continue

        # reserve as many doses as possible from this batch (FIFO inventory usage)
        take = min(
            batch["doses"],
            remaining
        )

        # accumulate activity-weighted doses: for ex. 100 doses at 50% activity contribute 50 activity-adjusted doses
        effective_treatments += (
            take
            * future_activity
        )

        # remove reserved doses from inventory and from dose requirement
        batch["doses"] -= take
        remaining -= take

    # Actual treatments started
    actual_doses_reserved = (
        doses_needed - remaining
    )

    # convert reserved doses back into complete treatment courses (patients treated)
    new_patients = (
        actual_doses_reserved
        / doses_per_treatment
    )

    # mean treatment activity among all doses deployed today. The average activity that patients will experience at treatment completion.
    if actual_doses_reserved > 0:
        mean_treatment_activity = (
            effective_treatments
            / actual_doses_reserved
        )
    else:
        mean_treatment_activity = 0

    return (
        inventory,
        new_patients,
        actual_doses_reserved,
        mean_treatment_activity,
        max_new_patients
    )

def allocate_general_patients(
        inventory,
        requested_patients,
        doses_per_treatment,
        treatment_duration,
        daily_activity_decay, 
        high_risk_use_threshold,
        minimum_usable_activity,
        available_stock,
        decay_mode = "exponential" # or "linear"
):
    # How many patients could start based on usable stock  (only doses that remain above activity threshold at treatment completion)
    max_new_patients = (
        available_stock
        / doses_per_treatment
    )
    # clinical demand, capped at maximum based on stock
    requested_starts = min(
        requested_patients,
        max_new_patients
    )
    # total number of doses that would be required to start all treatment courses requested today
    doses_needed = (
        requested_starts
        * doses_per_treatment
    )

    # Reserve stock (FIFO)
    remaining = doses_needed
    # "effective_treatments" accumulates the activity-
    # weighted dose volume and is later used to compute
    # mean treatment activity.
    effective_treatments = 0
    for batch in inventory:

        if remaining <= 0: # Stop if all required doses have been allocated.
            break

        # Activity expected at treatment completion
        future_activity = project_activity(
            current_activity=batch["activity"],
            daily_activity_decay=daily_activity_decay,
            days=treatment_duration,
            decay_mode=decay_mode
        )

        # Ignore stock that won't be used for general patients
        if (future_activity < minimum_usable_activity) or (future_activity >= high_risk_use_threshold):
            continue

        # reserve as many doses as possible from this batch (FIFO inventory usage)
        take = min(
            batch["doses"],
            remaining
        )

        # accumulate activity-weighted doses: for ex. 100 doses at 50% activity contribute 50 activity-adjusted doses
        effective_treatments += (
            take
            * future_activity
        )

        # remove reserved doses from inventory and from dose requirement
        batch["doses"] -= take
        remaining -= take

    # Actual treatments started
    actual_doses_reserved = (
        doses_needed - remaining
    )

    # convert reserved doses back into complete treatment courses (patients treated)
    new_patients = (
        actual_doses_reserved
        / doses_per_treatment
    )

    # mean treatment activity among all doses deployed today. The average activity that patients will experience at treatment completion.
    if actual_doses_reserved > 0:
        mean_treatment_activity = (
            effective_treatments
            / actual_doses_reserved
        )
    else:
        mean_treatment_activity = 0

    return (
        inventory,
        new_patients,
        actual_doses_reserved,
        mean_treatment_activity,
        max_new_patients
    )

def remove_empty_batches(
    inventory
):
    return [
        batch
        for batch in inventory
        if batch["doses"] > 0
    ]

def calculate_stock_metrics(inventory):
    # stock after reservation
    stock = sum(
        batch["doses"]
        for batch in inventory
    )

    mean_activity = (
        sum(
            batch["doses"] * batch["activity"]
            for batch in inventory
        )
        /
        max(stock, 1)
    )
    
    return (stock, mean_activity)

def calculate_coverage(
    treated_patients,
    eligible_patients,
    treatment_activity
):
    # Fraction of the eligible population that receives
    # treatment.    
    if eligible_patients > 0:
        coverage = (
            treated_patients
            / eligible_patients
        )
    else:
        coverage = 0

    # activity-adjusted (fraction of the target population receiving biologically meaningful CCP). later used to estimate hospitalization reduction.
    effective_coverage = (
        coverage
        * treatment_activity
    )

    return (
        coverage,
        effective_coverage
    )


def calculate_supply_coverage(
    treated_patients,
    requested_patients
):
    # how much of the treatment demand could be fulfilled (inventory performance, not hospitalization impact)
    if requested_patients > 0:

        return (
            treated_patients
            / requested_patients
        )
    # if nobody requests treatment, demand is fully satisfied by definition
    return 1.0

def run_model(
    H,
    I,
    variant_changes,
    initial_ccp_activity=0.70,
    high_risk_use_threshold=0.50,
    daily_activity_decay=0.002,
    decay_mode = "exponential",
    minimum_usable_activity = 0.05,
    A_max=1.0,
    capacity_per_day=450,
    treatment_duration=90,
    potential_donor_rate=0.2,
    over_titre_donor_rate=0.2,
    doses_per_patient_per_day=2,
    donation_volume=0.6,
    donations_per_donor=3,
    dose_volume=0.0012,
    delay_inf_to_hosp=7,
    t_start=30,
    T_rollout=30,
    window_start=30,
    window_end=180
):
    # Simple calculations:
    doses_per_donation = donation_volume/dose_volume
    doses_per_donor = donations_per_donor * doses_per_donation
    doses_per_treatment = doses_per_patient_per_day * treatment_duration
    treatments_per_donor = doses_per_donor/doses_per_treatment
    donor_rate = potential_donor_rate * over_titre_donor_rate

    # --------------------------
    # 3. BUILD TIME INDEX
    # --------------------------
    n_days = len(H)

    adoption = calculate_adoption(
    n_days,
    A_max,
    t_start,
    T_rollout
    )   

    daily_doses = calculate_daily_doses(
    I,
    n_days,
    donor_rate,
    capacity_per_day,
    doses_per_donor,
    window_start,
    window_end
    )

    high_risk_population, general_population = calculate_populations(
        H,
        I,
        delay_inf_to_hosp
    )


    stock = np.zeros_like(H, dtype=float)

    # Track patients in active treatment (cohort accumulation) 
    active_patients = np.zeros_like(H, dtype=float) # high risk and general
    high_risk_patients_series = np.zeros_like(H, dtype=float)
    high_risk_demand_series = np.zeros_like(H, dtype=float)
    general_demand_series = np.zeros_like(H, dtype=float) 
    reserved_doses_series = np.zeros_like(H, dtype=float)
    discarded_doses = np.zeros_like(H, dtype=float)
    treatment_efficacy_series = np.zeros_like(H, dtype=float)
    effective_coverage = np.zeros_like(H, dtype=float)
    stock_efficacy_series = np.zeros_like(H, dtype=float)
    general_patients_series = np.zeros_like(H, dtype=float)
    high_risk_stock_series = np.zeros_like(H)
    general_stock_series = np.zeros_like(H)
    wuhan_stock_series = np.zeros_like(H, dtype=float)
    alpha_stock_series = np.zeros_like(H, dtype=float)
    delta_stock_series = np.zeros_like(H, dtype=float)
    omicron_stock_series = np.zeros_like(H, dtype=float)

    C = np.zeros_like(H, dtype=float)
    C_supply = np.zeros_like(H, dtype=float)

    supply_limited = np.zeros_like(H, dtype=int)
    not_supply_limited = np.zeros_like(H, dtype=int)

    inventory = []
    for i in range(n_days):

        inventory, expired_today = update_inventory(
            inventory,
            daily_doses,
            i,
            daily_activity_decay,
            initial_ccp_activity,
            minimum_usable_activity,
            variant_changes
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

        high_risk_stock, _ = calculate_available_stock(inventory, treatment_duration, daily_activity_decay, minimum_usable_activity, high_risk_use_threshold, decay_mode)

        inventory, high_risk_patients, high_risk_reserved, high_risk_activity, max_high_risk_patients = allocate_high_risk_patients(
            inventory, 
            high_risk_requested, 
            doses_per_treatment, 
            treatment_duration, 
            daily_activity_decay, 
            high_risk_use_threshold, 
            high_risk_stock,
            decay_mode)
        
        _, general_stock = calculate_available_stock(inventory, treatment_duration, daily_activity_decay, minimum_usable_activity, high_risk_use_threshold, decay_mode)

        inventory, general_patients, general_reserved, general_activity, max_general_patients = allocate_general_patients(
            inventory,
            general_requested,
            doses_per_treatment,
            treatment_duration,
            daily_activity_decay,
            high_risk_use_threshold,
            minimum_usable_activity,
            general_stock,
            decay_mode)
        
        
        high_risk_demand_series[i] = high_risk_requested
        general_demand_series[i] = general_requested
        general_patients_series[i] = general_patients
        high_risk_patients_series[i] = high_risk_patients
        treatment_efficacy_series[i] = high_risk_activity
        high_risk_stock_series[i] = high_risk_stock
        general_stock_series[i] = general_stock


        inventory = remove_empty_batches(inventory)

        wuhan_stock = 0
        alpha_stock = 0
        delta_stock = 0
        omicron_stock = 0

        for batch in inventory:

            if batch["variant"] == "Wuhan":
                wuhan_stock += batch["doses"]

            elif batch["variant"] == "Alpha":
                alpha_stock += batch["doses"]

            elif batch["variant"] == "Delta":
                delta_stock += batch["doses"]

            elif batch["variant"] == "Omicron":
                omicron_stock += batch["doses"]

        wuhan_stock_series[i] = wuhan_stock
        alpha_stock_series[i] = alpha_stock
        delta_stock_series[i] = delta_stock
        omicron_stock_series[i] = omicron_stock

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
            calculate_stock_metrics(
                inventory
            )
        )

        # Coverage: defined as treated patients/eligible patients
        C[i], effective_coverage[i] = (
            calculate_coverage(
                high_risk_patients,
                high_risk_population[i],
                high_risk_activity
            )
        )

        # Supply coverage: fraction of the requested that was fulfilled
        C_supply[i] = calculate_supply_coverage(
            high_risk_patients,
            high_risk_requested)
        
        
        # identify limiting factor 
        if max_high_risk_patients < high_risk_requested: # more requests than availability

            supply_limited[i] = 1

        else:

            not_supply_limited[i] = 1

        # daily delivered doses (for reporting) - 
        # daily delivered today corrspond to the treatment courses started today
        reserved_doses_series[i] = (
                high_risk_reserved
                + general_reserved
            )


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

    return {
        # Epidemiological trajectories
        "H_ccp": H_ccp,
        "H_prevented": H_prevented,
        "H_reduction_pct": H_reduction_pct,

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

        # Coverage
        "coverage": C,
        "coverage_effective": C_effective,
        "supply_limit": C_supply,

        # Inventory
        "stock": stock,
        "daily_production": daily_doses,
        "daily_reserved_doses": reserved_doses_series,
        "discarded_doses": discarded_doses,
        "high_risk_stock": high_risk_stock_series,
        "general_stock": general_stock_series,
        "wuhan_stock": wuhan_stock_series,
        "alpha_stock": alpha_stock_series,
        "delta_stock": delta_stock_series,
        "omicron_stock": omicron_stock_series,

        # Activity
        "end_treatment_efficacy": treatment_efficacy_series,
        "mean_stock_efficacy": stock_efficacy_series,

        # Capacity diagnostics
        "supply_limited": supply_limited,
        "not_supply_limited": not_supply_limited,

        # Constants needed later
        "doses_per_treatment": doses_per_treatment,
        "treatments_per_donor": treatments_per_donor
    }

def summarize_results(results):

    return {
        "total_produced": np.sum(
            results["daily_production"]
        ),

        "total_delivered": np.sum(
            results["daily_reserved_doses"]
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

        "stockout_days": np.sum(
            results["stock"] <= 0
        ),

        "average_coverage": np.mean(
            results["coverage"]
        ),

        "peak_coverage": np.max(
            results["coverage"]
        ),

        "peak_treatment_starts": np.max(
            results["high_risk_patients"]
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

def plot_results(results, summary):
    # unpack
    H_ccp = results["H_ccp"]
    H_prevented = results["H_prevented"]
    H_reduction_pct = results["H_reduction_pct"]

    coverage = results["coverage"]
    adoption = results["adoption"]

    stock = results["stock"]

    doses_per_treatment = results["doses_per_treatment"]
    treatments_per_donor = results["treatments_per_donor"]
    daily_production = results["daily_production"]
    daily_reserved_doses = results["daily_reserved_doses"]

    high_risk_patients = results["high_risk_patients"]
    general_patients = results["general_patients"]

    high_risk_demand = results["high_risk_demand"]
    general_demand = results["general_demand"]

    supply_limited = results["supply_limited"]
    not_supply_limited = results["not_supply_limited"]

    ## PLOTS
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)


    # ==================================================
    # Hospitalizations
    # ==================================================
    ax = axs[0, 0]

    ax.plot(
        dates,
        H,
        label="Hospitalizations",
        color="black"
    )

    ax.plot(
        dates,
        H_ccp,
        label="With CCP",
        color="blue"
    )

    ax.set_title("Hospitalizations impact")
    ax.set_ylabel("Admissions/day")

    ax.legend()
    ax.grid()

    # ==================================================
    # Supply dynamics
    # ==================================================
    ax = axs[0, 1]

    ax.plot(
        dates,
        daily_production / doses_per_treatment,
        label="Production",
        color="green"
    )

    ax.plot(
        dates,
        daily_reserved_doses / doses_per_treatment,
        label="Delivered",
        color="red"
    )

    ax.set_ylabel("Treatment courses/day")

    ax2 = ax.twinx()

    ax2.plot(
        dates,
        stock / doses_per_treatment,
        label="Inventory",
        color="blue",
        linewidth=2
    )

    ax2.set_ylabel(
        "Treatment courses in inventory"
    )

    ax.set_title("Supply dynamics")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(
        lines + lines2,
        labels + labels2
    )

    ax.grid()

    # ==================================================
    # Demand, delivery and coverage
    # ==================================================
    axdem = axs[1, 0]

    axdem.plot(
        dates,
        high_risk_demand,
        "--",
        label="High-risk demand",
        color="black"
    )

    axdem.plot(
        dates,
        general_demand,
        "--",
        label="General demand",
        color="grey"
    )

    axdem.plot(
        dates,
        high_risk_patients,
        label="High-risk treated",
        color="green"
    )

    axdem.plot(
        dates,
        general_patients,
        label="General treated",
        color="purple"
    )

    axdem.set_ylabel(
        "Patients/day"
    )

    axdem2 = axdem.twinx()

    axdem2.plot(
        dates,
        coverage,
        color="blue",
        linewidth=1.5,
        label="Coverage"
    )

    axdem2.plot(
        dates,
        adoption,
        "--",
        color="orange",
        linewidth=1,
        label="Adoption"
    )

    axdem2.fill_between(
        dates,
        0,
        1,
        where=supply_limited.astype(bool),
        color="red",
        alpha=0.15,
        label="Supply limited"
    )

    axdem2.set_ylim(0, 1.1)
    axdem2.set_ylabel("Fraction")

    axdem.set_title(
        "Demand, Delivery and Coverage"
    )

    lines, labels = axdem.get_legend_handles_labels()
    lines2, labels2 = axdem2.get_legend_handles_labels()

    axdem.legend(
        lines + lines2,
        labels + labels2,
        loc="upper left"
    )

    axdem.grid()

    # ==================================================
    # Stock composition
    # ==================================================
    ax = axs[1, 1]

    ax.plot(
        dates,
        results["high_risk_stock"]
        / doses_per_treatment,
        label="High-risk stock"
    )

    ax.plot(
        dates,
        results["general_stock"]
        / doses_per_treatment,
        label="General-use stock"
    )

    ax.set_ylabel(
        "Treatment courses"
    )

    ax.set_title(
        "Stock quality composition"
    )

    ax.legend()
    ax.grid()

    # ==================================================
    # Date formatting
    # ==================================================
    for ax in axs.flat:

        ax.xaxis.set_major_locator(
            mdates.MonthLocator(interval=6)
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m")
        )

        for label in ax.get_xticklabels():

            label.set_rotation(45)
            label.set_horizontalalignment("right")

    plt.tight_layout()
    plt.show()


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

    variant_dates = {
        "Alpha": "2020-12-15",
        "Delta": "2021-06-15",
        "Omicron": "2021-12-15"
    }

    variant_penalties = {
        "Alpha": 0.15,
        "Delta": 0.15,
        "Omicron": 0.15
    }

    for variant, date_str in variant_dates.items():

        day = (
            pd.Timestamp(date_str)
            - pd.Timestamp(dates.iloc[0])
        ).days

        variant_changes.append({
            "day": day,
            "name": variant,
            "penalty": variant_penalties[variant]
        })
    
    variant_changes = sorted(
        variant_changes,
        key=lambda x: x["day"]
    )

    results = run_model(
        H=H,
        I=I,
        variant_changes = variant_changes
    )
    
    summary = summarize_results(results)

    # print_report(results)
    plot_results(results, summary)
