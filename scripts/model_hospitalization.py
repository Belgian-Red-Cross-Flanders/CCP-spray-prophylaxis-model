import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Debug output directory
debug_dir = (
    Path.cwd()
    / "outputs"
    / "debug plots"
)

debug_dir.mkdir(
    parents=True,
    exist_ok=True
)



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
    T_rollout,
    debug=False
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

    if debug:
        fig, ax = plt.subplots(
            figsize=(5, 3)
        )

        days = np.arange(n_days)

        ax.plot(
            days,
            adoption,
            color="blue",
            linewidth=2
        )

        ax.axvline(
            t_start,
            color="red",
            linestyle="--",
            label=f"t_start = {t_start}"
        )

        ax.axvline(
            t_start + T_rollout,
            color="green",
            linestyle="--",
            label=f"t_start + T_rollout = {t_start + T_rollout}"
        )

        ax.axhline(
            A_max,
            color="black",
            linestyle=":",
            label=f"A_max = {A_max:.2f}"
        )

        ax.scatter(
            [t_start, t_start + T_rollout],
            [0, A_max],
            color="black",
            zorder=3
        )

        ax.annotate(
            "Rollout starts",
            (t_start, 0),
            xytext=(10, 10),
            textcoords="offset points"
        )

        ax.annotate(
            "Maximum adoption reached",
            (t_start + T_rollout, A_max),
            xytext=(10, -15),
            textcoords="offset points"
        )

        ax.set_xlabel("Day")
        ax.set_ylabel("Adoption")
        ax.set_ylim(
            0,
            max(1.05 * A_max, 0.05)
        )
        ax.set_title(
            "Adoption function"
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        fig.savefig(
            debug_dir / "adoption_function.png",
            dpi=300,
            bbox_inches="tight"
        )  
        plt.close()

    return adoption

def calculate_populations(
    H,
    I,
    delay_inf_to_hosp,
    debug=False
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

    if debug:

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(8, 5),
            sharex=True,
            constrained_layout=True
        )

        days = np.arange(len(H))

        # -------------------------------------
        # TOP PANEL
        # Hospitalizations → high-risk population
        # -------------------------------------

        ax1.plot(
            days,
            H,
            color="black",
            label="Hospitalizations"
        )

        ax1.plot(
            days,
            high_risk_population,
            color="red",
            label=(
                f"Estimated high-risk infections "
            )
        )

        ax1.set_ylabel(
            "People/day"
        )

        ax1.set_title(
            "High-risk population reconstruction"
        )

        ax1.legend()
        ax1.grid(alpha=0.3)

        # -------------------------------------
        # BOTTOM PANEL
        # Infection decomposition
        # -------------------------------------
        high_risk_fraction = (
            100*high_risk_population/(general_population+high_risk_population)
        )

        ax2.plot(
            days,
            high_risk_fraction,
            color="red",
            linewidth=1.5
        )

        ax2.set_ylabel(
            "% high-risk"
        )

        ax2.set_xlabel(
            "Day"
        )

        ax2.set_title(
            "Estimated high-risk fraction of infections"
        )

        ax2.grid(alpha=0.3)

        ax2.set_ylim(
            0,
            max(high_risk_fraction) * 1.1
        )

        fig.savefig(
            debug_dir /
            (
                f"population_reconstruction_"
                f"delay{delay_inf_to_hosp}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    return (
        high_risk_population,
        general_population
    )

def calculate_batch_activity(
    infection_day,
    donation_day,
    initial_ccp_activity,
    donation_window_start,
    variant_changes
):

    # Peak activity occurs at donation_window_start
    days_since_peak = max(
        0,
        donation_day
        - infection_day
        - donation_window_start
    )

    activity = initial_ccp_activity

    # Variant changes that occurred while antibodies
    # were still in the donor
    if variant_changes:

        for event in variant_changes:

            if (
                infection_day
                < event["day"]
                <= donation_day
            ):

                activity *= (
                    1 - event["penalty"]
                )


    return activity
    
def build_donation_schedule(
        infection_day, donations_per_donor, min_donation_interval, window_start, window_end, variant_changes
):

    # first eligible donation. Donation 1 is fixed at window start
    first_donation_day = (infection_day + window_start)

    if donations_per_donor == 1:
        return np.array([first_donation_day], dtype=int)

    # after the first donation, we maximize the spacing between donations (constrained by the minimum spacing and the window_end and getting as many donations as possible before next variant)
    last_donation_day = (infection_day + window_end) # hypothetical

    # default schedule is evenly spread through window
    donation_days = np.linspace(first_donation_day, last_donation_day, donations_per_donor).round().astype(int)

    # look for the next variant after infection
    next_variant_day = None
    if variant_changes:
        for event in variant_changes:
            if event["day"] > infection_day:
                next_variant_day = event["day"] # found a variant transition after the infection
                break

    if next_variant_day is not None: # determine how many donations can happen before next variant
        n_pre_variant = 1
        
        while ((first_donation_day + (n_pre_variant * min_donation_interval)) < next_variant_day) and (n_pre_variant < donations_per_donor):
            n_pre_variant += 1

        # place the pre-variant donations as late as possible
        if n_pre_variant > 1:
            pre_variant_end = min(
                next_variant_day - 1,
                last_donation_day
            )

            pre_variant_days = np.linspace(
                first_donation_day,
                pre_variant_end,
                n_pre_variant
            ).round().astype(int)

            donation_days[:n_pre_variant] = pre_variant_days

        # spread remaining donations after variant
        n_remaining = (donations_per_donor - n_pre_variant)

        if n_remaining > 0:
            post_variant_days = np.linspace(next_variant_day, last_donation_day, n_remaining +1).round().astype(int)[1:]

            donation_days[n_pre_variant:] = post_variant_days

    donation_days = np.sort(donation_days)

    return donation_days

def calculate_daily_batches(
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
    debug=False
):
    # Convert infections to donors, create the repeated donations, apply the donation window, apply collection capacity, produce batches with activities
    daily_batches = {
        day: []
        for day in range(n_days)
    }

    average_interval = np.full(n_days, np.nan)
    min_interval_used = np.full(n_days, np.nan)
    max_interval_used = np.full(n_days, np.nan)

    for infection_day in range(n_days):

        # number of donors that were infected on this day
        donors = (
            donor_rate
            * I[infection_day]
        )

        donation_days = build_donation_schedule(infection_day, donations_per_donor, min_donation_interval, window_start, window_end, variant_changes)

        intervals = np.diff(donation_days)

        if len(intervals) > 0:

            average_interval[infection_day] = np.mean(intervals)
            min_interval_used[infection_day] = np.min(intervals)
            max_interval_used[infection_day] = np.max(intervals)

        for donation_day in donation_days:
            if donation_day >= n_days:
                continue

            days_post_infection = (
                donation_day
                - infection_day
            )

            # calculate activity of the batch donated at donation day - it is lower if the variant changed since the donor got infected
            activity = calculate_batch_activity(infection_day, donation_day, initial_ccp_activity, window_start, variant_changes)

            # determine donor infection variant
            donor_variant = "Wuhan"

            if variant_changes:
                for event in variant_changes:
                    if infection_day >= event["day"]:
                        donor_variant = event["name"]

            daily_batches[
                donation_day
            ].append(
                {
                    "donors": donors,
                    "doses": (
                        donors
                        * doses_per_donation
                    ),
                    "activity": activity,
                    "variant": donor_variant,
                    "infection_day": infection_day,
                    "donation_day": donation_day,
                    "days_post_infection": days_post_infection
                }
            )

    # Apply collection capacity
    for day in range(n_days):

        total_donations_today = sum(
            batch["donors"]
            for batch in daily_batches[day]
        )

        if total_donations_today <= capacity_per_day:
            continue

        scaling_factor = (
            capacity_per_day
            / total_donations_today
        )

        for batch in daily_batches[day]:
            batch["donors"] *= scaling_factor
            batch["doses"] *= scaling_factor


    if debug:

        total_donors = np.zeros(n_days)
        total_doses = np.zeros(n_days)
        mean_activity = np.full(n_days, np.nan)
        min_activity = np.full(n_days, np.nan)
        max_activity = np.full(n_days, np.nan)

        for day in range(n_days):

            batches_today = daily_batches[day]

            if len(batches_today) == 0:
                continue

            total_donors[day] = sum(
                batch["donors"]
                for batch in batches_today
            )

            total_doses[day] = sum(
                batch["doses"]
                for batch in batches_today
            )

            activities = [
                batch["activity"]
                for batch in batches_today
            ]

            weights = [
                batch["doses"]
                for batch in batches_today
            ]

            mean_activity[day] = np.average(
                activities,
                weights=weights
            )

            min_activity[day] = np.min(activities)
            max_activity[day] = np.max(activities)


        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            4,
            1,
            figsize=(8, 6),
            sharex=True,
            constrained_layout=True
        )

        # donations after capacity constraint
        ax1.plot(
            total_donors,
            color="tab:blue"
        )

        ax1.axhline(
            capacity_per_day,
            color="red",
            linestyle="--",
            label="Capacity"
        )

        ax1.set_ylabel(
            "Donations/day"
        )

        ax1.set_title(
            "Collected donations"
        )

        ax1.legend()

        # CCP production
        ax2.plot(
            total_doses,
            color="tab:green"
        )

        ax2.set_ylabel(
            "Doses/day"
        )

        ax2.set_title(
            "CCP production"
        )

        # mean activity of collected plasma
        ax3.plot(
            mean_activity,
            color="tab:orange"
        )

        ax3.fill_between(
            range(n_days),
            min_activity,
            max_activity,
            alpha=0.3
        )

        ax3.set_ylabel(
            "Activity"
        )

        ax3.set_title(
            "Mean activity of newly collected CCP"
        )

        ax4.plot(
            average_interval,
            label="Average"
        )

        ax4.plot(
            min_interval_used,
            label="Minimum"
        )

        ax4.plot(
            max_interval_used,
            label="Maximum"
        )

        ax4.fill_between(
            range(n_days),
            min_interval_used,
            max_interval_used,
            alpha=0.3,
            label="Min-Max"
        )

        ax4.set_title(
            "Days between donations"
        )

        ax4.set_xlabel(
            "Infection day"
        )


        # Variant change markers
        for event in variant_changes:

            for ax in [ax1, ax2, ax3, ax4]:

                ax.axvline(
                    event["day"],
                    color="black",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1
                )

            # only label on bottom panel
            ax4.text(
                event["day"],
                ax4.get_ylim()[1] * 0.98,
                event["name"],
                rotation=90,
                ha="right",
                va="top",
                fontsize=7,
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none"
                )
            )



        fig.savefig(
            debug_dir /
            "calculate_daily_batches.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)



    return daily_batches




def update_inventory(
    inventory,
    daily_batches,
    day,
    debug=False
):
    # Age batches and add new batches.

    # age the inventory (all plasma batch ages 1 day) (even though right now it does not lose efficacy when ageing)
    for batch in inventory:
        batch["age"] += 1


    # today's production (prepare to save the age of each stock addition)
    for batch in daily_batches[day]:
        inventory.append(
            {
                "age": 0,
                "doses": batch["doses"], 
                "activity": batch["activity"],
                "variant": batch["variant"],
                "donation_day": day,
                "treatment_class": None
            }
        )

    return inventory


def classify_inventory(
    inventory,
    current_day,
    treatment_duration,
    variant_changes,
    high_risk_use_threshold,
    minimum_usable_activity
):
    # Classifies the batches based on their activity at end of treatment - if it's enough, it goes to high-risk patients, if below threshold, goes to general population.
    # If end treatment activity is below minimum usable, the batch gets deleted

    treatment_end_day = current_day + treatment_duration
    expired_today = 0
    surviving_inventory = []
    for batch in inventory:

        future_activity = batch["activity"] # in principle, there's no decay unless there's variant change

        if variant_changes:
            for event in variant_changes:
                # if there is any variant change in the treatment period, the future activity is penalized
                if(current_day < event["day"] <= treatment_end_day):
                    future_activity *= (1-event["penalty"])
        batch["future_activity"] = future_activity

        if (
            batch["future_activity"]
            >= high_risk_use_threshold
        ):
            batch["treatment_class"] = "high_risk"
            surviving_inventory.append(batch)

        elif (
            batch["future_activity"]
            >= minimum_usable_activity
        ):
            batch["treatment_class"] = "general"
            surviving_inventory.append(batch)

        else:
            expired_today += (batch["doses"])

    return (surviving_inventory, expired_today)
    
def summarize_inventory(
    inventory
):

    high_risk_stock = sum(
        batch["doses"]
        for batch in inventory
        if batch["treatment_class"] == "high_risk"
    )

    general_stock = sum(
        batch["doses"]
        for batch in inventory
        if batch["treatment_class"] == "general"
    )

    return (
        high_risk_stock,
        general_stock
    )

def allocate_patients(
    inventory,
    requested_patients,
    doses_per_treatment,
    available_stock,
    treatment_class
):

    max_new_patients = (
        available_stock
        / doses_per_treatment
    )

    requested_starts = min(
        requested_patients,
        max_new_patients
    )

    doses_needed = (
        requested_starts
        * doses_per_treatment
    )

    remaining = doses_needed
    effective_treatments = 0

    for batch in inventory:

        if remaining <= 0:
            break

        if batch["treatment_class"] != treatment_class:
            continue

        take = min(
            batch["doses"],
            remaining
        )

        effective_treatments += (
            take
            * batch["future_activity"]
        )

        batch["doses"] -= take
        remaining -= take

    actual_doses_reserved = (
        doses_needed - remaining
    )

    new_patients = (
        actual_doses_reserved
        / doses_per_treatment
    )

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
        max_new_patients,
        actual_doses_reserved,
        mean_treatment_activity)

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

def calculate_hospitalization_reduction(
    treated_patients,
    eligible_patients,
    treatment_activity
):

    if eligible_patients > 0:
        treated_fraction = (
            treated_patients
            / eligible_patients
        )
    else:
        treated_fraction = 0

    hospitalization_reduction = (
        treated_fraction
        * treatment_activity
    )

    return (
        treated_fraction,
        hospitalization_reduction
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

def variant_accounting(inventory):

    counts = {
        "wuhan_high_risk": 0,
        "wuhan_general": 0,
        "alpha_high_risk": 0,
        "alpha_general": 0,
        "delta_high_risk": 0,
        "delta_general": 0,
        "omicron_high_risk": 0,
        "omicron_general": 0
    }

    for batch in inventory:

        variant = batch["variant"]
        treatment_class = batch["treatment_class"]
        doses = batch["doses"]

        key = f"{variant.lower()}_{treatment_class}"

        if key in counts:
            counts[key] += doses

    return counts

def run_model(
    H,
    I,
    variant_changes,
    initial_ccp_activity=0.70,
    high_risk_use_threshold=0.50,
    minimum_usable_activity = 0.05,
    A_max=1.0,
    capacity_per_day=100,
    treatment_duration=90,
    potential_donor_rate=0.2,
    over_titre_donor_rate=0.2,
    doses_per_patient_per_day=2,
    donation_volume=0.6,
    donations_per_donor=1,
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

    adoption = calculate_adoption(
    n_days,
    A_max,
    t_start,
    T_rollout,
    debug
    )   

    daily_batches = calculate_daily_batches(
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
    debug = True
    )

    daily_doses = np.zeros(n_days)

    for day in range(n_days):

        daily_doses[day] = sum(
            batch["doses"]
            for batch in daily_batches[day]
        )


    high_risk_population, general_population = calculate_populations(
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
    for i in range(n_days):

        inventory = update_inventory(
            inventory,
            daily_batches,
            i,
            debug=False
        )

        inventory, expired_today = classify_inventory(
            inventory,
            i,
            treatment_duration,
            variant_changes,
            high_risk_use_threshold,
            minimum_usable_activity
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
                summarize_inventory(inventory)
            )

        variant_counts = variant_accounting(inventory)

        for name, value in variant_counts.items():
            variant_series[name][i] = value

        (   inventory,
            high_risk_patients, # actual patients that started (can be less than capacity)
            max_high_risk_patients, # capacity implied by inventory
            high_risk_reserved,
            high_risk_activity,
        ) = allocate_patients(
                inventory,
                high_risk_requested,
                doses_per_treatment,
                high_risk_stock,
                "high_risk"
            )

        (   inventory,
            general_patients,
            _, # not analyzing general population stock limits for now
            general_reserved,
            general_activity,
        ) = allocate_patients(
                inventory,
                general_requested,
                doses_per_treatment,
                general_stock,
                "general"
            )
        
        
        high_risk_demand_series[i] = high_risk_requested
        general_demand_series[i] = general_requested
        general_patients_series[i] = general_patients
        high_risk_patients_series[i] = high_risk_patients
        high_risk_treatment_efficacy_series[i] = high_risk_activity
        general_treatment_efficacy_series[i] = general_activity
        high_risk_stock_series[i] = high_risk_stock
        general_stock_series[i] = general_stock


        inventory = remove_empty_batches(inventory)

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

        # Coverage:
        C[i], effective_coverage[i] = (
            calculate_hospitalization_reduction(
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
        **variant_series,


        # Activity
        "high_risk_end_treatment_efficacy": high_risk_treatment_efficacy_series,
        "general_end_treatment_efficacy": general_treatment_efficacy_series,
        "mean_stock_efficacy": stock_efficacy_series,

        # Capacity diagnostics
        "supply_limited": supply_limited,

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


def debug_calculate_batch_activity(
    donation_window_start,
    donation_window_end,
    min_donation_interval,
    donations_per_donor,
    initial_ccp_activity,
    variant_changes
):

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    representative_infections = [
        30,
        230,
        390,
        610,
        800
    ]

    # --------------------------
    # Shade variant periods
    # --------------------------

    variant_periods = [
        (0, "Wuhan")
    ]

    if variant_changes:
        variant_periods.extend(
            [
                (event["day"], event["name"])
                for event in variant_changes
            ]
        )

    variant_periods.append(
        (1200, "End")
    )

    colors = [
        "#d9d9d9",
        "#c6dbef",
        "#fcbba1",
        "#c7e9c0",
        "#fdd49e"
    ]

    for idx in range(
        len(variant_periods) - 1
    ):

        start_day = variant_periods[idx][0]
        end_day = variant_periods[idx + 1][0]
        name = variant_periods[idx][1]

        ax.axvspan(
            start_day,
            end_day,
            alpha=0.15,
            color=colors[idx % len(colors)]
        )

        y_variant = initial_ccp_activity * 1.05

        ax.text(
            (start_day + end_day) / 2,
            0.1,
            name,
            ha="center",
            va="bottom",
            fontsize=6,
            fontweight="bold"
            )
        
        ax.set_ylim(0, initial_ccp_activity * 1.15)
    # --------------------------
    # Representative donors
    # --------------------------

    for infection_day in representative_infections:

        activities = []

        donation_days = build_donation_schedule(
                infection_day,
                donations_per_donor,
                min_donation_interval,
                donation_window_start,
                donation_window_end,
                variant_changes
            )

        for donation_day in donation_days:
            if donation_day >= 1200: 
                continue

            activity = calculate_batch_activity(
                infection_day,
                donation_day,
                initial_ccp_activity,
                donation_window_start,
                variant_changes
            )

            activities.append(
                activity
            )

        # connect donations
        ax.plot(
            donation_days,
            activities,
            "-o",
            linewidth=1.5,
            markersize=5,
            label=f"Infected day {infection_day}"
        )

        # infection marker
        ax.axvline(
            infection_day,
            color=ax.lines[-1].get_color(),
            alpha=0.3,
            linestyle=":"
        )

        ax.annotate(
            f"Inf {infection_day}",
            (
                infection_day,
                initial_ccp_activity * 0.05
            ),
            rotation=90,
            fontsize=6,
            alpha=0.7
        )

        # donation labels
        for j, donation_day in enumerate(
            donation_days
        ):

            ax.annotate(
                f"D{j+1}\n{donation_day}",
                xy=(
                    donation_day,
                    activities[j]
                ),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=3
            )

        for j in range(len(donation_days) - 1):

            midpoint = (
                donation_days[j]
                + donation_days[j+1]
            ) / 2

            interval = (
                donation_days[j+1]
                - donation_days[j]
            )

            ax.text(
                midpoint,
                activities[j] - 0.03,
                f"{interval}d",
                ha="center",
                fontsize=3
            )
    # --------------------------
    # Variant change lines
    # --------------------------

    if variant_changes:

        for event in variant_changes:

            ax.axvline(
                event["day"],
                color="black",
                alpha=0.4,
                linestyle="--"
            )

    ax.set_title(
        "Donation activity by infection cohort"
    )

    ax.set_xlabel(
        "Simulation day"
    )

    ax.set_ylabel(
        "CCP activity"
    )

    ax.set_ylim(
        0,
        initial_ccp_activity * 1.1
    )

    ax.legend(
        fontsize=7
    )

    ax.grid(
        alpha=0.3
    )

    # --------------------------
    # Save
    # --------------------------

    debug_dir = (
        Path.cwd()
        / "outputs"
        / "debug plots"
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        debug_dir
        / "batch_activity_mechanism.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


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
        "Alpha": 0.3,
        "Delta": 0.3,
        "Omicron": 0.3
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
        variant_changes = variant_changes,
        debug=False
    )
    
    donation_window_start = 30
    donation_window_end = 180
    min_donation_interval = 14

    # debug_calculate_batch_activity(donation_window_start=donation_window_start, donation_window_end=donation_window_end, min_donation_interval=min_donation_interval , donations_per_donor=3, initial_ccp_activity=0.7, variant_changes=variant_changes)



    # summary = summarize_results(results)

    # print_report(results)
    # plot_results(results, summary)
