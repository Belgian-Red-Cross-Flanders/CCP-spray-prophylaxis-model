import debug_methods
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        debug_methods.debug_calculate_adoption(n_days, A_max, t_start, T_rollout, adoption)

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
        debug_methods.debug_calculate_populations(H, I, delay_inf_to_hosp, high_risk_population, general_population)

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
    """
    Constructs schedule of donations for a donor after infection. (all donors on this day have the same schedule)
    - First donation is at infection day + window start
    - produce exactly this number of donations per donor
    - attempts to spread donations evenly across the available sampling window, but when there is a variant transition it groups the donations before this
    - maintains a minimum donation interval when estimating how many donations can fit before a variant change

    """

    # first eligible donation. Donation 1 is fixed at window start
    first_donation_day = (infection_day + window_start)

    if donations_per_donor == 1:
        return np.array([first_donation_day], dtype=int)

    # after the first donation, we maximize the spacing between donations (constrained by the minimum spacing and the window_end and getting as many donations as possible before next variant)
    last_donation_day = (infection_day + window_end) # hypothetical

    # default schedule is evenly spread through window
    donation_days = np.linspace(first_donation_day, last_donation_day, donations_per_donor).round().astype(int)

    # look for the next variant dominance transition after 1st collection (we have a cross-neutralization table so it can be more than 1 variant of distance between infection and usage, that's ok)
    next_variant_day = None
    if variant_changes:
        for event in variant_changes:
            if event["day"] > first_donation_day:
                next_variant_day = event["day"] # found a variant transition after the donation
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

            activity = initial_ccp_activity # initialize with the full activity

            # determine donor infection variant
            donor_variant = get_variant(variant_changes, infection_day)

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
                    "donor_variant": donor_variant,
                    "infection_day": infection_day,
                    "donation_day": donation_day,
                    "days_post_infection": days_post_infection
                }
            )

    # Apply collection capacity
    daily_potential_donations = []
    for day in range(n_days):

        total_donations_today = sum(
            batch["donors"]
            for batch in daily_batches[day]
        )

        daily_potential_donations.append(total_donations_today)

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
        debug_methods.debug_calculate_daily_batches(n_days, daily_batches, capacity_per_day, average_interval, min_interval_used, max_interval_used, variant_changes)

    return daily_batches, daily_potential_donations


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
                "donor_variant": batch["donor_variant"],
                "donation_day": day,
                "treatment_class": None
            }
        )

    return inventory

def get_variant(variant_changes, day):
    """
    Returns the dominant variant on a given day.

    Assumes variant_changes is sorted chronologically:
    [
        {"day": 365, "name": "Alpha"},
        {"day": 500, "name": "Delta"},
        {"day": 700, "name": "Omicron"}
    ]
    """

    variant = "Wuhan"

    for event in variant_changes:
        if day >= event["day"]:
            variant = event["name"]
        else:
            break

    return variant

def classify_inventory(
    inventory,
    variant_today,
    high_risk_use_threshold,
    minimum_usable_activity,
    max_storage_age
):
    # Classifies the batches based on their activity if it would be deployed today - if it's enough, it goes to high-risk patients, if below threshold, goes to general population.
    # If end treatment activity is below minimum usable, the batch gets deleted

    cross_neutralization = {
        "Wuhan": {
            "Wuhan": 1.00,
            "Alpha": 1/np.sqrt(2.3), # 0.66
            "Delta": 1/np.sqrt(1.6), # 0.79
            "Omicron": 1/np.sqrt(20) # 0.22
        },

        "Alpha": {
            "Alpha": 1.00,
            "Delta": 1/np.sqrt(2.2), # 0.67
            "Omicron": 1/np.sqrt(50) # 0.14
        },

        "Delta": {
            "Delta": 1.00,
            "Omicron": 1/np.sqrt(11) # 0.30
        },

        "Omicron": {
            "Omicron": 1.00
        }
    }

    expired_today = 0
    surviving_inventory = []
    for batch in inventory:
        # storage expiry
        if batch["age"] > max_storage_age:
            expired_today += batch["doses"]
            continue

        donor_variant = batch["donor_variant"]

        future_activity = batch["activity"] * cross_neutralization[donor_variant][variant_today]

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

def match_fifo_allocate_patients(
    inventory,
    requested_patients,
    doses_per_treatment,
    available_stock,
    treatment_class,
    variant_today
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
    doses_age = []

    # Prioritize best-matched (highest future_activity) batches first, so
    # a patient draws the most cross-neutralization-matched stock before
    # older, worse-matched stock that merely still clears the threshold.
    # Age is only a tiebreaker among equally-matched batches, to keep
    # FIFO-style waste control within a match tier.
    candidates = sorted(
        (batch for batch in inventory if batch["treatment_class"] == treatment_class),
        key=lambda b: (-b["future_activity"], -b["age"])
    )

    for batch in candidates:

        if remaining <= 0:
            break

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

        doses_age.append({"doses":take, "age":batch["age"], "donor_variant":batch["donor_variant"], "patient_variant":variant_today})
    
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
        mean_treatment_activity,
        doses_age)

def fifo_allocate_patients(
    inventory,
    requested_patients,
    doses_per_treatment,
    available_stock,
    treatment_class,
    variant_today
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
    doses_age = []
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

        doses_age.append({"doses":take, "age":batch["age"], "donor_variant":batch["donor_variant"], "patient_variant":variant_today})
    
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
        mean_treatment_activity,
        doses_age)

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

        variant = batch["donor_variant"]
        treatment_class = batch["treatment_class"]
        doses = batch["doses"]

        key = f"{variant.lower()}_{treatment_class}"

        if key in counts:
            counts[key] += doses

    return counts
