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


def run_model(
    H,
    I,
    E,
    A_max,
    capacity_per_day,
    treatment_duration,
    ccp_lifetime,
    potential_donor_rate,
    over_titre_donor_rate,
    doses_per_patient_per_day,
    donation_volume,
    donations_per_donor,
    dose_volume,
    delay_inf_to_hosp,
    t_start,
    T_rollout,
    window_start,
    window_end,
    debug=True
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
    t = np.arange(len(H))

    # --------------------------
    # ADOPTION FUNCTION
    # Adoption(t) = A_max × ramp(t)
    # --------------------------
    Adoption = np.zeros_like(H, dtype=float)

    for i in range(len(t)):
        if i < t_start:
            Adoption[i] = 0
        elif t_start <= i <= (t_start + T_rollout):
            Adoption[i] = A_max * (i - t_start) / T_rollout
        else:
            Adoption[i] = A_max

    # --------------------------
    # SUPPLY → STOCK
    # --------------------------

    # Daily production (flow)
    daily_doses = np.zeros_like(H, dtype=float)
    
    window_length = window_end - window_start + 1
    
    # smoothed infection curve - infections generate potential donors 
    # but daily reports are noisy bc of reporting delays, weekends, short-term fluctuations
    I_smooth = pd.Series(I).rolling(14).mean().bfill().values
    for i in range(len(t)):

        # distribute the potential donors through the window 
        potential_donors = 0
        for k in range(window_start, window_end+1):
            if i - k >= 0:
                potential_donors += (
                    donor_rate
                    * I_smooth[i - k]
                    / window_length
                )

        # capacity constraint acts on DONORS (how many donors we can collect from in a day)
        actual_donors = min(potential_donors, capacity_per_day)

        # convert to doses (full donor yield)
        daily_doses[i] = actual_donors * doses_per_donor

    # estimate high-risk infections from observed hospitalizations
    high_risk_population = np.roll(
        H,
        -delay_inf_to_hosp
    ).astype(float)

    # last days have no future hospitalization data
    high_risk_population[-delay_inf_to_hosp:] = 0

    stock = np.zeros_like(H, dtype=float)

    # Track patients in active treatment (cohort accumulation) 
    active_patients = np.zeros_like(H, dtype=float)
    new_patients_series = np.zeros_like(H, dtype=float)
    demand_today_series = np.zeros_like(H, dtype=float)
    reserved_doses_series = np.zeros_like(H, dtype=float)
    discarded_doses = np.zeros_like(H, dtype=float)

    C = np.zeros_like(H, dtype=float)
    C_supply = np.zeros_like(H, dtype=float)

    supply_limited = np.zeros_like(H, dtype=int)
    not_supply_limited = np.zeros_like(H, dtype=int)

    inventory = []
    for i in range(len(t)):

        # age the inventory (all plasma batch ages 1 day)
        for batch in inventory:
            batch["age"] += 1

        # today's production (prepare to save the age of each stock addition)
        inventory.append(
            {
                "age": 0,
                "doses": daily_doses[i]
            }
        )

        # discard expired plasma (older than ccp_lifetime removed from inventory)
        expired_today = sum(
            batch["doses"]
            for batch in inventory
            if batch["age"] > ccp_lifetime
        )

        discarded_doses[i] = expired_today

        # remove expired stock
        inventory = [
            batch
            for batch in inventory
            if batch["age"] <= ccp_lifetime
        ]

        # calculate stock that remains after expiry
        available_stock = sum(
            batch["doses"]
            for batch in inventory
        )

        # adoption determines the fraction of these eligible patients that seek treatment
        eligible_today = high_risk_population[i]

        # demand driven only by adoption
        requested_patients = (
            Adoption[i]
            * eligible_today
        )

        demand_today_series[i] = requested_patients

        # treatment capacity from stock
        # because the model reserves an entire treatment when a patient starts tratment
        # so a patient can only start treatment if enough doses exist to cover a whole treatment
        max_new_patients = (
            available_stock
            / doses_per_treatment
        )
        
        # new treatment initiations
        # actual treatment starts are limited by 
        # demand (requestd_patients) and
        # inventory (max_new_patients)

        new_patients = min(
            requested_patients,
            max_new_patients
        )

        new_patients_series[i] = new_patients

        # reserve full treatment courses (FIFO)
        # entire treatment courses arer removed from stock immediatly when treatment starts
        # again, this guarantees that every patient that starts a treatment can complete it
        doses_needed = (
            new_patients
            * doses_per_treatment
        )

        remaining = doses_needed

        for batch in inventory:

            if remaining <= 0:
                break

            take = min(batch["doses"], remaining)

            batch["doses"] -= take
            remaining -= take

        inventory = [
            batch
            for batch in inventory
            if batch["doses"] > 0
        ]

        # patients remain active for the entire treatment duration 
        # even though inventory was already reserved at treatment initiation
        active_patients[i] = np.sum(
            new_patients_series[
                max(0, i - treatment_duration + 1): i + 1
            ]
        )

        # stock after reservation
        stock[i] = sum(
            batch["doses"]
            for batch in inventory
        )

        # Coverage: defined as treated patients/eligible patients
        if eligible_today > 0:

            C[i] = (
                new_patients
                / eligible_today
            )

        else:

            C[i] = 0

        # Supply coverage: fraction of the requested that was fulfilled
        if requested_patients > 0:

            C_supply[i] = (
                new_patients
                / requested_patients
            )

        else:
            # if nobody requests treatment, demand is fully satisfied by definition
            C_supply[i] = 1.0

        # identify limiting factor 
        if max_new_patients < requested_patients: # more requests than availability

            supply_limited[i] = 1

        else:

            not_supply_limited[i] = 1

        # daily delivered doses (for reporting) - 
        # daily delivered today corrspond to the treatment courses started today
        reserved_doses_series[i] = doses_needed
 
    # if debug:

    #     fig, axs = plt.subplots(
    #         2,
    #         2,
    #         figsize=(12, 8)
    #     )
    #     # ----------------------------------
    #     # Production and inventory
    #     # ----------------------------------
    #     axs[0,0].plot(
    #         daily_doses,
    #         label="Production (doses/day)"
    #     )
    #     axs[0,0].plot(
    #         stock,
    #         label="Inventory stock"
    #     )
    #     axs[0,0].set_title(
    #         "Production and Inventory"
    #     )
    #     axs[0,0].legend()
    #     axs[0,0].grid()
    #     # ----------------------------------
    #     # Treatment demand
    #     # ----------------------------------
    #     axs[0,1].plot(
    #         demand_today_series,
    #         label="Demand"
    #     )
    #     axs[0,1].plot(
    #         new_patients_series,
    #         label="Treatment starts"
    #     )
    #     axs[0,1].set_title(
    #         "Demand vs Treatment Starts"
    #     )
    #     axs[0,1].legend()
    #     axs[0,1].grid()
    #     # ----------------------------------
    #     # Coverage
    #     # ----------------------------------
    #     axs[1,0].plot(
    #         C,
    #         label="Coverage"
    #     )
    #     axs[1,0].plot(
    #         Adoption,
    #         label="Adoption"
    #     )
    #     axs[1,0].set_ylim(0,1.05)
    #     axs[1,0].set_title(
    #         "Coverage"
    #     )
    #     axs[1,0].legend()
    #     axs[1,0].grid()
    #     # ----------------------------------
    #     # Plasma wastage
    #     # ----------------------------------
    #     axs[1,1].plot(
    #         discarded_doses,
    #         label="Discarded doses"
    #     )
    #     axs[1,1].set_title(
    #         "Expired Inventory"
    #     )
    #     axs[1,1].legend()
    #     axs[1,1].grid()
    #     plt.tight_layout()
    #     plt.show()

    # Shift coverage forward
    C_effective = np.roll(C, delay_inf_to_hosp)
    C_effective[:delay_inf_to_hosp] = 0

    # --------------------------
    # APPLY MODEL
    # --------------------------
    # Assumption: The fraction of hospitalizations is reduced proportionally to the fraction of treated high‑risk infections
    H_ccp = H * (1 - E * C_effective)

    # Prevented hospitalizations
    H_prevented = H - H_ccp
    H_reduction_pct = (H - H_ccp) / np.maximum(H, 1) * 100

    
    total_produced = np.sum(daily_doses)

    total_delivered = np.sum(
        new_patients_series
    ) * doses_per_treatment

    total_discarded = np.sum(
        discarded_doses
    )

    average_stock = np.mean(stock)

    maximum_stock = np.max(stock)

    stockout_days = np.sum(stock <= 0)

    average_coverage = np.mean(C)

    peak_coverage = np.max(C)

    peak_daily_demand = np.max(demand_today_series)

    peak_treatment_starts = np.max(new_patients_series)

    peak_daily_production = np.max(daily_doses)

    peak_inventory = np.max(stock)

    fraction_supply_limited = np.mean(
        supply_limited
    )

    return {
        "H_ccp": H_ccp,
        "H_prevented": H_prevented,
        "H_reduction_pct": H_reduction_pct,
        "coverage": C,
        "coverage_effective": C_effective,
        "adoption": Adoption,
        "supply_limit": C_supply,
        "stock": stock,
        "doses_per_treatment": doses_per_treatment,
        "treatments_per_donor": treatments_per_donor,
        "daily_production": daily_doses,
        "daily_reserved_doses": reserved_doses_series,
        "active_patients": active_patients,
        "new_patients": new_patients_series,
        "discarded_doses": discarded_doses,
        "supply_limited": supply_limited,
        "not_supply_limited": not_supply_limited,
        "demand_today": demand_today_series,
        "total_produced": total_produced,
        "total_delivered": total_delivered,
        "total_discarded": total_discarded,
        "average_stock": average_stock,
        "maximum_stock": maximum_stock,
        "stockout_days": stockout_days,
        "average_coverage": average_coverage,
        "peak_coverage": peak_coverage,
        "peak_daily_demand": peak_daily_demand,
        "peak_treatment_starts": peak_treatment_starts,
        "peak_daily_production": peak_daily_production,
        "peak_inventory": peak_inventory,
        "fraction_supply_limited": fraction_supply_limited

    }


def plot_results(results):
    # unpack
    H_ccp = results["H_ccp"]
    H_prevented = results["H_prevented"]
    H_reduction_pct = results["H_reduction_pct"]

    coverage = results["coverage"]
    Adoption = results["adoption"]

    stock = results["stock"]

    doses_per_treatment = results["doses_per_treatment"]
    treatments_per_donor = results["treatments_per_donor"]
    daily_production = results["daily_production"]
    daily_reserved_doses = results["daily_reserved_doses"]

    new_patients = results["new_patients"]
    demand_today = results["demand_today"]

    supply_limited = results["supply_limited"]
    not_supply_limited = results["not_supply_limited"]

    ## PLOTS
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # Main hospitalizations plot
    ax = axs[0,0]
    ax.plot(dates, H, label="Hospitalizations", color="black")
    ax.plot(dates, H_ccp, label="With CCP", color="blue")
    ax2 = ax.twinx()
    ax2.plot(dates, H_reduction_pct, label="% reduction", color="red", linestyle="--")
    ax.set_title("Hospitalizations impact")
    ax.set_ylabel("Hospital admissions/day")
    ax2.set_ylabel("% reduction")
    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()


    ax.legend(lines + lines2, labels + labels2)
    ax.grid()


    # Supply dynamics plot
    ax_supply = axs[0,1]
    ax_supply.plot(dates, daily_production / doses_per_treatment, label="Production", color="green")
    ax_supply.plot(dates, daily_reserved_doses / doses_per_treatment, label="Delivered", color="red")
    ax_supply.set_ylabel("Treatment courses/day")
    ax_supply2 = ax_supply.twinx()
    ax_supply2.plot(dates, stock / doses_per_treatment, label="Inventory", color="blue", linewidth=2)
    ax_supply2.set_ylabel("Treatment courses in inventory")
    ax_supply.set_title("Supply dynamics")

    # Legend
    lines, labels = ax_supply.get_legend_handles_labels()
    lines2, labels2 = ax_supply2.get_legend_handles_labels()
    ax_supply.legend(lines + lines2, labels + labels2)

    ax_supply.grid()

    # Demand, supply, coverage
    ax = axs[1,0]
    ax.plot(
        dates,
        demand_today,
        label="Treatment demand",
        color="black"
    )
    ax.plot(
        dates,
        new_patients,
        label="Treatment starters",
        color="green"
    )
    ax.set_ylabel("Patients/day")

    ax2 = ax.twinx()
    ax2.plot(
        dates,
        coverage,
        color="blue",
        linewidth=1,
        label="Coverage"
    )
    ax2.set_ylabel("Fraction covered")
    ax2.set_ylim(0,1.1)

    ax2.fill_between(
        dates,
        0,
        1,
        where=supply_limited.astype(bool),
        color="red",
        alpha=0.15,
        label="Supply limited"
    )
    ax2.fill_between(
        dates,
        0,
        1,
        where=not_supply_limited.astype(bool),
        color="green",
        alpha=0.10,
        label="Sufficient supply"
    )
    ax2.plot(
        dates,
        Adoption,
        "--",
        color="orange",
        label="Adoption",
        linewidth=1
    )

    ax.set_title("Demand, Delivery and Coverage")

    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2)
    ax.grid()


    # Cumulative prevented
    cumulative_prevented = np.cumsum(H_prevented)
    ax = axs[1,1]
    ax.plot(
        dates,
        cumulative_prevented,
        color="black",
        linewidth=2
        # label="Cumulative hospitalizations prevented"
    )
    total_prevented = np.sum(H_prevented)
    ax.text(
        0.02,
        0.95,
        f"Prevented: {total_prevented:,.0f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )
    axs[1,1].set_ylabel("Cumulative hospitalizations prevented")
    axs[1,1].set_title("Hospital admissions prevented through i.n. CCP")
    # axs[1,1].legend()
    axs[1,1].grid()


    for ax in axs.flat:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # every 6 months
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # format

        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')

    plt.tight_layout()
    plt.show()

# print_report(results)
# plot_results(results)
