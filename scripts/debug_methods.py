import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates


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

def debug_calculate_adoption(n_days, A_max, t_start, T_rollout, adoption):
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

def debug_calculate_populations(H, I, delay_inf_to_hosp, high_risk_population, general_population):
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

def debug_donation_schedule(
    donation_window_start,
    donation_window_end,
    min_donation_interval,
    donations_per_donor,
    initial_ccp_activity,
    variant_changes,
    cross_neutralization,
    high_risk_threshold=0.40,
    minimum_usable_activity=0.05
):

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    representative_infections = [
        30,   # Wuhan early
        230,  # Wuhan late
        390,  # Alpha
        610,  # Delta
        800   # Omicron
    ]

    # --------------------------------------------------
    # Variant periods
    # --------------------------------------------------

    variant_periods = [(0, "Wuhan")]

    variant_periods.extend(
        [
            (event["day"], event["name"])
            for event in variant_changes
        ]
    )

    simulation_end = 1200
    variant_periods.append(
        (simulation_end, "End")
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

        ax.axvspan(
            start_day,
            end_day,
            alpha=0.15,
            color=colors[idx % len(colors)]
        )

        ax.text(
            (start_day + end_day) / 2,
            0.03,
            variant_periods[idx][1],
            ha="center",
            fontsize=8,
            fontweight="bold"
        )

    # --------------------------------------------------
    # Example donors
    # --------------------------------------------------

    donor_colors = {
        "Wuhan": "tab:blue",
        "Alpha": "tab:orange",
        "Delta": "tab:red",
        "Omicron": "tab:purple"
    }

    for infection_day in representative_infections:

        donor_variant = get_variant(
            variant_changes,
            infection_day
        )

        donation_days = build_donation_schedule(
            infection_day,
            donations_per_donor,
            min_donation_interval,
            donation_window_start,
            donation_window_end,
            variant_changes
        )

        activities = []

        for donation_day in donation_days:

            treatment_end_day = (
                donation_day
                + 90
            )

            target_variant = get_variant(
                variant_changes,
                treatment_end_day
            )

            multiplier = (
                cross_neutralization
                .get(donor_variant, {})
                .get(target_variant, 1.0)
            )

            projected_activity = (
                initial_ccp_activity
                * multiplier
            )

            activities.append(
                projected_activity
            )

        ax.plot(
            donation_days,
            activities,
            "-o",
            linewidth=2,
            label=(
                f"{donor_variant} donor "
                f"(infection day {infection_day})"
            ),
            color=donor_colors.get(
                donor_variant,
                "black"
            )
        )

        # infection marker
        ax.scatter(
            infection_day,
            initial_ccp_activity,
            marker="X",
            s=80,
            color="black",
            zorder=5
        )

        for day, activity in zip(
            donation_days,
            activities
        ):

            ax.annotate(
                f"{activity:.2f}",
                (day, activity),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=6
            )

    # --------------------------------------------------
    # Variant transitions
    # --------------------------------------------------

    for event in variant_changes:

        ax.axvline(
            event["day"],
            color="black",
            linestyle="--",
            alpha=0.5
        )

    # --------------------------------------------------
    # Activity thresholds
    # --------------------------------------------------

    ax.axhline(
        initial_ccp_activity,
        color="black",
        linestyle=":",
        label="Initial activity (0.70)"
    )

    ax.axhline(
        high_risk_threshold,
        color="orange",
        linestyle="--",
        label=f"High-risk threshold ({high_risk_threshold:.2f})"
    )

    ax.axhline(
        minimum_usable_activity,
        color="red",
        linestyle=":"
    )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    ax.set_title(
        "Projected end-of-treatment activity of representative donors"
    )

    ax.set_xlabel(
        "Simulation day"
    )

    ax.set_ylabel(
        "Projected end-of-treatment activity"
    )

    ax.set_ylim(
        0,
        0.8
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(
        fontsize=7
    )

    plt.tight_layout()
    plt.show()

def debug_calculate_daily_batches(n_days, daily_batches, capacity_per_day, average_interval, min_interval_used, max_interval_used, variant_changes):
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

def plot_results(dates, results, summary):
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