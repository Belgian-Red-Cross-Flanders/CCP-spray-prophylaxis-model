import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model_hospitalization import run_model, summarize_results

# HOW TO RUN: & "C:\Users\MCASTRO\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m streamlit run scripts/app.py   
# OPTIMAL PARAMETERS (DIFFERENTIAL EVOLUTION OPTIMIZER) RERUN
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
H_ori = df["NEW_IN"].values
I_ori = df["CASES"].values

variant_changes = []

variant_dates = {
    "Alpha": "2020-12-15",
    "Delta": "2021-06-15",
    "Omicron": "2021-12-15"
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

delay_inf_to_hosp = 7 # 7 days according to cross-correlation between hospitalizations and infections

#endregion

## BASIC PARAMETERS (commented)
#region

## Supply constants ##
# From Elise: 3 donations × 500 mL = 1.5 L per donor
# Total doses = total plasma volume / dose volume
# current spray = 600 µL per nostril = 1.2 mL per dose 
# (600 is the value for the current safety study, but then they need to assess volumes)
# 1.5 L / 0.0012 L ≈ 1250 doses per donor
# treatment: 2 doses/day for 3 months (≈90 days) (so 180 doses per patient)
# patients_per_donor ≈ 1250 / 180 ≈ 7 people

# dose_volume = 0.0012 # 600 µL per nostril = 1.2 mL per dose 
# how many donations we consider (1 to 3)
# donations_per_donor = 1
# doses_per_patient_per_day = 2
# treatment_duration = 90  # days

# Donor rate (% of the recovered that actually donate)
# potential_donor_rate = 0.1  # 10% of recovered donate (in Belgium - Elise was using Flanders)
# over_titre_donor_rate = 0.2 # 20% of the donors have antibody titres above 20 µg/mL (the threshold used for CP in our hamster study; see the EBioMedicine paper). 
# This estimate is based on donor data from the Meuri & Confident studies in 2021 (approximately n = 70).

#endregion

## FUNCTIONS
#region
def add_variant_lines(ax, start_date, variant_changes):
    """
    Draw vertical lines marking variant transitions.
    """

    # Wuhan dominance starts at simulation start
    ax.axvline(
        start_date,
        color="black",
        linestyle=":",
        linewidth=0.7,
        alpha=0.5
    )

    ymax = ax.get_ylim()[1]

    ax.text(
        start_date,
        ymax * 0.95,
        "Wuhan",
        rotation=90,
        va="top",
        fontsize=5,
        alpha=0.5
    )

    for event in variant_changes:

        transition_date = (
            start_date
            + pd.Timedelta(days=event["day"])
        )

        ax.axvline(
            transition_date,
            color="black",
            linestyle=":",
            linewidth=0.7,
            alpha=0.5
        )

        ax.text(
            transition_date,
            ymax * 0.95,
            event["name"],
            rotation=90,
            va="top",
            fontsize=5,
            alpha=0.5
        )

def format_axes(ax):

    # Align all plots
    ax.figure.subplots_adjust(
        left=0.20,
        right=0.97,
        bottom=0.22,
        top=0.92
    )

    # Date axis
    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=6)
    )
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m')
    )

    # Tick labels
    ax.tick_params(
        axis="both",
        labelsize=6
    )

    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    # Axis labels
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)

    # Title
    ax.title.set_size(8)

    ax.margins(x=0)

def run_tornado(parameter_list):

    tornado_data = []

    for parameter in parameter_list:

        low, high = parameters_dict[parameter]
        outcomes = []

        for value in [low, high]:

            # copy baseline parameters
            params = {
                "initial_ccp_activity": initial_ccp_activity,
                "minimum_usable_activity": minimum_usable_activity,
                "potential_donor_rate": potential_donor_rate,
                "capacity_per_day": capacity_per_day,
                "A_max": A_max,
                "t_start": t_start,
                "T_rollout": T_rollout,
                "donations_per_donor": donations_per_donor,
            }

            # overwrite one parameter
            match parameter:

                case "Initial CCP activity":
                    params["initial_ccp_activity"] = value

                case "Minimum usable activity":
                    params["minimum_usable_activity"] = value

                case "Potential donor rate":
                    params["potential_donor_rate"] = value

                case "Maximum donations/day":
                    params["capacity_per_day"] = value

                case "Maximum adoption":
                    params["A_max"] = value

                case "Start time":
                    params["t_start"] = int(value)

                case "Rollout time":
                    params["T_rollout"] = int(value)

                case "Donations per donor":
                    params["donations_per_donor"] = int(value)




            results = run_model(
                H,
                I,
                variant_changes,
                initial_ccp_activity=0.70,
                high_risk_use_threshold=0.50,
                minimum_usable_activity = 0.05,
                A_max=params["A_max"],
                capacity_per_day=params["capacity_per_day"],
                treatment_duration=treatment_duration,
                potential_donor_rate=params["potential_donor_rate"],
                over_titre_donor_rate=over_titre_donor_rate,
                doses_per_patient_per_day=doses_per_patient_per_day,
                donation_volume=donation_volume,
                donations_per_donor=params["donations_per_donor"],
                min_donation_interval=donation_interval,
                dose_volume=dose_volume,
                delay_inf_to_hosp=delay_inf_to_hosp,
                t_start=params["t_start"],
                T_rollout=params["T_rollout"],
                window_start=window_start,
                window_end=window_end
                )

            outcomes.append(
                np.sum(results["H_prevented"])
            )

        tornado_data.append({
            "parameter": parameter,
            "low": outcomes[0],
            "high": outcomes[1],
            "range": abs(outcomes[1] - outcomes[0])
        })

    return sorted(
        tornado_data,
        key=lambda x: x["range"],
        reverse=True
    )

def plot_tornado(
    tornado_data,
    title,
    baseline_prevented,
    baseline_values
):

    labels = [d["parameter"] for d in tornado_data]

    fig, ax = plt.subplots(
        figsize=(9, 0.8 * len(labels) + 2)
    )

    # overall limits for all bars
    xmin = min(
        min(d["low"], d["high"])
        for d in tornado_data
    )

    xmax = max(
        max(d["low"], d["high"])
        for d in tornado_data
    )

    padding = 0.12 * (xmax - xmin)
    offset = 0.04 * (xmax - xmin)
    label_sep = 0.04 * (xmax - xmin)

    ax.set_xlim(
        xmin - padding,
        xmax + padding
    )

    for i, d in enumerate(tornado_data):

        # outcomes at the low/high parameter values
        low_outcome = d["low"]
        high_outcome = d["high"]

        left = min(low_outcome, high_outcome)
        right = max(low_outcome, high_outcome)

        ax.barh(
            i,
            right - left,
            left=left,
            height=0.7
        )

        # parameter values
        param_low, param_high = parameters_dict[d["parameter"]]
        param_base = baseline_values[d["parameter"]]

        # force separation for tiny bars
        if abs(high_outcome - low_outcome) < label_sep:

            low_x = baseline_prevented - label_sep
            high_x = baseline_prevented + label_sep

        else:

            # IMPORTANT:
            # put 0.01 at the actual outcome produced by 0.01
            # put 1.0 at the actual outcome produced by 1.0

            low_x = low_outcome
            high_x = high_outcome

            if low_outcome < high_outcome:
                low_x -= offset
                high_x += offset
            else:
                low_x += offset
                high_x -= offset

        # low parameter value
        ax.text(
            low_x,
            i,
            f"{param_low:g}",
            ha="center",
            va="center",
            fontsize=7,
            color="navy"
        )

        # baseline parameter value
        ax.text(
            baseline_prevented,
            i,
            f"{param_base:g}",
            ha="center",
            va="center",
            fontsize=7,
            bbox=dict(
                facecolor="white",
                alpha=0.95,
                edgecolor="lightgray",
                pad=1
            )
        )

        # high parameter value
        ax.text(
            high_x,
            i,
            f"{param_high:g}",
            ha="center",
            va="center",
            fontsize=7,
            color="darkred"
        )


    ax.axvline(
        baseline_prevented,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Baseline = {baseline_prevented:.0f}"
    )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    ax.invert_yaxis()

    ax.set_xlabel(
        "Total hospitalizations prevented"
    )

    ax.set_title(title)

    ax.legend()

    plt.tight_layout()

    st.pyplot(fig, width="stretch")
#endregion
# ------------------------------------------------------
st.title(
    "Intranasal CCP Prophylaxis Model"
)
tab_pandemic, tab_model, tab_sensitivity = st.tabs(
    [
        "Pandemic",
        "Model",
        "Sensitivity"
    ]
)

with tab_pandemic:
    # change the pandemic here (infections, hospitalizations, variant appearance and cross-neutralization can also be here)
    st.header("Pandemic scenario")
    st.subheader("(relative to observed COVID-19 pandemic in Belgium)")
    infection_multiplier = st.slider(
        "Infection multiplier",
        min_value=1.0,
        max_value=10.0,
        value=1.0,
        step=0.5
    )
    hospitalization_multiplier = st.slider(
        "Hospitalization multiplier (severity - hospitalization risk of respiratory pandemic).",
        min_value=1.0,
        max_value=10.0,
        value=1.0,
        step=0.5
    )

    I = (
        infection_multiplier
        * I_ori
    )

    hospitalization_rate = H_ori / I_ori

    H = (
        I
        * hospitalization_multiplier
        * hospitalization_rate
    )

    # PLOTS
    #region
    show_covid = st.toggle(
    "Show COVID-19",
    value=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    # Infections 
    ax.plot(
        dates,
        I,
        label="Simulated pandemic",
        color="black",
        linewidth=1
    )
    ax.set_ylabel(
        "New infections/day"
    )
    if show_covid:       
        ax.plot(
            dates,
            I_ori,
            label="COVID-19",
            color="red",
            linewidth=1,
            alpha=0.8
        )

    # Combine legends
    ax.legend(loc='upper right', fontsize=7)
    ax.grid()
    add_variant_lines(
            ax,
            dates[0],
            variant_changes
        )
    format_axes(ax)
    st.pyplot(fig, width="stretch")

    fig, ax = plt.subplots(figsize=(6, 3))
    # Hospitalizations 
    ax.plot(
        dates,
        H,
        label="Simulated pandemic",
        color="black",
        linewidth=1
    )
    ax.set_ylabel(
        "Hospital admissions/day"
    )
    if show_covid:       
        ax.plot(
            dates,
            H_ori,
            label="COVID-19",
            color="red",
            linewidth=1,
            alpha=0.8
        )
    # Combine legends
    ax.legend(loc='upper right', fontsize=7)
    ax.grid()
    add_variant_lines(
            ax,
            dates[0],
            variant_changes
        )
    format_axes(ax)
    st.pyplot(fig, width="stretch")
    #endregion


with tab_model:
    ## SIDEBAR
    #region
    st.sidebar.header(
        "Scenario Parameters"
    )
    st.sidebar.subheader(
        "Clinical/Biological"
    )

    # % reduction in hospitalization risk
    initial_ccp_activity = st.sidebar.slider(
        "Initial CCP activity (effectiveness at 30 days after infection)",
        min_value=0.01,
        max_value=1.00,
        value=0.7,
        step=0.01
    )
    high_risk_use_threshold = st.sidebar.slider(
        "Activity threshold for high-risk usage",
        min_value=0.01,
        max_value=1.0,
        value=0.40,
        step=0.01,
        format="%0.2f"
    )    
    minimum_usable_activity = st.sidebar.slider(
        "CCP expiry activity (remove from stock)",
        min_value=0.001,
        max_value=0.2,
        value=0.05,
        step=0.001,
        format="%0.3f"
    )
    max_storage_age = st.sidebar.slider(
        "Max storage age",
        min_value=90,
        max_value=500,
        value=365,
        step=5
    )

    # ---------
    st.sidebar.subheader(
        "Treatment design"
    )
    dose_volume_ul = st.sidebar.slider(
        "Dose volume per nostril (µL)",
        min_value=100,
        max_value=600,
        value=600,
        step=50
    )
    dose_volume = dose_volume_ul / 500000 # converting back to model units (L), for the 2 nostrils
    doses_per_patient_per_day = int(st.sidebar.text_input("Doses per patient per day", 2))
    treatment_duration = int(st.sidebar.text_input("Treatment duration (days)", 90))
    
    # ---------
    st.sidebar.subheader(
        "Donor recruitment/availability"
    )
    # 10% of recovered donate (in Belgium - Elise was using Flanders) (% of the recovered that actually donate)
    potential_donor_rate = st.sidebar.slider(
        "Potential donor rate",
        min_value=0.01,
        max_value=1.00,
        value=0.20,
        step=0.01
    )
    # 20% of the donors have antibody titres above 20 µg/mL (the threshold used for CP in our hamster study; see the EBioMedicine paper). 
    # This estimate is based on donor data from the Meuri & Confident studies in 2021 (approximately n = 70).
    over_titre_donor_rate = st.sidebar.slider(
        "Over titre threshold donor rate",
        min_value=0.01,
        max_value=1.00,
        value=0.20,
        step=0.01
    )
    # Assumption of donation window:
    st.sidebar.caption("People can donate in a window between [window_start] and [window_end] days post-infection")
    #  People can donate in a window between [window_start] and [window_end] days post-infection
    window_start = int(st.sidebar.text_input("Window start (days)", 30))
    window_end = int(st.sidebar.text_input("Window end (days)", 180))
    # ---------
    st.sidebar.subheader(
        "Operational/supply chain"
    )
    donation_volume = float(st.sidebar.text_input("Donation volume (L)", 0.6))
    donations_per_donor = int(st.sidebar.text_input("Donations per donor", 3)) # how many times the donor can go donate in the window of donation pos-infection
    donation_interval = int(st.sidebar.text_input("Minimum interval between donations", 14)) # 2 weeks between donations
    capacity_per_day = st.sidebar.slider(
        "Maximum donations/day",
        min_value=1,
        max_value=450,
        value=100,
        step=1
    )
    # ---------
    st.sidebar.subheader(
        "Adoption/implementation"
    )
    A_max = st.sidebar.slider(
        "Maximum adoption",
        min_value=0.05,
        max_value=1.00,
        value=1.0,
        step=0.05
    )
    t_start = int(st.sidebar.text_input("Start time (days after pandemic/plot start)", 30))  # days after start
    T_rollout = int(st.sidebar.text_input("Rollout time (days after start time)", 30))  # days to reach max
    #endregion

    ## RUN MODEL
    #region
    results = run_model(
        H,
        I,
        variant_changes,
        initial_ccp_activity = initial_ccp_activity,
        high_risk_use_threshold = high_risk_use_threshold,
        minimum_usable_activity = minimum_usable_activity,
        max_storage_age = max_storage_age,
        A_max=A_max,
        capacity_per_day=capacity_per_day,
        treatment_duration=treatment_duration,
        potential_donor_rate=potential_donor_rate,
        over_titre_donor_rate=over_titre_donor_rate,
        doses_per_patient_per_day=doses_per_patient_per_day,
        donation_volume=donation_volume,
        donations_per_donor=donations_per_donor,
        min_donation_interval=donation_interval,
        dose_volume=dose_volume,
        delay_inf_to_hosp=delay_inf_to_hosp,
        t_start=t_start,
        T_rollout=T_rollout,
        window_start=window_start,
        window_end=window_end
    )
    summary = summarize_results(results)
    #endregion

    ## GLOSSARY
    #region
    with st.expander("Parameter glossary", expanded=False):

        st.markdown("""
                    
    ## Clinical / Biological Parameters

    ### Initial CCP activity
    Biological activity of newly produced CCP at the time of collection.
    - 0.70 = newly collected plasma starts at 70% activity

    Activity at collection represents the baseline biological activity of the plasma unit. 
    
    Activity is subsequently adjusted according to the degree of cross-neutralization between the donor infection variant and the dominant circulating variant at the projected end of treatment.

    ---

    ### Infection-to-hospitalization delay - 7 days (fixed)
    Average delay between infection and potential hospitalization. 
    
    Coverage is shifted forward by this number of days when estimating effects on hospital admissions.
    
    This was estimated by cross-correlation between the infections and hospitalizations.

    ---

    ### Activity threshold for high-risk usage
    Minimum projected (at the end of treatment) activity required for CCP to be allocated to the high-risk population.
    
    Projected activity is determined by applying the variant-specific cross-neutralization coefficient between the donor infection variant and the dominant circulating variant expected at treatment completion.

    - 0.50 = plasma projected to retain at least 50% activity at the last day of treatment is reserved for high-risk individuals.

    
    ---    

    ### CCP expiry activity
    Minimum projected end-of-treatment activity required for plasma to remain usable.
   
    Plasma is discarded when its projected activity at the end of a newly initiated treatment falls below this threshold.

    - 0.05 = CCP expected to retain less than 5% activity by the end of treatment is not allocated to new patients.

    ---

    ### Variant cross-neutralization matrix
    Variant-specific effectiveness coefficients describing how well plasma generated from infection with one variant is expected to neutralize a different dominant variant at treatment completion.
    
    The matrix is based on published reductions in neutralization titres observed between SARS-CoV-2 variants:
    
    Sullivan, David J., et al. "Analysis of anti-SARS-CoV-2 Omicron-neutralizing antibody titers in different vaccinated and unvaccinated convalescent plasma sources." Nature Communications 13.1 (2022): 6478.
    
    Dupont, Liane, et al. "Neutralizing antibody activity in convalescent sera from infection in humans with SARS-CoV-2 and variants of concern." Nature microbiology 6.11 (2021): 1433-1442.
    
    |                |     Wuhan    |     Alpha    |     Delta    |     Omicron    |
    |----------------|--------------|--------------|--------------|----------------|
    |     Wuhan      |     1        |     0.66     |     0.79     |     0.22       |
    |     Alpha      |     x        |     1        |     0.67     |     0.14       |
    |     Delta      |     x        |     x        |     1        |     0.30       |
    |     Omicron    |     x        |     x        |     x        |     1          |

    Projected treatment activity is calculated as collection activity * cross-neutralization coefficient.

    --- 
                                                
    ## Treatment Design Parameters

    ### Dose volume per nostril
    Volume administered into each nostril at a single administration. Then, the model assumes administration in both nostrils.
    - 600 µL per nostril
                    
    ---

    ### Doses per patient per day
    Number of administrations received each day as part of the treatment
    - 2 doses/day
                    
    ---

    ### Treatment duration
    Duration of prophylaxis (days). The model reserves the complete treatment course at treatment initiation, so on day 1 of treatment we already take from the stock all the doses for all the days.
                    
    Example:
    - 3 donations × 500 mL = 1.5 L per donor
    - Total doses = total plasma volume / dose volume
    - current spray = 600 µL per nostril = 1.2 mL per dose 
    - 1.5 L / 0.0012 L ≈ 1250 doses per donor
    - treatment: 2 doses/day for 3 months (≈90 days) (so 180 doses per patient (treatment course))
    - patients_per_donor ≈ 1250 / 180 ≈ 7 people
                    
    ---

    ## Donor Recruitment / Availability Parameters

    ### Potential donor rate
    Fraction of recovered individuals willing and eligible to donate plasma.
    - 0.20 = 20% of recovered individuals donate
                    
    ---

    ### Over-titre threshold donor rate
    Fraction of donors whose plasma meets the antibody titre threshold required for use. \n
    Info: 20% of the donors have antibody titres above 20 µg/mL (the threshold used for CP in hamster study; see the EBioMedicine paper). 
    
    This estimate is based on donor data from the Meuri & Confident studies in 2021 (approximately n = 70).
                    
    The effective donor rate is, then:

    Potential donor rate × Over-titre donor rate

    ---

    ### Window start
    Earliest time after infection at which donation becomes possible and the date of the first donation. \n
    Example:
    - 30 = donation possible beginning 30 days after infection
                    
    ---

    ### Window end
    Latest time after infection at which donation is considered possible.
    - 180 = donation possible until 180 days after infection.
                    
    Together, Window start and Window end define the donation window used for donor recruitment.
                    
    ---

    ### Minimum donation interval 
    Minimum time allowed between two donations from the same donor. \n    
    Defaults to 14 days. \n   
    When a variant transition is approaching, donations may be moved closer together to this min interval in order to collect plasma before activity is affected.

    ---

    ## Operational / Supply Chain Parameters

    ### Donation volume
    Amount of plasma collected during a single donation. \n
    Current default:
    - 0.6 L (600 mL)
                    
    Larger volumes produce more doses per donation.
                    
    ---

    ### Donations per donor
    Number of donations obtained from each donor. \n
    The collection schedule is optimized to maximize spacing between donations while preserving as many donations as possible before the next anticipated variant-dominance transition. 
    
    This represents anticipatory collection efforts to obtain plasma before a decline in expected cross-variant effectiveness.
    
    A value of 1 corresponds to a single collection 30 days after infection. Defaults to 3.
                    
    ---

    ### Maximum donations per day
    Operational collection capacity. \n
    Represents the maximum number of plasma donations that can be processed each day. \n
    If donor availability (from infections) exceeds this limit, capacity becomes the bottleneck.
                    
    ---

    ### Inventory classes
    Stored plasma is classified into:

    High-risk stock
    - Projected end-of-treatment activity above the high-risk threshold.

    General-use stock
    - Projected end-of-treatment activity between the high-risk threshold and expiry activity.

    Expired stock
    - Projected end-of-treatment activity below the expiry threshold and removed from inventory.
                    
    ---

    ## Adoption / Implementation Parameters

    ### Maximum adoption
    Maximum fraction of eligible individuals who seek prophylaxis once implementation is complete.
    - 1.00 = full adoption

    Adoption creates treatment demand.
                    
    ---

    ### Start time
    Time (days after model start) when prophylaxis becomes available. \n
    Before this date, adoption is assumed to be zero.
                    
    ---

    ### Rollout time
    Time required to reach maximum adoption. \n
    Longer rollout periods delay uptake and reduce early population impact.
                    
    ---

    # Key Model Outputs

    ### Coverage
    Fraction of the high-risk population that receives treatment.

    Coverage =
    treated high-risk patients /
    high-risk population

    ---

    ### Effective coverage
    Activity-adjusted coverage.

    Effective coverage =
    coverage × treatment activity \n

    This quantity is used to estimate hospitalization reduction.

    ---

    ### High-risk stock
    Inventory projected to retain sufficient activity for high-risk treatment.

    ---

    ### General-use stock
    Inventory no longer suitable for high-risk treatment but still usable for the general population.

    ---

    ### Stock activity
    Average activity of plasma currently stored in inventory.

    ---

    ### End-of-treatment activity
    Average activity expected at completion of treatment among patients starting treatment on a given day. \n
    This value depends on the donor infection variant, the dominant circulating variant expected at treatment completion, and the corresponding cross-neutralization coefficient.

                        
    """)
        
    #endregion

    ## KEY RESULTS
    #region
    st.subheader("Key Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Prevented hospitalizations",
            f"{np.sum(results['H_prevented']):,.0f}/{np.sum(H):,.0f}"
        )

    with col2:
        st.metric(
            "Average high-risk people reached",
            f"{100*summary['average_coverage']:.1f}%"
        )

    col3, col4 = st.columns(2)

    with col3:
        st.metric(
            "Average hospitalizations prevented",
            f"{100*summary['average_effective_coverage']:.1f}%"
        )

    with col4:
        st.metric(
            "Discard rate",
            f"{100*summary['total_discarded']/summary['total_produced']:.1f}%"
        )
    #endregion

    ## PLOTS
    #region
    fig, ax = plt.subplots(figsize=(6, 3))
    # Hospitalizations reduction (main plot)
    show_infections = st.toggle(
    "Show infections instead of % reduction",
    value=False)
    H_ccp_plot = np.where(
        results["coverage"] > 0,
        results["H_ccp"],
        np.nan
    )
    ax.plot(
        dates,
        H,
        label="Hospitalizations",
        color="black"
    )
    ax.plot(
        dates,
        H_ccp_plot,
        label="With CCP",
        color="blue"
    )
    ax.set_ylabel(
        "Hospital admissions/day"
    )
    ax2 = ax.twinx()
    if show_infections:       
        ax2.fill_between(
            dates,
            0,
            I,
            color="orange",
            alpha=0.15
        )
        ax2.set_ylabel("Infections/day")
    else:
        ax2.plot(
            dates,
            results["H_reduction_pct"],
            label="% reduction",
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.5
        )
        ax2.set_ylabel(
            "% reduction"
        )
    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles=lines + lines2, labels=labels + labels2, loc='center right', fontsize=7)
    ax.grid()
    format_axes(ax)
    format_axes(ax2)
    st.pyplot(fig, width="stretch")

    # Cumulative prevented
    fig, ax = plt.subplots(figsize=(6, 3))
    cumulative_prevented = np.cumsum(results["H_prevented"])
    ax.plot(
        dates,
        cumulative_prevented,
        color="black",
        linewidth=2
        # label="Cumulative hospitalizations prevented"
    )
    total_prevented = np.sum(results["H_prevented"])
    ax.text(
        0.03,
        0.96,
        f"Prevented: {total_prevented:,.0f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
        fontsize=6
    )
    ax.set_ylabel("Cumulative hospitalizations prevented")
    ax.set_title("Hospital admissions prevented through i.n. CCP")
    ax.grid()
    format_axes(ax)
    st.pyplot(fig, width="stretch")

    # Flows (produced, delivered, discarded)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(dates, results["daily_production"]/ results["doses_per_treatment"], label="Production", color="blue", linewidth=2, alpha=0.5)
    ax.plot(dates, results["daily_reserved_doses"] / results["doses_per_treatment"], label="Delivered", color="green", linewidth=1)
    ax.plot(dates, results["discarded_doses"] / results["doses_per_treatment"], label="Discarded", color="red", linewidth=1)
    ax.set_ylabel("Treatment courses/day")
    ax.legend(fontsize=7)
    ax.grid()
    ax.set_title("Operational flows")
    add_variant_lines(
        ax,
        dates[0],
        variant_changes
    )
    format_axes(ax)
    st.pyplot(fig, width="stretch")

    # Stock
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(dates, results["stock"] / results["doses_per_treatment"], label="Stock", color="blue", linewidth=2, alpha=0.5)
    ax.plot(dates, results["high_risk_stock"] / results["doses_per_treatment"], label="High-risk stock", color="red", linewidth=1)
    ax.plot(dates, results["general_stock"] / results["doses_per_treatment"], label="General-use stock", color="green", linewidth=1)
    ax.set_ylabel("Treatment courses in inventory")
    ax.set_title("Inventory")
    ax.legend(fontsize=7)
    ax.grid()
    add_variant_lines(
        ax,
        dates[0],
        variant_changes
    )    
    format_axes(ax)
    st.pyplot(fig, width="stretch")

    # Demand and treated (high-risk)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(
        dates,
        results["high_risk_demand"],
        label="High-risk demand",
        color="black"
    )
    ax.plot(
        dates,
        results["high_risk_patients"],
        label="High-risk treated",
        color="red",
        linewidth=0.5
    )
    ax.set_ylabel("Patients/day")
    ax.legend(fontsize=7)
    ax.grid()
    ax.set_title("High-risk: demand and delivery")
    format_axes(ax)
    add_variant_lines(
        ax,
        dates[0],
        variant_changes
    )
    st.pyplot(fig, width="stretch")

    # Demand and treated (general)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(
    dates,
    results["general_demand"],
    label="General demand",
    color="gray",
    alpha=0.5
    )
    ax.set_ylabel("Patients/day")
    ax.plot(
        dates,
        results["general_patients"],
        label="General treated",
        color="green"
    )
    ax.legend(fontsize=7)
    ax.grid()
    ax.set_title("General population: demand and delivery")
    add_variant_lines(
        ax,
        dates[0],
        variant_changes
    )
    format_axes(ax)
    st.pyplot(fig, width="stretch")


    # Stock by variant and eligibility
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.stackplot(
        dates,

        results["wuhan_high_risk"] / results["doses_per_treatment"],
        results["wuhan_general"] / results["doses_per_treatment"],

        results["alpha_high_risk"] / results["doses_per_treatment"],
        results["alpha_general"] / results["doses_per_treatment"],

        results["delta_high_risk"] / results["doses_per_treatment"],
        results["delta_general"] / results["doses_per_treatment"],

        results["omicron_high_risk"] / results["doses_per_treatment"],
        results["omicron_general"] / results["doses_per_treatment"],

        colors=[
            "#08306b",  # dark blue
            "#6baed6",  # light blue

            "#ad2121", 
            "#fd6b6b",  

            "#00441b",  
            "#74c476",  

            "#F7A205",  
            "#f8fb4a"   
        ],

        labels=[
            "Wuhan high-risk",
            "Wuhan general",

            "Alpha high-risk",
            "Alpha general",

            "Delta high-risk",
            "Delta general",

            "Omicron high-risk",
            "Omicron general"
        ]
    )

    add_variant_lines(
        ax,
        dates[0],
        variant_changes
    )

    ax.set_ylabel("Treatment courses")
    ax.legend(bbox_to_anchor=(0.4, 1.05))
    ax.grid()

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    ax.set_title("Inventory composition by variant and eligibility", pad=25)
    # Put a legend below current axis
    ax.legend(ncol=4, fontsize=5, loc='upper center', bbox_to_anchor=(0.5, 1.15))
    format_axes(ax)

    st.pyplot(fig, width="stretch")

    # Coverage, efficacy and adoption
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(
        dates,
        results["coverage"],
        label="High-risk coverage", 
        linewidth=0.5
    )
    ax.plot(
        dates,
        results["coverage_effective"],
        label="Effective coverage", 
        linewidth=1
    )
    ax.plot(
        dates,
        results["high_risk_end_treatment_efficacy"],
        "--",
        label="High-risk end of treatment efficacy", 
        linewidth=1
    )
    general_eff = np.where(
        results["general_end_treatment_efficacy"] > 0,
        results["general_end_treatment_efficacy"],
        np.nan
    )

    if np.any(~np.isnan(general_eff)):

        ax.plot(
            dates,
            general_eff,
            "--",
            label="General end of treatment efficacy",
            linewidth=1
        )
    ax.plot(
        dates,
        results["mean_stock_efficacy"],
        ":",
        label="Stock efficacy", 
        linewidth=1
    )
    ax.plot(
        dates,
        results["adoption"],
        "--",
        color="orange",
        label="Adoption",
        linewidth=1
    )
    ax.fill_between(
    dates,
    0,
    1,
    where=results["supply_limited"].astype(bool),
    color="red",
    alpha=0.1,
    label="Supply limited"
    )
    ax.set_ylabel("Fraction (activity)")
    ax.set_ylim(0,1.1)
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    ax.set_title("Coverage, efficacy and adoption", pad=25)
    ax.legend(ncol=4, fontsize=5, loc='upper center', bbox_to_anchor=(0.5, 1.15))
    ax.grid()
    add_variant_lines(
        ax,
        dates[0],
        variant_changes
    )
    format_axes(ax)
    st.pyplot(fig, width="stretch")

    # Relative inventory composition
    fig, ax = plt.subplots(figsize=(6, 3))

    total_classified_stock = (
        results["high_risk_stock"]
        + results["general_stock"]
    )

    high_risk_fraction = (
        100
        * results["high_risk_stock"]
        / np.maximum(total_classified_stock, 1)
    )

    general_fraction = (
        100
        * results["general_stock"]
        / np.maximum(total_classified_stock, 1)
    )

    ax.stackplot(
        dates,
        high_risk_fraction,
        general_fraction,
        labels=[
            "High-risk eligible",
            "General-use"
        ],
        colors=[
            "#d62728",
            "#2ca02c"
        ]
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("% of inventory")
    ax.set_title(
        "Inventory eligibility composition"
    )
    ax.legend()
    ax.grid()
    format_axes(ax)

    st.pyplot(fig, width="stretch")

    # configs
    st.set_page_config(
        layout="wide",
    )
    #endregion

    ## RESULTS TABLE
    #region
    doses_per_treatment = (
        doses_per_patient_per_day
        * treatment_duration
    )

    report = pd.DataFrame({
        "Metric": [
            "Total treatment courses produced",
            "Total treatment courses delivered",
            "Total treatment courses discarded",
            "Average inventory",
            "Peak inventory",
            "Peak high-risk inventory",
            "Peak general inventory",
            "Average high-risk coverage",
            "Peak high-risk coverage",
            "Stockout days",
            "% days supply limited",
            "Peak high-risk demand",
            "Peak high-risk treatment starts",
            "Peak general users",
            "Peak daily production",
            "Maximum daily hospitalization reduction"
        ],
        "Value": [
            f"{summary['total_produced'] / doses_per_treatment:,.0f}",

            f"{summary['total_delivered'] / doses_per_treatment:,.0f}",

            f"{summary['total_discarded'] / doses_per_treatment:,.0f}",

            f"{summary['average_stock'] / doses_per_treatment:,.0f} treatment courses",

            f"{summary['maximum_stock'] / doses_per_treatment:,.0f} treatment courses",
            f"{summary['maximum_high_risk_stock'] / doses_per_treatment:,.0f} treatment courses",
            f"{summary['maximum_general_stock'] / doses_per_treatment:,.0f} treatment courses",

            f"{100 * summary['average_coverage']:.1f}%",

            f"{100 * summary['peak_coverage']:.1f}%",

            f"{summary['stockout_days']}",

            f"{100 * summary['fraction_supply_limited']:.1f}%",

            f"{summary['peak_high_risk_demand']:.0f} patients/day",

            f"{summary['peak_treatment_starts']:.0f} patients/day",
            f"{summary['peak_general_users']:.0f} patients/day",

            f"{summary['peak_daily_production'] / doses_per_treatment:.0f} treatments/day",

            f"{summary['max_reduction_pct']:.1f}%"

        ]
    })

    st.dataframe(
        report,
        use_container_width=True
    )

    #endregion


with tab_sensitivity:
    st.subheader("Sensitivity Analysis")

    parameters_dict = {
                        "High-risk usage activity threshold": [0.1, 0.6],
                        "Max storage age": [90, 500],
                        "Dose volume per nostril": [100, 600],
                        "Donations per donor": [1, 10],
                        "Maximum donations/day": [1, 450],
                        "Maximum adoption": [0.01, 1.00],
                        "Start time": [0, 200],
                        "Rollout time": [10, 400]
                         }
            
    parameter = st.selectbox(
        "Parameter",
        parameters_dict.keys()
    )

    # Sensitivity plots
    n_points = 20
    ranges = {
        parameter:
            np.arange(int(low), int(high))
            if parameter == "Donations per donor"
            else np.linspace(low, high, n_points)

        for parameter, (low, high)
        in parameters_dict.items()
    }
    x_values = ranges[parameter]
    prevented = []
    for value in x_values:

        test_high_risk_use_threshold = high_risk_use_threshold
        test_max_storage_age = max_storage_age
        test_dose = dose_volume
        test_donations_per_donor = donations_per_donor
        test_capacity = capacity_per_day
        test_A_max = A_max
        test_t_start = t_start
        test_rollout = T_rollout

        if parameter == "High-risk usage activity threshold":
            test_high_risk_use_threshold = value

        elif parameter == "Max storage age":
            test_max_storage_age = int(value)

        elif parameter == "Dose volume per nostril":
            # Convert µL per nostril to model units (L total dose)
            test_dose = value / 500000

        elif parameter == "Donations per donor":
            test_donations_per_donor = int(value)

        elif parameter == "Maximum donations/day":
            test_capacity = value

        elif parameter == "Maximum adoption":
            test_A_max = value

        elif parameter == "Start time":
            test_t_start = int(value)

        elif parameter == "Rollout time":
            test_rollout = int(value)

        results_sens = run_model(
            H,
            I,
            variant_changes,
            initial_ccp_activity=initial_ccp_activity,
            high_risk_use_threshold=test_high_risk_use_threshold, #
            minimum_usable_activity=minimum_usable_activity,
            max_storage_age=test_max_storage_age, #
            A_max=test_A_max, #
            capacity_per_day=test_capacity, #
            treatment_duration=treatment_duration,
            potential_donor_rate=potential_donor_rate,
            over_titre_donor_rate=over_titre_donor_rate,
            doses_per_patient_per_day=doses_per_patient_per_day,
            donation_volume=donation_volume,
            donations_per_donor=test_donations_per_donor,
            min_donation_interval=donation_interval,
            dose_volume=test_dose,
            delay_inf_to_hosp=delay_inf_to_hosp,
            t_start=test_t_start,
            T_rollout=test_rollout,
            window_start=window_start,
            window_end=window_end
        )

        prevented.append(
            np.sum(results_sens["H_prevented"])
        )

        
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(
        x_values,
        prevented,
        marker="o"
    )

    ax.set_xlabel(parameter)

    ax.set_ylabel(
        "Total hospitalizations prevented"
    )

    ax.set_title(
        f"Sensitivity to {parameter}"
    )

    ax.grid(True)

    st.pyplot(fig, width="stretch")

    # ## Tornado plot analysis
    # baseline_results = run_model(
    # H,
    # I,
    # variant_changes,
    # initial_ccp_activity=initial_ccp_activity,
    # minimum_usable_activity=minimum_usable_activity,
    # A_max=A_max,
    # capacity_per_day=capacity_per_day,
    # treatment_duration=treatment_duration,
    # potential_donor_rate=potential_donor_rate,
    # over_titre_donor_rate=over_titre_donor_rate,
    # doses_per_patient_per_day=doses_per_patient_per_day,
    # donation_volume=donation_volume,
    # donations_per_donor=donations_per_donor,
    # dose_volume=dose_volume,
    # delay_inf_to_hosp=delay_inf_to_hosp,
    # t_start=t_start,
    # T_rollout=T_rollout,
    # window_start=window_start,
    # window_end=window_end
    # )

    # baseline_prevented = np.sum(
    #     baseline_results["H_prevented"]
    # )
    # operational_parameters = [
    #     "Potential donor rate",
    #     "Maximum donations/day",
    #     "Maximum adoption",
    #     "Start time",
    #     "Rollout time"
    # ]

    # system_parameters = [
    #     "Initial CCP activity",
    #     "Daily activity decay",
    #     "Minimum usable activity",
    #     "Donations per donor",
    # ]
   
    # baseline_values = {
    #     "Initial CCP activity": initial_ccp_activity,
    #     "Minimum usable activity": minimum_usable_activity,
    #     "Potential donor rate": potential_donor_rate,
    #     "Donations per donor": donations_per_donor,
    #     "Maximum donations/day": capacity_per_day,
    #     "Maximum adoption": A_max,
    #     "Start time": t_start,
    #     "Rollout time": T_rollout,
    # }

    # operational_tornado = run_tornado(
    #     operational_parameters
    # )

    # plot_tornado(
    #     operational_tornado,
    #     "Operational levers",
    #     baseline_prevented,
    #     baseline_values
    # )

    # system_tornado = run_tornado(
    #     system_parameters
    # )

    # plot_tornado(
    #     system_tornado,
    #     "Treatment and biological assumptions",
    #     baseline_prevented,
    #     baseline_values
    # )