import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from io import BytesIO

from model_hospitalization import run_model, summarize_results


# HOW TO RUN: & "C:\Users\MCASTRO\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m streamlit run scripts/app.py   

## FUNCTIONS
#region
def load_scaled_covid(
    hosp_path = "data_processed/hospitalizations_clean.csv",
    cases_path = "data_processed/cases_clean.csv",
    infection_multiplier = 1.0,
    hospitalization_multiplier = 1.0,
):
    """
    Load COVID-19 cases and hospitalizations and apply scaling factors to
    infection incidence and hospitalization severity.

    Returns scaled infections (I), scaled hospitalizations (H), and dates.
    """
    df_hosp = pd.read_csv(hosp_path)
    df_cases = pd.read_csv(cases_path)

    # Convert the csv DATE column to pandas datetime
    df_hosp["DATE"] = pd.to_datetime(df_hosp["DATE"])
    df_cases["DATE"] = pd.to_datetime(df_cases["DATE"])

    # Merge on DATE (inner join keeps only common dates, so the hospitalization dates)
    df = pd.merge(df_hosp, df_cases, on="DATE", how="inner")

    # Sort by date just in case
    df = df.sort_values("DATE")

    # Extract aligned series
    dates = df["DATE"]
    H_ori = df["NEW_IN"].values
    I_ori = df["CASES"].values

    # Multiply the infections by a constant to make the pandemic more "infectious"
    I = infection_multiplier * I_ori

    # Recalculate the hospitalization rate with the new infections
    hospitalization_rate = H_ori / I_ori

    # Recalculate the hospitalizations with the infections and the hospitalizations multiplier (makes the pandemic more "severe")
    H = (
        I
        * hospitalization_multiplier
        * hospitalization_rate
    )

    return I, H, dates

def load_covasim_pandemic(
    covasim_file
):
    """
    Load a Covasim-generated pandemic time series.

    Returns infections (I), hospitalizations (H), and dates.
    """

    df = pd.read_csv(
        covasim_file,
        parse_dates=["date"],
    )

    I = (
        df["infections"].to_numpy()
    )

    H = (
        df["hospitalizations"].to_numpy()
    )

    dates = pd.to_datetime(df["date"])

    return I, H, dates

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

def smart_format(x, pos):
    """
    Format numbers for display with an appropriate number of decimals.
    """
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    elif (abs(x) >= 1) or (abs(x) == 0):
        return f"{x:.0f}"
    else:
        return f"{x:.2f}"

def format_axes(ax, months=True):
    """
    Apply consistent formatting to matplotlib axes.
    """
    # Standardize subplot margins across figures
    ax.figure.subplots_adjust(
        left=0.20,
        right=0.97,
        bottom=0.22,
        top=0.92
    )

    if months:
        # Show date ticks every 6 months
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

    # Rotate date labels to avoid overlap
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    # Axis labels
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)

    ax.yaxis.set_major_formatter(
        FuncFormatter(smart_format)
    )

    # Title
    ax.title.set_size(8)

    # Remove default x-axis padding
    ax.margins(x=0)

#endregion
# ------------------------------------------------------
# Application title and top-level navigation tabs
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
    st.header("Pandemic scenario")     # Pandemic scenario selection and parameterization

    pandemic_source = st.radio(
        "Pandemic source",
        [
            "COVID-19 (possible scaling)",
            "Covasim pandemic"
        ],
        horizontal=True,
        )

    if pandemic_source == "COVID-19 (possible scaling)":
        st.markdown("###### (relative to observed COVID-19 pandemic in Belgium)")

        # Scale infection incidence and severity relative to observed COVID-19
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

        I, H, dates = load_scaled_covid(
            "data_processed/hospitalizations_clean.csv",
            "data_processed/cases_clean.csv",
            infection_multiplier,
            hospitalization_multiplier,
        )

        # Estimated lag between infections and hospital admissions
        delay_inf_to_hosp = 7 # 7 days according to cross-correlation between hospitalizations and infections


    else:
        st.markdown("###### (same population and demographics as Belgium, no treatments or containment measures)")
        I, H, dates = load_covasim_pandemic(
            "data_processed/covasim_belgium_unmitigated.csv",
        )

        # Estimated lag between infections and hospital admissions
        delay_inf_to_hosp = 11 # 11 days according to cross-correlation between hospitalizations and infections

    # Define dates at which a new variant becomes dominant
    variant_changes = []

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


    # PLOTS
    # Pandemic time-series visualizations
    #region

    fig, ax = plt.subplots(figsize=(6, 3))
    # Daily infections 
    ax.plot(
        dates,
        I,
        # label="Simulated pandemic", # Add this back if you want to use the simulations (plot was used to show just the regular COVID-19)
        color="black",
        linewidth=1
    )
    ax.set_ylabel(
        "New infections/day"
    )

    # ax.legend(loc='upper right', fontsize=7) # Add this back if you want to use the simulations (plot was used to show just the regular COVID-19)
    ax.grid(alpha=0.3)
    add_variant_lines(
            ax,
            dates[0],
            variant_changes
        )
    format_axes(ax)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    # Daily hospital admissions
    ax.plot(
        dates,
        H,
        # label="Simulated pandemic",
        color="black",
        linewidth=1
    )
    ax.set_ylabel(
        "Hospital admissions/day"
    )
    # Combine legends
    # ax.legend(loc='upper right', fontsize=7)
    ax.grid(alpha=0.3)
    add_variant_lines(
            ax,
            dates[0],
            variant_changes
        )
    format_axes(ax)
    st.pyplot(fig, width="stretch")
    #endregion


with tab_model:
    ## SIDEBAR - Model scenario configuration
    #region
    st.sidebar.header(
        "Scenario Parameters"
    )

    # Clinical and biological assumptions
    st.sidebar.subheader(
        "Clinical/Biological"
    )

    # CCP effectiveness against hospitalization at 30 days post-infection
    initial_ccp_activity = st.sidebar.slider(
        "Initial CCP activity (effectiveness at 30 days after infection)",
        min_value=0.01,
        max_value=1.00,
        value=0.7,
        step=0.01
    )
    # Activity threshold below which CCP is no longer reserved for high-risk patients
    high_risk_use_threshold = st.sidebar.slider(
        "Activity threshold for high-risk usage",
        min_value=0.01,
        max_value=1.0,
        value=0.40,
        step=0.01,
        format="%0.2f"
    )
    # Minimum activity required for a unit to remain in inventory    
    minimum_usable_activity = st.sidebar.slider(
        "CCP expiry activity (remove from stock)",
        min_value=0.001,
        max_value=0.8,
        value=0.05,
        step=0.001,
        format="%0.3f"
    )
    # Maximum storage duration before units expire
    max_storage_age = st.sidebar.slider(
        "Max storage age",
        min_value=90,
        max_value=800,
        value=730,
        step=10
    )

    # Treatment regimen assumptions
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
    
    # Donor recruitment and eligibility assumptions
    st.sidebar.subheader(
        "Donor recruitment/availability"
    )
    # x% of recovered donate (in Belgium - Elise was using Flanders) 
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
    # Assumption of donation window: Eligible donation period after infection
    st.sidebar.caption("People can donate in a window between [window_start] and [window_end] days post-infection")
    #  People can donate in a window between [window_start] and [window_end] days post-infection
    window_start = int(st.sidebar.text_input("Window start (days)", 30))
    window_end = int(st.sidebar.text_input("Window end (days)", 180))

    # Collection and inventory management assumptions
    st.sidebar.subheader(
        "Operational/supply chain"
    )
    donation_volume = float(st.sidebar.text_input("Donation volume (L)", 0.6))
    donations_per_donor = int(st.sidebar.text_input("Donations per donor", 3)) # how many times the donor can go donate in the window of donation pos-infection
    donation_interval = int(st.sidebar.text_input("Minimum interval between donations", 14)) # 2 weeks between donations
    capacity_per_day = st.sidebar.slider(
        "Maximum donations/day",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )

    # Conditions under which excess stock is released to the general population
    release_threshold = st.sidebar.slider(
        "Release threshold (on high-risk requested/stock rate)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.05
    )
    release_fraction = st.sidebar.slider(
        "Release fraction (release to general if under release_threshold)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.05
    )

    # Adoption and rollout assumptions
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

    # Rollout timing parameters
    t_start = int(st.sidebar.text_input("Start time (days after pandemic/plot start)", 30))  # days after start
    T_rollout = int(st.sidebar.text_input("Rollout time (days after start time)", 30))  # days to reach max after t_start
    #endregion

    ## RUN MODEL (using the selected scenario parameters)
    #region
    ## PARAMETERS USED FOR THIS RUN - Store parameter values for export and reproducibility
    params_df = pd.DataFrame([{
        "Initial CCP activity": initial_ccp_activity,
        "High-risk usage activity threshold": high_risk_use_threshold,
        "Minimum usable activity": minimum_usable_activity,
        "Max storage age": max_storage_age,
        "Maximum adoption": A_max,
        "Maximum donations/day": capacity_per_day,
        "Release threshold": release_threshold,
        "Release fraction": release_fraction,
        "Treatment duration": treatment_duration,
        "Potential donor rate": potential_donor_rate,
        "Over-titre donor rate": over_titre_donor_rate,
        "Doses per patient per day": doses_per_patient_per_day,
        "Donation volume": donation_volume,
        "Donations per donor": donations_per_donor,
        "Minimum donation interval": donation_interval,
        "Dose volume": dose_volume,
        "Infection-to-hospitalization delay": delay_inf_to_hosp,
        "Start time": t_start,
        "Rollout time": T_rollout,
        "Donation window start": window_start,
        "Donation window end": window_end
    }])

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
        release_threshold=release_threshold,
        release_fraction=release_fraction,
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
        window_end=window_end,
        debug=False
    )

    # Calculate aggregate outcome measures
    summary = summarize_results(results)
    #endregion

    ## GLOSSARY
    #region
    with st.expander("Parameter glossary", expanded=False):

        st.markdown("""
                    
    ## Clinical / Biological Parameters

    ### Initial CCP activity
    Maximum prophylactic effectiveness of a CCP unit against a matching viral variant.

    This represents the hospitalization reduction that could be achieved if the donor infection variant and circulating variant are identical.
    
    Activity is subsequently adjusted according to the degree of cross-neutralization between the donor infection variant and the dominant circulating variant at the time of treatment.

    ---

    ### Infection-to-hospitalization delay - 7 days for Belgium data, 11 days for simulated data (fixed)
    Average delay between infection and potential hospitalization. 
    
    Coverage is shifted forward by this number of days when estimating effects on hospital admissions.
    
    This was estimated by cross-correlation between the infections and hospitalizations.

    ---

    ### Activity threshold for high-risk usage
    Minimum activity, given the patient's current variant, required for CCP to be allocated to the high-risk population.
    
    Projected activity is determined by applying the variant-specific cross-neutralization coefficient between the donor infection variant and the dominant circulating variant expected at treatment completion.

    - 0.40 = plasma projected to retain at least 40% activity at the last day of treatment is reserved for high-risk individuals.

    
    ---    

    ### CCP expiry activity
    Minimum projected activity required for a CCP unit to remain eligible for allocation.

    Units with projected activity below this threshold are removed from inventory.

    ### Maximum storage age
    Maximum time (days) that a CCP unit can remain in inventory.

    Units exceeding this age are discarded regardless of their projected activity.

    ---

    ### Variant cross-neutralization matrix
    Variant-specific effectiveness coefficients describing how well plasma generated from infection with one variant is expected to neutralize a different dominant variant at the time of use.
    
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
    - 3 donations × 600 mL = 1.8 L per donor
    - Total doses = total plasma volume / dose volume
    - current spray = 600 µL per nostril = 1.2 mL per dose 
    - 1.8 L / 0.0012 L ≈ 1500 doses per donor
    - treatment: 2 doses/day for 3 months (≈90 days) (so 180 doses per patient (treatment course))
    - patients_per_donor ≈ 1500 / 180 ≈ 8.3 people
                    
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
    Maximum number of donations obtained from each eligible donor during the donation window. \n
    The collection schedule is optimized to maximize spacing between donations while preserving as many donations as possible before the next anticipated variant-dominance transition. 
    
    This represents anticipatory collection efforts to obtain plasma before a decline in expected cross-variant effectiveness.
    
    A value of 1 corresponds to a single collection 30 days after infection. Defaults to 3.
                    
    ---

    ### Maximum donations per day
    Operational collection capacity. \n
    Represents the maximum number of plasma donations that can be processed each day. \n
    If donor availability (from infections) exceeds this limit, capacity becomes the bottleneck.
                    
    ---

    ### Release threshold
    Minimum high-risk stock request rate below whcich a fraction of the high-risk inventory becomes available for allocation to the general population.

    The high-risk stock rate is calculated as:

    High-risk doses requested today/
    Current high-risk inventory
    

    A low value indicates that the current high-risk inventory substantially exceeds today's expected high-risk demand.
    Example:
    - 0.05 = release is activated when today's high-risk demand represents less than 5% of the available high-risk stock.

    ---
    
    ### Release fraction
    Fraction of the current high-risk inventory that may be allocated (today) to the general population when the release threshold condition is met.

    This parameter does not affect stock already classified as general-use stock.

    Example:
    - 0.05 = up to 5% of the high-risk inventory becomes available for general-population allocation when high-risk demand is sufficiently low (below release threshold).

    ---

    ### Inventory classes
    Stored CCP units are classified according to projected activity under the currently dominant variant:

    High-risk stock
    - Activity above the high-risk usage threshold.

    General-use stock
    - Activity below the high-risk threshold but above the expiry threshold.

    Expired stock
    - Activity below the expiry threshold or beyond the maximum storage age.

    ---

    ### Release policy

    When high-risk inventory substantially exceeds expected high-risk demand, a fraction of the reserved inventory can be made available for allocation to the general population.

    Release behavior is governed by the release threshold and release fraction parameters.

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

    Coverage reflects treatment allocation and does not account for differences in CCP activity.

    ---

    ### Effective coverage
    Activity-adjusted treatment coverage.

    Patients treated with higher-activity plasma contribute more to effective coverage than patients treated with lower-activity plasma.
    Effective coverage equals treatment coverage multiplied by the average projected activity of the CCP delivered.

    Effective coverage is used to estimate the reduction in hospitalization burden.

    ---

    ### Hospitalizations prevented

    Estimated reduction in hospital admissions attributable to CCP prophylaxis.

    This outcome is calculated from effective coverage and the infection-to-hospitalization delay.

    ---

    ### High-risk stock
    Inventory projected to retain sufficient activity for high-risk treatment.

    ---

    ### General-use stock
    Inventory no longer suitable for high-risk treatment but still usable for the general population.

    ---

    ### Stock activity
    Average activity of plasma currently stored in inventory.

                        
    """)
        
    #endregion

    ## KEY RESULTS - Summary outcomes for the current scenario
    #region
    st.subheader("Key Results")

    col1, col2 = st.columns(2)

    with col1: # Total hospitalizations prevented over the simulation period
        st.metric(
            "Prevented hospitalizations",
            f"{np.sum(results['H_prevented']):,.0f}/{np.sum(H):,.0f}"
        )

    with col2: # Mean coverage of the high-risk population
        st.metric(
            "Average high-risk people reached",
            f"{100*summary['average_coverage']:.1f}%"
        )

    col3, col4 = st.columns(2)

    with col3: # Hospitalizations prevented as a percentage of all hospitalizations
        st.metric(
            "Average hospitalizations prevented",
            f"{100*np.sum(results['H_prevented'])/np.sum(H):.1f}%"
        )

    with col4: # Fraction of produced CCP that was ultimately discarded
        st.metric(
            "Discard rate",
            f"{100*summary['total_discarded']/summary['total_produced']:.1f}%"
        )
    #endregion

    ## PLOTS
    #region
    fig, ax = plt.subplots(figsize=(6, 3))
    # Hospitalizations over time with and without CCP prophylaxis
    show_infections = st.toggle(
    "Show infections",
    value=False)
    # Hide periods before prophylaxis is deployed
    H_ccp_plot = np.where(
        results["coverage"] > 0,
        results["H_ccp"],
        np.nan
    )

    # Baseline and CCP-adjusted hospitalizations
    ax.plot(
        dates,
        H,
        label="Hospitalizations",
        color="black",
        linewidth=0.5
    )
    ax.plot(
        dates,
        H_ccp_plot,
        label="With CCP",
        color="blue",
        linewidth=0.5
    )
    ax.set_ylabel(
        "Hospital admissions/day"
    )

    # Optional infection incidence overlay
    if show_infections:  
        ax2 = ax.twinx()     
        ax2.fill_between(
            dates,
            0,
            I,
            color="orange",
            alpha=0.15
        )
        ax2.set_ylabel("Infections/day")

        # Combine legends from both y-axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles=lines + lines2, labels=labels + labels2, loc='center right', fontsize=7)
        format_axes(ax2)

    ax.legend(loc='center right', fontsize=7)
    ax.grid(alpha=0.3)
    format_axes(ax)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # Cumulative hospitalization burden
    fig, ax = plt.subplots(figsize=(6, 3))

    # Cumulative hospitalizations without prophylaxis
    cum_hosp = np.cumsum(H)

    # Cumulative hospitalizations after accounting for prevented admissions (with CCP)
    cum_hosp_ccp = np.cumsum(
        H - results["H_prevented"]
    )

    # Plot cumulative lines
    ax.plot(
        dates,
        cum_hosp,
        color="black",
        linewidth=2,
        label="Actual (no program)" #"Werkelijk (geen programma)"
    )

    ax.plot(
        dates,
        cum_hosp_ccp,
        color="tab:blue",
        linewidth=2,
        label="With CCP prophylaxis" #"Met CCP-profylaxe"
    )

    # Highlight prevented hospitalizations
    ax.fill_between(
        dates,
        cum_hosp_ccp,
        cum_hosp,
        color="red",
        alpha=0.25
    )

    # Summary statistics for annotation
    total_hosp = np.sum(H)
    total_prevented = np.sum(results["H_prevented"])
    pct_prevented = (
        100 * total_prevented / total_hosp
    )

    # annotation (option for dutch commented. uncomment if necessary)
    ax.text(
        0.78,
        0.65,
        #f"Totaal voorkomen:\n"
        f"Total prevented:\n"
        f"{total_prevented:,.0f}\n"
        f"({pct_prevented:.1f}%)",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5,
        bbox=dict(
            facecolor="white",
            alpha=0.9
        )
    )

    ax.set_ylabel(
        "Cumulative hospitalizations" #"Cumulatieve ziekenhuisopnames"
    )

    ax.set_title(
        "Cumulative hospitalizations with and without i.n. CCP" #"Cumulatieve ziekenhuisopnames met en zonder intranasaal CCP"
    )

    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    format_axes(ax)

    st.pyplot(fig, width="stretch")
    plt.close(fig)


    # Prepare model inputs and outputs for export
    params_export = params_df.T.reset_index()
    params_export.columns = ["Parameter", "Value"]


    # Export daily and cumulative hospitalization outcomes
    hosp_export = pd.DataFrame({
        "Date": dates,
        "Baseline hospitalizations": H,
        "Hospitalizations with CCP": H - results["H_prevented"],
        "Hospitalizations prevented": results["H_prevented"],
        "Cumulative baseline hospitalizations": cum_hosp,
        "Cumulative hospitalizations with CCP": cum_hosp_ccp,
    })


    # Create an in-memory Excel workbook
    excel_buffer = BytesIO()
    
    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        params_export.to_excel(
            writer,
            sheet_name="Parameters",
            index=False
        )

        hosp_export.to_excel(
            writer,
            sheet_name="Hospitalizations",
            index=False
        )

    # Reset buffer position before download
    excel_buffer.seek(0)
    
    # Download model results as an Excel workbook
    st.download_button(
        label="Download hospitalization results (Excel)",
        data=excel_buffer,
        file_name=f"hosp_results_{pct_prevented:.1f}pct.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        key="download_hosp_results"
    )


    # Hospitalizations by variant period / General population (infections) targeted per variant period
    general_reached = st.toggle(
        "Show general population reached",
        value=False)
    # Define variant periods from the dominance-transition dates
    period_names = ["Wuhan"]
    period_starts = [dates.iloc[0]]
    for event in variant_changes:
        period_names.append(event["name"])
        period_starts.append(
            dates.iloc[0] + pd.Timedelta(days=event["day"])
        )
    period_starts.append(dates.iloc[-1])

    # Aggregate outcomes within each variant period
    realized = []
    prevented = []
    for i in range(len(period_names)):
        start = period_starts[i]
        end = period_starts[i + 1]
        mask = (
            (dates >= start)
            &
            (dates < end)
        )
        if general_reached:
            realized.append(np.sum(results["general_population"][mask]))
            prevented.append(np.sum(results["general_patients"][mask]))
            
        else:
            realized.append(
                np.sum(results["H_ccp"][mask])
            )
            prevented.append(
                np.sum(results["H_prevented"][mask])
            )
    # Stacked bar chart of outcomes by variant period
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(period_names))
    if general_reached:
        label_realized = "Remaining infected population"
        label_prevented = "Infected population reached"
        ylabel = "Non-high-risk infections"
        title = "CCP delivery to non-high-risk infected population"
    else:
        label_realized = "Hospitalizations prevented with i.n. CCP" #"Ziekenhuisopnames met CCP"
        label_prevented = "Prevented hospitalizations" #"Voorkomen ziekenhuisopnames"
        ylabel = "Hospitalizations" #"Ziekenhuisopnames"
        title = "Hospitalizations prevented per variant period" #"Voorkomen ziekenhuisopnames per variantperiode"
    # Remaining hospitalizations (or infections) after CCP
    ax.bar(
        x,
        realized,
        color="steelblue",
        label=label_realized
    )
    # Prevented hospitalizations (or reached infections)
    ax.bar(
        x,
        prevented,
        bottom=realized,
        color="red",
        alpha=0.8,
        label=label_prevented
    )
    # Annotate the prevented/reached component of each bar
    for i in range(len(period_names)):
        if prevented[i] > 0:
            ax.text(
                x[i],
                realized[i] + prevented[i] * 0.5,
                f"{prevented[i]:,.0f}",
                ha="center",
                va="center",
                fontsize=6,
                color="black"
            )
    ax.set_xticks(x)
    ax.set_xticklabels(period_names)
    ax.tick_params(
        axis="both",
        labelsize=6
    )
    ax.set_ylabel(
        ylabel,
        fontsize=8
    )
    ax.set_title(
        title,
        fontsize=8
    )
    ax.legend(
        fontsize=6
    )
    ax.grid(
        alpha=0.3,
        axis="y"
    )
    ax.yaxis.set_major_formatter(
            FuncFormatter(smart_format)
        )
    plt.tight_layout()
    st.pyplot(
        fig,
        width="stretch"
    )
    plt.close(fig)

    # Waterfall plot summarizing the overall impact of CCP prophylaxis
    fig, ax = plt.subplots(figsize=(6,3))
    total_hosp = np.sum(H)
    total_prevented = np.sum(results["H_prevented"])
    remaining = total_hosp - total_prevented # Hospitalizations remaining after CCP impact
    labels = [
        "Expected\n(no CCP)",
        "Prevented",
        "Remaining\n(with CCP)"
    ]
    # Waterfall bar positions
    x = np.arange(3)
    # first bar - Baseline hospitalization burden
    ax.bar(
        x[0],
        total_hosp,
        color="black"
    )
    # Reduction attributable to CCP prophylaxis (negative)
    ax.bar(
        x[1],
        -total_prevented,
        bottom=total_hosp,
        color="red"
    )
    # Residual hospitalization burden after CCP
    ax.bar(
        x[2],
        remaining,
        color="tab:blue"
    )
    # connector lines between waterfall stages
    ax.plot(
        [x[0], x[1]],
        [total_hosp, total_hosp],
        color="gray",
        linestyle="--"
    )
    ax.plot(
        [x[1], x[2]],
        [remaining, remaining],
        color="gray",
        linestyle="--"
    )
    # Annotate bar values and overall percentage reduction
    ax.text(
        x[0],
        total_hosp + 5000,
        f"{total_hosp:,.0f}",
        ha="center",
        fontsize=7
    )
    ax.text(
        x[1],
        total_hosp - total_prevented/2,
        f"-{total_prevented:,.0f}",
        ha="center",
        va="center",
        fontsize=7,
        color="white"
    )
    ax.text(
        x[2],
        remaining + 5000,
        f"{remaining:,.0f}",
        ha="center",
        fontsize=7
    )
    pct = (
        100
        * total_prevented
        / total_hosp
    )
    ax.text(
        0.98,
        0.95,
        f"{pct:.1f}% reduction",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            facecolor="white",
            alpha=0.9
        ),
        fontsize=7
    )
    ax.set_ylim(0, 1.25*total_hosp)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(
        "Hospitalizations", 
        fontsize=8
    )
    ax.tick_params(
        axis="both",
        labelsize=6
    )
    ax.set_title(
        "Impact of intranasal CCP prophylaxis", 
        fontsize=8
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(smart_format)
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # CCP collection requirements relative to national plasma collection capacity
    total_belgium_plasma_capacity = 1000 # a maximum of 1000 donations per day in all plasma (Belgium)
    potential_pct = (np.array(results["daily_potential_donations"])/total_belgium_plasma_capacity * 100)
    donations_pct = (np.array(results["daily_donations"])/total_belgium_plasma_capacity * 100)
    capacity_pct = (capacity_per_day/total_belgium_plasma_capacity * 100)

    fig, ax = plt.subplots(figsize=(6, 3))
    # Potential and realized CCP collections as a percentage of national capacity
    ax.bar(dates, potential_pct, label="Potential donations (willing and over-titer)", color="orange", 
            width=0.3, alpha=0.5)
    # ax.axhline(y=np.mean(potential_pct), color="brown",
    #          linestyle="--", linewidth=0.6, 
    #          label="Average potential donations")
    ax.bar(dates, donations_pct, 
            label="Donations (capped by capacity)", color="green", 
            width=0.3, alpha=0.8)
    # ax.axhline(y=np.mean(donations_pct), color="blue",
    #         linestyle="--", linewidth=0.6, 
    #         label="Average donations")

    # Collection capacity allocated to the CCP program
    ax.axhline(
        y=capacity_pct,
        color="red",
        linestyle="--",
        linewidth=0.4,
        label="CCP collection capacity"
    )
    ax.set_ylim([0,100])

    ax.set_ylabel("% of total Belgian plasma donations (1000 per day)")
    ax.legend(fontsize=5, loc="center right")
    ax.grid(alpha=0.3)
    ax.set_title("CCP collection requirements relative to Belgian plasma donation capacity")
    add_variant_lines(
        ax,
        dates[0],
        variant_changes
    )
    format_axes(ax)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # Inventory available for treatment allocation
    # Convert doses to complete treatment courses
    stock = np.where(
    results["stock"] > 0,
    results["stock"] / results["doses_per_treatment"],
    np.nan
    )

    high_risk_stock = np.where(
        results["high_risk_stock"] > 0,
        results["high_risk_stock"] / results["doses_per_treatment"],
        np.nan
    )

    general_stock = np.where(
        results["general_stock"] > 0,
        results["general_stock"] / results["doses_per_treatment"],
        np.nan
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    # Inventory expressed as complete treatment courses
    ax.plot(dates, stock, label="Stock", color="blue", linewidth=2, alpha=0.5)
    ax.plot(dates, high_risk_stock, label="High-risk stock", color="red", linewidth=1)
    ax.plot(dates, general_stock, label="General-use only stock", color="green", linewidth=1)

    ax.set_ylabel("Treatment courses in inventory")
    ax.set_title("Inventory")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    add_variant_lines(
        ax,
        dates[0],
        variant_changes
    )    
    format_axes(ax)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # CCP utilization and wastage over time
    show_general_delivered = st.toggle(
        "Show treatment courses delivered to general population",
        value=False)
    fig, ax = plt.subplots(figsize=(6, 3))

    # Convert delivered and discarded doses to treatment courses
    delivered_hr = np.where(
        results["daily_reserved_doses_high_risk"] > 0,
        results["daily_reserved_doses_high_risk"] / results["doses_per_treatment"],
        np.nan
    )

    delivered_general = np.where(
        results["daily_reserved_doses_general"] > 0,
        results["daily_reserved_doses_general"] / results["doses_per_treatment"],
        np.nan
    )

    discarded = np.where(
        results["discarded_doses"] > 0,
        results["discarded_doses"] / results["doses_per_treatment"],
        np.nan
    )

    if show_general_delivered:
        # General-population delivery occurs at a much larger scale
        ax.plot(
            dates,
            delivered_general,
            label="Delivered (general pop.)",
            color="gray",
            linewidth=0.4
        )
        ax.set_yscale("log")
        ylabel = "Treatment courses/day (log scale)"
    else:
        ylabel = "Treatment courses/day"

    # High-risk delivery and discarded inventory
    ax.plot(dates, delivered_hr, label="Delivered (high-risk)", color="orange", linewidth=0.4)
    ax.plot(
        dates,
        discarded,
        label="Discarded",
        color="red",
        linewidth=0.4,
        alpha=0.5
    )
        
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=5)
    ax.grid(alpha=0.3)
    ax.set_title("Prophylaxis delivery")
    add_variant_lines(
        ax,
        dates[0],
        variant_changes
    )
    format_axes(ax)
    st.pyplot(fig, width="stretch")


    # Compare prophylaxis demand with actual treatment delivery
    general_dd = st.toggle(
                "Demand and delivery to the general population",
                value=False,
                key="dd")
    
    fig, ax = plt.subplots(figsize=(6, 3))
    if general_dd:
        title ="General population"
        demand = results["general_demand"]
        patients = results["general_patients"]
        ylabel = "Infections/day (log scale)" # General-population values span several orders of magnitude
        ax.set_yscale("log")
    else:
        title = "High-risk"
        demand = results["high_risk_demand"]
        patients = results["high_risk_patients"]
        ylabel = "Hospitalizations/day"

    # Requested versus delivered treatment courses
    ax.fill_between(
        dates,
        demand,
        label=f"{title} demand",
        color="black",
        alpha=0.2
        # width=0.5
    )
    ax.fill_between(
        dates,
        patients,
        label=f"{title} treated",
        alpha=0.4,
        color="red"
        # width=0.5
    )
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=5, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_title(f"{title}: demand and delivery")
    format_axes(ax)
    add_variant_lines(
        ax,
        dates[0],
        variant_changes
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # Inventory composition by donor variant and eligibility class
    fig, ax = plt.subplots(figsize=(6, 3))
    # Stack inventory by donor variant and whether units are currently
    # eligible for high-risk or general-population allocation
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
    ax.grid(alpha=0.3)

    # Move the plotting area down to make room for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    ax.set_title("Inventory composition by variant and eligibility", pad=25)
    # Compact legend above the plot
    ax.legend(ncol=4, fontsize=5, loc='upper center', bbox_to_anchor=(0.5, 1.15))
    format_axes(ax)

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # Delivered plasma age/activity by day (how variant matching and inventory aging influence the characteristics of CCP units delivered)
    # y = plasma age/projected activity
    # Point color = donor variant
    # background color = dominant circulating (patient) variant
    age_activity = st.radio(
    "Display age or activity of the delivered plasma",
    [
        "Age",
        "Activity"
    ],
    horizontal=True,
    )
    show_activity = age_activity == "Activity"
    if show_activity:
        key = "activity"
    else:
        key="age"
    show_general = st.toggle(
            f"{key} of delivered plasma to general population",
            value=False,
            key="age_var")
    fig, ax = plt.subplots(figsize=(6, 3))
    variant_colors = {
        "Wuhan": "#1f77b4",
        "Alpha": "#d62728",
        "Delta": "#2ca02c",
        "Omicron": "#ff7f0e"
    }
    # Background shading by patient variant
    patient_variant_day = []
    # Select deliveries to the high-risk or general population
    patient_variant_stock_delivered = results[
            "high_risk_patient_variant_stock_delivered"
        ]
    donor_variant_stock_delivered = results[
            "high_risk_donor_variant_stock_delivered"
        ]
    age_stock_delivered = results["high_risk_age_stock_delivered"]
    activity_stock_delivered = results["high_risk_activity_stock_delivered"]

    gen_high = "high-risk population"
    if show_general:
        patient_variant_stock_delivered = results[
                "general_patient_variant_stock_delivered"
            ]
        donor_variant_stock_delivered = results[
                "general_donor_variant_stock_delivered"
            ]
        age_stock_delivered = results["general_age_stock_delivered"]
        activity_stock_delivered = results["general_activity_stock_delivered"]
        gen_high = "general population"
    
    for variants in patient_variant_stock_delivered:
        if len(variants) == 0:
            patient_variant_day.append(None)
        else:
            # all deliveries from a day typically correspond to a single
            # dominant circulating variant
            patient_variant_day.append(
                variants[0]
            )
    start_idx = None
    # Shade periods according to the dominant circulating variant
    for i,_ in enumerate(patient_variant_day):
        if patient_variant_day[i] is None:
            continue
        if start_idx is None:
            start_idx = i
        is_end = (
            i == len(patient_variant_day) - 1
            or patient_variant_day[i + 1]
            != patient_variant_day[start_idx]
        )
        if is_end:
            variant = patient_variant_day[start_idx]
            ax.axvspan(
                dates[start_idx],
                dates[i],
                color=variant_colors.get(
                    variant,
                    "lightgray"
                ),
                alpha=0.10
            )
            start_idx = None

    # Scatter points - Plot delivered CCP units individually
    #
    # x-position: delivery date
    # y-position: plasma age or projected activity
    # color: donor infection variant
    for day_idx, (
        activities,
        ages,
        donor_variants,
        patient_variants
    ) in enumerate(
        zip(
            activity_stock_delivered,
            age_stock_delivered,
            donor_variant_stock_delivered,
            patient_variant_stock_delivered
        )
    ):
        if len(ages) == 0:
            continue
        x = [dates[day_idx]] * len(ages)
        colors = [
            variant_colors.get(
                donor_variant,
                "gray"
            )
            for donor_variant in donor_variants
        ]
        if show_activity:
            ax.scatter(
                x,
                activities,
                c=colors,
                s=0.2,
                alpha=0.2
            )
        else:
            ax.scatter(
                x,
                ages,
                c=colors,
                s=0.2,
                alpha=0.2
            )    

    # Legend (donor variant)
    for variant, color in variant_colors.items():
        ax.scatter(
            [],
            [],
            c=color,
            label=variant
        )
    ax.set_ylabel(
        f"{key} of delivered CCP plasma (days)"
    )
    ax.set_title(
        f"Delivered plasma {key} ({gen_high})\n"
        "point color = donor variant, background = patient variant"
    )
    ax.grid(
        alpha=0.3
    )
    add_variant_lines(
        ax,
        dates[0],
        variant_changes
    )
    ax.legend(
        fontsize=5,
        loc="lower right"
    )
    if show_general:
        ax.legend(
            fontsize=5,
            loc="lower left"
        )        
    format_axes(ax)
    st.pyplot(
        fig,
        width="stretch"
    )
    plt.close(fig)

    # Coverage, efficacy and adoption dynamics
    # Effective coverage combines treatment coverage with CCP activity
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(
        dates,
        results["coverage_effective"],
        label="Effective coverage", 
        linewidth=0.5
    )
    # Mean efficacy of CCP delivered to high-risk patients
    ax.plot(
        dates,
        results["high_risk_treatment_efficacy"],
        "--",
        label="High-risk treatment efficacy", 
        linewidth=0.4,
        alpha=0.5
    )
    # Hide periods with no general-population delivery
    general_eff = np.where(
        results["general_treatment_efficacy"] > 0,
        results["general_treatment_efficacy"],
        np.nan
    )

    if np.any(~np.isnan(general_eff)):
        # Mean efficacy of CCP delivered to the general population
        ax.plot(
            dates,
            general_eff,
            "--",
            label="General treatment efficacy",
            linewidth=0.4,
            alpha=0.5
        )
    # Average efficacy of all CCP units currently in inventory
    ax.plot(
        dates,
        results["mean_stock_efficacy"],
        ":",
        label="Mean stock efficacy", 
        linewidth=1
    )
    # Prophylaxis uptake over time
    ax.plot(
        dates,
        results["adoption"],
        "--",
        color="orange",
        label="Adoption",
        linewidth=1
    )
    # Highlight periods where demand exceeds available supply
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
    ax.grid(alpha=0.3)
    add_variant_lines(
        ax,
        dates[0],
        variant_changes
    )
    format_axes(ax)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # configs
    st.set_page_config(
        layout="wide",
    )
    #endregion

    ## RESULTS TABLE (Summary performance metrics for the current scenario)
    #region
    # Convert inventory and delivery outputs from doses to treatment courses
    doses_per_treatment = (
        doses_per_patient_per_day
        * treatment_duration
    )

    report = pd.DataFrame({
        "Metric": [
            "Total number of donors",
            "Total number of donations",
            "Average donations per donor",
            "Total treatment courses produced",
            "Total treatment courses delivered",
            "Total treatment courses delivered (high-risk)",
            "Total treatment courses delivered (general)",
            "Total treatment courses discarded (expiry/low activity)",
            "Remaining treatment courses at end of simulation",
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
            f"{summary['total_donors']:,.0f}",
            f"{summary['total_donations']:,.0f}",
            f"{results['average_donations_per_participating_donor']:,.2f}",

            f"{summary['total_produced'] / doses_per_treatment:,.0f}",

            f"{summary['total_delivered'] / doses_per_treatment:,.0f}",
            f"{summary['total_delivered_high_risk'] / doses_per_treatment:,.0f}",
            f"{summary['total_delivered_general'] / doses_per_treatment:,.0f}",

            f"{summary['total_discarded'] / doses_per_treatment:,.0f}",

            f"{results['remaining_doses'] / doses_per_treatment:,.0f}",

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

    # Parameter ranges explored in the one-way sensitivity analysis
    parameters_dict = { "Potential donor rate": [0.01, 1.0],
                        "Initial CCP activity": [0.1, 1.0],
                        "High-risk usage activity threshold": [0.1, 0.6],
                        "Minimum usable activity": [0.01, 0.5],
                        "Max storage age": [30, 500],
                        "Dose volume per nostril": [100/ 500000, 600/ 500000],
                        "Donations per donor": [1, 10],
                        "Maximum donations/day": [10, 1000],
                        "Release threshold": [0.01, 1.0],
                        "Release fraction": [0.01, 1.0],
                        "Maximum adoption": [0.01, 1.00],
                        "Start time": [0, 200],
                        "Rollout time": [10, 400]
                         }
            
    parameter = st.selectbox(
        "Parameter",
        parameters_dict.keys()
    )

    # Generate evaluation points across the selected parameter range
    n_points = 20
    # Integer-valued parameters are evaluated on discrete values;
    # all others are sampled evenly across their range
    ranges = {
        parameter:
            (
                np.arange(
                    int(low),
                    int(high) + 1
                )
                if parameter == "Donations per donor"
                else np.round(
                    np.linspace(
                        low,
                        high,
                        n_points
                    )
                ).astype(int)
            )
            if parameter in [
                "Donations per donor",
                "Max storage age",
                "Start time",
                "Rollout time"
            ]
            else np.linspace(
                low,
                high,
                n_points
            )

        for parameter, (low, high)
        in parameters_dict.items()
    }

    x_values = ranges[parameter]
    prevented = []
    general = []
    for value in x_values: # Vary one parameter while holding all others constant
        # Start from the baseline scenario
        test_potential_donor_rate = potential_donor_rate
        test_initial_ccp_activity = initial_ccp_activity
        test_high_risk_use_threshold = high_risk_use_threshold
        test_minimum_usable_activity = minimum_usable_activity
        test_max_storage_age = max_storage_age
        test_dose = dose_volume
        test_donations_per_donor = donations_per_donor
        test_capacity = capacity_per_day
        test_release_threshold = release_threshold
        test_release_fraction = release_fraction
        test_A_max = A_max
        test_t_start = t_start
        test_rollout = T_rollout

        # Apply the test value to the selected parameter
        if parameter == "Potential donor rate":
            test_potential_donor_rate = value

        elif parameter == "Initial CCP activity":
                    test_initial_ccp_activity = value

        elif parameter == "High-risk usage activity threshold":
            test_high_risk_use_threshold = value

        elif parameter == "Minimum usable activity":
            test_minimum_usable_activity = value

        elif parameter == "Max storage age":
            test_max_storage_age = value

        elif parameter == "Dose volume per nostril":
            # Convert µL per nostril to model units (L total dose)
            test_dose = value

        elif parameter == "Donations per donor":
            test_donations_per_donor = value

        elif parameter == "Maximum donations/day":
            test_capacity = value

        elif parameter == "Release threshold":
            test_release_threshold = value

        elif parameter == "Release fraction":
            test_release_fraction = value

        elif parameter == "Maximum adoption":
            test_A_max = value

        elif parameter == "Start time":
            test_t_start = value

        elif parameter == "Rollout time":
            test_rollout = value

        # Re-run the model with the modified parameter value
        results_sens = run_model(
            H,
            I,
            variant_changes,
            initial_ccp_activity=test_initial_ccp_activity,
            high_risk_use_threshold=test_high_risk_use_threshold, #
            minimum_usable_activity=test_minimum_usable_activity,
            max_storage_age=test_max_storage_age, #
            A_max=test_A_max, #
            capacity_per_day=test_capacity, #
            release_threshold=test_release_threshold, #
            release_fraction=test_release_fraction, #
            treatment_duration=treatment_duration,
            potential_donor_rate=test_potential_donor_rate,
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
            window_end=window_end,
            debug=False
        )

        # Store outcome metrics for plotting
        prevented.append(
            np.sum(results_sens["H_prevented"])
        )
        general.append(np.sum(results_sens["general_patients"]))

    # Total hospitalization reduction across the parameter range
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
    ax.set_xticks(x_values)
    ax.grid(alpha=0.3)
    format_axes(ax, False)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # Total general-population treatment allocation across the parameter range
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(
        x_values,
        general,
        marker="o"
    )
    ax.set_xlabel(parameter)
    ax.set_ylabel(
        "Total general population reached"
    )

    ax.set_xticks(x_values)
    ax.grid(alpha=0.3)
    format_axes(ax, False)
    st.pyplot(fig, width="stretch")
    plt.close(fig)