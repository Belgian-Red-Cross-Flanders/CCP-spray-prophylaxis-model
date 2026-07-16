import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model_hospitalization import run_model

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
H = df["NEW_IN"].values
I = df["CASES"].values
# Peak hospitalizations (for normalization)
H_peak = np.max(H)
#endregion

## BASIC PARAMETERS (commented)
#region
# Effectiveness (scenario)
# E = 0.25  # % reduction in hospitalization risk

# Assumption - Average delay infection → hospitalization (approx)
# delay_inf_to_hosp = 7  # days 

## Supply constants ##
# From Elise: 3 donations × 500 mL = 1.5 L per donor
# Total doses = total plasma volume / dose volume
# current spray = 600 µL per nostril = 1.2 mL per dose 
# (600 is the value for the current safety study, but then they need to assess volumes)
# 1.5 L / 0.0012 L ≈ 1250 doses per donor
# treatment: 2 doses/day for 3 months (≈90 days) (so 180 doses per patient)
# patients_per_donor ≈ 1250 / 180 ≈ 7 people

# donation_volume = 0.5 #500 mL
# slider with between 100 µL per nostril to 600 µL per nostril - simulation of their volume study
# dose_volume = 0.0012 # 600 µL per nostril = 1.2 mL per dose 
# how many donations we consider (1 to 3)
# donations_per_donor = 1
# doses_per_patient_per_day = 2
# treatment_duration = 90  # days

# capacity_per_day = 200  # TEMP placeholder (maximum donations/day)

#TODO: find out
# this is taking into account new variants and that the antigens keeps changing, but maybe it is too conservative
# ccp_lifetime = 90  # days plasma remains clinically relevant

# Assumption of donation window:
#  People can donate in a window between [window_start] and [window_end] days post-infection
#TODO: FIND OUT
# window_start = 30
# window_end = 50

# Donor rate (% of the recovered that actually donate)
# potential_donor_rate = 0.1  # 10% of recovered donate (in Belgium - Elise was using Flanders)
# over_titre_donor_rate = 0.2 # 20% of the donors have antibody titres above 20 µg/mL (the threshold used for CP in our hamster study; see the EBioMedicine paper). 
# This estimate is based on donor data from the Meuri & Confident studies in 2021 (approximately n = 70).

## Adoption constants
# A_max = 1  # max 50% adoption
# Timing
# t_start = 60   # days after start
# T_rollout = 200  # days to reach max



#endregion

## FUNCTIONS
#region
def run_tornado(parameter_list):

    tornado_data = []

    for parameter in parameter_list:

        low, high = parameters_dict[parameter]
        outcomes = []

        for value in [low, high]:

            # copy baseline parameters
            params = {
                "E": E,
                "potential_donor_rate": potential_donor_rate,
                "capacity_per_day": capacity_per_day,
                "A_max": A_max,
                "t_start": t_start,
                "T_rollout": T_rollout,
                "donations_per_donor": donations_per_donor,
                "ccp_lifetime": ccp_lifetime,
            }

            # overwrite one parameter
            match parameter:

                case "Effectiveness":
                    params["E"] = value

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

                case "CCP lifetime":
                    params["ccp_lifetime"] = int(value)

            results = run_model(
                H=H,
                I=I,
                E=params["E"],
                A_max=params["A_max"],
                capacity_per_day=params["capacity_per_day"],
                treatment_duration=treatment_duration,
                ccp_lifetime=params["ccp_lifetime"],
                potential_donor_rate=params["potential_donor_rate"],
                over_titre_donor_rate=over_titre_donor_rate,
                doses_per_patient_per_day=doses_per_patient_per_day,
                donation_volume=donation_volume,
                donations_per_donor=params["donations_per_donor"],
                dose_volume=dose_volume,
                delay_inf_to_hosp=delay_inf_to_hosp,
                t_start=params["t_start"],
                T_rollout=params["T_rollout"],
                window_start=window_start,
                window_end=window_end,
                debug=False,
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

    st.pyplot(fig)
#endregion
# ------------------------------------------------------
st.title(
    "Intranasal CCP Prophylaxis Model"
)
tab_model, tab_sensitivity = st.tabs(
    [
        "Model",
        "Sensitivity"
    ]
)

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
    E = st.sidebar.slider(
        "Effectiveness",
        min_value=0.01,
        max_value=1.00,
        value=0.7,
        step=0.01
    )

    # Average delay infection → hospitalization 
    delay_inf_to_hosp = int(st.sidebar.text_input("Infection to hospitalization delay (days)", 7))
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
    capacity_per_day = st.sidebar.slider(
        "Maximum donations/day",
        min_value=1,
        max_value=200,
        value=20,
        step=1
    )
    ccp_lifetime = int(st.sidebar.text_input("CCP lifetime (days)", 180))  # days plasma remains clinically relevant
    # this is taking into account new variants and that the antigens keeps changing, but maybe it is too conservative
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
        H=H,
        I=I,
        E=E,
        A_max=A_max,
        capacity_per_day=capacity_per_day,
        treatment_duration=treatment_duration,
        ccp_lifetime=ccp_lifetime,
        potential_donor_rate=potential_donor_rate,
        over_titre_donor_rate=over_titre_donor_rate,
        doses_per_patient_per_day=doses_per_patient_per_day,
        donation_volume=donation_volume,
        donations_per_donor=donations_per_donor,
        dose_volume=dose_volume,
        delay_inf_to_hosp=delay_inf_to_hosp,
        t_start=t_start,
        T_rollout=T_rollout,
        window_start=window_start,
        window_end=window_end,
        debug=False
    )
    #endregion

    ## GLOSSARY
    #region
    with st.expander("Parameter glossary", expanded=False):

        st.markdown("""
    ## Clinical / Biological Parameters

    ### Effectiveness
    Fractional reduction in hospitalization risk among individuals treated with the full prophylaxis.
    - 0.4 = 40% reduction in hospitalization risk
    ---

    ### Infection-to-hospitalization delay
    Average delay between infection and potential hospitalization. Coverage is shifted forward by this number of days when estimating effects on hospital admissions.

    ---

    ## Treatment Design Parameters

    ### Dose volume per nostril
    Volume administered into each nostril at a single administration.
    Examples:
    - 100 µL per nostril
    - 600 µL per nostril
                    
    ---

    ### Doses per patient per day
    Number of administrations received each day as part of the treatment
    Examples:
    - 1 dose/day
    - 2 doses/day
                    
    ---

    ### Treatment duration
    Duration of prophylaxis (days).
                    
    Example:
    - 3 donations × 500 mL = 1.5 L per donor
    - Total doses = total plasma volume / dose volume
    - current spray = 600 µL per nostril = 1.2 mL per dose 
    - 1.5 L / 0.0012 L ≈ 1250 doses per donor
    - treatment: 2 doses/day for 3 months (≈90 days) (so 180 doses per patient)
    - patients_per_donor ≈ 1250 / 180 ≈ 7 people
                    
    ---

    ## Donor Recruitment / Availability Parameters

    ### Potential donor rate
    Fraction of recovered individuals willing and eligible to donate plasma (regardless of it getting used to produce i.n. CCP).
    - 0.10 = 10% of recovered individuals donate
                    
    ---

    ### Over-titre threshold donor rate
    Fraction of donors whose plasma meets the antibody titre threshold required for use. \n
    Info: 20% of the donors have antibody titres above 20 µg/mL (the threshold used for CP in hamster study; see the EBioMedicine paper). 
    This estimate is based on donor data from the Meuri & Confident studies in 2021 (approximately n = 70).

    ---

    ### Window start
    Earliest time after infection at which donation becomes possible.
    Example:
    - 30 = donation possible beginning 30 days after infection
                    
    ---

    ### Window end
    Latest time after infection at which donation is considered possible.
    Example:
    - 50 = donation possible until 50 days after infection
    Together, Window start and Window end define the donation window used for donor recruitment.
                    
    ---

    ## Operational / Supply Chain Parameters

    ### Donation volume
    Amount of plasma collected during a single donation.
    Current default:
    - 0.5 L (500 mL)
    Larger volumes produce more doses per donation.
                    
    ---

    ### Donations per donor
    Number of donations obtained from each donor.
    Higher values increase total production.
                    
    ---

    ### Maximum donations per day
    Operational collection capacity.
    Represents the maximum number of plasma donations that can be processed each day.
    If donor availability (from infections) exceeds this limit, capacity becomes the bottleneck.
                    
    ---

    ### CCP lifetime
    Number of days plasma remains clinically useful in inventory.
    Older plasma is discarded once this threshold is reached.
                    
    ---

    ## Adoption / Implementation Parameters

    ### Maximum adoption
    Maximum fraction of eligible individuals who seek prophylaxis once implementation is complete.
    Examples:
    - 0.50 = 50% adoption
    - 1.00 = full adoption

    This affects treatment demand.
                    
    ---

    ### Start time
    Time (days after model start) when prophylaxis becomes available.
    Before this date, adoption is assumed to be zero.
                    
    ---

    ### Rollout time
    Time required to reach maximum adoption.
    Longer rollout periods delay uptake and reduce early population impact.
                    
    """)
        
    #endregion

    ## KEY RESULTS
    #region
    st.subheader("Key Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Prevented hospitalizations",
            f"{np.sum(results['H_prevented']):,.0f}"
        )

    with col2:
        st.metric(
            "Average coverage",
            f"{100*results['average_coverage']:.1f}%"
        )

    with col3:
        st.metric(
            "Discard rate",
            f"{100*results['total_discarded']/results['total_produced']:.1f}%"
        )
    #endregion

    ## PLOTS
    #region
    fig, axes = plt.subplots(
        4, 1, 
        figsize=(15,40)
    )

    # Hospitalizations reduction (main plot)
    ax = axes[0]
    ax.plot(
        dates,
        H,
        label="Hospitalizations",
        color="black"
    )
    ax.plot(
        dates,
        results["H_ccp"],
        label="With CCP",
        color="blue"
    )
    ax.set_ylabel(
        "Hospital admissions/day"
    )
    ax2 = ax.twinx()
    ax2.plot(
        dates,
        results["H_reduction_pct"],
        label="% reduction",
        color="red",
        linestyle="--"
    )
    ax2.set_ylabel(
        "% reduction"
    )
    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles=lines + lines2, labels=labels + labels2, loc='center right')
    ax.grid()

    # Supply dynamics (produced, delivered, inventory)
    ax3 = axes[1]
    ax3.plot(dates, results["daily_production"]/ results["doses_per_treatment"], label="Production", color="green")
    ax3.plot(dates, results["daily_reserved_doses"] / results["doses_per_treatment"], label="Delivered", color="red")
    ax3.set_ylabel("Treatment courses/day")
    ax4 = ax3.twinx()
    ax4.plot(dates, results["stock"] / results["doses_per_treatment"], label="Inventory", color="blue", linewidth=2)
    ax4.set_ylabel("Treatment courses in inventory")
    ax4.set_title("Supply dynamics")
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4)
    ax3.grid()

    # Demand, supply, coverage
    ax5 = axes[2]
    ax5.plot(
        dates,
        results["demand_today"],
        label="Treatment demand",
        color="black"
    )
    ax5.plot(
        dates,
        results["new_patients"],
        label="Treatment starters",
        color="green"
    )
    ax5.set_ylabel("Patients/day")
    ax6 = ax5.twinx()
    ax6.plot(
        dates,
        results["coverage"],
        color="blue",
        linewidth=1,
        label="Coverage"
    )
    ax6.set_ylabel("Fraction covered")
    ax6.set_ylim(0,1.1)

    ax6.fill_between(
        dates,
        0,
        1,
        where=results["supply_limited"].astype(bool),
        color="red",
        alpha=0.15,
        label="Supply limited"
    )
    ax6.fill_between(
        dates,
        0,
        1,
        where=results["not_supply_limited"].astype(bool),
        color="green",
        alpha=0.10,
        label="Sufficient supply"
    )
    ax6.plot(
        dates,
        results["adoption"],
        "--",
        color="orange",
        label="Adoption",
        linewidth=1
    )

    ax5.set_title("Demand, Delivery and Coverage")

    # Combine legends
    lines5, labels5 = ax5.get_legend_handles_labels()
    lines6, labels6 = ax6.get_legend_handles_labels()
    ax5.legend(lines5 + lines6, labels5 + labels6, loc='center right')
    ax5.grid()


    # Cumulative prevented
    cumulative_prevented = np.cumsum(results["H_prevented"])
    ax7 = axes[3]
    ax7.plot(
        dates,
        cumulative_prevented,
        color="black",
        linewidth=2
        # label="Cumulative hospitalizations prevented"
    )
    total_prevented = np.sum(results["H_prevented"])
    ax7.text(
        0.01,
        0.99,
        f"Prevented: {total_prevented:,.0f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )
    ax7.set_ylabel("Cumulative hospitalizations prevented")
    ax7.set_title("Hospital admissions prevented through i.n. CCP")
    # ax7.legend()
    ax7.grid()


    for ax in axes.flat:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # every 6 months
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # format

        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')


    # configs
    st.set_page_config(
        layout="wide",
    )
    st.pyplot(fig)


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
            "Average coverage",
            "Peak coverage",
            "Stockout days",
            "% days supply limited",
            "Peak daily demand",
            "Peak treatment starts",
            "Peak treatment production",
            "Hospitalizations prevented",
            "Maximum daily hospitalizations reduction",        
            "Total hospitalizations reduction"
        ],
        "Value": 
    [
            f"{results['total_produced']/doses_per_treatment:,.0f}",
            f"{results['total_delivered']/doses_per_treatment:,.0f}",
            f"{results['total_discarded']/doses_per_treatment:,.0f}",
            f"{results['average_stock']/doses_per_treatment:,.0f} treatment courses",
            f"{results['maximum_stock']/doses_per_treatment:,.0f} treatment courses",
            f"{100*results['average_coverage']:.1f}%",
            f"{100*results['peak_coverage']:.1f}%",
            f"{results["stockout_days"]}",
            f"{100*results['fraction_supply_limited']:.1f}%",
            f"{results['peak_daily_demand']:.0f} patients/day",
            f"{results['peak_treatment_starts']:.0f} patients/day",
            f"{results['peak_daily_production']/doses_per_treatment:.0f} treatments/day",
            f"{np.sum(results['H_prevented']):,.0f}",
            f"{np.max(results['H_reduction_pct']):.1f}%",
            f"{100*np.sum(results['H_prevented'])/np.sum(H):.1f}%"
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
                        "Effectiveness": [0.05, 1.00],
                        "Dose volume per nostril": [100, 600],
                        "Potential donor rate": [0.01, 1.00],
                        "Donations per donor": [1, 4],
                        "Maximum donations/day": [1, 50],
                        "CCP lifetime": [30, 360],
                        "Maximum adoption": [0.01, 1.00],
                        "Start time": [0, 200],
                        "Rollout time": [10, 400]
                         }
            
    parameter = st.selectbox(
        "Parameter",
        parameters_dict.keys()
    )

    # Sensitivity plots
    n_points = 50
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

        test_E = E
        test_dose = dose_volume
        test_potential_donor_rate = potential_donor_rate
        test_donations_per_donor = donations_per_donor
        test_capacity = capacity_per_day
        test_lifetime = ccp_lifetime
        test_A_max = A_max
        test_t_start = t_start
        test_rollout = T_rollout

        if parameter == "Effectiveness":
            test_E = value

        elif parameter == "Dose volume per nostril":
            # Convert µL per nostril to model units (L total dose)
            test_dose = value / 500000

        elif parameter == "Potential donor rate":
            test_potential_donor_rate = value

        elif parameter == "Donations per donor":
            test_donations_per_donor = int(value)

        elif parameter == "Maximum donations/day":
            test_capacity = value

        elif parameter == "CCP lifetime":
            test_lifetime = int(value)

        elif parameter == "Maximum adoption":
            test_A_max = value

        elif parameter == "Start time":
            test_t_start = int(value)

        elif parameter == "Rollout time":
            test_rollout = int(value)

        results_sens = run_model(
            H=H,
            I=I,
            E=test_E,
            A_max=test_A_max,
            capacity_per_day=test_capacity,
            treatment_duration=treatment_duration,
            ccp_lifetime=test_lifetime,
            potential_donor_rate=test_potential_donor_rate,
            over_titre_donor_rate=over_titre_donor_rate,
            doses_per_patient_per_day=doses_per_patient_per_day,
            donation_volume=donation_volume,
            donations_per_donor=test_donations_per_donor,
            dose_volume=test_dose,
            delay_inf_to_hosp=delay_inf_to_hosp,
            t_start=test_t_start,
            T_rollout=test_rollout,
            window_start=window_start,
            window_end=window_end,
            debug=False
        )

        prevented.append(
            np.sum(results_sens["H_prevented"])
        )

        
    fig, ax = plt.subplots(figsize=(8, 4))

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

    st.pyplot(fig)

    ## Tornado plot analysis
    baseline_results = run_model(
    H=H,
    I=I,
    E=E,
    A_max=A_max,
    capacity_per_day=capacity_per_day,
    treatment_duration=treatment_duration,
    ccp_lifetime=ccp_lifetime,
    potential_donor_rate=potential_donor_rate,
    over_titre_donor_rate=over_titre_donor_rate,
    doses_per_patient_per_day=doses_per_patient_per_day,
    donation_volume=donation_volume,
    donations_per_donor=donations_per_donor,
    dose_volume=dose_volume,
    delay_inf_to_hosp=delay_inf_to_hosp,
    t_start=t_start,
    T_rollout=T_rollout,
    window_start=window_start,
    window_end=window_end,
    debug=False
    )

    baseline_prevented = np.sum(
        baseline_results["H_prevented"]
    )
    operational_parameters = [
        "Potential donor rate",
        "Maximum donations/day",
        "Maximum adoption",
        "Start time",
        "Rollout time"
    ]

    system_parameters = [
        "Effectiveness",
        "Donations per donor",
        "CCP lifetime"
    ]
   
    baseline_values = {
        "Effectiveness": E,
        "Potential donor rate": potential_donor_rate,
        "Donations per donor": donations_per_donor,
        "Maximum donations/day": capacity_per_day,
        "CCP lifetime": ccp_lifetime,
        "Maximum adoption": A_max,
        "Start time": t_start,
        "Rollout time": T_rollout,
    }

    operational_tornado = run_tornado(
        operational_parameters
    )

    plot_tornado(
        operational_tornado,
        "Operational levers",
        baseline_prevented,
        baseline_values
    )

    system_tornado = run_tornado(
        system_parameters
    )

    plot_tornado(
        system_tornado,
        "Treatment and biological assumptions",
        baseline_prevented,
        baseline_values
    )