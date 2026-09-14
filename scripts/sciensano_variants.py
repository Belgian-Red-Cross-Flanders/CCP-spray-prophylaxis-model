import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# I am interested in the variants (variant in data sheet) Alpha, Delta, Gamma, Omicron, and whatever precedes Alpha (we'll figure that out later)
# plot these variants in the data period in the file. make them different color trends. top plot is ma_variant_perc_14, bottom is ma_variant_freq_7

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

DATA_FILE = Path("data_raw/sciensano_data_covid_variants.xlsx")

VARIANTS_OF_INTEREST = [
    "Alpha",
    "Gamma",
    "Delta",
    "Omicron",
]

# ------------------------------------------------------------------------------
# Read data
# ------------------------------------------------------------------------------
# sheet "data" has columns date (in format yyy-mm-dd), variant (string), ma_variant_perc_14d, ma_variant_freq_7d
# (the last 2 are the values to plot per day per variant group)
df = pd.read_excel(
    DATA_FILE,
    sheet_name="data",
)

# the sheet "table" has the glossary for the variant groups. it has columns Variant, Genes, and Remarks (to ignore)
glossary = pd.read_excel(
    DATA_FILE,
    sheet_name="table",
)

df["date"] = pd.to_datetime(df["date"])

available_variants = set(df["variant"].dropna().unique())

variants_to_plot = VARIANTS_OF_INTEREST

# ------------------------------------------------------------------------------
# Determine variants to plot
# ------------------------------------------------------------------------------

base_variants = ["Alpha", "Gamma", "Delta", "Omicron"]

# Find all Omicron-derived variants from glossary
omicron_variants = (
    glossary.loc[
        glossary["Variant"].str.startswith("Omicron ", na=False),
        "Variant"
    ]
    .str.replace("Omicron ", "", regex=False)
    .tolist()
)

variants_to_plot = base_variants + omicron_variants

# Keep only those that exist in the data sheet
variants_to_plot = [
    v for v in variants_to_plot
    if v in available_variants
]

print("Variants plotted:")
for v in variants_to_plot:
    print(f"  - {v}")

# ------------------------------------------------------------------------------
# Variant display names (for legend)
# ------------------------------------------------------------------------------

display_names = {
    v: v for v in variants_to_plot
}

omicron_glossary = glossary.loc[
    glossary["Variant"].str.startswith("Omicron ", na=False),
    "Variant"
]

for full_name in omicron_glossary:

    short_name = full_name.replace("Omicron ", "")

    if short_name in variants_to_plot:
        display_names[short_name] = full_name

# ------------------------------------------------------------------------------
# Find first appearance date of each variant
# ------------------------------------------------------------------------------

variant_start_dates = []

for variant in variants_to_plot:

    d = (
        df.loc[df["variant"] == variant]
        .sort_values("date")
    )

    # First date with measurable presence
    started = d.loc[d["ma_variant_perc_14d"] > 0]

    if len(started):

        first_date = started["date"].iloc[0]

        variant_start_dates.append(
            (variant, first_date)
        )

print("\nVariant introduction dates:")
for variant, date in variant_start_dates:
    print(f"{variant:15s} : {date.date()}")
    
# ------------------------------------------------------------------------------
# Find dominance transitions
# ------------------------------------------------------------------------------

dominance_df = (
    df[df["variant"].isin(variants_to_plot)]
    .pivot_table(
        index="date",
        columns="variant",
        values="ma_variant_perc_14d",
        aggfunc="first",
    )
    .fillna(0)
    .sort_index()
)

# dominant variant for each day
dominance_df["dominant_variant"] = dominance_df.idxmax(axis=1)

# dates where dominant variant changes
change_mask = (
    dominance_df["dominant_variant"]
    != dominance_df["dominant_variant"].shift()
)

candidate_dates = dominance_df.index[change_mask]

transitions = []
# require a new dominant variant to stay dominant
# for at least 7 consecutive days
for date in candidate_dates:
    new_variant = dominance_df.loc[date, "dominant_variant"]
    future = dominance_df.loc[date:date+pd.Timedelta(days=6), "dominant_variant"] # the variant after 7 days
    if len(future) >= 7 and (future == new_variant).all():
        transitions.append((date, new_variant))


print("\nDominance transitions:")
for date, variant in transitions:
    print(f"{date.date()} : {variant}")

transition_dates = [date for date, _ in transitions]

# ------------------------------------------------------------------------------
# Prepare colors
# ------------------------------------------------------------------------------

cmap = plt.get_cmap("tab10")
colors = {
    variant: cmap(i)
    for i, variant in enumerate(variants_to_plot)
}

# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------

study_start = pd.Timestamp("2020-03-06")
study_end = pd.Timestamp("2023-06-27")

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(14, 10),
    sharex=True,
)

# Top panel: percentage
ax = axes[0]

# shaded study period
ax.axvspan(
    study_start,
    study_end,
    color="lightgrey",
    alpha=0.3,
    zorder=0,
)

for variant in variants_to_plot:
    d = (
        df[df["variant"] == variant]
        .sort_values("date")
    )

    ax.plot(
        d["date"],
        d["ma_variant_perc_14d"],
        label=display_names[variant],
        color=colors[variant],
        linewidth=2,
    )
# ------------------------------------------------------------------------------
# Plot dominance transitions
# ------------------------------------------------------------------------------

for date, variant in transitions:

    ax.axvline(
        date,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    ax.text(
        date,
        1.02,  # assumes percentages roughly 0-100
        variant,
        rotation=90,
        fontsize=8,
        ha="right",
        va="top",
    )

ax.set_title("COVID Variant Share (14-day Moving Average)")
ax.set_ylabel("Percentage (%)")
ax.grid(True, alpha=0.3)
ax.legend()

# Bottom panel: frequency
ax = axes[1]

# shaded study period
ax.axvspan(
    study_start,
    study_end,
    color="lightgrey",
    alpha=0.3,
    zorder=0,
)

for variant in variants_to_plot:
    d = (
        df[df["variant"] == variant]
        .sort_values("date")
    )

    ax.plot(
        d["date"],
        d["ma_variant_freq_7d"],
        label=display_names[variant],
        color=colors[variant],
        linewidth=2,
    )

ax.set_title("COVID Variant Frequency (7-day Moving Average)")
ax.set_ylabel("Frequency")
ax.set_xlabel("Date")
ax.grid(True, alpha=0.3)

# force x-axis to include the full study period,
# creating blank shaded space before variant data starts
axes[1].set_xlim(
    study_start,
    max(study_end, df["date"].max())
)

# ------------------------------------------------------------------------------
# Save figure
# ------------------------------------------------------------------------------

output_file = Path("outputs/covid_variants.png")
output_file.parent.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

print(f"Figure saved to: {output_file}")

# Optional
# plt.show()
plt.close()

# ------------------------------------------------------------------------------
# Optional: display glossary entries for plotted variants
# ------------------------------------------------------------------------------

print("\nGlossary:")
print(
    glossary.loc[
        glossary["Variant"].isin(variants_to_plot),
        ["Variant", "Genes"],
    ]
)