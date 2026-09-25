import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# open excel file in data_processed/hosp_results_min_st_max.xlsx and the sheet inside it called hospitalizatoins
# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
input_file = "data_processed/hosp_results_min_st_max.xlsx"
output_file = "outputs/cumulative_hosp_minmax.png"

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df = pd.read_excel(
    input_file,
    sheet_name="hospitalizations",
    engine="openpyxl"
    )

df["Date"] = pd.to_datetime(df["Date"])

# plot against the "Date", the column "Cumulative baseline hospitalizations"
# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))

# Baseline hospitalizations
ax.plot(
    df["Date"],
    df["Cumulative baseline hospitalizations"],
    color="black",
    linewidth=2,
    label="No CCP program"
    )

# Shading between optimistic and conservative scenarios
ax.fill_between(
    df["Date"],
    df["max_Cumulative hospitalizations with CCP"],
    df["min_Cumulative hospitalizations with CCP"],
    color="lightblue",
    alpha=0.30
)

# then plot the lines "min_Cumulative hospitalizations with CCP", "max_Cumulative hospitalizations with CCP", and shade the area between them
#then plot the line "std_Cumulative hospitalizations with CCP", which will fall between the previous 2

# Optimistic scenario
ax.plot(
    df["Date"],
    df["max_Cumulative hospitalizations with CCP"],
    color="tab:green",
    linestyle=":",
    linewidth=1.2,
    label="Optimistic scenario"
    )

# Standard scenario
ax.plot(
    df["Date"],
    df["std_Cumulative hospitalizations with CCP"],
    color="tab:blue",
    linewidth=1.2,
    label="Standard scenario"
    )

# Conservative scenario
ax.plot(
    df["Date"],
    df["min_Cumulative hospitalizations with CCP"],
    color="tab:red",
    linestyle=":",
    linewidth=1.2,
    label="Conservative scenario"
    )

# ---------------------------------------------------------
# Formatting
# ---------------------------------------------------------

ax.set_xlabel("Date")
ax.set_ylabel("Cumulative hospitalizations")

ax.set_title(
    "Cumulative hospitalizations under CCP prophylaxis scenarios"
    )

# Format y-axis: 150000 -> 150k
ax.yaxis.set_major_formatter(
    FuncFormatter(
        lambda x, pos: f"{x/1000:.0f}k"
    )
)

# ---------------------------------------------------------
# Right-side labels: hospitalizations prevented
# ---------------------------------------------------------

baseline_final = (
    df["Cumulative baseline hospitalizations"]
    .iloc[-1]
)

scenarios = [
    (
    "Optimistic",
    "max_Cumulative hospitalizations with CCP",
    "tab:green"
    ),
    (
    "Standard",
    "std_Cumulative hospitalizations with CCP",
    "tab:blue"
    ),
    (
    "Conservative",
    "min_Cumulative hospitalizations with CCP",
    "tab:red"
    ),
    ]

lines = []
for name, column, color in scenarios:
    final_hosp = df[column].iloc[-1]
    prevented = baseline_final - final_hosp
    pct_prevented = (
    100 * prevented / baseline_final
    )
    lines.append((f"{name}: "
        f"{prevented/1000:.0f}k prevented "
        f"({pct_prevented:.1f}%)",
        color))

# position of box
x0 = 0.58
y0 = 0.1


# three colored lines
line_spacing = 0.055
for i, (txt, color) in enumerate(lines):
    ax.text(
        x0 + 0.02,
        y0 + (2-i)*line_spacing,
        txt,
        transform=ax.transAxes,
        fontsize=8,
        color=color,
        ha="left",
        va="center"
        )
    
ax.legend(
    fontsize=8,
    frameon=False
    )

ax.grid(
    alpha=0.25
    )

fig.autofmt_xdate()

plt.tight_layout()

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
    )

plt.show()
plt.close(fig)