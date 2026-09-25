# imports
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

# import csv at "outputs/global_sensitivity_results.csv",
# this csv is global_sensitivity_results
global_sensitivity_results = pd.read_csv("outputs/global_sensitivity_results.csv")

outcome1 = "Hospitalizations prevented"
# outcome1 = "General treated"
y = global_sensitivity_results[outcome1]

test_params = global_sensitivity_results.columns[:-2]
fig, axes = plt.subplots(4, 4, figsize=(14,9))
axes_flat = axes.flatten()

#formatter: 2000 to 2k
k_formatter = FuncFormatter(lambda x, pos: f"{int(x/1000)}k")

for idx, parameter in enumerate(test_params):
    x = global_sensitivity_results[parameter]

    loess_fit = lowess(
        y,
        x,
        frac=0.3,
        return_sorted=True
    )
    ax = axes_flat[idx]
    ax.scatter(
        x,
        y,
        s=2,
        alpha=0.05,
        color="gray"
    )

    ax.plot(
        loess_fit[:,0],
        loess_fit[:,1],
        color="red",
        linewidth=2
    )

    ax.set_xlabel(parameter, fontsize=8)
    ax.yaxis.set_major_formatter(k_formatter)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0,80000)

    # only the first column gets y-axis label
    if idx%4==0:
        ax.set_ylabel("Hosp. prevented", fontsize=8)
        # ax.set_ylabel("General treated", fontsize=8)
    else:
        ax.set_ylabel("")

#remove unused axes
for ax in axes_flat[len(test_params):]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()

global_sensitivity_results[
    [
        "Minimum usable activity",
        "High-risk usage activity threshold",
        "Initial CCP activity"
    ]
].corr()


quantiles = np.quantile(
    x,
    [0.01, 0.25, 0.50, 0.75, 0.99]
)

predictor = interp1d(
    loess_fit[:,0],
    loess_fit[:,1],
    bounds_error=False,
    fill_value="extrapolate"
)

y_pred = predictor(
    quantiles
)

effect = y_pred[-1] - y_pred[0]
importance = abs(effect)

importance_results = []

for outcome in [
    "Hospitalizations prevented",
    "General treated"
]:

    for parameter in test_params:

        x = global_sensitivity_results[parameter]
        y = global_sensitivity_results[outcome]

        loess_fit = lowess(
            y,
            x,
            frac=0.3,
            return_sorted=True
        )

        predictor = interp1d(
            loess_fit[:,0],
            loess_fit[:,1],
            bounds_error=True
        )

        quantiles = np.quantile(
            x,
            [0.01, 0.25, 0.50, 0.75, 0.99]
        )

        yq = predictor(quantiles)

        effect = (yq[-1]- yq[0])

        importance_results.append({
            "Outcome": outcome,
            "Parameter": parameter,
            "Q01 prediction": yq[0],
            "Q25 prediction": yq[1],
            "Q50 prediction": yq[2],
            "Q75 prediction": yq[3],
            "Q99 prediction": yq[4],
            "Effect":effect,
            "LOESS effect size":
                abs(effect)
        })

importance_df = pd.DataFrame(
    importance_results
)

importance_df.sort_values(
    ["Outcome","LOESS effect size"],
    ascending=[True,False]
)        

print(importance_df)

plot_df = importance_df[
    [
        "Outcome",
        "Parameter",
        "Q01 prediction",
        "Q25 prediction",
        "Q50 prediction",
        "Q75 prediction",
        "Q99 prediction",
        "Effect",
        "LOESS effect size"
    ]
]

plot_df = (
    plot_df[
        plot_df["Outcome"]
        == "Hospitalizations prevented"
    ]
    .sort_values(
        "LOESS effect size",
        ascending=True
    )
)

fig, ax = plt.subplots(
    figsize=(8,6)
)

ypos = np.arange(len(plot_df))

for i, (_, row) in enumerate(
    plot_df.iterrows()
):

    ax.plot(
        [
            row["Q01 prediction"],
            row["Q99 prediction"]
        ],
        [i, i],
        color="black",
        linewidth=1
    )

    ax.scatter(
        row["Q01 prediction"],
        i,
        color="blue",
        s=30,
        label="1%" if i == 0 else None
    )

    ax.scatter(
        row["Q25 prediction"],
        i,
        color="skyblue",
        s=30,
        label="25%" if i == 0 else None
    )

    ax.scatter(
        row["Q50 prediction"],
        i,
        color="black",
        s=35,
        label="50%" if i == 0 else None
    )

    ax.scatter(
        row["Q75 prediction"],
        i,
        color="orange",
        s=30,
        label="75%" if i == 0 else None
    )

    ax.scatter(
        row["Q99 prediction"],
        i,
        color="red",
        s=30,
        label="99%" if i == 0 else None
    )

ax.set_yticks(ypos)
ax.set_yticklabels(
    plot_df["Parameter"]
)

ax.set_xlabel(
    "Predicted hospitalizations prevented"
)

ax.legend()

plt.tight_layout()
plt.show()