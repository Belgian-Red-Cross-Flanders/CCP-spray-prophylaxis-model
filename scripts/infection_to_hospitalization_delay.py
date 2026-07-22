import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

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

I = pd.Series(I_ori).rolling(7, center=True).mean()
H = pd.Series(H_ori).rolling(7, center=True).mean()

tmp = pd.DataFrame({
    "I": I,
    "H": H
}).dropna()

I = tmp["I"].values
H = tmp["H"].values

# Peak hospitalizations (for normalization)
H_peak = np.max(H)
#endregion

# through cross-correlation

I_centered = I - np.mean(I)
H_centered = H - np.mean(H)

corr = correlate(H_centered, I_centered, mode="full")

lags = np.arange(-len(I) + 1, len(I))

best_lag = lags[np.argmax(corr)]

print(f"Estimated infection→hospitalization delay = {best_lag} days")

fig, ax = plt.subplots(figsize=(8,4))

ax.plot(lags, corr)

ax.axvline(
    best_lag,
    color="red",
    linestyle="--",
    label=f"Best lag = {best_lag} days"
)

ax.legend()
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Correlation")

plt.show()

plt.figure(figsize=(10,4))


plt.plot(
    dates[7:],
    I_ori[:-7] / np.nanmax(I_ori),
    label="Cases shifted +7d"
)

plt.plot(
    dates,
    H_ori / np.nanmax(H_ori),
    label="Hospitalizations"
)


plt.legend()
plt.show()