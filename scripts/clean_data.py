import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_hosp = pd.read_csv("data_raw/COVID19BE_HOSP.csv")

# Parse date
df_hosp["DATE"] = pd.to_datetime(df_hosp["DATE"])

# Keep only needed columns
df_hosp = df_hosp[["DATE", "NEW_IN"]]

# Aggregate to national level (sum over provinces)
df_hosp_daily = df_hosp.groupby("DATE").sum().reset_index()

# Sort (important)
df_hosp_daily = df_hosp_daily.sort_values("DATE")

# Optional: handle missing values
df_hosp_daily["NEW_IN"] = df_hosp_daily["NEW_IN"].fillna(0)

# Save outputs
df_hosp_daily.to_csv("data_processed/hospitalizations_clean.csv", index=False)
df_hosp_daily.to_excel("data_processed/hospitalizations_clean.xlsx", index=False)


print(df_hosp_daily.head())
print(df_hosp_daily.describe())
print(df_hosp_daily["NEW_IN"].max())


print("Clean data saved.")


# Plot the time series for sanity check
# Convert DATE
df_hosp_daily["DATE"] = pd.to_datetime(df_hosp_daily["DATE"])

# Plot
plt.figure(figsize=(12,5))
plt.plot(df_hosp_daily["DATE"], df_hosp_daily["NEW_IN"], label="Daily hospitalizations (NEW_IN)")

plt.xlabel("Date")
plt.ylabel("Hospital admissions")
plt.title("COVID-19 hospital admissions in Belgium")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# # Plot 7 day moving average 
# df["NEW_IN_MA7"] = df["NEW_IN"].rolling(window=7).mean()

# plt.figure(figsize=(12,5))
# plt.plot(df["DATE"], df["NEW_IN"], alpha=0.3, label="Raw")
# plt.plot(df["DATE"], df["NEW_IN_MA7"], label="7-day average")

# plt.xlabel("Date")
# plt.ylabel("Hospital admissions")
# plt.title("COVID-19 hospital admissions (smoothed)")
# plt.legend()
# plt.grid(True)
# plt.show()

# Read and clean infection data (NEW confirmed cases)
# Load data
df_cases = pd.read_csv("data_raw/COVID19BE_CASES_AGESEX.csv")

# Parse date
df_cases["DATE"] = pd.to_datetime(df_cases["DATE"])

# Keep only needed columns
df_cases = df_cases[["DATE", "CASES"]]

# Aggregate to national level (sum over provinces)
df_cases_daily = df_cases.groupby("DATE").sum().reset_index()

# Sort (important)
df_cases_daily = df_cases_daily.sort_values("DATE")

# Optional: handle missing values
df_cases_daily["CASES"] = df_cases_daily["CASES"].fillna(0)

# Save outputs
df_cases_daily.to_csv("data_processed/cases_clean.csv", index=False)
df_cases_daily.to_excel("data_processed/cases_clean.xlsx", index=False)


print(df_cases_daily.head())
print(df_cases_daily.describe())
print(df_cases_daily["CASES"].max())


print("Clean data saved.")


# Plot the time series for sanity check
# Convert DATE
df_cases_daily["DATE"] = pd.to_datetime(df_cases_daily["DATE"])

# Plot
plt.figure(figsize=(12,5))
plt.plot(df_cases_daily["DATE"], df_cases_daily["CASES"], label="Daily cases (CASES)")

plt.xlabel("Date")
plt.ylabel("Confirmed cases")
plt.title("COVID-19 cases daily in Belgium")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

