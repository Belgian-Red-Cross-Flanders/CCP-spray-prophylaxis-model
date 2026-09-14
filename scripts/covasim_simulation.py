# a COVID-19 pandemic in Belgium without other mitigation strategies and treatments
import covasim as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Statbel population data
pop_bel_2020 = 11492641 
pop_bel_2023 = 11697557
average_pop_bel = np.mean([pop_bel_2020,pop_bel_2023])

# beta value scenarios
default_beta = 0.016 # from covasim 
low_beta     = 0.8 * default_beta
baseline     = default_beta
high_beta    = 1.2 * default_beta


# n_imports is the first n infected people with this variant, like travelers or external introductions
alpha = cv.variant(
    'alpha',
    days='2020-12-15',
    n_imports=10
)

delta = cv.variant(
    'delta',
    days='2021-04-05',
    n_imports=10
)

omicron = cv.variant(
    label='omicron_like',
    variant=dict(
        rel_beta=3.0,
        rel_severe_prob=0.5,
    ),
    days='2021-11-21',
    n_imports=20,
)
custom_immunity = np.array([
    [1.00, 0.66, 0.79, 0.22],
    [0.66, 1.00, 0.67, 0.14],
    [0.79, 0.67, 1.00, 0.30],
    [0.22, 0.14, 0.30, 1.00],
])

pars = dict(
    pop_type = 'hybrid', # synthpops is the best pop_type but slower. hybrid is still good
    location = 'belgium', # Use population characteristics for Japan
    pop_size = 100000, #average_pop_bel is too much for memory. scale afterwards, 
    pop_infected = 100, # initial number of people infected
    # Study period is between March 6, 2020 and June 27, 2023. 
    start_day = '2020-03-06', 
    end_day = '2023-06-27',
    variants=[alpha, delta, omicron],
    #beta is the transmission parameter - higher beta, more infectons and larger waves. now set to the standard
    #standard beta:0.016
    immunity = custom_immunity,
    interventions=[], # no interventions
    
)

sim = cv.Sim(pars)
sim.run()
fig = sim.plot()

scale = average_pop_bel / sim.pars["pop_size"]
df = pd.DataFrame({
    "date": sim.datevec,
    "infections": sim.results["new_infections"].values * scale,
    "hospitalizations": sim.results["new_severe"].values * scale,
    "deaths": sim.results["new_deaths"].values * scale,
    "severe_prevalence": sim.results["n_severe"].values * scale,
})

output_file = "data_processed/covasim_belgium_unmitigated.csv"
df.to_csv(output_file, index=False)

fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

ax[0].plot(df["date"], df["infections"])
ax[0].set_title("New infections/day")

ax[1].plot(df["date"], df["hospitalizations"])
ax[1].set_title("Hospital admissions/day")

ax[2].plot(df["date"], df["deaths"])
ax[2].set_title("Deaths/day")

plt.tight_layout()

fig, ax = plt.subplots(2, 1, figsize=(12,6), sharex=True)

ax[0].plot(df["date"], df["infections"])
ax[0].set_title("Potential CCP donors")

ax[1].plot(df["date"], df["hospitalizations"])
ax[1].set_title("Potential CCP demand")

plt.tight_layout()
a=1
