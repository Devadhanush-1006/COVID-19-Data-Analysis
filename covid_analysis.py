import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ==============================
# 1. Load Data from GitHub
# ==============================
def load_data():
    base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    confirmed = pd.read_csv(base_url + "time_series_covid19_confirmed_global.csv")
    deaths    = pd.read_csv(base_url + "time_series_covid19_deaths_global.csv")
    recovered = pd.read_csv(base_url + "time_series_covid19_recovered_global.csv")
    return confirmed, deaths, recovered

# ==============================
# 2. Prepare Data
# ==============================
def prepare_data(confirmed, deaths, recovered, output_folder):
    confirmed_long = confirmed.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"],
        var_name="Date", value_name="Confirmed"
    )
    deaths_long = deaths.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"],
        var_name="Date", value_name="Deaths"
    )
    recovered_long = recovered.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"],
        var_name="Date", value_name="Recovered"
    )

    covid = confirmed_long.merge(
        deaths_long, on=["Province/State", "Country/Region", "Lat", "Long", "Date"]
    ).merge(
        recovered_long, on=["Province/State", "Country/Region", "Lat", "Long", "Date"]
    )

    covid["Date"] = pd.to_datetime(covid["Date"], format="%m/%d/%y", errors='coerce')

    covid_country = covid.groupby(["Country/Region", "Date"]).sum().reset_index()
    covid_country["Active"] = covid_country["Confirmed"] - covid_country["Deaths"] - covid_country["Recovered"]

    for col in ["Confirmed", "Deaths", "Recovered"]:
        covid_country[f"New{col}"] = covid_country.groupby("Country/Region")[col].diff().fillna(0)

    # Fix negative values
    for col in ["NewConfirmed", "NewDeaths", "NewRecovered"]:
        covid_country[col] = covid_country[col].clip(lower=0)

    # Save cleaned dataset
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "covid_cleaned_new.csv")
    covid_country.to_csv(output_path, index=False)
    print(f"✅ Cleaned dataset saved at: {output_path}")

    return covid_country

# ==============================
# 3. Plot COVID Trends
# ==============================
def plot_country(covid_country, country="India", output_folder=".", stacked=True):
    df = covid_country[covid_country["Country/Region"] == country].copy()
    df["Confirmed_7d"] = df["NewConfirmed"].rolling(7).mean()

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 1, figsize=(14,10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # --- Cumulative cases ---
    axes[0].plot(df["Date"], df["Confirmed"], label="Confirmed", color="#1f77b4", linewidth=2.5)
    axes[0].plot(df["Date"], df["Deaths"], label="Deaths", color="#d62728", linewidth=2.5)
    axes[0].plot(df["Date"], df["Recovered"], label="Recovered", color="#2ca02c", linewidth=2.5)
    axes[0].plot(df["Date"], df["Active"], label="Active", color="#ff7f0e", linewidth=2.5)
    axes[0].set_title(f"COVID-19 Cumulative Trend in {country}", fontsize=18, fontweight="bold", pad=15, color="#333333")
    axes[0].set_ylabel("Cumulative Cases", fontsize=13)
    axes[0].legend(fontsize=11, frameon=True, shadow=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # --- Daily new cases ---
    if stacked:
        axes[1].bar(df["Date"], df["NewConfirmed"], label="New Confirmed", color="#1f77b4", alpha=0.7)
        axes[1].bar(df["Date"], df["NewRecovered"], bottom=df["NewConfirmed"], label="New Recovered", color="#2ca02c", alpha=0.7)
        axes[1].bar(df["Date"], df["NewDeaths"], bottom=df["NewConfirmed"] + df["NewRecovered"], label="New Deaths", color="#d62728", alpha=0.7)
    else:
        axes[1].bar(df["Date"], df["NewConfirmed"], label="New Confirmed", color="#1f77b4", alpha=0.6)
        axes[1].bar(df["Date"], df["NewDeaths"], label="New Deaths", color="#d62728", alpha=0.6)
        axes[1].bar(df["Date"], df["NewRecovered"], label="New Recovered", color="#2ca02c", alpha=0.6)

    axes[1].plot(df["Date"], df["Confirmed_7d"], color="black", linewidth=2, label="7-day avg (Confirmed)")
    axes[1].set_title(f"Daily New COVID-19 Cases in {country}", fontsize=16, pad=12, color="#333333")
    axes[1].set_xlabel("Date", fontsize=13)
    axes[1].set_ylabel("Daily Cases", fontsize=13)
    axes[1].legend(fontsize=10, frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45, fontsize=11)

    plt.tight_layout()
    fig_path = os.path.join(output_folder, f"{country.lower()}_trend.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"📊 Plot saved at: {fig_path}")

    plt.show()

# ==============================
# 4. Run the Pipeline
# ==============================
if __name__ == "__main__":
    output_folder = r"C:\Users\CHANDU\OneDrive\Desktop\COVID_output"

    confirmed, deaths, recovered = load_data()
    covid_country = prepare_data(confirmed, deaths, recovered, output_folder)
    plot_country(covid_country, country="India", output_folder=output_folder, stacked=True)
