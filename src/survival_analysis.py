from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_survival_data.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA)

    kmf = KaplanMeierFitter()
    plt.figure()

    for label, group in df.groupby("biomarker_positive"):
        name = "Biomarker positive" if label == 1 else "Biomarker negative"
        kmf.fit(group["time_months"], group["event"], label=name)
        kmf.plot_survival_function(ci_show=True)

    plt.xlabel("Time (months)")
    plt.ylabel("Survival probability")
    plt.title("Kaplan-Meier survival by biomarker group")
    plt.tight_layout()
    plt.savefig(FIGURES / "km_survival_by_biomarker.png", dpi=300)
    plt.close()

    cox_df = df.copy()
    cox_df["stage_IV"] = (cox_df["stage"] == "IV").astype(int)
    cox_df = cox_df[["time_months", "event", "biomarker_positive", "age", "stage_IV"]]

    model = CoxPHFitter()
    model.fit(cox_df, duration_col="time_months", event_col="event")
    model.summary.to_csv(RESULTS / "cox_model_summary.csv")

    print(model.summary[["coef", "exp(coef)", "p"]].to_string())

if __name__ == "__main__":
    main()
