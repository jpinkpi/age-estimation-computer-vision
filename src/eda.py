from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_data, validate_data

sns.set_theme(
    style="whitegrid",
    context="notebook",
    font="Times New Roman",
    rc={
        "grid.color": ".6",
        "grid.linestyle": "--",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 15,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16
    }
)
ROOT = Path(__file__).resolve().parents[1]
FIGURES_PATH = ROOT / "reports" / "figures"
def plot_age_distribution(df):
    '''
    Valida y visualiza la distribución de edades en el DataFrame.
    '''
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df['real_age'], bins=30, kde=True, color='skyblue', ax=ax)
    ax.axvline(df['real_age'].mean(), color='red', linestyle='--', label=f'Media: {df["real_age"].mean():.2f}')
    ax.axvline(df['real_age'].median(), color='green', linestyle='--', label=f'Mediana: {df["real_age"].median():.2f}')
    ax.set_title('Distribución de Edades', fontsize=16)
    ax.set_xlabel('Edad (años)', fontsize=14)
    ax.set_ylabel("Numero de fotografías", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "age_distribution.png", dpi=300)
    plt.show()

def analyze_age_groups(df):
    '''
    Analiza la distribución de edades en diferentes grupos de edad.
    '''
    age_bins = [0, 18, 30, 45, 60, 75, 90, 120]
    age_labels = ['0-17', '18-29', '30-44', '45-59', '60-74', '75-89', '90+']
    age_groups = pd.cut(df['real_age'], bins=age_bins, labels=age_labels, right=False)

    distribution = age_groups.value_counts().sort_index()

    percentages = (age_groups.value_counts(normalize=True).sort_index() * 100)
    summary = pd.DataFrame({'Frequency': distribution, 'Percentage': percentages})
    print("\nDistribución por grupos etarios:")
    print(summary)
    return summary

def plot_age_groups_distribution(summary):
    '''
    Visualiza la distribución de edades por grupos etarios.
    '''
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=summary.index, y=summary['Frequency'], palette='rocket', ax=ax)
    ax.set_title('Distribución de Edades por Grupos Etarios', fontsize=16)
    ax.set_xlabel('Grupos Etarios', fontsize=14)
    ax.set_ylabel("Número de fotografías", fontsize=14)
    for i, v in enumerate(summary['Frequency']):
        ax.text(i, v + 5, str(v), color='black', ha='center', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "age_groups_distribution.png", dpi=300)
    plt.show()

def analyze_minors(df):
    '''
    Analiza la proporción de menores de edad en el DataFrame.
    '''
    minors_count = (df['real_age'] < 18).sum()
    total_count = len(df)
    minors_percentage = (minors_count / total_count) * 100
    adult_count = (df['real_age'] >= 18).sum()
    adult_percentage = (adult_count / total_count) * 100
    print(f"\nNúmero de menores de edad: {minors_count}")
    print(f"Porcentaje de menores de edad: {minors_percentage:.2f}%")
    print(f"\nNúmero de adultos: {adult_count}")
    print(f"Porcentaje de adultos: {adult_percentage:.2f}%")

    return {'minors': minors_count, 'adults': adult_count, 'minors_percentage': minors_percentage, 'adults_percentage': adult_percentage}

def analyze_critical_ages(df):
    """
    Analiza la distribución de edades
    próximas a la mayoría de edad.
    """

    critical_ages = df[
        df["real_age"].between(15, 21)
    ]

    distribution = (
        critical_ages["real_age"]
        .value_counts()
        .sort_index()
    )

    print("\nDistribución de edades entre 15 y 21 años:")
    print(distribution)

    print(
        f"\nObservaciones en este intervalo: "
        f"{len(critical_ages)}"
    )

    return distribution

def main():
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    df = load_data()
    validate_data(df)
    plot_age_distribution(df)
    summary = analyze_age_groups(df)
    plot_age_groups_distribution(summary)
    analyze_minors(df)
    analyze_critical_ages(df)

if __name__ == "__main__":
    main()