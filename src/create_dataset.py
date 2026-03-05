import pandas as pd
import numpy as np
from pathlib import Path
from folktables import ACSDataSource, ACSIncome

# ACS PUMS code mappings — taken manually from the 2018 data dictionary
# hardcoding these because the folktables API doesn't expose them directly
OCCP_MAP = {
    '10': 'Chief-Executives', '20': 'General-Operations-Managers', '120': 'Financial-Managers',
    '440': 'Other-Operations-Managers', '800': 'Accountants', '710': 'Management-Analysts',
    '1021': 'Software-Developers', '1010': 'Computer-Systems-Analysts', '2100': 'Lawyers',
    '2310': 'Elementary-Teachers', '3255': 'Registered-Nurses', '1020': 'Software-Apps-Developers',
    '4110': 'Waiters-Waitresses', '4720': 'Cashiers', '4760': 'Retail-Salespersons',
    '5110': 'Receptionists', '5700': 'Secretaries-Admin', '4220': 'Janitors',
    '2320': 'Secondary-Teachers', '2545': 'Teaching-Assistants', '2723': 'Designers-Artists',
    '4622': 'Hotel-Clerks', '5120': 'Reservation-Agents', '9620': 'Laborers',
    '0': 'N/A-Unemployed'
}

POBP_MAP = {
    '36': 'New-York', '34': 'New-Jersey', '42': 'Pennsylvania', '9': 'Connecticut',
    '25': 'Massachusetts', '12': 'Florida', '6': 'California', '48': 'Texas',
    '13': 'Georgia', '37': 'North-Carolina', '45': 'South-Carolina',
    '72': 'PlaceCode_72', '313': 'PlaceCode_313', '303': 'PlaceCode_303', '1': 'PlaceCode_1'
}

# the two regions we compare — chosen to maximize socioeconomic contrast
REGIONS = {
    'northeast': ['NY', 'NJ', 'CT', 'MA', 'PA'],
    'south':     ['TX', 'FL', 'GA', 'NC', 'SC'],
}


def create_region_dataset(states, year="2018"):
    """
    Download ACS data for a list of states and concatenate them into one DataFrame.
    Each row gets a 'state' column so we can trace where it came from if needed.
    """
    dfs = []
    data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')

    for state in states:
        print(f"    - downloading {state} {year}...")
        acs_data = data_source.get_data(states=[state], download=True)
        features, labels, _ = ACSIncome.df_to_pandas(acs_data)
        df = features.copy()
        df['target'] = labels
        df['state'] = state
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"    - combined: {len(combined):,} samples from {states}")
    return combined


def categorize_dataset(input_path, output_path):
    """
    Convert continuous/coded columns into readable categorical bins.
    Needed because FP-Growth works on discrete items, not raw numbers.
    """
    print(f"  > Categorizing {Path(input_path).name}...")
    df = pd.read_csv(input_path)

    # map occupation and place-of-birth codes to readable strings
    df['OCCP'] = df['OCCP'].astype(str).map(OCCP_MAP).fillna('Other-Occupation')
    df['POBP'] = df['POBP'].astype(str).map(POBP_MAP).fillna('Other-Place')

    # SCHL: education level codes from the ACS codebook
    def map_schl(x):
        if x <= 15:
            return 'No-HS-Diploma'
        if x in [16, 17]:
            return 'HS-Diploma-GED'
        if x == 21:
            return 'Bachelor-Degree'
        if x >= 22:
            return 'Advanced-Degree'
        return 'Some-College-Vocational'

    df['SCHL'] = df['SCHL'].apply(map_schl)

    # AGEP: ACSIncome filters to 16+ so the lower bound starts at 15
    df['AGEP'] = pd.cut(df['AGEP'], bins=[15, 29, 44, 59, 150],
                        labels=['Young-Adults', 'Adults', 'Middle-Aged', 'Seniors'])

    # WKHP: BLS defines part-time as <35h, FLSA standard full-time is 40h/week
    df['WKHP'] = pd.cut(df['WKHP'], bins=[-1, 34, 40, 49, 150],
                        labels=['Part-Time', 'Full-Time', 'Overtime', 'Intensive'])

    # drop the group column folktables sometimes adds, and the state column
    # added by create_region_dataset (not a feature, just a provenance tag)
    for col in ['group', 'state']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    df['target'] = df['target'].map({False: '<=50k', True: '>50k'})

    df.to_csv(output_path, index=False)
    print(f"    - saved to {Path(output_path).name}")


def balance_datasets(df1, df2, random_state=42):
    """
    Stratified downsample the larger dataset to match the size of the smaller one.
    Stratification is on 'target' so the class ratio is preserved after sampling.

    Returns (df1_balanced, df2_balanced) — one of them is unchanged, the other
    is the downsampled version.
    """
    n1, n2 = len(df1), len(df2)
    target_n = min(n1, n2)

    print(f"  > Balancing datasets: {n1:,} vs {n2:,} → target {target_n:,} each")

    def stratified_sample(df, n, seed):
        # sample proportionally within each target class
        return (
            df.groupby('target', group_keys=False)
            .apply(lambda g: g.sample(frac=n / len(df), random_state=seed))
            .reset_index(drop=True)
        )

    if n1 > n2:
        df1 = stratified_sample(df1, target_n, random_state)
    elif n2 > n1:
        df2 = stratified_sample(df2, target_n, random_state)

    print(f"    - northeast: {len(df1):,} samples  "
          f"(>50k: {(df1['target'] == '>50k').mean():.1%})")
    print(f"    - south:     {len(df2):,} samples  "
          f"(>50k: {(df2['target'] == '>50k').mean():.1%})")

    return df1, df2


if __name__ == "__main__":
    if Path("/content").exists():
        data_dir = Path("/content/data")
    else:
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data"

    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("DATA PREPARATION — NORTHEAST vs SOUTH 2018")
    print("="*70 + "\n")

    for region, states in REGIONS.items():
        raw_csv = data_dir / f"ACSIncome_{region}_2018_clean.csv"
        cat_csv = data_dir / f"ACSIncome_{region}_2018_categorized.csv"

        print(f"  > Region: {region.upper()} {states}")

        if not raw_csv.exists():
            df_raw = create_region_dataset(states)
            df_raw.to_csv(raw_csv, index=False)
            print(f"    - raw saved to {raw_csv.name}\n")
        else:
            print(f"    - raw already exists ({raw_csv.name}), skipping.\n")

        if not cat_csv.exists():
            categorize_dataset(raw_csv, cat_csv)
            print()
        else:
            print(f"    - categorized already exists ({cat_csv.name}), skipping.\n")

    # balance the two categorized datasets
    ne_csv = data_dir / "ACSIncome_northeast_2018_categorized.csv"
    so_csv = data_dir / "ACSIncome_south_2018_categorized.csv"
    ne_bal = data_dir / "ACSIncome_northeast_2018_balanced.csv"
    so_bal = data_dir / "ACSIncome_south_2018_balanced.csv"

    if not ne_bal.exists() or not so_bal.exists():
        print("  > Balancing datasets...")
        df_ne = pd.read_csv(ne_csv)
        df_so = pd.read_csv(so_csv)
        df_ne, df_so = balance_datasets(df_ne, df_so)
        df_ne.to_csv(ne_bal, index=False)
        df_so.to_csv(so_bal, index=False)
        print(f"    - balanced datasets saved.\n")
    else:
        print(f"  > Balanced datasets already exist, skipping.\n")

    print("="*70)
    print("Done.")
    print("="*70 + "\n")
