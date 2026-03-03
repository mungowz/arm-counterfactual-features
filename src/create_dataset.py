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
    '72': 'PlaceCode_72', '313': 'PlaceCode_313', '303': 'PlaceCode_303', '1': 'PlaceCode_1'
}


def create_ny_2018_dataset(state="NY", year="2018"):
    """Download ACS data via folktables and return it as a DataFrame with the target column."""
    print(f"  > Downloading ACS {year} 1-Year survey for {state}...")

    data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=[state], download=True)
    features, labels, _ = ACSIncome.df_to_pandas(acs_data)

    df = features.copy()
    df['target'] = labels
    return df


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

    # drop the group column folktables sometimes adds, encode target as strings
    if 'group' in df.columns:
        df.drop(columns=['group'], inplace=True)
    df['target'] = df['target'].map({False: '<=50k', True: '>50k'})

    df.to_csv(output_path, index=False)
    print(f"    - saved to {Path(output_path).name}")


if __name__ == "__main__":
    if Path("/content").exists():
        data_dir = Path("/content/data")
    else:
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data"

    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("DATA PREPARATION")
    print("="*70 + "\n")

    raw_csv = data_dir / "ACSIncome_NY_2018_clean.csv"
    cat_csv = data_dir / "ACSIncome_NY_2018_categorized.csv"

    if not raw_csv.exists():
        df_raw = create_ny_2018_dataset()
        df_raw.to_csv(raw_csv, index=False)
        print(f"  > Raw dataset saved to {raw_csv.name}\n")
    else:
        print(f"  > Raw dataset already exists ({raw_csv.name}), skipping download.\n")

    if not cat_csv.exists():
        categorize_dataset(raw_csv, cat_csv)
        print()
    else:
        print(f"  > Categorized dataset already exists ({cat_csv.name}), skipping.\n")

    print("="*70)
    print("Done.")
    print("="*70 + "\n")