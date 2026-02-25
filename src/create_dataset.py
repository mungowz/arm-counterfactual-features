import pandas as pd
import numpy as np
from pathlib import Path
from folktables import ACSDataSource, ACSIncome

# Manual mapping dictionaries based on ACS 2018 documentation.
# Hardcoding these to avoid API changes breaking the pipeline.
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
    """ Downloads the ACS dataset via Folktables and returns it as a DataFrame. """
    print("  > Downloading ACS dataset via Folktables...")
    print(f"    - Fetching data for {year} 1-Year person survey for {state}...")
    
    data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=[state], download=True)
    features, labels, _ = ACSIncome.df_to_pandas(acs_data)
    
    df = features.copy()
    df['target'] = labels
    return df


def categorize_dataset(input_path, output_path):
    """ Cleans and categorizes continuous features into discrete bins. """
    print(f"  > Categorizing dataset from {Path(input_path).name}...")
    df = pd.read_csv(input_path)

    # Applying manual mappings for specific categorical features
    df['OCCP'] = df['OCCP'].astype(str).map(OCCP_MAP).fillna('Other-Occupation')
    df['POBP'] = df['POBP'].astype(str).map(POBP_MAP).fillna('Other-Place')

    # Binning continuous variables so FP-growth can process them later
    def map_schl(x):
        if x <= 15: return 'No-HS-Diploma'
        if x in [16, 17]: return 'HS-Diploma-GED'
        if x == 21: return 'Bachelor-Degree'
        if x >= 22: return 'Advanced-Degree'
        return 'Some-College-Vocational'
    
    df['SCHL'] = df['SCHL'].apply(map_schl)
    df['AGEP'] = pd.cut(df['AGEP'], bins=[0, 29, 44, 59, 150], labels=['Young-Adults', 'Adults', 'Middle-Aged', 'Seniors'])
    df['WKHP'] = pd.cut(df['WKHP'], bins=[-1, 29, 39, 49, 150], labels=['Part-Time', 'Full-Time', 'Overtime', 'Intensive'])

    if 'group' in df.columns: 
        df.drop(columns=['group'], inplace=True)
    df['target'] = df['target'].map({False: '<=50k', True: '>50k'})

    df.to_csv(output_path, index=False)
    print(f"    - Categorized dataset saved to: {Path(output_path).name}\n")


if __name__ == "__main__":
    # Detect the environment (Local vs Colab) and set paths accordingly
    if Path("/content").exists():
        data_dir = Path("/content/data")
    else:
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data"

    data_dir.mkdir(parents=True, exist_ok=True)
    
    raw_csv = data_dir / "ACSIncome_NY_2018_clean.csv"
    cat_csv = data_dir / "ACSIncome_NY_2018_categorized.csv"

    print("\n--- Standalone Data Preparation ---\n")
    if not raw_csv.exists():
        df_raw = create_ny_2018_dataset()
        df_raw.to_csv(raw_csv, index=False)
    else:
        print(f"  > Raw dataset already exists at {raw_csv.name}")

    if not cat_csv.exists():
        categorize_dataset(raw_csv, cat_csv)
    else:
        print(f"  > Categorized dataset already exists at {cat_csv.name}\n")
        
    print("Standalone execution completed successfully.")