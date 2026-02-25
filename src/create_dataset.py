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
    '36': 'New-York', '34': 'New-Jersey', '42': 'Pennsylvania', '06': 'California',
    '12': 'Florida', '25': 'Massachusetts', '303': 'Mexico', '207': 'China',
    '210': 'India', '329': 'Dominican-Republic', '332': 'Haiti', '333': 'Jamaica',
    '109': 'Philippines', '233': 'Italy', '22': 'Louisiana-Maine'
}


def create_ny_2018_dataset():
    """ Downloads the raw 2018 Person data for NY using folktables. """
    print("Downloading ACS NY 2018 data via Folktables...")
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["NY"], download=True)

    features, label, group = ACSIncome.df_to_numpy(acs_data)
    df = pd.DataFrame(features, columns=ACSIncome.features)
    df['target'] = label
    return df


def categorize_dataset(input_path, output_path):
    """ Converts numerical codes to strings and bins continuous variables. """
    print(f"Categorizing dataset from {input_path}...")
    df = pd.read_csv(input_path)

    # Cast high-cardinality features to string to use the dictionaries
    df['OCCP'] = df['OCCP'].fillna(0).astype(int).astype(str)
    df['POBP'] = df['POBP'].fillna(0).astype(int).astype(str)

    df['OCCP'] = df['OCCP'].map(lambda x: OCCP_MAP.get(x, f"OccCode_{x}"))
    df['POBP'] = df['POBP'].map(lambda x: POBP_MAP.get(x, f"PlaceCode_{x}"))

    # Map standard socioeconomic features
    df['COW'] = df['COW'].map({1.0:'Private-profit', 2.0:'Private-non-profit', 3.0:'Local-gov',
                               4.0:'State-gov', 5.0:'Federal-gov', 6.0:'Self-employed-inc',
                               7.0:'Self-employed-not-inc'}).fillna('Other')
    df['MAR'] = df['MAR'].map({1.0:'Married', 2.0:'Widowed', 3.0:'Divorced', 4.0:'Separated', 5.0:'Never-married'}).fillna('Other')
    df['SEX'] = df['SEX'].map({1.0: 'Male', 2.0: 'Female'})
    df['RAC1P'] = df['RAC1P'].map({1.0:'White', 2.0:'Black', 6.0:'Asian'}).fillna('Other-Race')


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
    print(f"Categorized dataset saved to: {output_path}")


if __name__ == "__main__":
    current_dir = Path.cwd()
    data_dir = current_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = data_dir / "ACSIncome_NY_2018_clean.csv"
    cat_csv = data_dir / "ACSIncome_NY_2018_categorized.csv"

    if not raw_csv.exists():
        create_ny_2018_dataset().to_csv(raw_csv, index=False)

    categorize_dataset(raw_csv, cat_csv)