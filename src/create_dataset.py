"""
build_acs_datasets.py

Downloads ACS Income PUMS data for two US regions and saves them as
classification-ready CSVs with human-readable categorical labels.

    Northeast states — income threshold $110,000
    South states     — income threshold $90,000

Survey year: 2024 | Horizon: 1-Year | Survey: person (PUMS)

Usage:
    pip install folktables pandas numpy
    python build_acs_datasets.py
"""

import os
import time
import numpy as np
import pandas as pd
import folktables
from folktables import ACSDataSource
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- parallelism -----------------------------------------------------------------

_CORES           = os.cpu_count() or 4
DOWNLOAD_WORKERS = min(_CORES * 2, 16)   # capped to avoid Census server rate-limits
REGION_WORKERS   = 2


# --- state groups ----------------------------------------------------------------

NORTHEAST_STATES = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA']
SOUTH_STATES     = ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'DC', 'WV',
                    'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX']

SURVEY_YEAR = '2024'
HORIZON     = '1-Year'


# --- prediction tasks ------------------------------------------------------------

_INCOME_FEATURES = [
    'AGEP',   # age
    'COW',    # class of worker
    'SCHL',   # educational attainment
    'MAR',    # marital status
    'OCCP',   # occupation
    'POBP',   # place of birth
    'RELSHIPP',  # relationship to reference person (replaces RELP from 2019 onward)
    'WKHP',   # usual hours worked per week
    'SEX',    # sex
    'RAC1P',  # race
]

ACSIncomeNortheast = folktables.BasicProblem(
    features=_INCOME_FEATURES,
    target='PINCP',
    target_transform=lambda x: x > 110_000,
    group='RAC1P',
    preprocess=folktables.adult_filter,
    postprocess=lambda x: np.nan_to_num(x, nan=-1),
)

ACSIncomeSouth = folktables.BasicProblem(
    features=_INCOME_FEATURES,
    target='PINCP',
    target_transform=lambda x: x > 90_000,
    group='RAC1P',
    preprocess=folktables.adult_filter,
    postprocess=lambda x: np.nan_to_num(x, nan=-1),
)


# --- categorical mappings --------------------------------------------------------
# Source: 2024 ACS PUMS Data Dictionary (census.gov, October 16, 2025)
# Codes arrive from folktables as float64; the lookup layer converts them to
# integers before indexing, so keys here are plain int-strings without leading zeros.

COW_MAP = {
    '1': 'Employee-Private-For-Profit',
    '2': 'Employee-Private-Non-Profit',
    '3': 'Local-Government-Employee',
    '4': 'State-Government-Employee',
    '5': 'Federal-Government-Employee',
    '6': 'Self-Employed-Not-Incorporated',
    '7': 'Self-Employed-Incorporated',
    '8': 'Unpaid-Family-Worker',
    '9': 'Unemployed-5plus-Years-Or-Never-Worked',
}

SCHL_MAP = {
    '1':  'No-Schooling-Completed',
    '2':  'Nursery-School-Preschool',
    '3':  'Kindergarten',
    '4':  'Grade-1',
    '5':  'Grade-2',
    '6':  'Grade-3',
    '7':  'Grade-4',
    '8':  'Grade-5',
    '9':  'Grade-6',
    '10': 'Grade-7',
    '11': 'Grade-8',
    '12': 'Grade-9',
    '13': 'Grade-10',
    '14': 'Grade-11',
    '15': 'Grade-12-No-Diploma',
    '16': 'Regular-HS-Diploma',
    '17': 'GED-Or-Alt-Credential',
    '18': 'Some-College-Less-Than-1yr',
    '19': 'Some-College-1yr-Or-More-No-Degree',
    '20': 'Associates-Degree',
    '21': 'Bachelors-Degree',
    '22': 'Masters-Degree',
    '23': 'Professional-Degree-Beyond-Bachelors',
    '24': 'Doctorate-Degree',
}

MAR_MAP = {
    '1': 'Married',
    '2': 'Widowed',
    '3': 'Divorced',
    '4': 'Separated',
    '5': 'Never-Married-Or-Under-15',
}

# RELSHIPP replaced RELP starting with 2019 ACS PUMS; codes are 20-38 (two digits).
# The new scheme adds separate codes for same-sex vs opposite-sex couples.
RELSHIPP_MAP = {
    '20': 'Reference-Person',
    '21': 'Opposite-Sex-Husband-Wife-Spouse',
    '22': 'Opposite-Sex-Unmarried-Partner',
    '23': 'Same-Sex-Husband-Wife-Spouse',
    '24': 'Same-Sex-Unmarried-Partner',
    '25': 'Biological-Son-Or-Daughter',
    '26': 'Adopted-Son-Or-Daughter',
    '27': 'Stepson-Or-Stepdaughter',
    '28': 'Brother-Or-Sister',
    '29': 'Father-Or-Mother',
    '30': 'Grandchild',
    '31': 'Parent-In-Law',
    '32': 'Son-In-Law-Or-Daughter-In-Law',
    '33': 'Other-Relative',
    '34': 'Roommate-Or-Housemate',
    '35': 'Foster-Child',
    '36': 'Other-Nonrelative',
    '37': 'Institutionalized-Group-Quarters',
    '38': 'Noninstitutionalized-Group-Quarters',
}

SEX_MAP = {
    '1': 'Male',
    '2': 'Female',
}

RAC1P_MAP = {
    '1': 'White-Alone',
    '2': 'Black-Or-African-American-Alone',
    '3': 'American-Indian-Alone',
    '4': 'Alaska-Native-Alone',
    '5': 'American-Indian-And-Alaska-Native-Tribes',
    '6': 'Asian-Alone',
    '7': 'Native-Hawaiian-And-Other-Pacific-Islander-Alone',
    '8': 'Some-Other-Race-Alone',
    '9': 'Two-Or-More-Races',
}

POBP_MAP = {
    # US states and DC — FIPS numeric codes
    '1':  'Alabama',        '2':  'Alaska',         '4':  'Arizona',
    '5':  'Arkansas',       '6':  'California',     '8':  'Colorado',
    '9':  'Connecticut',    '10': 'Delaware',        '11': 'DC',
    '12': 'Florida',        '13': 'Georgia',         '15': 'Hawaii',
    '16': 'Idaho',          '17': 'Illinois',        '18': 'Indiana',
    '19': 'Iowa',           '20': 'Kansas',          '21': 'Kentucky',
    '22': 'Louisiana',      '23': 'Maine',           '24': 'Maryland',
    '25': 'Massachusetts',  '26': 'Michigan',        '27': 'Minnesota',
    '28': 'Mississippi',    '29': 'Missouri',        '30': 'Montana',
    '31': 'Nebraska',       '32': 'Nevada',          '33': 'New-Hampshire',
    '34': 'New-Jersey',     '35': 'New-Mexico',      '36': 'New-York',
    '37': 'North-Carolina', '38': 'North-Dakota',    '39': 'Ohio',
    '40': 'Oklahoma',       '41': 'Oregon',          '42': 'Pennsylvania',
    '44': 'Rhode-Island',   '45': 'South-Carolina',  '46': 'South-Dakota',
    '47': 'Tennessee',      '48': 'Texas',           '49': 'Utah',
    '50': 'Vermont',        '51': 'Virginia',        '53': 'Washington',
    '54': 'West-Virginia',  '55': 'Wisconsin',       '56': 'Wyoming',
    '72': 'Puerto-Rico',
    # Foreign born
    '100': 'Born-Abroad-US-Parents',
    '301': 'Cuba',            '302': 'Jamaica',           '303': 'Dominican-Republic',
    '308': 'Haiti',           '313': 'Other-Caribbean',
    '400': 'Mexico',          '414': 'Guatemala',         '416': 'Honduras',
    '417': 'El-Salvador',     '422': 'Nicaragua',         '423': 'Panama',
    '424': 'Other-Central-America',
    '501': 'Colombia',        '507': 'Peru',              '508': 'Brazil',
    '516': 'Venezuela',       '523': 'Other-South-America',
    '600': 'Armenia',         '601': 'China',             '603': 'India',
    '607': 'Japan',           '613': 'Philippines',       '615': 'South-Korea',
    '618': 'Vietnam',         '619': 'Other-Southeast-Asia', '620': 'Other-Asia',
    '700': 'United-Kingdom',  '703': 'Germany',           '706': 'Greece',
    '708': 'Ireland',         '710': 'Italy',             '714': 'Poland',
    '716': 'Portugal',        '720': 'Russia',            '724': 'Ukraine',
    '730': 'Other-Europe',
    '800': 'Nigeria',         '803': 'Ethiopia',          '804': 'Egypt',
    '820': 'Other-Africa',
    '900': 'Canada',          '999': 'Other-NEC',
}

# Only the most frequent SOC codes are named; the rest fall back to 'Other-Occupation'
OCCP_MAP = {
    '0':    'Not-In-Labor-Force-Or-Under-16',
    '10':   'Chief-Executives',
    '20':   'General-Operations-Managers',
    '120':  'Financial-Managers',
    '136':  'HR-Managers',
    '220':  'Advertising-And-Marketing-Managers',
    '300':  'Purchasing-Managers',
    '310':  'Transportation-Managers',
    '330':  'Food-Service-Managers',
    '410':  'Medical-And-Health-Services-Managers',
    '430':  'Construction-Managers',
    '440':  'Other-Managers',
    '500':  'Agents-Of-Performing-Arts',
    '510':  'Compliance-Officers',
    '520':  'Cost-Estimators',
    '530':  'Human-Resources-Workers',
    '710':  'Management-Analysts',
    '720':  'Meeting-And-Event-Planners',
    '800':  'Accountants-And-Auditors',
    '840':  'Financial-Analysts',
    '850':  'Personal-Financial-Advisors',
    '900':  'Financial-Examiners',
    '950':  'Other-Financial-Specialists',
    '1010': 'Computer-Systems-Analysts',
    '1020': 'Software-Developers',
    '1021': 'Software-Quality-Assurance',
    '1022': 'Web-Developers',
    '1032': 'Software-Dev-And-Programmers-NEC',
    '1050': 'Computer-Support-Specialists',
    '1060': 'Database-Administrators',
    '1100': 'Network-Architects',
    '1110': 'Network-And-Systems-Administrators',
    '1200': 'Actuaries',
    '1220': 'Operations-Research-Analysts',
    '1230': 'Statisticians',
    '1240': 'Data-Scientists',
    '2100': 'Lawyers',
    '2105': 'Judicial-Law-Clerks',
    '2110': 'Judges-And-Magistrates',
    '2310': 'Elementary-School-Teachers',
    '2320': 'Middle-School-Teachers',
    '2330': 'Secondary-School-Teachers',
    '2540': 'Special-Education-Teachers',
    '2550': 'Other-Teachers',
    '2560': 'Tutors-And-Instructors',
    '2630': 'Postsecondary-Teachers',
    '2640': 'Preschool-And-Kindergarten-Teachers',
    '2720': 'Art-Directors',
    '2740': 'Graphic-Designers',
    '2750': 'Interior-Designers',
    '3010': 'Chiropractors',
    '3050': 'Dietitians-And-Nutritionists',
    '3090': 'Emergency-Medical-Technicians',
    '3100': 'Exercise-Physiologists',
    '3130': 'Pharmacists',
    '3160': 'Physical-Therapists',
    '3230': 'Physicians-And-Surgeons',
    '3250': 'Registered-Nurses',
    '3255': 'Nurse-Practitioners',
    '3260': 'Occupational-Therapists',
    '3300': 'Dentists',
    '3420': 'Dental-Assistants',
    '3500': 'Licensed-Practical-Nurses',
    '3600': 'Medical-Assistants',
    '4000': 'Cooks-Restaurant',
    '4020': 'Food-Preparation-Workers',
    '4040': 'Bartenders',
    '4055': 'Fast-Food-Workers',
    '4110': 'Waiters-And-Waitresses',
    '4120': 'Dining-Room-Attendants',
    '4140': 'Dishwashers',
    '4220': 'Janitors-And-Cleaners',
    '4230': 'Maids-And-Housekeeping',
    '4700': 'First-Line-Retail-Supervisors',
    '4720': 'Cashiers',
    '4740': 'Counter-And-Rental-Clerks',
    '4760': 'Retail-Salespersons',
    '4800': 'Insurance-Sales-Agents',
    '4810': 'Securities-And-Financial-Sales',
    '4820': 'Real-Estate-Brokers-And-Agents',
    '4840': 'Telemarketers',
    '4850': 'Sales-Representatives',
    '5000': 'First-Line-Office-Supervisors',
    '5110': 'Receptionists',
    '5120': 'Information-Clerks',
    '5160': 'Customer-Service-Representatives',
    '5230': 'Payroll-And-Timekeeping-Clerks',
    '5240': 'Human-Resources-Assistants',
    '5260': 'Eligibility-Interviewers',
    '5420': 'Postal-Service-Workers',
    '5600': 'Production-Planning-Clerks',
    '5700': 'Secretaries-And-Admin-Assistants',
    '5820': 'Data-Entry-Keyers',
    '9600': 'Cleaners-Of-Vehicles-And-Equipment',
    '9620': 'Laborers-And-Material-Movers',
    '9800': 'Military-Officer-Special-Operations',
    '9810': 'Military-First-Line-Supervisors',
    '9820': 'Military-Enlisted-Tactical-Operations',
    '9830': 'Military-Rank-Not-Specified',
}

# Codes not found in the maps above get this label, so each column stays
# semantically uniform (no raw integer strings mixed with text labels).
COLUMN_FALLBACKS = {
    'OCCP': 'Other-Occupation',
    'POBP': 'Other-NEC',
}

COLUMN_MAPS = {
    'COW':   COW_MAP,
    'SCHL':  SCHL_MAP,
    'MAR':   MAR_MAP,
    'RELSHIPP': RELSHIPP_MAP,
    'SEX':   SEX_MAP,
    'RAC1P': RAC1P_MAP,
    'POBP':  POBP_MAP,
    'OCCP':  OCCP_MAP,
}


# --- lookup arrays ---------------------------------------------------------------
# Each mapping is compiled into a numpy object array indexed by integer code.
# Index = code + 1, so the NaN sentinel (-1) maps to index 0 and gets the
# fallback label.  This avoids per-row dict lookups and string conversions.

def _build_lookup(mapping: dict, fallback: str) -> np.ndarray:
    max_code = max(int(k) for k in mapping) if mapping else 0
    arr = np.full(max_code + 2, fill_value=fallback, dtype=object)
    for code_str, label in mapping.items():
        arr[int(code_str) + 1] = label
    return arr


_LOOKUPS: dict[str, np.ndarray] = {
    col: _build_lookup(mapping, COLUMN_FALLBACKS.get(col, 'Unknown'))
    for col, mapping in COLUMN_MAPS.items()
}


# --- helpers ---------------------------------------------------------------------

def _decode_column(series: pd.Series, lookup: np.ndarray) -> np.ndarray:
    """Map a float64 code series to string labels via integer array indexing."""
    codes = pd.to_numeric(series, errors='coerce').fillna(-1).to_numpy(dtype=np.int32)
    idx   = np.clip(codes + 1, 0, len(lookup) - 1)
    return lookup[idx]


def apply_categorical_mappings(df: pd.DataFrame) -> None:
    """Decode all categorical columns in-place using the pre-built lookup arrays."""
    for col, lookup in _LOOKUPS.items():
        if col not in df.columns:
            continue
        df[col] = pd.Categorical(_decode_column(df[col], lookup))


def _download_state(data_source: ACSDataSource, state: str) -> pd.DataFrame:
    return data_source.get_data(states=[state], download=True)


def parallel_get_data(
    data_source: ACSDataSource,
    states: list[str],
    label: str,
) -> pd.DataFrame:
    """Download one region's states concurrently and concatenate the results."""
    n_workers = min(len(states), DOWNLOAD_WORKERS)
    print(f"  [{label}] fetching {len(states)} states ({n_workers} workers)")

    results: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_download_state, data_source, s): s for s in states}
        for fut in as_completed(futures):
            state = futures[fut]
            try:
                results[state] = fut.result()
                print(f"    {state} done")
            except Exception as exc:
                raise RuntimeError(
                    f"Download failed for {state}: {exc}\n"
                    f"If {SURVEY_YEAR} data is not yet available, "
                    f"try setting SURVEY_YEAR = '{int(SURVEY_YEAR) - 1}'."
                ) from exc

    # maintain original state order
    return pd.concat([results[s] for s in states], ignore_index=True)


def build_dataset(
    task: folktables.BasicProblem,
    states: list[str],
    data_source: ACSDataSource,
    label: str,
) -> pd.DataFrame:
    t0 = time.perf_counter()
    print(f"\n{label}")
    print("-" * len(label))

    raw = parallel_get_data(data_source, states, label)
    print(f"  raw rows: {len(raw):,}  ({time.perf_counter() - t0:.1f}s)")

    features_df, labels, _ = task.df_to_pandas(raw)
    features_df['INCOME_ABOVE_THRESHOLD'] = labels.to_numpy(dtype=np.int8)

    apply_categorical_mappings(features_df)

    for col in ('AGEP', 'WKHP'):
        if col in features_df.columns:
            features_df[col] = (
                pd.to_numeric(features_df[col], errors='coerce')
                .fillna(-1)
                .astype(np.int32)
            )

    pos = features_df['INCOME_ABOVE_THRESHOLD'].mean() * 100
    print(f"  kept rows: {len(features_df):,}  |  positive class: {pos:.1f}%"
          f"  ({time.perf_counter() - t0:.1f}s total)")

    return features_df


# --- main ------------------------------------------------------------------------

def main() -> None:
    t0 = time.perf_counter()

    data_source = ACSDataSource(survey_year=SURVEY_YEAR, horizon=HORIZON, survey='person')

    out_ne = f'acs_income_northeast_{SURVEY_YEAR}.csv'
    out_s  = f'acs_income_south_{SURVEY_YEAR}.csv'

    print(f"ACS Income {SURVEY_YEAR} — building datasets\n")

    with ThreadPoolExecutor(max_workers=REGION_WORKERS) as pool:
        fut_ne = pool.submit(
            build_dataset, ACSIncomeNortheast, NORTHEAST_STATES,
            data_source, 'Northeast ($110,000 threshold)'
        )
        fut_s = pool.submit(
            build_dataset, ACSIncomeSouth, SOUTH_STATES,
            data_source, 'South ($90,000 threshold)'
        )
        df_ne = fut_ne.result()
        df_s  = fut_s.result()

    with ThreadPoolExecutor(max_workers=2) as pool:
        w1 = pool.submit(df_ne.to_csv, out_ne, index=False)
        w2 = pool.submit(df_s.to_csv,  out_s,  index=False)
        w1.result()
        w2.result()

    print(f"\ndone in {time.perf_counter() - t0:.1f}s")
    print(f"  {out_ne}  ({len(df_ne):,} rows)")
    print(f"  {out_s}  ({len(df_s):,} rows)")
    print(f"\ndtypes:\n{df_ne.dtypes.to_string()}")


if __name__ == '__main__':
    main()