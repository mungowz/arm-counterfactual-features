"""
create_dataset.py
=================
Downloads ACS PUMS person-level microdata for two US regions (Northeast and
South), applies adult_filter, bins continuous variables, maps categorical
codes to human-readable labels, and writes one CSV per region to disk.

Public API
----------
main(survey_year, horizon, random_seed, output_dir,
     income_threshold_northeast, income_threshold_south)
    Full pipeline: download → transform → undersample → save.

build_dataset(task, states, data_source, label)
    Build a processed DataFrame for one region.

undersample_to(df, n, seed)
    Stratified undersample to at most n rows.
"""

import os
import time
import numpy as np
import pandas as pd
import folktables
from folktables import ACSDataSource
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Hardware-aware worker counts
# ---------------------------------------------------------------------------

_CORES           = os.cpu_count() or 4
_DOWNLOAD_WORKERS = min(_CORES * 2, 16)
_REGION_WORKERS   = 2

# ---------------------------------------------------------------------------
# Geographic scope
# DC is excluded: not present in the folktables state list.
# ---------------------------------------------------------------------------

NORTHEAST_STATES = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA']
SOUTH_STATES     = [
    'DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV',
    'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX',
]

# ---------------------------------------------------------------------------
# ACS PUMS feature set used for both region tasks
# Source: 2024 ACS PUMS Data Dictionary — census.gov
# ---------------------------------------------------------------------------

_INCOME_FEATURES = [
    'AGEP',     # age
    'COW',      # class of worker
    'SCHL',     # educational attainment
    'MAR',      # marital status
    'OCCP',     # occupation
    'POBP',     # place of birth
    'RELSHIPP', # relationship to reference person
    'WKHP',     # usual hours worked per week past 12 months
    'SEX',      # sex
    'RAC1P',    # race
]

# ---------------------------------------------------------------------------
# Continuous → categorical binning
#
# AGEP  Census Bureau standard working-age groups.
#       adult_filter retains respondents aged ≥16 who reported ≥1 work-hour
#       and personal income > $100. Source: ACS PUMS 2024, variable AGEP (0–99).
#
# WKHP  BLS part-time/full-time boundary at 35 hours, with finer granularity
#       for overtime analysis. Source: ACS PUMS 2024, variable WKHP (1–99).
# ---------------------------------------------------------------------------

AGEP_BINS   = [0, 24, 34, 44, 54, 64, 200]
AGEP_LABELS = [
    'Young',           # 16–24
    'Young-Adult',     # 25–34
    'Mid-Career',      # 35–44
    'Experienced',     # 45–54
    'Late-Career',     # 55–64
    'Retirement-Age',  # 65+
]

WKHP_BINS   = [0, 19, 34, 39, 40, 49, 200]
WKHP_LABELS = [
    'Part-Time-Low',    #  1–19 hrs/wk
    'Part-Time',        # 20–34 hrs/wk
    'Near-Full-Time',   # 35–39 hrs/wk
    'Full-Time',        # 40    hrs/wk
    'Over-Full-Time',   # 41–49 hrs/wk
    'Extended-Hours',   # 50–99 hrs/wk
]

# ---------------------------------------------------------------------------
# Categorical mappings — source: 2024 ACS PUMS Data Dictionary (census.gov)
# Codes arrive from folktables as float64 and are cast to int before lookup;
# keys are plain integer strings without leading zeros.
# ---------------------------------------------------------------------------

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

# Two-digit codes (20–38) introduced in 2019; distinguishes same-sex couples.
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
    '1': 'Alabama',        '2': 'Alaska',          '4': 'Arizona',
    '5': 'Arkansas',       '6': 'California',      '8': 'Colorado',
    '9': 'Connecticut',   '10': 'Delaware',        '11': 'DC',
    '12': 'Florida',      '13': 'Georgia',         '15': 'Hawaii',
    '16': 'Idaho',        '17': 'Illinois',        '18': 'Indiana',
    '19': 'Iowa',         '20': 'Kansas',          '21': 'Kentucky',
    '22': 'Louisiana',    '23': 'Maine',           '24': 'Maryland',
    '25': 'Massachusetts','26': 'Michigan',        '27': 'Minnesota',
    '28': 'Mississippi',  '29': 'Missouri',        '30': 'Montana',
    '31': 'Nebraska',     '32': 'Nevada',          '33': 'New-Hampshire',
    '34': 'New-Jersey',   '35': 'New-Mexico',      '36': 'New-York',
    '37': 'North-Carolina','38': 'North-Dakota',   '39': 'Ohio',
    '40': 'Oklahoma',     '41': 'Oregon',          '42': 'Pennsylvania',
    '44': 'Rhode-Island', '45': 'South-Carolina',  '46': 'South-Dakota',
    '47': 'Tennessee',    '48': 'Texas',           '49': 'Utah',
    '50': 'Vermont',      '51': 'Virginia',        '53': 'Washington',
    '54': 'West-Virginia','55': 'Wisconsin',       '56': 'Wyoming',
    '72': 'Puerto-Rico',
    # Foreign-born
    '100': 'Born-Abroad-US-Parents',
    '301': 'Cuba',                  '302': 'Jamaica',
    '303': 'Dominican-Republic',    '308': 'Haiti',
    '313': 'Other-Caribbean',       '400': 'Mexico',
    '414': 'Guatemala',             '416': 'Honduras',
    '417': 'El-Salvador',           '422': 'Nicaragua',
    '423': 'Panama',                '424': 'Other-Central-America',
    '501': 'Colombia',              '507': 'Peru',
    '508': 'Brazil',                '516': 'Venezuela',
    '523': 'Other-South-America',   '600': 'Armenia',
    '601': 'China',                 '603': 'India',
    '607': 'Japan',                 '613': 'Philippines',
    '615': 'South-Korea',           '618': 'Vietnam',
    '619': 'Other-Southeast-Asia',  '620': 'Other-Asia',
    '700': 'United-Kingdom',        '703': 'Germany',
    '706': 'Greece',                '708': 'Ireland',
    '710': 'Italy',                 '714': 'Poland',
    '716': 'Portugal',              '720': 'Russia',
    '724': 'Ukraine',               '730': 'Other-Europe',
    '800': 'Nigeria',               '803': 'Ethiopia',
    '804': 'Egypt',                 '820': 'Other-Africa',
    '900': 'Canada',                '999': 'Other-NEC',
}

# Most-frequent SOC codes only; any unrecognised code maps to 'Other-Occupation'.
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
    '540':  'Training-And-Development-Specialists',
    '565':  'Logisticians',
    '600':  'Accountants-And-Auditors',
    '630':  'Budget-Analysts',
    '640':  'Credit-Analysts',
    '650':  'Financial-Analysts',
    '700':  'Management-Analysts',
    '726':  'Market-Research-Analysts',
    '740':  'Business-Operations-Specialists',
    '800':  'Buyers-And-Purchasing-Agents',
    '840':  'Claims-Adjusters',
    '1005': 'Computer-And-Info-Research-Scientists',
    '1006': 'Computer-Systems-Analysts',
    '1007': 'Information-Security-Analysts',
    '1010': 'Computer-Programmers',
    '1021': 'Software-Developers',
    '1022': 'Software-Quality-Assurance',
    '1031': 'Web-Developers',
    '1032': 'Web-And-Digital-Interface-Designers',
    '1050': 'Database-Administrators',
    '1065': 'Network-And-Computer-Systems-Admins',
    '1100': 'Computer-Support-Specialists',
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

# Default label for codes absent from the mapping.
_COLUMN_FALLBACKS = {
    'OCCP': 'Other-Occupation',
    'POBP': 'Other-NEC',
}

_COLUMN_MAPS = {
    'COW':      COW_MAP,
    'SCHL':     SCHL_MAP,
    'MAR':      MAR_MAP,
    'RELSHIPP': RELSHIPP_MAP,
    'SEX':      SEX_MAP,
    'RAC1P':    RAC1P_MAP,
    'POBP':     POBP_MAP,
    'OCCP':     OCCP_MAP,
}


# ---------------------------------------------------------------------------
# Vectorised lookup arrays
# Built once at import time; index = code + 1 so that the NaN sentinel
# (-1) maps to index 0 (the fallback label).
# ---------------------------------------------------------------------------

def _build_lookup(mapping: dict, fallback: str) -> np.ndarray:
    max_code = max(int(k) for k in mapping) if mapping else 0
    arr = np.full(max_code + 2, fill_value=fallback, dtype=object)
    for k, label in mapping.items():
        arr[int(k) + 1] = label
    return arr


_LOOKUPS: dict[str, np.ndarray] = {
    col: _build_lookup(m, _COLUMN_FALLBACKS.get(col, 'Unknown'))
    for col, m in _COLUMN_MAPS.items()
}


# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------

def _decode_column(series: pd.Series, lookup: np.ndarray) -> np.ndarray:
    """Vectorised integer-code → label conversion using a pre-built lookup array."""
    codes = pd.to_numeric(series, errors='coerce').fillna(-1).to_numpy(dtype=np.int32)
    idx   = np.clip(codes + 1, 0, len(lookup) - 1)
    return lookup[idx]


def apply_categorical_mappings(df: pd.DataFrame) -> None:
    """
    In-place: replace raw numeric codes with human-readable category labels
    for every column present in _COLUMN_MAPS.
    """
    for col, lookup in _LOOKUPS.items():
        if col in df.columns:
            df[col] = pd.Categorical(_decode_column(df[col], lookup))


def _make_income_task(threshold: int) -> folktables.BasicProblem:
    """
    Return a binary income-classification task for the given annual-income
    threshold (in USD).  The positive class is income > threshold.
    """
    return folktables.BasicProblem(
        features         = _INCOME_FEATURES,
        target           = 'PINCP',
        target_transform = lambda x: x > threshold,
        group            = 'RAC1P',
        preprocess       = folktables.adult_filter,
        postprocess      = lambda x: np.nan_to_num(x, nan=-1),
    )


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_state(data_source: ACSDataSource, state: str) -> pd.DataFrame:
    return data_source.get_data(states=[state], download=True)


def parallel_get_data(
    data_source: ACSDataSource,
    states: list[str],
    label: str,
) -> pd.DataFrame:
    """
    Download ACS data for all states in parallel and return the concatenated
    raw DataFrame.

    Parameters
    ----------
    data_source : ACSDataSource
    states      : list of state abbreviations to download
    label       : human-readable region name used in progress messages
    """
    n_workers = min(len(states), _DOWNLOAD_WORKERS)
    print(f'  > [{label}] {len(states)} states — {n_workers} workers')

    results: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_download_state, data_source, s): s for s in states}
        for fut in as_completed(futures):
            state = futures[fut]
            try:
                results[state] = fut.result()
                print(f'    - {state} done')
            except Exception as exc:
                raise RuntimeError(
                    f'Download failed for {state}: {exc}\n'
                    f'If the requested survey year is not yet available, '
                    f'try an earlier year.'
                ) from exc

    return pd.concat([results[s] for s in states], ignore_index=True)


def build_dataset(
    task: folktables.BasicProblem,
    states: list[str],
    data_source: ACSDataSource,
    label: str,
) -> pd.DataFrame:
    """
    Download raw ACS data, apply the prediction task filter, bin continuous
    variables, and map categorical codes to labels.

    Returns a fully processed DataFrame with an INCOME_ABOVE_THRESHOLD column.
    """
    t0 = time.perf_counter()
    print(f'\n{"=" * 70}')
    print(label)
    print(f'{"=" * 70}')

    raw = parallel_get_data(data_source, states, label)
    print(f'  > raw rows: {len(raw):,}  ({time.perf_counter() - t0:.1f}s)')

    features_df, labels, _ = task.df_to_pandas(raw)
    features_df['INCOME_ABOVE_THRESHOLD'] = labels.to_numpy(dtype=np.int8)

    apply_categorical_mappings(features_df)

    if 'AGEP' in features_df.columns:
        features_df['AGEP'] = pd.Categorical(
            pd.cut(
                pd.to_numeric(features_df['AGEP'], errors='coerce'),
                bins=AGEP_BINS, labels=AGEP_LABELS, right=True,
            )
        )
    if 'WKHP' in features_df.columns:
        features_df['WKHP'] = pd.Categorical(
            pd.cut(
                pd.to_numeric(features_df['WKHP'], errors='coerce'),
                bins=WKHP_BINS, labels=WKHP_LABELS, right=True,
            )
        )

    pos_pct = features_df['INCOME_ABOVE_THRESHOLD'].mean() * 100
    print(
        f'  > filtered rows: {len(features_df):,}  |  '
        f'positive class: {pos_pct:.1f}%  ({time.perf_counter() - t0:.1f}s)'
    )

    return features_df


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def undersample_to(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Stratified random undersample to at most n rows, preserving the class
    ratio of INCOME_ABOVE_THRESHOLD.
    """
    if len(df) <= n:
        return df

    target_col = 'INCOME_ABOVE_THRESHOLD'
    groups     = [g for _, g in df.groupby(target_col, observed=False)]
    per_class  = [max(1, round(len(g) / len(df) * n)) for g in groups]

    # Absorb rounding error into the majority class.
    per_class[0] += n - sum(per_class)

    sampled = pd.concat(
        [g.sample(k, random_state=seed) for g, k in zip(groups, per_class)],
        ignore_index=True,
    )
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    survey_year: str                 = '2024',
    horizon: str                     = '1-Year',
    random_seed: int                 = 42,
    output_dir: str                  = 'data',
    income_threshold_northeast: int  = 110_000,
    income_threshold_south: int      = 90_000,
) -> None:
    """
    Download ACS data for Northeast and South, build processed datasets,
    and save them as CSV files under output_dir.

    Parameters
    ----------
    survey_year                  ACS survey year (e.g. '2024').
    horizon                      Survey horizon ('1-Year' or '5-Year').
    random_seed                  Random seed for reproducibility.
    output_dir                   Directory where CSV files are written.
    income_threshold_northeast   Annual income threshold (USD) for the
                                 Northeast positive class.
    income_threshold_south       Annual income threshold (USD) for the
                                 South positive class.
    """
    t0 = time.perf_counter()

    os.makedirs(output_dir, exist_ok=True)

    data_source = ACSDataSource(
        survey_year=survey_year, horizon=horizon, survey='person'
    )

    out_ne = os.path.join(output_dir, f'acs_income_northeast_{survey_year}.csv')
    out_s  = os.path.join(output_dir, f'acs_income_south_{survey_year}.csv')

    task_ne = _make_income_task(income_threshold_northeast)
    task_s  = _make_income_task(income_threshold_south)

    print(f'\n{"=" * 70}')
    print(f'ACS INCOME DATASET — {survey_year}')
    print(f'{"=" * 70}')
    print(f'  > Northeast threshold : ${income_threshold_northeast:,}')
    print(f'  > South threshold     : ${income_threshold_south:,}')

    with ThreadPoolExecutor(max_workers=_REGION_WORKERS) as pool:
        fut_ne = pool.submit(
            build_dataset, task_ne, NORTHEAST_STATES, data_source,
            f'Northeast (income > ${income_threshold_northeast:,})',
        )
        fut_s = pool.submit(
            build_dataset, task_s, SOUTH_STATES, data_source,
            f'South (income > ${income_threshold_south:,})',
        )
        df_ne = fut_ne.result()
        df_s  = fut_s.result()

    df_s = undersample_to(df_s, len(df_ne), seed=random_seed)

    pos_ne = df_ne['INCOME_ABOVE_THRESHOLD'].mean() * 100
    pos_s  = df_s['INCOME_ABOVE_THRESHOLD'].mean()  * 100
    print(f'\n  > Northeast : {len(df_ne):>7,} rows  |  positive class: {pos_ne:.1f}%')
    print(f'  > South     : {len(df_s):>7,} rows  |  positive class: {pos_s:.1f}%')

    with ThreadPoolExecutor(max_workers=2) as pool:
        w1 = pool.submit(df_ne.to_csv, out_ne, index=False)
        w2 = pool.submit(df_s.to_csv,  out_s,  index=False)
        w1.result()
        w2.result()

    print(f'\n{"=" * 70}')
    print(f'Completed in {time.perf_counter() - t0:.1f}s')
    print(f'{"=" * 70}')
    print(f'  > {out_ne}')
    print(f'  > {out_s}')
    print(f'\n  > dtypes:\n{df_ne.dtypes.to_string()}')


if __name__ == '__main__':
    main()
