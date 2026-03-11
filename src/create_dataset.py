"""
create_dataset.py
=================
Downloads ACS PUMS (Public Use Microdata Sample) person-level microdata for
three US geographic regions — Northeast, South, and national (USA) — applies
the folktables adult_filter, discretises continuous variables, maps numeric
ACS codes to human-readable labels, and writes one CSV per region to disk.

Pipeline position
-----------------
    [create_dataset.py]  →  feature_importance.py  →  macroscopic_experiment_association_rules.py

The output CSVs are the sole input to feature_importance.py.  Every column
written here (ST, YEAR, INCOME_ABOVE_THRESHOLD, …) must be consistent with
the assumptions in that module.

Design principles
-----------------
- All thresholds, states, and feature lists are module-level constants or
  explicit function parameters — no magic numbers buried in logic.
- State downloads are parallelised with ThreadPoolExecutor; the first state
  download is serialised under _zip_lock to prime the shared ACS cache file
  before concurrent workers start reading it.
- Three regions can be built concurrently (up to _REGION_WORKERS = 3).
- Undersampling is applied only to the NE/South pair to equalise class
  distributions for a fair cross-regional comparison.  The USA dataset is
  intentionally left at its full post-filter size for independent national
  analysis.
"""

import os
import time
import threading
import numpy as np
import pandas as pd
import folktables
from folktables import ACSDataSource
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ---------------------------------------------------------------------------
# Hardware-aware worker counts  — tuned for Apple M1 Ultra
# ---------------------------------------------------------------------------
#
# M1 Ultra topology: 16 P-cores (performance) + 4 E-cores (efficiency) = 20 cores.
# macOS exposes all 20 as logical CPUs; os.cpu_count() returns 20.
#
# Work taxonomy:
#   I/O-bound (network + NVMe reads)  → ThreadPoolExecutor   — GIL is released
#   CPU-bound (DataFrame transforms)  → ProcessPoolExecutor  — bypasses GIL
#
# _CORES            : 20 on M1 Ultra (os.cpu_count())
# _DOWNLOAD_WORKERS : I/O threads per region.  ACS server rate-limits at ~8-12
#                     concurrent connections; 16 gives head-room while the NVMe
#                     cache absorbs disk bursts.  Capped at 16 to avoid
#                     exhausting file-descriptor limits across 3 concurrent regions.
# _REGION_WORKERS   : exactly 3 regions; use all 3 concurrently.
# _CPU_WORKERS      : ProcessPoolExecutor for CPU-bound postprocessing.
#                     16 = P-core count; leaves 4 E-cores for the OS + GCD threads.
#                     Spawning more than 16 processes on M1 Ultra causes thermal
#                     throttling with no throughput benefit.
# _CATEGORICAL_WORKERS: parallelism inside apply_categorical_mappings().
#                     8 columns in _COLUMN_MAPS; 8 processes = one per column.

_CORES               = os.cpu_count() or 4
_DOWNLOAD_WORKERS    = min(_CORES, 16)         # I/O threads — capped at 16
_REGION_WORKERS      = 3                        # one per region
_CPU_WORKERS         = min(_CORES - 4, 16)     # CPU processes — leave 4 E-cores free
_CATEGORICAL_WORKERS = min(len({'COW','SCHL','MAR','RELSHIPP','SEX','RAC1P','POBP','OCCP'}),
                            _CPU_WORKERS)       # one process per categorical column

# ---------------------------------------------------------------------------
# Geographic scope
# ---------------------------------------------------------------------------

# DC (FIPS 11) is excluded: folktables does not include it in the standard
# state list and its data would distort both region sizes and distributions.

# Northeast: New England + Middle Atlantic (BLS definition, 9 states).
NORTHEAST_STATES = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA']

# South: South Atlantic + East South Central + West South Central
# (BLS definition, 16 states, excluding DC).
SOUTH_STATES = [
    'DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV',   # South Atlantic
    'AL', 'KY', 'MS', 'TN',                             # East South Central
    'AR', 'LA', 'OK', 'TX',                             # West South Central
]

# USA: all 50 states minus AK — used for a national-level experiment that is NOT
# undersampled (see sampling note in main() docstring).
# AK (Alaska) is excluded: ACS 1-Year estimates are only published for geographic
# areas with population ≥ 65,000.  Alaska does not meet this threshold for the
# 1-Year survey, so folktables returns a malformed/placeholder CSV that causes
# a pandas ParserError ("Expected 286 fields, saw 457").  Use the 5-Year horizon
# if Alaska coverage is required.
USA_STATES = [
    'AL',       'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]

# ---------------------------------------------------------------------------
# ACS PUMS feature set
# ---------------------------------------------------------------------------

# These are the raw ACS variable codes that folktables will pull from the
# PUMS files.  They must match the folktables BasicProblem features list
# exactly so that df_to_pandas() returns them in the correct order.
#
# Notable omissions:
#   - PINCP (wage/salary income) is the target, not a feature.
#   - DC/military-only variables are excluded to keep the feature space clean.
#
# ST is listed last and handled specially: it is injected as the state
# abbreviation string (e.g. 'NY') before df_to_pandas() is called, so it
# arrives in the feature matrix already decoded.  It is therefore absent
# from _COLUMN_MAPS (no numeric→label translation needed).
_INCOME_FEATURES = [
    'AGEP',     # Age — continuous; binned into 6 career-stage categories below
    'COW',      # Class of worker (employee type, self-employed, etc.)
    'SCHL',     # Educational attainment (24 levels from no schooling to doctorate)
    'MAR',      # Marital status
    'OCCP',     # Occupation code (SOC-based, ~500 codes mapped to readable labels)
    'POBP',     # Place of birth (state or country code)
    'RELSHIPP', # Relationship to household reference person
    'WKHP',     # Usual hours worked per week — continuous; binned into 6 bands
    'SEX',      # Sex (binary in ACS PUMS)
    'RAC1P',    # Race (9 categories)
    'ST',       # State of residence — injected as abbreviation, bypasses FIPS
]

# ---------------------------------------------------------------------------
# Continuous → categorical binning
# ---------------------------------------------------------------------------

# AGEP bins: right-inclusive intervals capturing the six conventional US
# career stages.  The upper bound (200) is intentionally large to absorb
# any data quality outliers without triggering a NaN.
AGEP_BINS   = [0, 24, 34, 44, 54, 64, 200]
AGEP_LABELS = [
    'Young',           # (0–24]  entry level / student (adult_filter ensures age ≥ 16)
    'Young-Adult',     # (25–34] early career
    'Mid-Career',      # (35–44] peak productivity
    'Experienced',     # (45–54] senior contributor
    'Late-Career',     # (55–64] approaching retirement
    'Retirement-Age',  # (65–200] beyond standard retirement age
]

# WKHP bins: the 40-hour threshold is standard for US full-time employment.
# Near-Full-Time (35–39) captures workers just below the full-time threshold,
# which is policy-relevant for benefits eligibility.
WKHP_BINS   = [0, 19, 34, 39, 40, 49, 200]
WKHP_LABELS = [
    'Part-Time-Low',    #  (0–19]  hrs/wk  marginal attachment
    'Part-Time',        # (20–34]  hrs/wk  standard part-time
    'Near-Full-Time',   # (35–39]  hrs/wk  benefits threshold zone
    'Full-Time',        #    [40]  hrs/wk  exact standard full-time
    'Over-Full-Time',   # (41–49]  hrs/wk  modest overtime
    'Extended-Hours',   # (50–200] hrs/wk  heavy overtime / multiple jobs (upper bound 200 absorbs outliers)
]

# ---------------------------------------------------------------------------
# Categorical mappings
# ---------------------------------------------------------------------------

# Each map converts ACS numeric codes (stored as strings in the CSV) to
# human-readable dash-separated labels.  Dash separation is deliberate:
# FP-Growth itemsets are stored as strings; spaces inside labels would make
# parsing ambiguous in downstream CSV files.

COW_MAP = {
    # ACS COW variable: Class of Worker
    '1': 'Employee-Private-For-Profit',
    '2': 'Employee-Private-Non-Profit',
    '3': 'Local-Government-Employee',
    '4': 'State-Government-Employee',
    '5': 'Federal-Government-Employee',
    '6': 'Self-Employed-Not-Incorporated',   # unincorporated sole proprietor
    '7': 'Self-Employed-Incorporated',        # incorporated business owner
    '8': 'Unpaid-Family-Worker',
    '9': 'Unemployed-5plus-Years-Or-Never-Worked',
}

SCHL_MAP = {
    # ACS SCHL variable: Educational Attainment (24 levels)
    '1': 'No-Schooling-Completed', '2': 'Nursery-School-Preschool',
    '3': 'Kindergarten', '4': 'Grade-1', '5': 'Grade-2', '6': 'Grade-3',
    '7': 'Grade-4', '8': 'Grade-5', '9': 'Grade-6', '10': 'Grade-7',
    '11': 'Grade-8', '12': 'Grade-9', '13': 'Grade-10', '14': 'Grade-11',
    '15': 'Grade-12-No-Diploma', '16': 'Regular-HS-Diploma',
    '17': 'GED-Or-Alt-Credential', '18': 'Some-College-Less-Than-1yr',
    '19': 'Some-College-1yr-Or-More-No-Degree', '20': 'Associates-Degree',
    '21': 'Bachelors-Degree', '22': 'Masters-Degree',
    '23': 'Professional-Degree-Beyond-Bachelors', '24': 'Doctorate-Degree',
}

MAR_MAP = {
    # ACS MAR variable: Marital Status
    '1': 'Married', '2': 'Widowed', '3': 'Divorced',
    '4': 'Separated', '5': 'Never-Married-Or-Under-15',
}

RELSHIPP_MAP = {
    # ACS RELSHIPP variable: Relationship to Reference Person (2019+ codes)
    '20': 'Reference-Person', '21': 'Opposite-Sex-Husband-Wife-Spouse',
    '22': 'Opposite-Sex-Unmarried-Partner', '23': 'Same-Sex-Husband-Wife-Spouse',
    '24': 'Same-Sex-Unmarried-Partner', '25': 'Biological-Son-Or-Daughter',
    '26': 'Adopted-Son-Or-Daughter', '27': 'Stepson-Or-Stepdaughter',
    '28': 'Brother-Or-Sister', '29': 'Father-Or-Mother', '30': 'Grandchild',
    '31': 'Parent-In-Law', '32': 'Son-In-Law-Or-Daughter-In-Law',
    '33': 'Other-Relative', '34': 'Roommate-Or-Housemate', '35': 'Foster-Child',
    '36': 'Other-Nonrelative', '37': 'Institutionalized-Group-Quarters',
    '38': 'Noninstitutionalized-Group-Quarters',
}

SEX_MAP = {'1': 'Male', '2': 'Female'}

RAC1P_MAP = {
    # ACS RAC1P variable: Race (recoded to single-race categories)
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
    # ACS POBP variable: Place of Birth (state FIPS + country codes).
    # Only the most frequently occurring codes are listed; OCCP uses
    # 'Other-NEC' (Not Elsewhere Classified) as fallback for unmapped codes.
    '1': 'Alabama', '2': 'Alaska', '4': 'Arizona', '5': 'Arkansas', '6': 'California',
    '8': 'Colorado', '9': 'Connecticut', '10': 'Delaware', '11': 'DC', '12': 'Florida',
    '13': 'Georgia', '15': 'Hawaii', '16': 'Idaho', '17': 'Illinois', '18': 'Indiana',
    '19': 'Iowa', '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana', '23': 'Maine',
    '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota',
    '28': 'Mississippi', '29': 'Missouri', '30': 'Montana', '31': 'Nebraska',
    '32': 'Nevada', '33': 'New-Hampshire', '34': 'New-Jersey', '35': 'New-Mexico',
    '36': 'New-York', '37': 'North-Carolina', '38': 'North-Dakota', '39': 'Ohio',
    '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania', '44': 'Rhode-Island',
    '45': 'South-Carolina', '46': 'South-Dakota', '47': 'Tennessee', '48': 'Texas',
    '49': 'Utah', '50': 'Vermont', '51': 'Virginia', '53': 'Washington',
    '54': 'West-Virginia', '55': 'Wisconsin', '56': 'Wyoming', '72': 'Puerto-Rico',
    '100': 'Born-Abroad-US-Parents', '301': 'Cuba', '302': 'Jamaica',
    '303': 'Dominican-Republic', '308': 'Haiti', '313': 'Other-Caribbean',
    '400': 'Mexico', '414': 'Guatemala', '416': 'Honduras', '417': 'El-Salvador',
    '422': 'Nicaragua', '423': 'Panama', '424': 'Other-Central-America',
    '501': 'Colombia', '507': 'Peru', '508': 'Brazil', '516': 'Venezuela',
    '523': 'Other-South-America', '600': 'Armenia', '601': 'China', '603': 'India',
    '607': 'Japan', '613': 'Philippines', '615': 'South-Korea', '618': 'Vietnam',
    '619': 'Other-Southeast-Asia', '620': 'Other-Asia', '700': 'United-Kingdom',
    '703': 'Germany', '706': 'Greece', '708': 'Ireland', '710': 'Italy',
    '714': 'Poland', '716': 'Portugal', '720': 'Russia', '724': 'Ukraine',
    '730': 'Other-Europe', '800': 'Nigeria', '803': 'Ethiopia', '804': 'Egypt',
    '820': 'Other-Africa', '900': 'Canada', '999': 'Other-NEC',
}

OCCP_MAP = {
    # ACS OCCP variable: Occupation (SOC-based codes).
    # Covers ~80 of the most common codes; anything not listed maps to
    # 'Other-Occupation' via _COLUMN_FALLBACKS.
    '0':  'Not-In-Labor-Force-Or-Under-16',  # NOTE: adult_filter keeps these rows; exclude downstream if needed
    '10': 'Chief-Executives',
    '20': 'General-Operations-Managers', '120': 'Financial-Managers',
    '136': 'HR-Managers', '220': 'Advertising-And-Marketing-Managers',
    '300': 'Purchasing-Managers', '310': 'Transportation-Managers',
    '330': 'Food-Service-Managers', '410': 'Medical-And-Health-Services-Managers',
    '430': 'Construction-Managers', '440': 'Other-Managers',
    '500': 'Agents-Of-Performing-Arts', '510': 'Compliance-Officers',
    '520': 'Cost-Estimators', '530': 'Human-Resources-Workers',
    '540': 'Training-And-Development-Specialists', '565': 'Logisticians',
    '600': 'Accountants-And-Auditors', '630': 'Budget-Analysts',
    '640': 'Credit-Analysts', '650': 'Financial-Analysts',
    '700': 'Management-Analysts', '726': 'Market-Research-Analysts',
    '740': 'Business-Operations-Specialists', '800': 'Buyers-And-Purchasing-Agents',
    '840': 'Claims-Adjusters', '1005': 'Computer-And-Info-Research-Scientists',
    '1006': 'Computer-Systems-Analysts', '1007': 'Information-Security-Analysts',
    '1010': 'Computer-Programmers', '1021': 'Software-Developers',
    '1022': 'Software-Quality-Assurance', '1031': 'Web-Developers',
    '1032': 'Web-And-Digital-Interface-Designers', '1050': 'Database-Administrators',
    '1065': 'Network-And-Computer-Systems-Admins', '1100': 'Computer-Support-Specialists',
    '1200': 'Actuaries', '1220': 'Operations-Research-Analysts',
    '1230': 'Statisticians', '1240': 'Data-Scientists', '2100': 'Lawyers',
    '2105': 'Judicial-Law-Clerks', '2110': 'Judges-And-Magistrates',
    '2310': 'Elementary-School-Teachers', '2320': 'Middle-School-Teachers',
    '2330': 'Secondary-School-Teachers', '2540': 'Special-Education-Teachers',
    '2550': 'Other-Teachers', '2560': 'Tutors-And-Instructors',
    '2630': 'Postsecondary-Teachers', '2640': 'Preschool-And-Kindergarten-Teachers',
    '2720': 'Art-Directors', '2740': 'Graphic-Designers',
    '2750': 'Interior-Designers', '3010': 'Chiropractors',
    '3050': 'Dietitians-And-Nutritionists', '3090': 'Emergency-Medical-Technicians',
    '3100': 'Exercise-Physiologists', '3130': 'Pharmacists',
    '3160': 'Physical-Therapists', '3230': 'Physicians-And-Surgeons',
    '3250': 'Registered-Nurses', '3255': 'Nurse-Practitioners',
    '3260': 'Occupational-Therapists', '3300': 'Dentists',
    '3420': 'Dental-Assistants', '3500': 'Licensed-Practical-Nurses',
    '3600': 'Medical-Assistants', '4000': 'Cooks-Restaurant',
    '4020': 'Food-Preparation-Workers', '4040': 'Bartenders',
    '4055': 'Fast-Food-Workers', '4110': 'Waiters-And-Waitresses',
    '4120': 'Dining-Room-Attendants', '4140': 'Dishwashers',
    '4220': 'Janitors-And-Cleaners', '4230': 'Maids-And-Housekeeping',
    '4700': 'First-Line-Retail-Supervisors', '4720': 'Cashiers',
    '4740': 'Counter-And-Rental-Clerks', '4760': 'Retail-Salespersons',
    '4800': 'Insurance-Sales-Agents', '4810': 'Securities-And-Financial-Sales',
    '4820': 'Real-Estate-Brokers-And-Agents', '4840': 'Telemarketers',
    '4850': 'Sales-Representatives', '5000': 'First-Line-Office-Supervisors',
    '5110': 'Receptionists', '5120': 'Information-Clerks',
    '5160': 'Customer-Service-Representatives', '5230': 'Payroll-And-Timekeeping-Clerks',
    '5240': 'Human-Resources-Assistants', '5260': 'Eligibility-Interviewers',
    '5420': 'Postal-Service-Workers', '5600': 'Production-Planning-Clerks',
    '5700': 'Secretaries-And-Admin-Assistants', '5820': 'Data-Entry-Keyers',
    '9600': 'Cleaners-Of-Vehicles-And-Equipment', '9620': 'Laborers-And-Material-Movers',
    '9800': 'Military-Officer-Special-Operations', '9810': 'Military-First-Line-Supervisors',
    '9820': 'Military-Enlisted-Tactical-Operations', '9830': 'Military-Rank-Not-Specified',
}

# ---------------------------------------------------------------------------
# Fallback labels for columns with unmapped codes
# ---------------------------------------------------------------------------

# OCCP and POBP have hundreds of valid codes; codes not in the maps above
# receive the fallback label instead of 'Unknown' to preserve analytical
# interpretability.
_COLUMN_FALLBACKS = {'OCCP': 'Other-Occupation', 'POBP': 'Other-NEC'}

# Master dispatch table: column name → its code-to-label map.
# ST is intentionally absent: it is handled by injecting the state
# abbreviation string directly in _download_state(), bypassing the FIPS
# numeric code entirely.  No further numeric→label translation is needed.
_COLUMN_MAPS = {
    'COW': COW_MAP, 'SCHL': SCHL_MAP, 'MAR': MAR_MAP,
    'RELSHIPP': RELSHIPP_MAP, 'SEX': SEX_MAP, 'RAC1P': RAC1P_MAP,
    'POBP': POBP_MAP, 'OCCP': OCCP_MAP,
    # 'ST' is handled dynamically (state abbreviations injected directly, bypassing FIPS codes).
}

# ---------------------------------------------------------------------------
# Vectorised lookup arrays
# ---------------------------------------------------------------------------

def _build_lookup(mapping: dict, fallback: str) -> np.ndarray:
    """
    Build a fixed-size NumPy array for O(1) integer-indexed label lookup.

    Layout: index 0 is reserved for 'unknown/missing' (numeric code -1 maps
    to index 0).  Index k+1 holds the label for numeric code k.

    This avoids a Python dict lookup for every cell during apply_categorical_mappings,
    reducing the inner loop from O(n) dict lookups to a single NumPy fancy-index
    operation on the entire column at once (see _decode_column).

    Parameters
    ----------
    mapping  : code (str) → label (str) dictionary
    fallback : label returned for any code not in mapping
    """
    max_code = max(int(k) for k in mapping) if mapping else 0
    # +2: index 0 = fallback, index max_code+1 = last valid code.
    arr = np.full(max_code + 2, fill_value=fallback, dtype=object)
    for k, label in mapping.items():
        arr[int(k) + 1] = label   # code k → slot k+1
    return arr

# Pre-build all lookup arrays at import time to avoid rebuilding them on
# every call to apply_categorical_mappings.
_LOOKUPS: dict[str, np.ndarray] = {
    col: _build_lookup(m, _COLUMN_FALLBACKS.get(col, 'Unknown'))
    for col, m in _COLUMN_MAPS.items()
}

# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------

def _decode_column(series: pd.Series, lookup: np.ndarray) -> np.ndarray:
    """
    Vectorised decode of a numeric-coded pandas Series using a pre-built
    lookup array.

    Steps:
    1. Coerce to numeric, turning non-parseable strings into NaN.
    2. Fill NaN with -1 (maps to index 0 = fallback in the lookup array).
    3. Cast to int32 and compute slot indices (code + 1).
    4. Clip to [0, len(lookup)-1] to guard against unexpected out-of-range codes.
    5. Fancy-index the lookup array in one vectorised operation.

    Returns a NumPy object array of label strings.
    """
    codes = pd.to_numeric(series, errors='coerce').fillna(-1).to_numpy(dtype=np.int32)
    idx   = np.clip(codes + 1, 0, len(lookup) - 1)
    return lookup[idx]


def _decode_column_task(args: tuple) -> tuple[str, np.ndarray]:
    """
    Top-level picklable worker for parallel categorical decoding.

    ProcessPoolExecutor requires top-level functions (lambdas and closures are
    not picklable).  Receives (col_name, series_values, lookup_array) and
    returns (col_name, decoded_array) so the caller can reassemble the DataFrame.

    Parameters
    ----------
    args : (col, values, lookup)
        col    – column name (str)
        values – raw Series values as a NumPy object array
        lookup – pre-built label lookup array from _LOOKUPS
    """
    col, values, lookup = args
    codes = pd.to_numeric(pd.Series(values), errors='coerce').fillna(-1).to_numpy(dtype=np.int32)
    idx   = np.clip(codes + 1, 0, len(lookup) - 1)
    return col, lookup[idx]


def apply_categorical_mappings(df: pd.DataFrame) -> None:
    """
    Decode all numerically-coded ACS columns in-place using _LOOKUPS.

    Parallelisation strategy
    ------------------------
    Each column decode is independent (pure function, no shared state), making
    this embarrassingly parallel.  We dispatch one task per column to a
    ProcessPoolExecutor so that all P-cores are utilised simultaneously and the
    GIL is bypassed entirely.

    The worker (_decode_column_task) is a top-level function (picklable) that
    receives pre-serialised NumPy arrays — avoiding the cost of pickling the
    full DataFrame.  Results are written back into df in the main process.

    Only columns present in both df and _LOOKUPS are processed; ST and others
    not in the dispatch table are silently skipped.
    """
    cols_to_decode = [col for col in _LOOKUPS if col in df.columns]
    if not cols_to_decode:
        return

    # Serialise only the columns we need (object arrays, not the full DataFrame).
    tasks = [(col, df[col].to_numpy(), _LOOKUPS[col]) for col in cols_to_decode]

    n_workers = min(len(tasks), _CATEGORICAL_WORKERS)

    if n_workers > 1:
        # spawn context avoids inheriting open file descriptors / locks from the
        # parent process — critical when called from inside a ThreadPoolExecutor.
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures_map = {pool.submit(_decode_column_task, t): t[0] for t in tasks}
            decoded: dict[str, np.ndarray] = {}
            for fut in as_completed(futures_map):
                col_name, arr = fut.result()
                decoded[col_name] = arr
    else:
        decoded = dict(_decode_column_task(t) for t in tasks)

    # Reassemble: write decoded arrays back as pd.Categorical in original column order.
    for col in cols_to_decode:
        df[col] = pd.Categorical(decoded[col])


def _decode_column_task(args: tuple) -> tuple[str, np.ndarray]:
    """
    Top-level picklable worker for parallel categorical decoding.

    ProcessPoolExecutor requires top-level (picklable) functions — lambdas and
    closures cannot be pickled.  Receives a packed tuple so the pool can dispatch
    it without keyword-argument serialisation overhead.

    Parameters
    ----------
    args : (col, values, lookup)
        col    – column name string
        values – column data as a NumPy object array (cheap to pickle vs Series)
        lookup – pre-built label lookup array from _LOOKUPS

    Returns
    -------
    (col, decoded_array) — caller reassembles the DataFrame column-by-column.
    """
    col, values, lookup = args
    codes = pd.to_numeric(pd.Series(values), errors='coerce').fillna(-1).to_numpy(dtype=np.int32)
    idx   = np.clip(codes + 1, 0, len(lookup) - 1)
    return col, lookup[idx]


def apply_categorical_mappings(df: pd.DataFrame) -> None:
    """
    Decode all numerically-coded ACS columns in-place using _LOOKUPS.

    Parallelisation strategy
    ------------------------
    Each column decode is a pure function with no shared state → embarrassingly
    parallel.  We dispatch one task per column to a ProcessPoolExecutor so that
    all M1 P-cores are utilised simultaneously and the GIL is bypassed entirely.

    Serialisation cost is minimised by passing pre-extracted NumPy object arrays
    (not the full DataFrame) to each worker.  Results are written back into df
    in the main process after all workers complete.

    'spawn' multiprocessing context is used to avoid inheriting open file
    descriptors or threading locks from the parent process, which is especially
    important when called from inside a ThreadPoolExecutor (region-level pool).

    Only columns present in both df and _LOOKUPS are processed; ST and other
    columns not in the dispatch table are silently skipped.

    Mutates df in place; returns None.
    """
    cols_to_decode = [col for col in _LOOKUPS if col in df.columns]
    if not cols_to_decode:
        return

    # Pack each task as a plain tuple (picklable) containing only the arrays needed.
    tasks = [(col, df[col].to_numpy(), _LOOKUPS[col]) for col in cols_to_decode]
    n_workers = min(len(tasks), _CATEGORICAL_WORKERS)

    if n_workers > 1:
        ctx = mp.get_context('spawn')   # safe context: no inherited state
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            fut_map = {pool.submit(_decode_column_task, t): t[0] for t in tasks}
            decoded: dict[str, np.ndarray] = {}
            for fut in as_completed(fut_map):
                col_name, arr = fut.result()
                decoded[col_name] = arr
    else:
        decoded = dict(_decode_column_task(t) for t in tasks)

    # Write decoded arrays back as pd.Categorical in the original column order.
    for col in cols_to_decode:
        df[col] = pd.Categorical(decoded[col])


def _make_income_task(threshold: int) -> folktables.BasicProblem:
    """
    Build a folktables BasicProblem with a region-specific income threshold.

    The target_transform converts the continuous PINCP (personal income) field
    into a binary label: 1 if income > threshold, 0 otherwise.

    The postprocess lambda fills ACS missing-value sentinels (NaN) with -1
    across all feature columns.  folktables passes a *numpy.ndarray* (not a
    DataFrame) to postprocess, so we use np.vectorize instead of .apply().

    Parameters
    ----------
    threshold : annual income in USD above which INCOME_ABOVE_THRESHOLD = 1
    """
    _is_str = np.vectorize(lambda v: isinstance(v, str))
    return folktables.BasicProblem(
        features         = _INCOME_FEATURES,
        target           = 'PINCP',
        target_transform = lambda x: x > threshold,
        group            = 'RAC1P',
        preprocess       = folktables.adult_filter,
        # np.vectorize works on ndarray; .apply() would require a DataFrame.
        postprocess      = lambda x: np.where(pd.isna(x) & ~_is_str(x), -1, x),
    )

# ---------------------------------------------------------------------------
# State-level download helpers
# ---------------------------------------------------------------------------

def _download_state(data_source: ACSDataSource, state: str) -> pd.DataFrame:
    """
    Download the raw ACS PUMS person file for a single state and inject the
    state abbreviation before returning.

    Why inject ST here (before df_to_pandas)?
    -----------------------------------------
    folktables reads ST as a numeric FIPS code (e.g. 36 for NY).  If we left
    it as a number, it would pass through df_to_pandas() as an integer, then
    need a separate FIPS→abbreviation mapping step.  By overwriting the ST
    column with the abbreviation string *before* df_to_pandas() is called,
    the abbreviation flows through the entire pipeline transparently.  It is
    subsequently converted to pd.Categorical in build_dataset().

    Parameters
    ----------
    data_source : ACSDataSource configured for a given survey year/horizon
    state       : two-letter state abbreviation (e.g. 'NY', 'CA')

    Returns
    -------
    Raw ACS DataFrame with ST already replaced by the abbreviation string.
    """
    df = data_source.get_data(states=[state], download=True)
    # Inject the state abbreviation (e.g. 'CA') directly into the raw DataFrame.
    # This overwrites the numeric FIPS code before df_to_pandas() is called,
    # so the abbreviation appears in the feature matrix without any extra mapping.
    df['ST'] = state
    return df


# Module-level lock that serialises the *first* state download in any thread
# group.  folktables downloads a zip archive and caches it locally; if two
# threads call get_data() simultaneously before the cache exists, both may
# attempt to write the same file, corrupting it.  Acquiring the lock for only
# the first download (the "cache primer") keeps the serialisation window
# minimal — all subsequent downloads skip the lock and run in parallel.
_zip_lock = threading.Lock()


def parallel_get_data(
    data_source: ACSDataSource,
    states: list[str],
    label: str,
) -> pd.DataFrame:
    """
    Download ACS data for a list of states concurrently using ThreadPoolExecutor.

    Strategy
    --------
    1. The first state is downloaded under _zip_lock to ensure the ACS zip
       archive is fully written to disk before any other thread tries to read
       it.  This prevents the 'cache corruption' race condition.
    2. All remaining states are submitted to the thread pool and downloaded
       in parallel.  Downloads are I/O-bound (network + disk), so threads
       saturate bandwidth without GIL contention.
    3. Results are re-ordered to match the original `states` list order so
       that pd.concat produces a deterministic row sequence.

    Parameters
    ----------
    data_source : configured ACSDataSource
    states      : list of two-letter state abbreviations
    label       : human-readable name shown in progress output (e.g. 'northeast')

    Returns
    -------
    Concatenated DataFrame of all states, in input order.
    """
    n_workers = min(len(states), _DOWNLOAD_WORKERS)
    print(f'  > [{label}] {len(states)} states — {n_workers} workers')

    # Download the first state under the lock to prime the ACS zip cache.
    # All subsequent downloads will find the cache already populated and
    # can run concurrently without risk of corruption.
    with _zip_lock:
        first_df = _download_state(data_source, states[0])
    print(f'    - {states[0]} done (cache primed)')

    results: dict[str, pd.DataFrame] = {states[0]: first_df}
    remaining = states[1:]

    if remaining:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_download_state, data_source, s): s
                for s in remaining
            }
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

    # Concatenate in deterministic input order (not completion order).
    return pd.concat([results[s] for s in states], ignore_index=True)


def _bin_continuous_columns(df: pd.DataFrame) -> None:
    """
    Bin AGEP and WKHP in-place, running both pd.cut calls concurrently.

    pd.cut on large numeric arrays releases the GIL, so two threads can
    genuinely overlap on separate M1 P-cores.  For typical region sizes
    (100 k–1 M rows) this saves 20–60 ms versus sequential binning.
    """
    def _bin_agep() -> None:
        if 'AGEP' in df.columns:
            df['AGEP'] = pd.Categorical(pd.cut(
                pd.to_numeric(df['AGEP'], errors='coerce'),
                bins=AGEP_BINS, labels=AGEP_LABELS, right=True,
            ))
            df['AGEP'] = df['AGEP'].cat.add_categories('Unknown').fillna('Unknown')

    def _bin_wkhp() -> None:
        if 'WKHP' in df.columns:
            df['WKHP'] = pd.Categorical(pd.cut(
                pd.to_numeric(df['WKHP'], errors='coerce'),
                bins=WKHP_BINS, labels=WKHP_LABELS, right=True,
            ))
            df['WKHP'] = df['WKHP'].cat.add_categories('Unknown').fillna('Unknown')

    t_agep = threading.Thread(target=_bin_agep, daemon=True)
    t_wkhp = threading.Thread(target=_bin_wkhp, daemon=True)
    t_agep.start(); t_wkhp.start()
    t_agep.join();  t_wkhp.join()


def build_dataset(
    task: folktables.BasicProblem,
    states: list[str],
    data_source: ACSDataSource,
    label: str,
) -> pd.DataFrame:
    """
    Download and process ACS data for one region, returning a fully encoded
    feature DataFrame ready for CSV export.

    Processing steps
    ----------------
    1. parallel_get_data()          — download all states (I/O threads)
    2. task.df_to_pandas()          — adult_filter, feature selection, binarise target
    3. del raw                      — free raw DataFrame memory early
    4. apply_categorical_mappings() — decode ACS codes → labels (ProcessPoolExecutor)
    5. ST column                    — already a string abbreviation → pd.Categorical
    6. _bin_continuous_columns()    — AGEP + WKHP binned concurrently in two threads

    Memory notes
    ------------
    - raw DataFrame freed immediately after df_to_pandas() to reclaim ~100-200 MB
      before the CPU-intensive steps.
    - INCOME_ABOVE_THRESHOLD stored as int8 (8× smaller than int64).
    - All decoded columns stored as pd.Categorical (~4-8× smaller than object).

    Note: YEAR is NOT injected here.  It is inserted in main() as column 0
    immediately before writing so that it appears as the leftmost CSV column.

    Parameters
    ----------
    task        : folktables BasicProblem configured with the region's threshold
    states      : list of state abbreviations for this region
    data_source : ACSDataSource instance
    label       : display name used in progress output

    Returns
    -------
    pd.DataFrame with all features encoded and INCOME_ABOVE_THRESHOLD column.
    """
    t0 = time.perf_counter()
    print(f'\n{"=" * 70}\n{label}\n{"=" * 70}')
    raw = parallel_get_data(data_source, states, label)
    print(f'  > raw rows: {len(raw):,}  ({time.perf_counter() - t0:.1f}s)')

    # df_to_pandas applies adult_filter (age 16–90, PINCP > 100) and returns
    # (features_df, labels, group).  The 'group' column (RAC1P) is discarded.
    features_df, labels, _ = task.df_to_pandas(raw)
    del raw   # release raw data (~100-200 MB) before CPU-intensive processing

    # Attach binary income label.  int8 is sufficient (0/1) → 8× less memory than int64.
    features_df['INCOME_ABOVE_THRESHOLD'] = labels.to_numpy(dtype=np.int8)
    del labels

    # Decode numeric ACS codes → human-readable labels in parallel across P-cores.
    apply_categorical_mappings(features_df)

    # ST is already a string abbreviation (injected in _download_state).
    if 'ST' in features_df.columns:
        features_df['ST'] = pd.Categorical(features_df['ST'])

    # Bin AGEP and WKHP concurrently in two threads (pd.cut releases GIL on large arrays).
    _bin_continuous_columns(features_df)

    pos_pct = features_df['INCOME_ABOVE_THRESHOLD'].mean() * 100
    print(
        f'  > filtered rows: {len(features_df):,}  |  '
        f'positive class: {pos_pct:.1f}%  '
        f'({time.perf_counter() - t0:.1f}s)'
    )
    return features_df

# ---------------------------------------------------------------------------
# Undersampling
# ---------------------------------------------------------------------------

def undersample_to(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Stratified undersampling: reduce df to exactly n rows while preserving
    the original class ratio of INCOME_ABOVE_THRESHOLD.

    Why stratified?
    ---------------
    Simple random sampling on an imbalanced dataset could accidentally shift
    the class ratio significantly at small n.  Stratified sampling guarantees
    the same proportion of positive/negative examples in the reduced set,
    keeping the classification problem difficulty constant across regions.

    Per-class allocation uses proportional rounding; any rounding remainder
    is added to the majority class (class 0) to ensure the total is exactly n.

    Returns df unchanged if len(df) <= n (no upsampling is performed).

    Parameters
    ----------
    df   : DataFrame containing INCOME_ABOVE_THRESHOLD column
    n    : target row count
    seed : random seed for reproducible sampling
    """
    if len(df) <= n:
        return df   # already at or below target size — nothing to do

    target_col = 'INCOME_ABOVE_THRESHOLD'
    groups     = [g for _, g in df.groupby(target_col, observed=False)]

    # Compute per-class sample counts proportional to class frequencies.
    per_class = [max(1, round(len(g) / len(df) * n)) for g in groups]

    # Correct for rounding errors so the total is exactly n.
    # Apply remainder to the largest group to avoid accidentally shrinking
    # a minority class — do not assume class 0 is always the majority.
    largest_idx = max(range(len(groups)), key=lambda i: len(groups[i]))
    per_class[largest_idx] += n - sum(per_class)

    # Sample each class independently then shuffle the combined result.
    sampled = pd.concat(
        [g.sample(k, random_state=seed) for g, k in zip(groups, per_class)],
        ignore_index=True,
    )
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    survey_year: str                      = '2024',
    horizon: str                          = '1-Year',
    random_seed: int                      = 42,
    output_dir: str                       = 'data',
    income_threshold_northeast: int       = 110_000,
    income_threshold_south: int           = 90_000,
    income_threshold_usa: int             = 100_000,
    regions_to_build: list[str] | None    = None,
) -> None:
    """
    Download ACS data for the specified regions, build processed datasets,
    and save them as CSV files under output_dir.

    Sampling note
    -------------
    When both 'northeast' and 'south' are present, the South dataset is
    stratified-undersampled to match the Northeast row count (seed=random_seed).
    The 'usa' dataset, if requested, is written at its full post-filter size
    and is intentionally not undersampled — it is meant for independent
    national-level analysis.

    Parameters
    ----------
    survey_year                  ACS survey year (e.g. '2024').
    horizon                      Survey horizon ('1-Year' or '5-Year').
    random_seed                  Random seed for reproducibility.
    output_dir                   Directory where CSV files are written.
    income_threshold_northeast   Annual income threshold (USD) for the Northeast positive class.
    income_threshold_south       Annual income threshold (USD) for the South positive class.
    income_threshold_usa         Annual income threshold (USD) for the USA positive class.
    regions_to_build             List of region names to build; defaults to ['northeast', 'south'].
    """
    t0 = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    # ACSDataSource is the folktables entry point; it caches downloaded zip files
    # locally so repeated runs skip the network download entirely.
    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey='person')

    if regions_to_build is None:
        regions_to_build = ['northeast', 'south']   # default: NE/South comparison

    print(f'\n{"=" * 70}')
    print(f'ACS INCOME DATASET — {survey_year}')
    print(f'{"=" * 70}')

    # Each region config pairs its state list with its income threshold.
    # Using a dict keyed by region name avoids passing positional arguments
    # and makes it trivial to add new regions without changing call sites.
    region_configs = {
        'northeast': {'states': NORTHEAST_STATES, 'threshold': income_threshold_northeast},
        'south':     {'states': SOUTH_STATES,     'threshold': income_threshold_south},
        'usa':       {'states': USA_STATES,        'threshold': income_threshold_usa},
    }

    datasets: dict[str, pd.DataFrame] = {}

    # Build all requested regions concurrently (up to _REGION_WORKERS = 3).
    # Each region's build_dataset call itself parallelises state downloads,
    # so this is a two-level parallelism: region-level and state-level.
    with ThreadPoolExecutor(max_workers=_REGION_WORKERS) as pool:
        futures = {}
        for reg in regions_to_build:
            cfg   = region_configs[reg]
            task  = _make_income_task(cfg['threshold'])   # threshold baked into task
            label = f"{reg.capitalize()} (income > ${cfg['threshold']:,})"
            futures[reg] = pool.submit(build_dataset, task, cfg['states'], data_source, label)

        # Resolve futures in input order (not completion order) so that
        # logging is deterministic and the undersample step below finds
        # both 'northeast' and 'south' already populated.
        for reg in regions_to_build:
            datasets[reg] = futures[reg].result()

    # Undersample South to match the Northeast row count so both datasets are
    # the same size for a fair cross-regional comparison.
    # USA is deliberately excluded: it is used for independent national analysis
    # and must not be artificially reduced.
    if 'northeast' in datasets and 'south' in datasets:
        print('\n  > Undersampling South to match Northeast size...')
        datasets['south'] = undersample_to(
            datasets['south'], len(datasets['northeast']), seed=random_seed
        )

    print('\n  > Saving datasets...')

    # Write all CSVs in parallel (up to _REGION_WORKERS writers) to overlap disk I/O.
    # M1 Ultra's NVMe controller handles multiple concurrent writes efficiently.
    with ThreadPoolExecutor(max_workers=min(len(datasets), _REGION_WORKERS)) as pool:
        write_futs = []
        for reg, df in datasets.items():

            # Insert the survey year as the first column for longitudinal analysis.
            # col index 0 ensures it is the leftmost column in the CSV, making it
            # easy to group or filter by year when loading multi-year datasets.
            df.insert(0, 'YEAR', str(survey_year))

            out_path = os.path.join(output_dir, f'acs_income_{reg}_{survey_year}.csv')
            pos_pct  = df['INCOME_ABOVE_THRESHOLD'].mean() * 100
            print(f'  > {reg.capitalize():<10}: {len(df):>8,} rows  |  positive class: {pos_pct:.1f}%')

            # df.to_csv is submitted to the pool; result() is called below to
            # propagate any write exceptions back to the main thread.
            write_futs.append(pool.submit(df.to_csv, out_path, index=False))

        # Wait for all writes to complete and raise any exceptions.
        for w in write_futs:
            w.result()

    print(f'\n{"=" * 70}')
    print(f'Completed in {time.perf_counter() - t0:.1f}s')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    # freeze_support() is required on macOS/Windows when using ProcessPoolExecutor
    # with the 'spawn' start method.  It is a no-op on Linux and in normal runs,
    # but prevents a RuntimeError when the script is frozen (e.g. with PyInstaller)
    # or run via multiprocessing on macOS where 'spawn' is the default context.
    mp.freeze_support()
    main()