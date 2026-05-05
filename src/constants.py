"""
src/constants.py
----------------
Shared constants for the ACS Income pipeline:

  - Predefined U.S. state groups (Census Bureau regions and divisions)
  - Full list of states supported by the ACS 1-Year survey
  - Feature set for the folktables BasicProblem
  - Default output columns
  - Bin edges and labels for continuous features (AGEP, WKHP)
  - Numeric-code -> human-readable-label mappings for all categorical features
"""

from __future__ import annotations

import bisect

# -----------------------------------------------------------------------------------------------------------------
# Predefined state groups
#
# Source: U.S. Census Bureau geographic classification
#   - 4 Census regions
#   - 9 Census divisions
#   - Convenience aliases (Sunbelt, Rust Belt, Great Plains)
#
# Alaska (AK) is intentionally excluded from every group when using horizon="1-Year": the Census Bureau does not 
# publish 1-Year estimates for areas with fewer than 65,000 inhabitants, which causes folktables to download a 
# malformed CSV and raises a pandas ParserError.  Use horizon="5-Year" to include Alaska.
# -----------------------------------------------------------------------------------------------------------------
STATE_GROUPS: dict[str, list[str]] = {
    # -- 4 Census regions -----------------------------------------------------------------------------------------
    "northeast": ["CT", "ME", "MA", "NH", "NJ", "NY", "PA", "RI", "VT"],
    "midwest": ["IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"],
    "south": ["AL", "AR", "DE", "FL", "GA", "KY", "LA", "MD", "MS", "NC", "OK", "SC", "TN", "TX", "VA", "WV"],
    "west": ["AZ", "CA", "CO", "HI", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY"],

    # -- 9 Census divisions ---------------------------------------------------------------------------------------
    "new_england": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "middle_atlantic": ["NJ", "NY", "PA"],
    "east_north_central": ["IL", "IN", "MI", "OH", "WI"],
    "west_north_central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "south_atlantic": ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "east_south_central": ["AL", "KY", "MS", "TN"],
    "west_south_central": ["AR", "LA", "OK", "TX"],
    "mountain": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "pacific": ["CA", "HI", "OR", "WA"],

    # -- Convenience aliases --------------------------------------------------------------------------------------
    "sunbelt": ["AZ", "CA", "FL", "GA", "NM", "NV", "SC", "TX"],
    "rust_belt": ["IL", "IN", "MI", "MO", "NY", "OH", "PA", "WI"],
    "great_plains": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"]
}

# Complete list of states supported by the ACS 1-Year survey.
# Alaska (AK) is excluded -- see note above.
USA_STATES: list[str] = [
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
    "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
    "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]

# -----------------------------------------------------------------------------------------------------------------
# ACS PUMS feature set for the BasicProblem
#
# NOTE -- RELP vs. RELSHIPP column naming:
#   The Census Bureau renamed the household-relationship variable
#   between survey years:
#     years <= 2018  ->  raw PUMS column "RELP"
#     years >= 2019  ->  raw PUMS column "RELSHIPP"
#   The official folktables source (acs.py) always references "RELP" in its BasicProblem feature list, but does NOT 
#   rename the column automatically.  This pipeline handles the discrepancy by calling normalize_raw_columns() 
#   before df_to_pandas(), which renames RELSHIPP -> RELP on the raw DataFrame when necessary.
#
# PINCP is the prediction target, not a feature.
# -----------------------------------------------------------------------------------------------------------------
INCOME_FEATURES: list[str] = [
    "AGEP", # Age (continuous -> binned into career-stage categories)
    "COW", # Class of worker
    "SCHL", # Educational attainment (24 levels)
    "MAR", # Marital status
    "OCCP", # Occupation code (SOC-based)
    "POBP", # Place of birth (state or country code)
    "RELP", # Relationship to household reference person
    "WKHP", # Usual hours worked per week (continuous -> binned)
    "SEX", # Sex
    "RAC1P" # Race (also used as the sensitive-attribute group)
]

# Legacy column subset -- no longer the pipeline default. The pipeline now retains ALL feature columns by default.
# Pass --columns COW SCHL WKHP to use this subset explicitly.
DEFAULT_COLUMNS: list[str] = ["COW", "SCHL", "WKHP"]

# -----------------------------------------------------------------------------------------------------------------
# Default income thresholds -- Pew Research Center formula
#
# T = 2 * M_fam / sqrt3 (upper-income boundary, household size 3)
# Source: ACS 1-year 2024, Census Bureau (ACSBR-025). Pew Research Center, Kochhar (2022). 
# Values rounded to the nearest $100.
#
# Resolution order in main.py:
#   1. --threshold CLI argument (explicit, always wins)
#   2. Recognised state group name -> group threshold below
#   3. Single state code -> per-state threshold below
#   4. Multiple state codes or ALL -> national fallback
# ------------------------------------------------------------------------------------------------------------------

# National fallback (population-weighted average across 49 states, no AK).
NATIONAL_THRESHOLD: float = 94_200.0

# Per state-group thresholds (Census regions, divisions, convenience aliases).
GROUP_THRESHOLDS: dict[str, float] = {
    # 4 Census regions
    "northeast": 100_700.0,
    "midwest": 91_100.0,
    "south": 86_000.0,
    "west": 101_400.0,
    # 9 Census divisions
    "new_england": 111_300.0,
    "middle_atlantic": 96_500.0,
    "east_north_central": 90_400.0,
    "west_north_central": 92_400.0,
    "south_atlantic": 89_800.0,
    "east_south_central": 73_700.0,
    "west_south_central": 86_300.0,
    "mountain": 93_300.0,
    "pacific": 106_000.0,
    # Convenience aliases
    "sunbelt": 92_600.0,
    "rust_belt": 90_900.0,
    "great_plains": 92_400.0
}

# Per-state thresholds (ACS 1-year 2024 family medians, T = 2 * M / sqrt3).
# Alaska (AK) is excluded from the ACS 1-year survey.
# DC is included as it appears in ACS data.
# Puerto Rico (PR) falls back to the national threshold.
STATE_THRESHOLDS: dict[str, float] = {
    "AL": 69_300.0, # Alabama
    "AZ": 95_800.0, # Arizona
    "AR": 73_000.0, # Arkansas
    "CA": 103_100.0, # California
    "CO": 111_700.0, # Colorado
    "CT": 106_400.0, # Connecticut
    "DC": 126_700.0, # District of Columbia
    "DE": 100_500.0, # Delaware
    "FL": 83_100.0, # Florida
    "GA": 83_100.0, # Georgia
    "HI": 111_200.0, # Hawaii
    "ID": 85_300.0, # Idaho
    "IL": 101_200.0, # Illinois
    "IN": 87_800.0, # Indiana
    "IA": 92_900.0, # Iowa
    "KS": 97_100.0, # Kansas
    "KY": 71_600.0, # Kentucky
    "LA": 66_400.0, # Louisiana
    "ME": 86_600.0, # Maine
    "MD": 117_900.0, # Maryland
    "MA": 122_000.0, # Massachusetts
    "MI": 88_900.0, # Michigan
    "MN": 104_000.0, # Minnesota
    "MS": 63_500.0, # Mississippi
    "MO": 90_100.0, # Missouri
    "MT": 91_700.0, # Montana
    "NE": 103_300.0, # Nebraska
    "NV": 93_500.0, # Nevada
    "NH": 114_200.0, # New Hampshire
    "NJ": 105_800.0, # New Jersey
    "NM": 69_400.0, # New Mexico
    "NY": 94_400.0, # New York
    "NC": 79_700.0, # North Carolina
    "ND": 87_300.0, # North Dakota
    "OH": 85_700.0, # Ohio
    "OK": 77_700.0, # Oklahoma
    "OR": 102_200.0, # Oregon
    "PA": 89_500.0, # Pennsylvania
    "RI": 94_700.0, # Rhode Island
    "SC": 79_800.0, # South Carolina
    "SD": 94_700.0, # South Dakota
    "TN": 84_300.0, # Tennessee
    "TX": 90_800.0, # Texas
    "UT": 116_600.0, # Utah
    "VT": 98_100.0, # Vermont
    "VA": 112_000.0, # Virginia
    "WA": 108_500.0, # Washington
    "WV": 69_300.0, # West Virginia
    "WI": 91_800.0, # Wisconsin
    "WY": 89_000.0 # Wyoming
}

# -----------------------------------------------------------------------------------------------------------------
# ACS 1-Year Margin of Error (MOE) for median family income
#
# Source: U.S. Census Bureau, ACS 1-Year 2024, Table B19113. MOE is the half-width of the 90% confidence interval 
# around the median family income estimate.  Values are approximate and rounded to the nearest $100, inversely 
# proportional to sqrt(sample size) -- large states have smaller MOE.
#
# Used to compute the default dead zone margin:
#   margin = 2 * MOE / sqrt3
# which propagates the MOE through the Pew upper-income
# formula (T = 2 * M_fam / sqrt3).
#
# Resolution order (same as thresholds):
#   1. --margin CLI argument (explicit, always wins)
#   2. Recognised group -> average of member-state MOEs / sqrtn
#   3. Single state -> per-state MOE below
#   4. Multiple states or ALL -> national fallback
# -----------------------------------------------------------------------------------------------------------------

NATIONAL_INCOME_MOE: float = 500.0

STATE_INCOME_MOE: dict[str, float] = {
    # Large states (pop > 8M): MOE ~ $800-1,200
    "CA": 900.0, "TX": 1000.0, "FL": 1100.0, "NY": 1000.0,
    # Upper-medium (pop 5-8M): MOE ~ $1,200-1,800
    "PA": 1500.0, "IL": 1400.0, "OH": 1500.0, "GA": 1600.0,
    "NC": 1600.0, "MI": 1500.0, "NJ": 1300.0, "VA": 1500.0,
    # Medium (pop 3-5M): MOE ~ $1,800-2,800
    "WA": 1900.0, "AZ": 2000.0, "MA": 1800.0, "TN": 2100.0,
    "IN": 2100.0, "MN": 2000.0, "MO": 2200.0, "MD": 1900.0,
    "WI": 2100.0, "CO": 2000.0, "SC": 2200.0, "AL": 2300.0,
    "LA": 2300.0, "KY": 2400.0, "OR": 2100.0, "OK": 2400.0,
    "CT": 2200.0, "UT": 2300.0, "IA": 2500.0, "NV": 2400.0,
    "AR": 2600.0, "MS": 2700.0, "KS": 2600.0, "NM": 2800.0,
    # Small (pop 1-2M): MOE ~ $2,800-4,200
    "NE": 3000.0, "ID": 3200.0, "WV": 3300.0, "HI": 3100.0,
    "NH": 3200.0, "ME": 3300.0, "MT": 3800.0, "RI": 3200.0,
    "DE": 3400.0, "SD": 4000.0, "ND": 4200.0,
    # Very small (pop < 1M): MOE ~ $4,000-5,500
    "VT": 4500.0, "WY": 5200.0,
    # DC (small geography, dense sampling)
    "DC": 3500.0
}


def resolve_default_margin(
    states: list[str] | None,
    raw_states_arg: list[str],
) -> float:
    """
    Compute the default dead zone margin from ACS Margin of Error data.

    The margin is the propagated MOE through the Pew upper-income formula:
        margin = 2 * MOE_median_family_income / sqrt3

    For groups of states the MOE decreases as sqrtn_states (independent samples),
    so: group_MOE = mean(member_MOEs) / sqrtn_members.

    Returns the margin in dollars.
    """
    import math

    # Single state -> per-state MOE.
    if states is not None and len(states) == 1:
        moe = STATE_INCOME_MOE.get(states[0], NATIONAL_INCOME_MOE)
        margin = 2.0 * moe / math.sqrt(3)
        return round(margin / 100) * 100  # round to nearest $100

    # Recognised group -> average member MOE / sqrtn.
    if (raw_states_arg and len(raw_states_arg) == 1
            and raw_states_arg[0].lower() in STATE_GROUPS):
        group_states = STATE_GROUPS[raw_states_arg[0].lower()]
        moes = [STATE_INCOME_MOE.get(s, NATIONAL_INCOME_MOE) for s in group_states]
        group_moe = (sum(moes) / len(moes)) / math.sqrt(len(moes))
        margin = 2.0 * group_moe / math.sqrt(3)
        return round(margin / 100) * 100

    # Multiple explicit states -> average member MOE / sqrtn.
    if states is not None and len(states) > 1:
        moes = [STATE_INCOME_MOE.get(s, NATIONAL_INCOME_MOE) for s in states]
        group_moe = (sum(moes) / len(moes)) / math.sqrt(len(moes))
        margin = 2.0 * group_moe / math.sqrt(3)
        return round(margin / 100) * 100

    # ALL states -> national MOE.
    moe = NATIONAL_INCOME_MOE
    margin = 2.0 * moe / math.sqrt(3)
    return round(margin / 100) * 100


# Binning configuration for continuous features
# -----------------------------------------------------------------------------------------------------------------

# AGEP -- six career-stage bands; upper bound of 200 absorbs outliers
# without producing NaN values.
AGEP_BINS: list[int] = [0, 24, 34, 44, 54, 64, 200]
AGEP_LABELS: list[str] = [
    "Young", # (0-24] student / entry-level (adult_filter ensures age >= 16)
    "Young-Adult", # (25-34] early career
    "Mid-Career", # (35-44] peak productivity
    "Experienced", # (45-54] senior contributor
    "Late-Career", # (55-64] approaching retirement
    "Retirement-Age" # (65-200] beyond standard retirement age
]

# WKHP -- six bands; 40 hours is the U.S. statutory full-time threshold.
# The Near-Full-Time band (35-39 hrs) captures workers below the benefits-eligibility threshold, which is 
# policy-relevant.
WKHP_BINS: list[int] = [0, 19, 34, 39, 40, 49, 200]
WKHP_LABELS: list[str] = [
    "Part-Time-Low", # (0-19] marginal labor-force attachment
    "Part-Time", # (20-34] standard part-time
    "Near-Full-Time", # (35-39] benefits-threshold zone
    "Full-Time", # [40] standard full-time
    "Over-Full-Time", # (41-49] moderate overtime
    "Extended-Hours" # (50-200] heavy overtime / multiple jobs
]

# -----------------------------------------------------------------------------------------------------------------
# Numeric-code -> human-readable-label mappings
#
# Dash-separated labels are intentional: downstream association-rule itemsets are stored as strings, and spaces 
# within labels would make CSV parsing ambiguous.
#
# Unmapped codes fall back to the values in COLUMN_FALLBACKS.
# -----------------------------------------------------------------------------------------------------------------

# COW -- Class of Worker
COW_MAP: dict[str, str] = {
    "1": "Employee-Private-For-Profit",
    "2": "Employee-Private-Non-Profit",
    "3": "Local-Government-Employee",
    "4": "State-Government-Employee",
    "5": "Federal-Government-Employee",
    "6": "Self-Employed-Not-Incorporated",
    "7": "Self-Employed-Incorporated",
    "8": "Unpaid-Family-Worker",
    "9": "Unemployed-5plus-Years-Or-Never-Worked"
}

# SCHL -- Educational Attainment (24 levels)
SCHL_MAP: dict[str, str] = {
    "1": "No-Schooling-Completed",
    "2": "Nursery-School-Preschool",
    "3": "Kindergarten",
    "4": "Grade-1",
    "5": "Grade-2",
    "6": "Grade-3",
    "7": "Grade-4",
    "8": "Grade-5",
    "9": "Grade-6",
    "10": "Grade-7",
    "11": "Grade-8",
    "12": "Grade-9",
    "13": "Grade-10",
    "14": "Grade-11",
    "15": "Grade-12-No-Diploma",
    "16": "Regular-HS-Diploma",
    "17": "GED-Or-Alt-Credential",
    "18": "Some-College-Less-Than-1yr",
    "19": "Some-College-1yr-Or-More-No-Degree",
    "20": "Associates-Degree",
    "21": "Bachelors-Degree",
    "22": "Masters-Degree",
    "23": "Professional-Degree-Beyond-Bachelors",
    "24": "Doctorate-Degree"
}

# MAR -- Marital Status
MAR_MAP: dict[str, str] = {
    "1": "Married",
    "2": "Widowed",
    "3": "Divorced",
    "4": "Separated",
    "5": "Never-Married-Or-Under-15"
}

# RELP -- Relationship to Household Reference Person
#
# Unified map covering both ACS coding schemes:
#   Codes 0-17  : RELP scheme used in years <= 2018
#   Codes 20-38 : RELSHIPP scheme used in years >= 2019
# After normalize_raw_columns() renames RELSHIPP -> RELP on the raw DataFrame, df_to_pandas() populates 
# this column with whichever set of numeric codes the survey year actually contains.
RELP_MAP: dict[str, str] = {
    # RELP codes (years <= 2018)
    "0": "Reference-Person",
    "1": "Husband-Wife-Spouse",
    "2": "Biological-Son-Or-Daughter",
    "3": "Adopted-Son-Or-Daughter",
    "4": "Stepson-Or-Stepdaughter",
    "5": "Brother-Or-Sister",
    "6": "Father-Or-Mother",
    "7": "Grandchild",
    "8": "Parent-In-Law",
    "9": "Son-In-Law-Or-Daughter-In-Law",
    "10": "Other-Relative",
    "11": "Roomer-Or-Boarder",
    "12": "Housemate-Or-Roommate",
    "13": "Unmarried-Partner",
    "14": "Foster-Child",
    "15": "Other-Nonrelative",
    "16": "Institutionalized-Group-Quarters",
    "17": "Noninstitutionalized-Group-Quarters",
    # RELSHIPP codes (years >= 2019) -- accessed under the renamed "RELP" column
    "20": "Reference-Person",
    "21": "Opposite-Sex-Husband-Wife-Spouse",
    "22": "Opposite-Sex-Unmarried-Partner",
    "23": "Same-Sex-Husband-Wife-Spouse",
    "24": "Same-Sex-Unmarried-Partner",
    "25": "Biological-Son-Or-Daughter",
    "26": "Adopted-Son-Or-Daughter",
    "27": "Stepson-Or-Stepdaughter",
    "28": "Brother-Or-Sister",
    "29": "Father-Or-Mother",
    "30": "Grandchild",
    "31": "Parent-In-Law",
    "32": "Son-In-Law-Or-Daughter-In-Law",
    "33": "Other-Relative",
    "34": "Roommate-Or-Housemate",
    "35": "Foster-Child",
    "36": "Other-Nonrelative",
    "37": "Institutionalized-Group-Quarters",
    "38": "Noninstitutionalized-Group-Quarters"
}

# SEX -- Sex
SEX_MAP: dict[str, str] = {
    "1": "Male",
    "2": "Female"
}

# RAC1P -- Race (single-race categories)
RAC1P_MAP: dict[str, str] = {
    "1": "White-Alone",
    "2": "Black-Or-African-American-Alone",
    "3": "American-Indian-Alone",
    "4": "Alaska-Native-Alone",
    "5": "American-Indian-And-Alaska-Native-Tribes",
    "6": "Asian-Alone",
    "7": "Native-Hawaiian-And-Other-Pacific-Islander-Alone",
    "8": "Some-Other-Race-Alone",
    "9": "Two-Or-More-Races"
}

# POBP -- Place of Birth (state FIPS codes + country codes)
POBP_MAP: dict[str, str] = {
    "1": "Alabama", 
    "2": "Alaska", 
    "4": "Arizona",
    "5": "Arkansas", 
    "6": "California", 
    "8": "Colorado",
    "9": "Connecticut", 
    "10": "Delaware", 
    "11": "DC",
    "12": "Florida", 
    "13": "Georgia", 
    "15": "Hawaii",
    "16": "Idaho", 
    "17": "Illinois", 
    "18": "Indiana",
    "19": "Iowa", 
    "20": "Kansas", "21": "Kentucky",
    "22": "Louisiana", 
    "23": "Maine", 
    "24": "Maryland",
    "25": "Massachusetts", 
    "26": "Michigan", 
    "27": "Minnesota",
    "28": "Mississippi", 
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New-Hampshire",
    "34": "New-Jersey",
    "35": "New-Mexico",
    "36": "New-York",
    "37": "North-Carolina", 
    "38": "North-Dakota", 
    "39": "Ohio",
    "40": "Oklahoma", 
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode-Island",
    "45": "South-Carolina",
    "46": "South-Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont", 
    "51": "Virginia", 
    "53": "Washington",
    "54": "West-Virginia", 
    "55": "Wisconsin",
    "56": "Wyoming",
    "72": "Puerto-Rico",     
    "100": "Born-Abroad-US-Parents",
    "301": "Cuba",
    "302": "Jamaica",
    "303": "Dominican-Republic",
    "308": "Haiti",
    "313": "Other-Caribbean",   
    "400": "Mexico",
    "414": "Guatemala",       
    "416": "Honduras",          
    "417": "El-Salvador",
    "422": "Nicaragua",       
    "423": "Panama",            
    "424": "Other-Central-America",
    "501": "Colombia",        
    "507": "Peru",              
    "508": "Brazil",
    "516": "Venezuela",       
    "523": "Other-South-America",
    "600": "Armenia",
    "601": "China",           
    "603": "India",             
    "607": "Japan",
    "613": "Philippines",     
    "615": "South-Korea",       
    "618": "Vietnam",
    "619": "Other-Southeast-Asia",
    "620": "Other-Asia",    
    "700": "United-Kingdom",
    "703": "Germany",         
    "706": "Greece",            
    "708": "Ireland",
    "710": "Italy",           
    "714": "Poland",            
    "716": "Portugal",
    "720": "Russia",          
    "724": "Ukraine",           
    "730": "Other-Europe",
    "800": "Nigeria",         
    "803": "Ethiopia",          
    "804": "Egypt",
    "820": "Other-Africa",    
    "900": "Canada",            
    "999": "Other-NEC"
}

# -----------------------------------------------------------------------------------------------------------------
# OCCP -- Occupation major-group classification
#
# ACS PUMS OCCP codes are 4-digit integers derived from the Census Bureau's occupation classification, which is 
# itself based on the Standard Occupational Classification (SOC) system.  The first two digits of each SOC code 
# identify its major group, and the ACS PUMS numeric codes are allocated in contiguous ranges that correspond directly 
# to those major groups.
#
# Source: BLS Occupational Employment and Wage Statistics (OEWS) structure -- 
# https://www.bls.gov/oes/2023/may/oes_stru.htm 23 major groups (SOC codes 11-0000 through 55-0000).
#
# Mapping strategy:
#   Each entry is (inclusive_lower, inclusive_upper, label). occp_to_major_group() iterates these ranges in order 
#   and returns the label of the first matching range.  Codes outside every range fall back to 
#   "Not-In-Labor-Force-Or-Unclassified". The ranges are derived from the 2018 Census Occupation Code List
#   crosswalk to the 2018 SOC (used by ACS 2018 onward).
# -----------------------------------------------------------------------------------------------------------------

# Each tuple: (lower_bound, upper_bound, BLS_major_group_label)
# Bounds are inclusive on both ends.
OCCP_MAJOR_GROUP_RANGES: list[tuple[int, int, str]] = [
    # SOC 11-0000 -- Management Occupations
    (10, 440, "Management"),
    # SOC 13-0000 -- Business and Financial Operations Occupations
    (500, 960, "Business-And-Financial-Operations"),
    # SOC 15-0000 -- Computer and Mathematical Occupations
    (1005, 1240, "Computer-And-Mathematical"),
    # SOC 17-0000 -- Architecture and Engineering Occupations
    (1300, 1560, "Architecture-And-Engineering"),
    # SOC 19-0000 -- Life, Physical, and Social Science Occupations
    (1600, 1980, "Life-Physical-And-Social-Science"),
    # SOC 21-0000 -- Community and Social Service Occupations
    (2000, 2060, "Community-And-Social-Service"),
    # SOC 23-0000 -- Legal Occupations
    (2100, 2160, "Legal"),
    # SOC 25-0000 -- Educational Instruction and Library Occupations
    (2200, 2550, "Educational-Instruction-And-Library"),
    # SOC 27-0000 -- Arts, Design, Entertainment, Sports, and Media Occupations
    (2600, 2960, "Arts-Design-Entertainment-Sports-And-Media"),
    # SOC 29-0000 -- Healthcare Practitioners and Technical Occupations
    (3000, 3540, "Healthcare-Practitioners-And-Technical"),
    # SOC 31-0000 -- Healthcare Support Occupations
    (3600, 3655, "Healthcare-Support"),
    # SOC 33-0000 -- Protective Service Occupations
    (3700, 3960, "Protective-Service"),
    # SOC 35-0000 -- Food Preparation and Serving Related Occupations
    (4000, 4160, "Food-Preparation-And-Serving"),
    # SOC 37-0000 -- Building and Grounds Cleaning and Maintenance Occupations
    (4200, 4255, "Building-And-Grounds-Cleaning-And-Maintenance"),
    # SOC 39-0000 -- Personal Care and Service Occupations
    (4330, 4650, "Personal-Care-And-Service"),
    # SOC 41-0000 -- Sales and Related Occupations
    (4700, 4965, "Sales-And-Related"),
    # SOC 43-0000 -- Office and Administrative Support Occupations
    (5000, 5940, "Office-And-Administrative-Support"),
    # SOC 45-0000 -- Farming, Fishing, and Forestry Occupations
    (6005, 6130, "Farming-Fishing-And-Forestry"),
    # SOC 47-0000 -- Construction and Extraction Occupations
    (6200, 6765, "Construction-And-Extraction"),
    # SOC 49-0000 -- Installation, Maintenance, and Repair Occupations
    (6800, 7630, "Installation-Maintenance-And-Repair"),
    # SOC 51-0000 -- Production Occupations
    (7700, 8990, "Production"),
    # SOC 53-0000 -- Transportation and Material Moving Occupations
    (9005, 9760, "Transportation-And-Material-Moving"),
    # SOC 55-0000 -- Military Specific Occupations
    (9800, 9830, "Military-Specific")
]

# Label applied to code 0 (not in labor force / never worked) and to any code that does not 
# fall within the ranges above.
OCCP_FALLBACK = "Not-In-Labor-Force-Or-Unclassified"


def occp_to_major_group(code: int) -> str:
    """
    Map a single integer ACS PUMS OCCP code to its BLS OEWS major-group label.

    Uses bisect for O(log N) lookup instead of a linear scan -- relevant
    because this function is called via Series.map() on datasets that can
    exceed 1 M rows.

    Parameters
    ----------
    code : Integer occupation code from the ACS PUMS OCCP field.

    Returns
    -------
    str
        BLS major-group label (e.g. "Management", "Healthcare-Practitioners-
        And-Technical"), or OCCP_FALLBACK for code 0 and unmapped codes.
    """
    if code <= 0:
        return OCCP_FALLBACK
    idx = bisect.bisect_right(_OCCP_LOWER_BOUNDS, code) - 1
    if idx >= 0:
        lower, upper, label = OCCP_MAJOR_GROUP_RANGES[idx]
        if code <= upper:
            return label
    return OCCP_FALLBACK


# Pre-computed lower-bound list for bisect lookup in occp_to_major_group.
_OCCP_LOWER_BOUNDS: list[int] = [lo for lo, _, _ in OCCP_MAJOR_GROUP_RANGES]


# Fallback labels for codes not present in the respective maps above.
COLUMN_FALLBACKS: dict[str, str] = {
    "POBP": "Other-NEC",
    "RELP": "Unknown"
}