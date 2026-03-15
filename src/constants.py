"""
src/constants.py
────────────────
Shared constants for the ACS Income pipeline:

  - Predefined U.S. state groups (Census Bureau regions and divisions)
  - Full list of states supported by the ACS 1-Year survey
  - Feature set for the folktables BasicProblem
  - Default output columns
  - Bin edges and labels for continuous features (AGEP, WKHP)
  - Numeric-code → human-readable-label mappings for all categorical features
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# Predefined state groups
#
# Source: U.S. Census Bureau geographic classification
#   - 4 Census regions
#   - 9 Census divisions
#   - Convenience aliases (Sunbelt, Rust Belt, Great Plains)
#
# Alaska (AK) is intentionally excluded from every group when
# using horizon="1-Year": the Census Bureau does not publish
# 1-Year estimates for areas with fewer than 65,000 inhabitants,
# which causes folktables to download a malformed CSV and raises
# a pandas ParserError.  Use horizon="5-Year" to include Alaska.
# ─────────────────────────────────────────────────────────────
STATE_GROUPS: dict[str, list[str]] = {
    # ── 4 Census regions ─────────────────────────────────────
    "northeast": ["CT", "ME", "MA", "NH", "NJ", "NY", "PA", "RI", "VT"],
    "midwest":   ["IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE",
                  "ND", "OH", "SD", "WI"],
    "south":     ["AL", "AR", "DE", "FL", "GA", "KY", "LA", "MD",
                  "MS", "NC", "OK", "SC", "TN", "TX", "VA", "WV"],
    "west":      ["AZ", "CA", "CO", "HI", "ID", "MT", "NV", "NM",
                  "OR", "UT", "WA", "WY"],

    # ── 9 Census divisions ───────────────────────────────────
    "new_england":          ["CT", "ME", "MA", "NH", "RI", "VT"],
    "middle_atlantic":      ["NJ", "NY", "PA"],
    "east_north_central":   ["IL", "IN", "MI", "OH", "WI"],
    "west_north_central":   ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "south_atlantic":       ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "east_south_central":   ["AL", "KY", "MS", "TN"],
    "west_south_central":   ["AR", "LA", "OK", "TX"],
    "mountain":             ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "pacific":              ["CA", "HI", "OR", "WA"],

    # ── Convenience aliases ──────────────────────────────────
    "sunbelt":     ["AZ", "CA", "FL", "GA", "NM", "NV", "SC", "TX"],
    "rust_belt":   ["IL", "IN", "MI", "MO", "NY", "OH", "PA", "WI"],
    "great_plains":["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
}

# Complete list of states supported by the ACS 1-Year survey.
# Alaska (AK) is excluded — see note above.
USA_STATES: list[str] = [
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
    "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
    "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

# ─────────────────────────────────────────────────────────────
# ACS PUMS feature set for the BasicProblem
#
# NOTE — RELP vs. RELSHIPP column naming:
#   The Census Bureau renamed the household-relationship variable
#   between survey years:
#     years ≤ 2018  →  raw PUMS column "RELP"
#     years ≥ 2019  →  raw PUMS column "RELSHIPP"
#   The official folktables source (acs.py) always references "RELP"
#   in its BasicProblem feature list, but does NOT rename the column
#   automatically.  This pipeline handles the discrepancy by calling
#   normalize_raw_columns() before df_to_pandas(), which renames
#   RELSHIPP → RELP on the raw DataFrame when necessary.
#
# PINCP is the prediction target, not a feature.
# ─────────────────────────────────────────────────────────────
INCOME_FEATURES: list[str] = [
    "AGEP",   # Age (continuous → binned into career-stage categories)
    "COW",    # Class of worker
    "SCHL",   # Educational attainment (24 levels)
    "MAR",    # Marital status
    "OCCP",   # Occupation code (SOC-based)
    "POBP",   # Place of birth (state or country code)
    "RELP",   # Relationship to household reference person
    "WKHP",   # Usual hours worked per week (continuous → binned)
    "SEX",    # Sex
    "RAC1P",  # Race (also used as the sensitive-attribute group)
]

# Feature columns retained in the final output CSV by default.
# Pass --columns ALL to keep every feature, or supply a custom list.
DEFAULT_COLUMNS: list[str] = ["COW", "SCHL", "WKHP"]

# ─────────────────────────────────────────────────────────────
# Binning configuration for continuous features
# ─────────────────────────────────────────────────────────────

# AGEP — six career-stage bands; upper bound of 200 absorbs outliers
# without producing NaN values.
AGEP_BINS: list[int] = [0, 24, 34, 44, 54, 64, 200]
AGEP_LABELS: list[str] = [
    "Young",           # (0–24]   student / entry-level (adult_filter ensures age ≥ 16)
    "Young-Adult",     # (25–34]  early career
    "Mid-Career",      # (35–44]  peak productivity
    "Experienced",     # (45–54]  senior contributor
    "Late-Career",     # (55–64]  approaching retirement
    "Retirement-Age",  # (65–200] beyond standard retirement age
]

# WKHP — six bands; 40 hours is the U.S. statutory full-time threshold.
# The Near-Full-Time band (35–39 hrs) captures workers below the
# benefits-eligibility threshold, which is policy-relevant.
WKHP_BINS: list[int] = [0, 19, 34, 39, 40, 49, 200]
WKHP_LABELS: list[str] = [
    "Part-Time-Low",    # (0–19]   marginal labor-force attachment
    "Part-Time",        # (20–34]  standard part-time
    "Near-Full-Time",   # (35–39]  benefits-threshold zone
    "Full-Time",        # [40]     standard full-time
    "Over-Full-Time",   # (41–49]  moderate overtime
    "Extended-Hours",   # (50–200] heavy overtime / multiple jobs
]

# ─────────────────────────────────────────────────────────────
# Numeric-code → human-readable-label mappings
#
# Dash-separated labels are intentional: downstream association-rule
# itemsets are stored as strings, and spaces within labels would
# make CSV parsing ambiguous.
#
# Unmapped codes fall back to the values in COLUMN_FALLBACKS.
# ─────────────────────────────────────────────────────────────

# COW — Class of Worker
COW_MAP: dict[str, str] = {
    "1": "Employee-Private-For-Profit",
    "2": "Employee-Private-Non-Profit",
    "3": "Local-Government-Employee",
    "4": "State-Government-Employee",
    "5": "Federal-Government-Employee",
    "6": "Self-Employed-Not-Incorporated",
    "7": "Self-Employed-Incorporated",
    "8": "Unpaid-Family-Worker",
    "9": "Unemployed-5plus-Years-Or-Never-Worked",
}

# SCHL — Educational Attainment (24 levels)
SCHL_MAP: dict[str, str] = {
    "1":  "No-Schooling-Completed",
    "2":  "Nursery-School-Preschool",
    "3":  "Kindergarten",
    "4":  "Grade-1",
    "5":  "Grade-2",
    "6":  "Grade-3",
    "7":  "Grade-4",
    "8":  "Grade-5",
    "9":  "Grade-6",
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
    "24": "Doctorate-Degree",
}

# MAR — Marital Status
MAR_MAP: dict[str, str] = {
    "1": "Married",
    "2": "Widowed",
    "3": "Divorced",
    "4": "Separated",
    "5": "Never-Married-Or-Under-15",
}

# RELP — Relationship to Household Reference Person
#
# Unified map covering both ACS coding schemes:
#   Codes 0–17  : RELP scheme used in years ≤ 2018
#   Codes 20–38 : RELSHIPP scheme used in years ≥ 2019
# After normalize_raw_columns() renames RELSHIPP → RELP on the raw
# DataFrame, df_to_pandas() populates this column with whichever set
# of numeric codes the survey year actually contains.
RELP_MAP: dict[str, str] = {
    # RELP codes (years ≤ 2018)
    "0":  "Reference-Person",
    "1":  "Husband-Wife-Spouse",
    "2":  "Biological-Son-Or-Daughter",
    "3":  "Adopted-Son-Or-Daughter",
    "4":  "Stepson-Or-Stepdaughter",
    "5":  "Brother-Or-Sister",
    "6":  "Father-Or-Mother",
    "7":  "Grandchild",
    "8":  "Parent-In-Law",
    "9":  "Son-In-Law-Or-Daughter-In-Law",
    "10": "Other-Relative",
    "11": "Roomer-Or-Boarder",
    "12": "Housemate-Or-Roommate",
    "13": "Unmarried-Partner",
    "14": "Foster-Child",
    "15": "Other-Nonrelative",
    "16": "Institutionalized-Group-Quarters",
    "17": "Noninstitutionalized-Group-Quarters",
    # RELSHIPP codes (years ≥ 2019) — accessed under the renamed "RELP" column
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
    "38": "Noninstitutionalized-Group-Quarters",
}

# SEX — Sex
SEX_MAP: dict[str, str] = {
    "1": "Male",
    "2": "Female",
}

# RAC1P — Race (single-race categories)
RAC1P_MAP: dict[str, str] = {
    "1": "White-Alone",
    "2": "Black-Or-African-American-Alone",
    "3": "American-Indian-Alone",
    "4": "Alaska-Native-Alone",
    "5": "American-Indian-And-Alaska-Native-Tribes",
    "6": "Asian-Alone",
    "7": "Native-Hawaiian-And-Other-Pacific-Islander-Alone",
    "8": "Some-Other-Race-Alone",
    "9": "Two-Or-More-Races",
}

# POBP — Place of Birth (state FIPS codes + country codes)
POBP_MAP: dict[str, str] = {
    "1":   "Alabama",         "2":   "Alaska",           "4":   "Arizona",
    "5":   "Arkansas",        "6":   "California",       "8":   "Colorado",
    "9":   "Connecticut",     "10":  "Delaware",          "11":  "DC",
    "12":  "Florida",         "13":  "Georgia",           "15":  "Hawaii",
    "16":  "Idaho",           "17":  "Illinois",          "18":  "Indiana",
    "19":  "Iowa",            "20":  "Kansas",            "21":  "Kentucky",
    "22":  "Louisiana",       "23":  "Maine",             "24":  "Maryland",
    "25":  "Massachusetts",   "26":  "Michigan",          "27":  "Minnesota",
    "28":  "Mississippi",     "29":  "Missouri",          "30":  "Montana",
    "31":  "Nebraska",        "32":  "Nevada",            "33":  "New-Hampshire",
    "34":  "New-Jersey",      "35":  "New-Mexico",        "36":  "New-York",
    "37":  "North-Carolina",  "38":  "North-Dakota",      "39":  "Ohio",
    "40":  "Oklahoma",        "41":  "Oregon",            "42":  "Pennsylvania",
    "44":  "Rhode-Island",    "45":  "South-Carolina",    "46":  "South-Dakota",
    "47":  "Tennessee",       "48":  "Texas",             "49":  "Utah",
    "50":  "Vermont",         "51":  "Virginia",          "53":  "Washington",
    "54":  "West-Virginia",   "55":  "Wisconsin",         "56":  "Wyoming",
    "72":  "Puerto-Rico",     "100": "Born-Abroad-US-Parents",
    "301": "Cuba",            "302": "Jamaica",           "303": "Dominican-Republic",
    "308": "Haiti",           "313": "Other-Caribbean",   "400": "Mexico",
    "414": "Guatemala",       "416": "Honduras",          "417": "El-Salvador",
    "422": "Nicaragua",       "423": "Panama",            "424": "Other-Central-America",
    "501": "Colombia",        "507": "Peru",              "508": "Brazil",
    "516": "Venezuela",       "523": "Other-South-America","600": "Armenia",
    "601": "China",           "603": "India",             "607": "Japan",
    "613": "Philippines",     "615": "South-Korea",       "618": "Vietnam",
    "619": "Other-Southeast-Asia","620": "Other-Asia",    "700": "United-Kingdom",
    "703": "Germany",         "706": "Greece",            "708": "Ireland",
    "710": "Italy",           "714": "Poland",            "716": "Portugal",
    "720": "Russia",          "724": "Ukraine",           "730": "Other-Europe",
    "800": "Nigeria",         "803": "Ethiopia",          "804": "Egypt",
    "820": "Other-Africa",    "900": "Canada",            "999": "Other-NEC",
}

# OCCP — Occupation (SOC-based codes, ~80 most common entries)
# Codes not present in this map fall back to COLUMN_FALLBACKS["OCCP"].
OCCP_MAP: dict[str, str] = {
    "0":    "Not-In-Labor-Force-Or-Under-16",
    "10":   "Chief-Executives",
    "20":   "General-Operations-Managers",         "120":  "Financial-Managers",
    "136":  "HR-Managers",                          "220":  "Advertising-And-Marketing-Managers",
    "300":  "Purchasing-Managers",                  "310":  "Transportation-Managers",
    "330":  "Food-Service-Managers",                "410":  "Medical-And-Health-Services-Managers",
    "430":  "Construction-Managers",                "440":  "Other-Managers",
    "500":  "Agents-Of-Performing-Arts",            "510":  "Compliance-Officers",
    "520":  "Cost-Estimators",                      "530":  "Human-Resources-Workers",
    "540":  "Training-And-Development-Specialists", "565":  "Logisticians",
    "600":  "Accountants-And-Auditors",             "630":  "Budget-Analysts",
    "640":  "Credit-Analysts",                      "650":  "Financial-Analysts",
    "700":  "Management-Analysts",                  "726":  "Market-Research-Analysts",
    "740":  "Business-Operations-Specialists",      "800":  "Buyers-And-Purchasing-Agents",
    "840":  "Claims-Adjusters",
    "1005": "Computer-And-Info-Research-Scientists","1006": "Computer-Systems-Analysts",
    "1007": "Information-Security-Analysts",        "1010": "Computer-Programmers",
    "1021": "Software-Developers",                  "1022": "Software-Quality-Assurance",
    "1031": "Web-Developers",                       "1032": "Web-And-Digital-Interface-Designers",
    "1050": "Database-Administrators",              "1065": "Network-And-Computer-Systems-Admins",
    "1100": "Computer-Support-Specialists",         "1200": "Actuaries",
    "1220": "Operations-Research-Analysts",         "1230": "Statisticians",
    "1240": "Data-Scientists",
    "2100": "Lawyers",           "2105": "Judicial-Law-Clerks",    "2110": "Judges-And-Magistrates",
    "2310": "Elementary-School-Teachers",  "2320": "Middle-School-Teachers",
    "2330": "Secondary-School-Teachers",   "2540": "Special-Education-Teachers",
    "2550": "Other-Teachers",              "2560": "Tutors-And-Instructors",
    "2630": "Postsecondary-Teachers",      "2640": "Preschool-And-Kindergarten-Teachers",
    "2720": "Art-Directors",     "2740": "Graphic-Designers",      "2750": "Interior-Designers",
    "3010": "Chiropractors",     "3050": "Dietitians-And-Nutritionists",
    "3090": "Emergency-Medical-Technicians","3100": "Exercise-Physiologists",
    "3130": "Pharmacists",       "3160": "Physical-Therapists",    "3230": "Physicians-And-Surgeons",
    "3250": "Registered-Nurses", "3255": "Nurse-Practitioners",    "3260": "Occupational-Therapists",
    "3300": "Dentists",          "3420": "Dental-Assistants",      "3500": "Licensed-Practical-Nurses",
    "3600": "Medical-Assistants",
    "4000": "Cooks-Restaurant",        "4020": "Food-Preparation-Workers",  "4040": "Bartenders",
    "4055": "Fast-Food-Workers",       "4110": "Waiters-And-Waitresses",    "4120": "Dining-Room-Attendants",
    "4140": "Dishwashers",             "4220": "Janitors-And-Cleaners",     "4230": "Maids-And-Housekeeping",
    "4700": "First-Line-Retail-Supervisors","4720": "Cashiers",
    "4740": "Counter-And-Rental-Clerks",   "4760": "Retail-Salespersons",
    "4800": "Insurance-Sales-Agents",      "4810": "Securities-And-Financial-Sales",
    "4820": "Real-Estate-Brokers-And-Agents","4840": "Telemarketers",
    "4850": "Sales-Representatives",
    "5000": "First-Line-Office-Supervisors","5110": "Receptionists",
    "5120": "Information-Clerks",           "5160": "Customer-Service-Representatives",
    "5230": "Payroll-And-Timekeeping-Clerks","5240": "Human-Resources-Assistants",
    "5260": "Eligibility-Interviewers",     "5420": "Postal-Service-Workers",
    "5600": "Production-Planning-Clerks",   "5700": "Secretaries-And-Admin-Assistants",
    "5820": "Data-Entry-Keyers",
    "9600": "Cleaners-Of-Vehicles-And-Equipment",  "9620": "Laborers-And-Material-Movers",
    "9800": "Military-Officer-Special-Operations", "9810": "Military-First-Line-Supervisors",
    "9820": "Military-Enlisted-Tactical-Operations","9830": "Military-Rank-Not-Specified",
}

# Fallback labels for codes not present in the respective maps above.
COLUMN_FALLBACKS: dict[str, str] = {
    "OCCP": "Other-Occupation",
    "POBP": "Other-NEC",
    "RELP": "Unknown",
}
