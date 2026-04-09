# Data Directory

## Files

- `water_parameters.csv` - **Main dataset** (not tracked in Git)
  - Water samples with chemical parameters
  - Columns: FID, Lat, long, Town, pH, EC, TDS, NO3, Cl, SO4, TA, TH, Ca, Mg, Na, K, Iron, WQI
  - **Excluded from Git** via .gitignore

- `sample_data.csv` - Sample template (3 rows for reference)

## Data Format

**Required columns:**
- Chemical: `pH`, `EC`, `TDS`, `NO3`, `Cl`, `SO4`, `Ca`, `Mg`, `Na`, `Iron`
- Location (optional): `FID`, `Lat`, `long`, `Town`
- Additional (optional): `TA`, `TH`, `K`

## Setup

Place your dataset in this directory as `water_parameters.csv`.