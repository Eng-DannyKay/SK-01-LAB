import pandas as pd
import numpy as np
import re
import json
import requests
from datetime import datetime
from pathlib import Path
import logging
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CONFIG = {
    "input_dir": Path("data/raw"),
    "output_dir": Path("data/processed"),
    "crm_api_url": "https://api.shopstream.example.com/v2/customers",
    "crm_api_key": "sk-xxxx",          # Use environment variable in production
    "valid_regions": ["US", "EU", "APAC"],
    "email_regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "quality_threshold": 0.95,         # 95% of records must pass each check
    "source_priority": {"crm": 1, "website": 2, "erp": 3, "marketing": 4},
}

# Create directories
for d in [CONFIG["input_dir"], CONFIG["output_dir"]]:
    d.mkdir(parents=True, exist_ok=True)

logger.info("Step 1.1 complete: directories ready.")
logger.info(f"  Input  : {CONFIG['input_dir'].resolve()}")
logger.info(f"  Output : {CONFIG['output_dir'].resolve()}")


def generate_synthetic_data():
    np.random.seed(42)
    n = 1000

    emails_pool = [
        f"customer{i}@{'gmail' if i % 3 == 0 else 'yahoo' if i % 3 == 1 else 'company'}.com"
        for i in range(800)
    ]
    emails_pool += ["not-an-email", "missing@", "@nodomain.com", "", "double@@sign.com"]

    first_names = [
        "Maria", "Jose", "Andre", "Lea", "Francois", "Muller", "O'Brien",
        "John", "Jane", "Mike", "Sarah", "Alex", "Chris", "Pat", "Sam"
    ] * 70
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
        "Martinez", "Diaz", "Lopez", "Gonzalez", "Wang", "Kim"
    ] * 84

    regions_messy = (
        ["US", "us", "USA", "united states", "North America"] * 150 +
        ["EU", "eu", "Europe", "EMEA", "europe"] * 150 +
        ["APAC", "apac", "Asia Pacific", "Asia", "AP"] * 100 +
        [None, "", "N/A"] * 80
    )
    np.random.shuffle(regions_messy)

    phones_messy = (
        ["+1 (555) 123-4567", "555.123.4567", "5551234567",
         "+44 20 7946 0958", "020 7946 0958",
         "+81-3-1234-5678", "invalid-phone", None] * 125
    )
    np.random.shuffle(phones_messy)

    website_df = pd.DataFrame({
        "CustomerEmail": np.random.choice(emails_pool, n),
        "First Name":    [first_names[i] for i in np.random.randint(0, len(first_names), n)],
        "Last Name":     [last_names[i]  for i in np.random.randint(0, len(last_names),  n)],
        "Phone":         [phones_messy[i % len(phones_messy)] for i in range(n)],
        "Region":        [regions_messy[i % len(regions_messy)] for i in range(n)],
        "Registration Date": pd.date_range("2020-01-01", periods=n, freq="4h").strftime("%Y-%m-%d"),
        "OptOut":        np.random.choice([0, 1], n, p=[0.85, 0.15]),
    })
    test_accounts = pd.DataFrame({
        "CustomerEmail":     [f"test{i}@test.shopstream.com" for i in range(20)],
        "First Name":        ["Test"] * 20,
        "Last Name":         ["Account"] * 20,
        "Phone":             [None] * 20,
        "Region":            ["US"] * 20,
        "Registration Date": ["2023-01-01"] * 20,
        "OptOut":            [0] * 20,
    })
    website_df = pd.concat([website_df, test_accounts], ignore_index=True)
    website_df.to_csv(CONFIG["input_dir"] / "website_customers.csv",
                      index=False, encoding="iso-8859-1")
    logger.info(f"  Generated website CSV   : {len(website_df)} records")

    crm_records = []
    for i in range(n // 2):
        crm_records.append({
            "id": f"CRM-{i:06d}",
            "email": np.random.choice(emails_pool),
            "profile": {
                "first_name": first_names[np.random.randint(0, len(first_names))],
                "last_name":  last_names[np.random.randint(0, len(last_names))],
            },
            "phone":             phones_messy[i % len(phones_messy)],
            "region":            regions_messy[i % len(regions_messy)],
            "registration_date": f"202{np.random.randint(0, 4)}-{np.random.randint(1, 13):02d}-01",
            "opt_out":           bool(np.random.choice([0, 1], p=[0.85, 0.15])),
            "lifetime_value":    round(np.random.uniform(50, 5000), 2),
        })
    crm_path = CONFIG["input_dir"] / "crm_export.json"
    crm_path.write_text(json.dumps({"customers": crm_records}), encoding="utf-8")
    logger.info(f"  Generated CRM JSON      : {len(crm_records)} records")

    erp_lines = []
    for i in range(n // 4):
        email  = np.random.choice(emails_pool)
        name   = f"{first_names[i % len(first_names)]} {last_names[i % len(last_names)]}"
        phone  = str(phones_messy[i % len(phones_messy)] or "")
        region = str(regions_messy[i % len(regions_messy)] or "")
        date   = f"2019-{np.random.randint(1, 13):02d}-01"
        status = np.random.choice(["ACTIV", "INACT"])
        line = (
            f"{str(i):>10}"
            f"{name:<50}"
            f"{email:<60}"
            f"{phone:<20}"
            f"{region:<5}"
            f"{date:<10}"
            f"{status:<5}"
        )
        erp_lines.append(line)
    (CONFIG["input_dir"] / "erp_customers.txt").write_text(
        "\n".join(erp_lines), encoding="utf-8"
    )
    logger.info(f"  Generated ERP fixed-width: {len(erp_lines)} records")
    logger.info("Step 1.2 complete: synthetic data generated.")


def ingest_website_csv(filepath: Path) -> pd.DataFrame:
    logger.info(f"  Ingesting website CSV: {filepath}")

    df = pd.read_csv(
        filepath,
        encoding="iso-8859-1",
        dtype={"Phone": str},
        parse_dates=["Registration Date"],
        na_values=["", "N/A", "null", "NULL", "none", "NaN"],
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    df = df.rename(columns={
        "customeremail":     "email",
        "registration_date": "registration_date",
        "optout":            "opt_out",
    })

    test_mask = df["email"].str.contains(r"@test\.shopstream\.com$", na=False, case=False)
    logger.info(f"    Removed {test_mask.sum()} test accounts")
    df = df[~test_mask].copy()

    df["source"] = "website"
    logger.info(f"    Loaded {len(df)} records")
    return df


def ingest_crm_json(filepath: Path) -> pd.DataFrame:
    logger.info(f"  Ingesting CRM JSON: {filepath}")

    raw = json.loads(filepath.read_text(encoding="utf-8"))
    df = pd.json_normalize(raw["customers"], sep="_")

    # Flatten profile_* columns to standard names
    df = df.rename(columns={
        "profile_first_name": "first_name",
        "profile_last_name":  "last_name",
    })

    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["source"] = "crm"
    logger.info(f"    Loaded {len(df)} records")
    return df


def ingest_erp_fixed_width(filepath: Path) -> pd.DataFrame:
    logger.info(f"  Ingesting ERP fixed-width: {filepath}")

    colspecs = [(0,10),(10,60),(60,120),(120,140),(140,145),(145,155),(155,160)]
    col_names = ["customer_id","full_name","email","phone",
                 "region_code","registration_date","status"]

    df = pd.read_fwf(
        filepath,
        colspecs=colspecs,
        names=col_names,
        dtype=str,
        encoding="utf-8",
    )

    for col in df.columns:
        df[col] = df[col].str.strip()

    split = df["full_name"].str.split(n=1, expand=True)
    df["first_name"] = split[0] if 0 in split.columns else np.nan
    df["last_name"]  = split[1] if 1 in split.columns else np.nan

    df["registration_date"] = pd.to_datetime(df["registration_date"],
                                              format="%Y-%m-%d", errors="coerce")
    df["region"] = df["region_code"]
    df["source"] = "erp"
    logger.info(f"    Loaded {len(df)} records")
    return df


STANDARD_SCHEMA = [
    "email", "first_name", "last_name", "phone",
    "region", "registration_date", "opt_out", "source",
]


def align_schema(df: pd.DataFrame) -> pd.DataFrame:
    for col in STANDARD_SCHEMA:
        if col not in df.columns:
            df[col] = np.nan
    return df[STANDARD_SCHEMA].copy()


def ingest_all_sources() -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STEP 2: Data Ingestion")

    website_df = ingest_website_csv(CONFIG["input_dir"] / "website_customers.csv")
    crm_df     = ingest_crm_json(CONFIG["input_dir"] / "crm_export.json")
    erp_df     = ingest_erp_fixed_width(CONFIG["input_dir"] / "erp_customers.txt")

    combined = pd.concat(
        [align_schema(website_df), align_schema(crm_df), align_schema(erp_df)],
        ignore_index=True,
    )

    logger.info(f"  Total records combined: {len(combined)}")
    for src, count in combined["source"].value_counts().items():
        logger.info(f"    {src}: {count} records")

    return combined


def standardize_emails(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "", regex=True)
        .replace({"nan": np.nan, "none": np.nan, "": np.nan})
    )


def validate_emails(series: pd.Series) -> pd.Series:
    return series.str.match(CONFIG["email_regex"], na=False)


def standardize_phone_numbers(series: pd.Series) -> pd.Series:
    def _clean(phone):
        if pd.isna(phone) or str(phone).strip() in ("", "nan", "None"):
            return np.nan
        phone = str(phone).strip()
        has_plus = phone.startswith("+")
        digits = re.sub(r"[^\d]", "", phone)
        if len(digits) < 7:
            return np.nan
        return f"+{digits}" if has_plus else digits

    return series.apply(_clean)


def standardize_names(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
        .replace({"Nan": np.nan, "None": np.nan, "": np.nan})
    )


REGION_MAP = {
    "us": "US", "usa": "US", "united states": "US",
    "north america": "US", "na": "US", "amer": "US", "america": "US",
    "eu": "EU", "europe": "EU", "emea": "EU",
    "eur": "EU", "european union": "EU",
    "apac": "APAC", "asia": "APAC", "asia pacific": "APAC",
    "ap": "APAC", "asia-pacific": "APAC",
}


def standardize_regions(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(REGION_MAP)
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STEP 3: Cleaning & Standardization")
    df = df.copy()

    df["email_raw"] = df["email"].copy()

    df["email"]             = standardize_emails(df["email"])
    df["email_valid"]       = validate_emails(df["email"])
    df["first_name"]        = standardize_names(df["first_name"])
    df["last_name"]         = standardize_names(df["last_name"])
    df["phone"]             = standardize_phone_numbers(df["phone"])
    df["region"]            = standardize_regions(df["region"])
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")

    invalid_emails = (~df["email_valid"]).sum()
    null_regions   = df["region"].isna().sum()
    null_phones    = df["phone"].isna().sum()

    logger.info(f"  Invalid emails  : {invalid_emails}")
    logger.info(f"  Null regions    : {null_regions}  (will stay null until dedup/LLM step)")
    logger.info(f"  Null phones     : {null_phones}")
    logger.info(f"  Records cleaned : {len(df)}")
    return df


if __name__ == "__main__":
    generate_synthetic_data()
    combined = ingest_all_sources()
    cleaned  = clean_dataframe(combined)
    print(cleaned[["email", "email_valid", "phone", "region", "first_name"]].head(10).to_string())
