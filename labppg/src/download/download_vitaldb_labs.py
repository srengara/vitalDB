"""
Download VitalDB Laboratory Results (Labs.csv) with Blood Glucose Data

This script downloads the laboratory results file from VitalDB which contains
the 35,358 blood glucose (BGL) time-series records mentioned in the paper.

The Labs.csv file contains:
- 34 laboratory parameters
- Time-series measurements (before, during, and after surgery)
- Blood glucose levels (BGL) with timestamps

Usage:
    python download_vitaldb_labs.py --output_dir ./data/labs

References:
- VitalDB Paper: https://www.nature.com/articles/s41597-022-01411-5
- VitalDB Dataset: https://vitaldb.net/dataset/
"""

import os
import sys
import argparse
import pandas as pd
import requests
from tqdm import tqdm


def download_file(url, output_path):
    """
    Download file with progress bar.

    Parameters:
    -----------
    url : str
        URL to download from
    output_path : str
        Local path to save file
    """
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"[OK] Downloaded: {output_path}")


def download_vitaldb_labs(output_dir='./data/labs'):
    """
    Download VitalDB laboratory results file.

    The Labs.csv file contains time-series laboratory measurements including
    blood glucose levels (BGL) for surgical patients.

    Parameters:
    -----------
    output_dir : str
        Directory to save the downloaded file
    """
    print("=" * 80)
    print("DOWNLOADING VITALDB LABORATORY RESULTS")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # VitalDB Labs.csv URL (from PhysioNet or VitalDB OSF repository)
    # Try multiple sources
    urls = [
        # PhysioNet (official mirror)
        "https://physionet.org/files/vitaldb/1.0.0/Labs.csv",
        # Alternative: Open Science Framework
        "https://osf.io/r6yag/download"  # This may need to be updated
    ]

    output_file = os.path.join(output_dir, 'Labs.csv')

    # Try each URL until one works
    for url in urls:
        try:
            print(f"\nAttempting to download from: {url}")
            download_file(url, output_file)
            break
        except Exception as e:
            print(f"[ERROR] Failed to download from {url}: {e}")
            continue
    else:
        print("\n[ERROR] Failed to download from all sources.")
        print("\nMANUAL DOWNLOAD INSTRUCTIONS:")
        print("1. Go to https://vitaldb.net/dataset/")
        print("2. Find and download 'Labs.csv' or 'lab_results.csv'")
        print(f"3. Save it to: {output_file}")
        print("\nOR")
        print("1. Go to https://physionet.org/content/vitaldb/1.0.0/")
        print("2. Download 'Labs.csv'")
        print(f"3. Save it to: {output_file}")
        return None

    # Load and analyze the labs data
    print("\n" + "=" * 80)
    print("ANALYZING LABORATORY RESULTS")
    print("=" * 80)

    try:
        labs_df = pd.read_csv(output_file)

        print(f"\nLabs.csv Statistics:")
        print(f"  Total rows: {len(labs_df):,}")
        print(f"  Columns: {labs_df.columns.tolist()}")
        print(f"  Unique cases: {labs_df['caseid'].nunique() if 'caseid' in labs_df.columns else 'N/A'}")

        # Look for glucose-related columns
        glucose_cols = [col for col in labs_df.columns if 'glu' in col.lower() or 'bgl' in col.lower()]
        print(f"\nGlucose-related columns: {glucose_cols}")

        if glucose_cols:
            for col in glucose_cols:
                glucose_data = labs_df[col].dropna()
                print(f"\n{col} Statistics:")
                print(f"  Non-null records: {len(glucose_data):,}")
                print(f"  Mean: {glucose_data.mean():.2f} mg/dL")
                print(f"  Range: {glucose_data.min():.2f} - {glucose_data.max():.2f} mg/dL")

        # Show first few rows
        print("\nFirst 10 rows:")
        print(labs_df.head(10))

        return labs_df

    except Exception as e:
        print(f"[ERROR] Failed to analyze Labs.csv: {e}")
        return None


def extract_glucose_for_case(labs_df, caseid):
    """
    Extract blood glucose time-series for a specific case.

    Parameters:
    -----------
    labs_df : pd.DataFrame
        Laboratory results dataframe
    caseid : int
        Case ID to extract

    Returns:
    --------
    glucose_df : pd.DataFrame
        DataFrame with timestamp and glucose values
    """
    if labs_df is None:
        return None

    case_labs = labs_df[labs_df['caseid'] == caseid]

    # Find glucose column
    glucose_cols = [col for col in case_labs.columns if 'glu' in col.lower() or 'bgl' in col.lower()]

    if not glucose_cols:
        print(f"[ERROR] No glucose column found for case {caseid}")
        return None

    glucose_col = glucose_cols[0]

    # Extract timestamp and glucose
    if 'time' in case_labs.columns:
        glucose_df = case_labs[['time', glucose_col]].dropna()
        glucose_df.columns = ['timestamp', 'glucose_mg_dl']
    else:
        glucose_df = case_labs[[glucose_col]].dropna()
        glucose_df.columns = ['glucose_mg_dl']
        glucose_df['timestamp'] = glucose_df.index

    return glucose_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download VitalDB laboratory results with blood glucose data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/labs',
        help='Output directory for Labs.csv (default: ./data/labs)'
    )

    parser.add_argument(
        '--caseid',
        type=int,
        default=None,
        help='Extract glucose data for specific case ID'
    )

    args = parser.parse_args()

    # Download labs data
    labs_df = download_vitaldb_labs(args.output_dir)

    # Extract glucose for specific case if requested
    if args.caseid and labs_df is not None:
        print(f"\n" + "=" * 80)
        print(f"EXTRACTING GLUCOSE FOR CASE {args.caseid}")
        print("=" * 80)

        glucose_df = extract_glucose_for_case(labs_df, args.caseid)

        if glucose_df is not None:
            output_file = os.path.join(args.output_dir, f'case_{args.caseid}_glucose.csv')
            glucose_df.to_csv(output_file, index=False)
            print(f"\n[OK] Saved glucose data to: {output_file}")
            print(f"  Records: {len(glucose_df)}")
            print(f"\nFirst 10 records:")
            print(glucose_df.head(10))


if __name__ == '__main__':
    main()
