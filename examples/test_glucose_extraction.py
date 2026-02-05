"""
Test script for glucose extraction from VitalDB

Usage:
    python test_glucose_extraction.py --case_id 2
"""

import argparse
from glucose_extractor import GlucoseExtractor


def main():
    parser = argparse.ArgumentParser(description='Test glucose extraction from VitalDB')
    parser.add_argument('--case_id', type=int, default=2, help='VitalDB case ID (default: 2)')
    args = parser.parse_args()

    print("=" * 70)
    print("Testing Glucose Extraction from VitalDB")
    print("=" * 70)
    print(f"\nCase ID: {args.case_id}")
    print()

    # Create extractor
    extractor = GlucoseExtractor()

    # Check available glucose tracks
    print("Checking for glucose tracks...")
    glucose_tracks = extractor.get_available_glucose_tracks(args.case_id)

    if glucose_tracks:
        print(f"\nFound {len(glucose_tracks)} glucose track(s):")
        for i, track in enumerate(glucose_tracks, 1):
            print(f"  {i}. {track['name']} (type: {track['type']})")

        # Extract from first track
        track_name = glucose_tracks[0]['name']
        print(f"\nExtracting data from: {track_name}")
        print("-" * 70)

        glucose_df = extractor.extract_glucose_data(args.case_id, track_name)

        if glucose_df is not None and len(glucose_df) > 0:
            print(f"\nSuccessfully extracted {len(glucose_df)} glucose measurements")
            print("\nStatistics:")
            print(f"  Time range: {glucose_df['time'].min():.1f} - {glucose_df['time'].max():.1f} seconds")
            print(f"  Duration: {(glucose_df['time'].max() - glucose_df['time'].min()) / 60:.1f} minutes")
            print(f"  Mean glucose: {glucose_df['glucose_mg_dl'].mean():.1f} mg/dL")
            print(f"  Std glucose: {glucose_df['glucose_mg_dl'].std():.1f} mg/dL")
            print(f"  Min glucose: {glucose_df['glucose_mg_dl'].min():.1f} mg/dL")
            print(f"  Max glucose: {glucose_df['glucose_mg_dl'].max():.1f} mg/dL")

            print("\nFirst 10 measurements:")
            print(glucose_df.head(10).to_string(index=False))

            print("\n" + "=" * 70)
            print("SUCCESS: Glucose data extraction working!")
            print("=" * 70)
        else:
            print("\nNo glucose data extracted")

    else:
        print(f"\nNo glucose tracks found for case {args.case_id}")
        print("\nSuggestions:")
        print("  - Try a different case ID")
        print("  - Check VitalDB documentation for cases with glucose data")
        print("  - Use manual glucose entry in the web app")


if __name__ == '__main__':
    main()
