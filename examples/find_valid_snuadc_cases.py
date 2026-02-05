"""
Find Valid SNUADC/PLETH Cases
==============================
Script to find cases with valid SNUADC/PLETH data (non-NaN values).
"""

from ppg_extractor import PPGExtractor
import sys


def find_valid_snuadc_cases(max_cases: int = 10, start_from: int = 1):
    """
    Find cases with valid SNUADC/PLETH data.

    Args:
        max_cases: Maximum number of valid cases to find
        start_from: Case ID to start searching from

    Returns:
        List of valid case IDs with details
    """
    print("=" * 70)
    print("FINDING CASES WITH VALID SNUADC/PLETH DATA")
    print("=" * 70)
    print(f"Looking for {max_cases} cases with valid data...")
    print(f"Starting from case {start_from}")
    print("=" * 70)

    extractor = PPGExtractor()
    valid_cases = []
    tested = 0
    current_case = start_from

    # First, find cases that have SNUADC/PLETH track
    print("\nStep 1: Finding cases with SNUADC/PLETH track...")
    cases_with_track = extractor.find_cases_with_ppg('SNUADC/PLETH')
    print(f"Found {len(cases_with_track)} cases with SNUADC/PLETH track listed")

    # Filter to cases starting from start_from
    cases_to_check = [c for c in cases_with_track if c >= start_from]
    print(f"Will check {len(cases_to_check)} cases (from case {start_from} onwards)")

    print("\nStep 2: Checking for valid (non-NaN) data...")
    print("-" * 70)

    for case_id in cases_to_check:
        tested += 1
        print(f"\n[{tested}] Testing case {case_id}...", end=" ")

        try:
            # Try to extract the track
            result = extractor.extract_ppg(
                case_id=case_id,
                track_name='SNUADC/PLETH',
                output_dir='./temp_validation'
            )

            # If we got here, the data is valid (non-empty, non-NaN)
            print(f"[OK] VALID")
            print(f"    Samples: {result['num_samples']:,}")
            print(f"    Duration: {result['duration_seconds']:.1f} seconds")
            print(f"    Sampling rate: {result['estimated_sampling_rate']:.1f} Hz")

            valid_cases.append({
                'case_id': case_id,
                'num_samples': result['num_samples'],
                'duration': result['duration_seconds'],
                'sampling_rate': result['estimated_sampling_rate'],
                'file': result['csv_file']
            })

            # Clean up test file
            import os
            try:
                os.remove(result['csv_file'])
                os.remove(result['metadata_file'])
            except:
                pass

            if len(valid_cases) >= max_cases:
                print(f"\n[OK] Found {max_cases} valid cases. Search complete!")
                break

        except ValueError as e:
            print(f"[FAIL] Invalid")
            print(f"    Reason: {e}")
            continue
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            continue

    # Summary
    print("\n" + "=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)
    print(f"Cases checked: {tested}")
    print(f"Valid cases found: {len(valid_cases)}")

    if valid_cases:
        print(f"\n[OK] VALID SNUADC/PLETH CASES:")
        print("-" * 70)
        for i, case in enumerate(valid_cases, 1):
            print(f"{i}. Case {case['case_id']}")
            print(f"   Samples: {case['num_samples']:,}")
            print(f"   Duration: {case['duration']:.1f} seconds")
            print(f"   Rate: {case['sampling_rate']:.1f} Hz")
            print()

        print("=" * 70)
        print("CASE IDs FOR ANALYSIS:")
        case_ids = [c['case_id'] for c in valid_cases]
        print(f"{case_ids}")
        print()
        print("You can analyze these cases with:")
        for case_id in case_ids[:3]:  # Show first 3 examples
            print(f"  python ppg_analysis_pipeline.py --case-id {case_id}")

    else:
        print("\n[FAIL] No valid SNUADC/PLETH data found.")
        print("Try starting from a different case ID or checking more cases.")

    # Save results
    if valid_cases:
        import json
        output_file = 'valid_snuadc_cases.json'
        with open(output_file, 'w') as f:
            json.dump(valid_cases, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return valid_cases


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Find VitalDB cases with valid SNUADC/PLETH data'
    )
    parser.add_argument('--max-cases', type=int, default=10,
                       help='Maximum number of valid cases to find (default: 10)')
    parser.add_argument('--start-from', type=int, default=1,
                       help='Case ID to start searching from (default: 1)')

    args = parser.parse_args()

    valid_cases = find_valid_snuadc_cases(
        max_cases=args.max_cases,
        start_from=args.start_from
    )

    sys.exit(0 if valid_cases else 1)
