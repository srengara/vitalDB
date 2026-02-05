"""
Find Cases with Valid PPG Data
================================
Script to find VitalDB cases that have valid PPG signals.

Many cases have PPG tracks listed but the actual signal data may be empty or sparse.
This script tests multiple cases to find ones with usable PPG data.
"""

import argparse
from ppg_extractor import PPGExtractor
import sys


def test_case_ppg(case_id: int, extractor: PPGExtractor) -> dict:
    """
    Test if a case has valid PPG data.

    Args:
        case_id: Case ID to test
        extractor: PPGExtractor instance

    Returns:
        Dictionary with test results
    """
    result = {
        'case_id': case_id,
        'has_ppg_tracks': False,
        'tracks': [],
        'valid_track': None,
        'error': None
    }

    try:
        # Get available PPG tracks
        tracks = extractor.get_available_ppg_tracks(case_id)

        if not tracks:
            result['error'] = "No PPG tracks found"
            return result

        result['has_ppg_tracks'] = True
        result['tracks'] = [t['tname'] for t in tracks]

        # Try to extract best track
        try:
            extraction = extractor.extract_best_ppg_track(case_id, './temp_ppg_test')
            result['valid_track'] = extraction['track_name']
            result['num_samples'] = extraction['num_samples']
            result['duration'] = extraction['duration_seconds']
            result['sampling_rate'] = extraction['estimated_sampling_rate']

            # Clean up test file
            import os
            try:
                os.remove(extraction['csv_file'])
                os.remove(extraction['metadata_file'])
            except:
                pass

        except ValueError as e:
            result['error'] = str(e)
        except Exception as e:
            result['error'] = f"Extraction failed: {e}"

    except Exception as e:
        result['error'] = f"Error: {e}"

    return result


def find_valid_cases(start_case: int = 1,
                    num_cases: int = 100,
                    max_valid: int = 10) -> list:
    """
    Find cases with valid PPG data.

    Args:
        start_case: Starting case ID
        num_cases: Number of cases to check
        max_valid: Stop after finding this many valid cases

    Returns:
        List of valid case IDs with details
    """
    print("=" * 70)
    print("FINDING CASES WITH VALID PPG DATA")
    print("=" * 70)
    print(f"Checking cases {start_case} to {start_case + num_cases - 1}")
    print(f"Will stop after finding {max_valid} valid cases")
    print("=" * 70)

    extractor = PPGExtractor()
    valid_cases = []
    tested = 0

    for case_id in range(start_case, start_case + num_cases):
        tested += 1
        print(f"\n[{tested}/{num_cases}] Testing case {case_id}...", end=" ")

        result = test_case_ppg(case_id, extractor)

        if result['valid_track']:
            print(f"✓ VALID")
            print(f"  Track: {result['valid_track']}")
            print(f"  Samples: {result['num_samples']:,}")
            print(f"  Duration: {result['duration']:.1f}s")
            print(f"  Rate: {result['sampling_rate']:.1f} Hz")

            valid_cases.append(result)

            if len(valid_cases) >= max_valid:
                print(f"\n✓ Found {max_valid} valid cases. Stopping search.")
                break
        else:
            print(f"✗ Invalid")
            if result['has_ppg_tracks']:
                print(f"  Tracks present: {', '.join(result['tracks'])}")
            print(f"  Reason: {result['error']}")

    # Summary
    print("\n" + "=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)
    print(f"Cases tested: {tested}")
    print(f"Valid cases found: {len(valid_cases)}")

    if valid_cases:
        print(f"\nValid case IDs:")
        for case in valid_cases:
            print(f"  - Case {case['case_id']}: {case['valid_track']} "
                  f"({case['num_samples']:,} samples, {case['duration']:.1f}s)")

        print(f"\nTo analyze a valid case, run:")
        print(f"  python ppg_analysis_pipeline.py --case-id {valid_cases[0]['case_id']}")
    else:
        print("\nNo valid PPG data found in the tested range.")
        print("Try checking more cases or a different range.")

    return valid_cases


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Find VitalDB cases with valid PPG data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find first 10 valid cases starting from case 1
  python find_valid_ppg_cases.py

  # Check cases 100-199
  python find_valid_ppg_cases.py --start 100 --num-cases 100

  # Find 20 valid cases quickly
  python find_valid_ppg_cases.py --max-valid 20

  # Check specific range
  python find_valid_ppg_cases.py --start 500 --num-cases 50 --max-valid 5
        """
    )

    parser.add_argument('--start', type=int, default=1,
                       help='Starting case ID (default: 1)')
    parser.add_argument('--num-cases', type=int, default=100,
                       help='Number of cases to check (default: 100)')
    parser.add_argument('--max-valid', type=int, default=10,
                       help='Stop after finding this many valid cases (default: 10)')

    args = parser.parse_args()

    valid_cases = find_valid_cases(
        start_case=args.start,
        num_cases=args.num_cases,
        max_valid=args.max_valid
    )

    # Save results to file
    if valid_cases:
        import json
        output_file = 'valid_ppg_cases.json'
        with open(output_file, 'w') as f:
            json.dump(valid_cases, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    sys.exit(0 if valid_cases else 1)


if __name__ == "__main__":
    main()
