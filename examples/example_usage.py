"""
Example usage of VitalDB Utility Class
======================================

This script demonstrates how to use the VitalDBUtility class
to list and filter case IDs from the VitalDB dataset.
"""

from vitaldb_utility import VitalDBUtility


def main():
    """Main function demonstrating VitalDB utility usage."""

    # Initialize the utility
    util = VitalDBUtility()

    print("VitalDB Case ID Utility - Demo")
    print("=" * 60)

    # Example 1: Get all case IDs
    print("\n[Example 1] Getting all case IDs...")
    try:
        all_cases = util.get_all_case_ids()
        print(f"[OK] Successfully retrieved {len(all_cases)} case IDs")
        print(f"  Case ID range: {min(all_cases)} to {max(all_cases)}")
        print(f"  First 20 cases: {all_cases[:20]}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")

    # Example 2: Get total case count
    print("\n[Example 2] Getting total case count...")
    try:
        count = util.get_case_count()
        print(f"[OK] Total cases in dataset: {count}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")

    # Example 3: Get available tracks
    print("\n[Example 3] Getting available tracks...")
    try:
        tracks = util.get_available_tracks()
        print(f"[OK] Found {len(tracks)} unique tracks")
        print(f"  Sample tracks (first 30):")
        for i, track in enumerate(tracks[:30], 1):
            print(f"    {i}. {track}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")

    # Example 4: Get cases with specific tracks (both required)
    print("\n[Example 4] Finding cases with BOTH specified tracks...")
    try:
        required_tracks = ['SNUADC/ECG_II', 'SNUADC/ART']
        matching_cases = util.get_cases_with_tracks(required_tracks)
        print(f"[OK] Found {len(matching_cases)} cases with tracks: {required_tracks}")
        if matching_cases:
            print(f"  Sample matching case IDs: {matching_cases[:10]}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")

    # Example 5: Get cases with any of the specified tracks
    print("\n[Example 5] Finding cases with ANY of the specified tracks...")
    try:
        any_tracks = ['SNUADC/ECG_II', 'SNUADC/ART', 'SNUADC/PLETH']
        cases_with_any = util.get_cases_with_any_track(any_tracks)
        print(f"[OK] Found {len(cases_with_any)} cases with any of: {any_tracks}")
        if cases_with_any:
            print(f"  Sample case IDs: {cases_with_any[:10]}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")

    # Example 6: Get cases organized by track
    print("\n[Example 6] Getting cases organized by track...")
    try:
        cases_by_track = util.get_case_ids_by_track()
        print(f"[OK] Retrieved data for {len(cases_by_track)} tracks")

        # Show statistics for a few tracks
        sample_tracks = list(cases_by_track.keys())[:5]
        print("  Sample track statistics:")
        for track in sample_tracks:
            case_count = len(cases_by_track[track])
            print(f"    - {track}: {case_count} cases")
    except Exception as e:
        print(f"[ERROR] Error: {e}")

    print("\n" + "=" * 60)
    print("Demo completed!")


if __name__ == "__main__":
    main()
