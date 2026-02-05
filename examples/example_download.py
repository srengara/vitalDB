"""
Example: Downloading VitalDB Case Data
======================================
This script demonstrates how to download case data using the VitalDB utility.
"""

from vitaldb_utility import VitalDBUtility


def example_1_get_case_info():
    """Example 1: Get clinical information for a case"""
    print("\n[Example 1] Getting case information...")
    print("-" * 60)

    util = VitalDBUtility()

    # Get info for case 1
    case_info = util.get_case_info(1)

    if case_info:
        print(f"[OK] Retrieved info for case 1\n")
        print("Clinical Information:")
        # Display key fields
        key_fields = ['caseid', 'age', 'sex', 'height', 'weight', 'asa', 'opname', 'optype']
        for field in key_fields:
            if field in case_info:
                print(f"  {field.capitalize()}: {case_info[field]}")
    else:
        print("[ERROR] Case not found")


def example_2_list_case_tracks():
    """Example 2: List all available tracks for a case"""
    print("\n[Example 2] Listing available tracks for a case...")
    print("-" * 60)

    util = VitalDBUtility()

    # Get tracks for case 1
    tracks = util.get_case_tracks(1)

    if tracks:
        print(f"[OK] Found {len(tracks)} tracks for case 1\n")
        print("Available tracks (first 20):")
        for i, track in enumerate(tracks[:20], 1):
            print(f"  {i}. {track['tname']} (tid: {track['tid']})")

        if len(tracks) > 20:
            print(f"  ... and {len(tracks) - 20} more tracks")
    else:
        print("[ERROR] No tracks found")


def example_3_download_specific_tracks():
    """Example 3: Download specific tracks for a case"""
    print("\n[Example 3] Downloading specific tracks...")
    print("-" * 60)

    util = VitalDBUtility()

    # Download ECG and arterial pressure for case 1
    tracks_to_download = ['SNUADC/ECG_II', 'SNUADC/ART']

    print(f"Downloading tracks: {tracks_to_download}")
    print("This may take a minute...\n")

    try:
        result = util.download_case_data(
            case_id=1,
            track_names=tracks_to_download,
            output_dir='./downloads'
        )

        print(f"[OK] Download completed!")
        print(f"\nSummary:")
        print(f"  Case ID: {result['case_id']}")
        print(f"  Tracks downloaded: {len(result['data_files'])}")
        print(f"  Output directory: {result['output_dir']}")
        print(f"\nDownloaded files:")
        for file_path in result['data_files']:
            import os
            print(f"  - {os.path.basename(file_path)}")

    except Exception as e:
        print(f"[ERROR] Download failed: {e}")


def example_4_download_single_track():
    """Example 4: Download a single track by track ID"""
    print("\n[Example 4] Downloading a single track by track ID...")
    print("-" * 60)

    util = VitalDBUtility()

    # First, get a track ID for case 1
    tracks = util.get_case_tracks(1)
    if not tracks:
        print("[ERROR] No tracks available")
        return

    # Use the first track as an example
    example_track = tracks[0]
    track_name = example_track['tname']
    track_id = example_track['tid']

    print(f"Downloading track: {track_name} (tid: {track_id})")

    try:
        data = util.download_track_data(track_id)

        print(f"[OK] Downloaded {len(data)} data points")

        if data:
            print(f"\nFirst 5 data points:")
            for i, point in enumerate(data[:5], 1):
                print(f"  {i}. {point}")

            # Save to file
            import csv
            import os
            os.makedirs('./downloads', exist_ok=True)
            output_file = f"./downloads/track_{track_id}.csv"

            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

            print(f"\nSaved to: {output_file}")

    except Exception as e:
        print(f"[ERROR] Download failed: {e}")


def main():
    print("VitalDB Download Examples")
    print("=" * 60)

    # Run examples
    example_1_get_case_info()
    example_2_list_case_tracks()

    # Ask user before downloading (to avoid large downloads)
    print("\n" + "=" * 60)
    response = input("\nRun download examples? This will download data (y/n): ")

    if response.lower() == 'y':
        example_3_download_specific_tracks()
        example_4_download_single_track()
        print("\n" + "=" * 60)
        print("All examples completed!")
    else:
        print("\nDownload examples skipped.")

    print("\n" + "=" * 60)
    print("Examples completed!")


if __name__ == "__main__":
    main()
