"""
Download VitalDB Case Data
==========================
Script to download case data from VitalDB by case ID.

Usage:
    python download_case.py <case_id> [track1] [track2] ...

Examples:
    # Download all tracks for case 1
    python download_case.py 1

    # Download specific tracks for case 1
    python download_case.py 1 SNUADC/ECG_II SNUADC/ART

    # Download to specific directory
    python download_case.py 1 --output ./data
"""

import sys
import argparse
from vitaldb_utility import VitalDBUtility


def main():
    parser = argparse.ArgumentParser(
        description='Download VitalDB case data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download all tracks for case 1:
    python download_case.py 1

  Download specific tracks:
    python download_case.py 1 SNUADC/ECG_II SNUADC/ART

  Download to specific directory:
    python download_case.py 1 --output ./data
        """
    )

    parser.add_argument('case_id', type=int, help='Case ID to download')
    parser.add_argument('tracks', nargs='*', help='Specific track names to download (optional)')
    parser.add_argument('--output', '-o', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--list-tracks', '-l', action='store_true',
                       help='List available tracks for the case without downloading')

    args = parser.parse_args()

    util = VitalDBUtility()

    print(f"VitalDB Case Downloader")
    print("=" * 60)
    print(f"Case ID: {args.case_id}")

    # List tracks mode
    if args.list_tracks:
        print("\nFetching available tracks...")
        try:
            tracks = util.get_case_tracks(args.case_id)
            if not tracks:
                print(f"[ERROR] No tracks found for case {args.case_id}")
                sys.exit(1)

            print(f"\n[OK] Found {len(tracks)} tracks for case {args.case_id}:\n")
            for i, track in enumerate(tracks, 1):
                print(f"  {i}. {track['tname']} (tid: {track['tid']})")

        except Exception as e:
            print(f"[ERROR] Failed to fetch tracks: {e}")
            sys.exit(1)

        sys.exit(0)

    # Download mode
    track_names = args.tracks if args.tracks else None

    if track_names:
        print(f"Tracks to download: {', '.join(track_names)}")
    else:
        print("Download mode: ALL tracks")

    print(f"Output directory: {args.output}")
    print("\nStarting download...")
    print("-" * 60)

    try:
        result = util.download_case_data(
            case_id=args.case_id,
            track_names=track_names,
            output_dir=args.output
        )

        print("-" * 60)
        print("\n[OK] Download completed successfully!")
        print(f"\nSummary:")
        print(f"  Case ID: {result['case_id']}")
        print(f"  Tracks downloaded: {len(result['data_files'])}")
        print(f"  Output directory: {result['output_dir']}")
        print(f"\nFiles created:")
        print(f"  - case_info.json (clinical information)")
        print(f"  - tracks.json (track metadata)")
        for i, file in enumerate(result['data_files'], 1):
            import os
            filename = os.path.basename(file)
            print(f"  - {filename}")

        # Show sample case info
        case_info = result['case_info']
        if case_info:
            print(f"\nCase Information:")
            key_fields = ['age', 'sex', 'height', 'weight', 'asa', 'opname', 'optype', 'approach']
            for field in key_fields:
                if field in case_info and case_info[field]:
                    print(f"  {field.capitalize()}: {case_info[field]}")

    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("VitalDB Case Downloader")
        print("=" * 60)
        print("\nUsage: python download_case.py <case_id> [track1] [track2] ...")
        print("\nOptions:")
        print("  -h, --help           Show help message")
        print("  -l, --list-tracks    List available tracks without downloading")
        print("  -o, --output DIR     Output directory (default: current directory)")
        print("\nExamples:")
        print("  Download all tracks for case 1:")
        print("    python download_case.py 1")
        print("\n  List available tracks:")
        print("    python download_case.py 1 --list-tracks")
        print("\n  Download specific tracks:")
        print("    python download_case.py 1 SNUADC/ECG_II SNUADC/ART")
        print("\n  Download to specific directory:")
        print("    python download_case.py 1 --output ./data")
        sys.exit(0)

    main()
