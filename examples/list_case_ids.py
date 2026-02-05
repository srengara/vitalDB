"""
Simple script to list all VitalDB case IDs
"""

from vitaldb_utility import VitalDBUtility


def main():
    # Initialize utility
    util = VitalDBUtility()

    # Get all case IDs
    print("Fetching all case IDs from VitalDB...")
    case_ids = util.get_all_case_ids()

    # Display results
    print(f"\nTotal cases: {len(case_ids)}")
    print(f"Case ID range: {min(case_ids)} to {max(case_ids)}")
    print(f"\nAll case IDs:")
    print(case_ids)

    # Save to file (optional)
    with open('case_ids.txt', 'w') as f:
        f.write("VitalDB Case IDs\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total cases: {len(case_ids)}\n")
        f.write(f"Case ID range: {min(case_ids)} to {max(case_ids)}\n\n")
        f.write("All case IDs:\n")
        for case_id in case_ids:
            f.write(f"{case_id}\n")

    print(f"\nCase IDs saved to: case_ids.txt")


if __name__ == "__main__":
    main()
