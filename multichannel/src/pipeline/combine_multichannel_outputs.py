#!/usr/bin/env python
"""
Combine Multi-Channel Outputs
==============================
Utility script to combine outputs from multiple channels into a single training dataset.

Usage:
    # Combine all channels from output folder
    python combine_multichannel_outputs.py --input ./output --output combined_training_data.csv

    # Combine specific channels only
    python combine_multichannel_outputs.py --input ./output --channels force Signal1 --output combined.csv

    # Filter for high-quality windows only (8080)
    python combine_multichannel_outputs.py --input ./output --output combined.csv --quality-only
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_output_files(input_dir: str, channels: Optional[List[str]] = None) -> List[Path]:
    """
    Find all *-output.csv files in the input directory.

    Args:
        input_dir: Directory containing processed case folders
        channels: Optional list of channels to filter (e.g., ['force', 'Signal1'])

    Returns:
        List of output CSV file paths
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    output_files = []

    # Search through all subdirectories
    for case_folder in input_path.iterdir():
        if not case_folder.is_dir():
            continue

        # Find output files in this case folder
        for output_file in case_folder.glob("*-output.csv"):
            # Check channel filter if specified
            if channels:
                file_channel = None
                for channel in channels:
                    if output_file.name.startswith(channel):
                        file_channel = channel
                        break

                if file_channel is None:
                    continue

            output_files.append(output_file)

    return sorted(output_files)


def load_and_combine(output_files: List[Path], quality_only: bool = False) -> pd.DataFrame:
    """
    Load and combine multiple output CSV files.

    Args:
        output_files: List of CSV file paths
        quality_only: If True, only include 8080 (pure quality) windows

    Returns:
        Combined DataFrame
    """
    all_data = []

    for output_file in output_files:
        logger.info(f"Loading: {output_file.name}")

        try:
            df = pd.read_csv(output_file)

            # Filter for quality if requested
            if quality_only:
                before_count = len(df)
                df['window_id_str'] = df['window_index'].astype(str)
                df = df[df['window_id_str'].str.startswith('8080')].copy()
                df = df.drop(columns=['window_id_str'])
                after_count = len(df)

                if after_count < before_count:
                    logger.info(f"  Quality filter: {before_count} -> {after_count} windows")

            if len(df) > 0:
                all_data.append(df)
                logger.info(f"  Added {len(df)} windows")
            else:
                logger.warning(f"  Skipped (no windows after filtering)")

        except Exception as e:
            logger.error(f"  Error loading {output_file}: {e}")
            continue

    if not all_data:
        raise ValueError("No data to combine!")

    # Combine all DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)

    return combined_df


def generate_summary(combined_df: pd.DataFrame) -> dict:
    """Generate summary statistics for combined data."""
    summary = {
        'total_windows': len(combined_df),
        'unique_channels': combined_df['channel'].nunique() if 'channel' in combined_df.columns else 0,
        'unique_glucose_values': combined_df['glucose_mg_dl'].nunique() if 'glucose_mg_dl' in combined_df.columns else 0,
    }

    # Channel breakdown
    if 'channel' in combined_df.columns:
        summary['windows_per_channel'] = combined_df['channel'].value_counts().to_dict()

    # Glucose range
    if 'glucose_mg_dl' in combined_df.columns:
        summary['glucose_range'] = {
            'min': float(combined_df['glucose_mg_dl'].min()),
            'max': float(combined_df['glucose_mg_dl'].max()),
            'mean': float(combined_df['glucose_mg_dl'].mean()),
            'std': float(combined_df['glucose_mg_dl'].std())
        }

    # Quality breakdown
    if 'window_index' in combined_df.columns:
        combined_df['quality_tag'] = combined_df['window_index'].astype(str).str[:4]
        summary['quality_breakdown'] = combined_df['quality_tag'].value_counts().to_dict()
        combined_df = combined_df.drop(columns=['quality_tag'])

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Combine multi-channel output files into single training dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine all channels
  python combine_multichannel_outputs.py --input ./output --output combined_training_data.csv

  # Combine specific channels only
  python combine_multichannel_outputs.py --input ./output --channels force Signal1 --output combined.csv

  # High-quality windows only
  python combine_multichannel_outputs.py --input ./output --output high_quality.csv --quality-only
        """
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing processed case folders')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file path')
    parser.add_argument('--channels', type=str, nargs='+', default=None,
                       help='Specific channels to include (default: all)')
    parser.add_argument('--quality-only', action='store_true',
                       help='Only include high-quality (8080) windows')

    args = parser.parse_args()

    try:
        logger.info("=" * 70)
        logger.info("Multi-Channel Output Combiner")
        logger.info("=" * 70)

        # Find output files
        logger.info(f"\nSearching for output files in: {args.input}")
        if args.channels:
            logger.info(f"Filtering for channels: {', '.join(args.channels)}")

        output_files = find_output_files(args.input, args.channels)

        if not output_files:
            logger.error("No output files found!")
            return 1

        logger.info(f"Found {len(output_files)} output files")

        # Load and combine
        logger.info("\nLoading and combining data...")
        if args.quality_only:
            logger.info("Quality filter: 8080 windows only")

        combined_df = load_and_combine(output_files, args.quality_only)

        # Generate summary
        logger.info("\n" + "=" * 70)
        logger.info("Combined Dataset Summary")
        logger.info("=" * 70)

        summary = generate_summary(combined_df)

        logger.info(f"Total windows: {summary['total_windows']}")
        logger.info(f"Unique channels: {summary['unique_channels']}")
        logger.info(f"Unique glucose values: {summary['unique_glucose_values']}")

        if 'windows_per_channel' in summary:
            logger.info("\nWindows per channel:")
            for channel, count in summary['windows_per_channel'].items():
                logger.info(f"  {channel}: {count}")

        if 'glucose_range' in summary:
            gr = summary['glucose_range']
            logger.info(f"\nGlucose range:")
            logger.info(f"  Min: {gr['min']:.1f} mg/dL")
            logger.info(f"  Max: {gr['max']:.1f} mg/dL")
            logger.info(f"  Mean: {gr['mean']:.1f} ± {gr['std']:.1f} mg/dL")

        if 'quality_breakdown' in summary:
            logger.info("\nQuality breakdown:")
            for quality_tag, count in summary['quality_breakdown'].items():
                tag_name = "Pure (8080)" if quality_tag == '8080' else "Repaired (4040)"
                logger.info(f"  {tag_name}: {count} ({count/summary['total_windows']*100:.1f}%)")

        # Save combined dataset
        logger.info(f"\nSaving combined dataset to: {args.output}")
        combined_df.to_csv(args.output, index=False)

        # Get file size
        file_size_mb = Path(args.output).stat().st_size / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")

        logger.info("\n" + "=" * 70)
        logger.info("SUCCESS!")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"\nERROR: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
