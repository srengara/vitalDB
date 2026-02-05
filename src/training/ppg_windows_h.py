"""Process and combine PPG windows and glucose labels into final_datasets.

Usage:
1) Place per-case folders under training_data/ with ppg_windows.csv (and optional glucose_labels.csv).
2) Run this script: python Ashwanth/ppg_windows_h.py
3) Enter how many cases to process when prompted (blank = all).
4) Outputs are written to final_datasets/ppg_windows.csv and final_datasets/glucose_labels.csv.

Notes:
- Source files are not overwritten; only combined outputs are written.
- Final combined outputs are sorted by case_id and window_id.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\IITM\vitalDB")
DATA_ROOT = ROOT / "new_data_format/case_1"
FINAL_DIR = ROOT / "new_data_format/final_datasets"
FINAL_PPG = FINAL_DIR / "ppg_windows.csv"
FINAL_GLU = FINAL_DIR / "glucose_labels.csv"
CSV_CHUNKSIZE = 200_000  # write in chunks to avoid pandas' extra memory spikes
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"


def format_case_label(ppg_path: Path) -> str:
    """Return a short case label based on the PPG file location."""
    return ppg_path.parent.name


def print_status(label: str, ok: bool, detail: str | None = None) -> None:
    """Print a short, colored status line."""
    color = ANSI_GREEN if ok else ANSI_RED
    mark = "OK" if ok else "X"
    tail = f" - {detail}" if detail else ""
    print(f"{color}{mark}{ANSI_RESET} {label}{tail}")


def pick_window_col(df: pd.DataFrame) -> str:
    """Ensure we operate on window_id, renaming legacy window_index if present."""
    if "window_id" in df.columns:
        return "window_id"
    if "window_index" in df.columns:
        df.rename(columns={"window_index": "window_id"}, inplace=True)
        return "window_id"
    raise ValueError("No window_id column found")


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink common integer/float columns to cut memory use."""
    for col in ("case_id", "window_id", "window_index", "sample_index"):
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in ("amplitude", "glucose_dt", "glucose_mg_dl"):
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def write_append(df: pd.DataFrame, dest: Path, index: bool = False) -> None:
    """Append a dataframe to dest, writing header only once."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = not dest.exists()
    df.to_csv(dest, mode="a", header=header, index=index, chunksize=CSV_CHUNKSIZE)


def sort_csv(
    path: Path,
    sort_columns: list[str],
    output_path: Path | None = None,
    return_df: bool = False,
    write_output: bool = True,
) -> tuple[int, pd.DataFrame | None]:
    """Sort a CSV by the given columns (using those that exist), add combined key, and optionally write output."""
    df = _downcast_numeric(pd.read_csv(path))
    window_col = pick_window_col(df) if any(col.startswith("window") for col in df.columns) else None

    needed = ["case_id"] + ([window_col] if window_col else [])
    missing_needed = [col for col in needed if col not in df.columns]
    if missing_needed:
        raise ValueError(f"{path.name} is missing expected columns: {missing_needed}")

    effective_sort_cols = [col for col in sort_columns if col in df.columns]
    if not effective_sort_cols:
        effective_sort_cols = needed  # fallback to case_id and window

    missing_sort = [col for col in effective_sort_cols if col not in df.columns]
    if missing_sort:
        raise ValueError(f"{path.name} is missing expected sort columns: {missing_sort}")

    sorted_df = df.sort_values(effective_sort_cols, kind="mergesort")  # stable sort keeps existing order within ties
    if window_col:
        sorted_df["window_index"] = sorted_df["case_id"].astype(str) + "8080" + sorted_df[window_col].astype(str)

    # Write in chunks to avoid numpy memory errors on wide files.
    if write_output:
        sorted_df.to_csv(output_path or path, index=False, chunksize=CSV_CHUNKSIZE)
    return len(sorted_df), (sorted_df if return_df else None)


def pivot_ppg_samples_df(
    df: pd.DataFrame,
    output_path: Path | None = None,
    return_df: bool = False,
    write_output: bool = True,
) -> tuple[int, pd.DataFrame | None]:
    """Wide-format the PPG windows so sample indices become columns and keep window IDs as index."""
    df = _downcast_numeric(df)
    window_col = pick_window_col(df)
    needed = ["case_id", window_col, "sample_index", "amplitude"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        # If sample_index is missing, the file is already wide; skip pivot.
        if "sample_index" in missing:
            return 0, None
        raise ValueError(f"{path.name} is missing expected columns for pivot: {missing}")

    df = df.sort_values(["case_id", window_col, "sample_index"], kind="mergesort")
    df["window_index"] = df["case_id"].astype(str) + "8080" + df[window_col].astype(str)

    wide = (
        df.pivot_table(
            index=["window_index", "case_id", window_col],
            columns="sample_index",
            values="amplitude",
            aggfunc="first",
        )
        .sort_index(axis=1)
    )

    # Rename numeric sample columns to a consistent prefix.
    wide.columns = [
        f"amplitude_sample_{int(col)}" if float(col).is_integer() else f"amplitude_sample_{col}"
        for col in wide.columns
    ]

    # Keep the multi-index (window_index -> case_id -> window_id) as the CSV index.
    if write_output:
        wide.to_csv(output_path, index=True, chunksize=CSV_CHUNKSIZE)
    return len(wide), (wide if return_df else None)

def pivot_ppg_samples(
    path: Path, output_path: Path | None = None, return_df: bool = False, write_output: bool = True
) -> tuple[int, pd.DataFrame | None]:
    """Wide-format the PPG windows so sample indices become columns and keep window IDs as index."""
    df = pd.read_csv(path)
    effective_output = output_path or path
    return pivot_ppg_samples_df(
        df, output_path=effective_output, return_df=return_df, write_output=write_output
    )


def process_pair(ppg_path: Path, glu_path: Path | None = None, out_dir: Path | None = None) -> None:
    """Run sort and pivot for one PPG (and optional glucose) file."""
    dest_ppg = None
    dest_glu = None
    write_outputs = False
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        dest_ppg = out_dir / "ppg_windows.csv"
        dest_glu = out_dir / "glucose_labels.csv" if glu_path else None
        write_outputs = True

    # Sort PPG then pivot to wide format without overwriting the source file.
    ppg_rows, sorted_ppg_df = sort_csv(
        ppg_path,
        ["case_id", "window_id", "sample_index"],
        output_path=dest_ppg,
        return_df=True,
        write_output=write_outputs,
    )
    ppg_wide_rows, wide_df = (0, None)
    if sorted_ppg_df is not None:
        ppg_wide_rows, wide_df = pivot_ppg_samples_df(
            sorted_ppg_df, output_path=dest_ppg, return_df=True, write_output=write_outputs
        )
    case_label = format_case_label(ppg_path)
    if ppg_wide_rows:
        print_status(case_label, True, "PPG processed")
    else:
        print_status(case_label, True, "PPG sorted (already wide)")

    # Append PPG output to final combined file (use pivoted if available).
    append_df = wide_df if wide_df is not None else sorted_ppg_df
    if append_df is not None:
        write_append(append_df, FINAL_PPG, index=wide_df is not None)

    if glu_path:
        if glu_path.exists():
            glu_rows, glu_df = sort_csv(
                glu_path,
                ["case_id", "window_id"],
                output_path=dest_glu,
                return_df=True,
                write_output=write_outputs,
            )
            print_status(case_label, True, "glucose labels sorted")
            if glu_df is not None:
                write_append(glu_df, FINAL_GLU)
        else:
            print_status(case_label, False, "glucose labels missing")


def main() -> None:
    # Start fresh combined outputs
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    if FINAL_PPG.exists():
        FINAL_PPG.unlink()
    if FINAL_GLU.exists():
        FINAL_GLU.unlink()

    # Process selected per-case training_data directories and append to final outputs
    ppg_files = sorted(DATA_ROOT.rglob("ppg_windows.csv"))
    if not ppg_files:
        print(f"No ppg_windows.csv files found under {DATA_ROOT}")
        return

    total_files = len(ppg_files)
    raw = input(f"How many files to process? (1-{total_files}, blank for all): ").strip()
    if raw:
        try:
            requested = int(raw)
        except ValueError:
            print("Invalid number; defaulting to all files.")
            requested = total_files
        else:
            if requested < 1:
                print("Requested count < 1; defaulting to all files.")
                requested = total_files
            elif requested > total_files:
                print("Requested count exceeds available files; defaulting to all files.")
                requested = total_files
        ppg_files = ppg_files[:requested]

    for ppg_file in ppg_files:
        glu_file = ppg_file.parent / "glucose_labels.csv"
        process_pair(ppg_file, glu_file)

    # Sort final combined outputs by case_id and window_id.
    if FINAL_PPG.exists():
        sort_csv(FINAL_PPG, ["case_id", "window_id"], output_path=FINAL_PPG, return_df=False)
    if FINAL_GLU.exists():
        sort_csv(FINAL_GLU, ["case_id", "window_id"], output_path=FINAL_GLU, return_df=False)


if __name__ == "__main__":
    main()

