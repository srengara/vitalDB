import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from pathlib import Path
import glob
import logging

logger = logging.getLogger(__name__)


class MultiChannelDataset(Dataset):
    """
    Multi-channel signal dataset for early fusion architecture.
    """

    def __init__(self, data: np.ndarray, targets: np.ndarray):
        """
        Args:
            data: Shape (num_samples, sequence_length, num_channels)
            targets: Shape (num_samples,)
        """
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class MultiChannelDataLoader:
    """
    Data loader for consolidated multi-channel CSV format.

    Expected CSV format:
        channel, window_index, glucose_mg_dl, systolic_mmhg, diastolic_mmhg,
        amplitude_sample_0, amplitude_sample_1, ..., amplitude_sample_99

    Each row represents one channel's signal for a given window.
    Multiple channels are stacked by matching on window_index.
    """

    def __init__(self, csv_path: str, config: dict):
        """
        Args:
            csv_path: Path to consolidated CSV or directory containing channel CSVs
            config: Configuration dictionary
        """
        self.csv_path = csv_path
        self.config = config
        self.scalers = {}
        self.channel_names = config['dataset']['channels']
        self.target_col = config['dataset']['target']

    def load_data(self) -> pd.DataFrame:
        """
        Load data from a single consolidated CSV or multiple channel CSVs
        in a directory.

        Returns:
            Combined DataFrame with all channels
        """
        path = Path(self.csv_path)

        if path.is_file():
            df = pd.read_csv(path)
            logger.info(f"Loaded consolidated CSV: {df.shape}")
        elif path.is_dir():
            csv_files = list(path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {path}")
            frames = []
            for f in csv_files:
                part = pd.read_csv(f)
                frames.append(part)
                logger.info(f"Loaded {f.name}: {part.shape}")
            df = pd.concat(frames, ignore_index=True)
            logger.info(f"Combined data shape: {df.shape}")
        else:
            raise FileNotFoundError(f"Path not found: {path}")

        # Validate required columns
        required = ['channel', 'window_index', self.target_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Remove duplicate rows (same channel + window_index)
        before = len(df)
        df = df.drop_duplicates(subset=['channel', 'window_index'], keep='first')
        if len(df) < before:
            logger.info(f"Removed {before - len(df)} duplicate rows")

        # Filter to configured channels only
        available_channels = df['channel'].unique().tolist()
        logger.info(f"Available channels in data: {available_channels}")

        selected = [ch for ch in self.channel_names if ch in available_channels]
        if not selected:
            raise ValueError(
                f"None of the configured channels {self.channel_names} "
                f"found in data. Available: {available_channels}"
            )
        if len(selected) < len(self.channel_names):
            missing_ch = set(self.channel_names) - set(selected)
            logger.warning(f"Channels not found in data (skipped): {missing_ch}")

        df = df[df['channel'].isin(selected)]
        logger.info(f"Using channels: {selected}")

        return df

    def build_multichannel_tensors(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pivot the long-format DataFrame into multi-channel input tensors.

        Groups by window_index, stacks channels to produce:
            X: (num_windows, sequence_length, num_channels)
            y: (num_windows,)

        Only keeps windows that have ALL selected channels present.
        """
        amp_cols = [c for c in df.columns if c.startswith('amplitude_sample_')]
        amp_cols = sorted(amp_cols, key=lambda c: int(c.split('_')[-1]))
        sequence_length = len(amp_cols)

        logger.info(f"Sequence length from amplitude columns: {sequence_length}")

        # Get channels present in data (in config order)
        available = df['channel'].unique().tolist()
        channels_ordered = [ch for ch in self.channel_names if ch in available]
        num_channels = len(channels_ordered)

        # Find windows that have all channels
        window_channel_counts = df.groupby('window_index')['channel'].nunique()
        complete_windows = window_channel_counts[
            window_channel_counts == num_channels
        ].index.tolist()

        if not complete_windows:
            raise ValueError("No windows found with all channels present")

        logger.info(
            f"Complete windows (all {num_channels} channels): "
            f"{len(complete_windows)} out of {df['window_index'].nunique()}"
        )

        df_complete = df[df['window_index'].isin(complete_windows)]

        # Build tensors
        X_list = []
        y_list = []

        for win_idx in sorted(complete_windows):
            win_data = df_complete[df_complete['window_index'] == win_idx]

            # Stack channels: (sequence_length, num_channels)
            channel_arrays = []
            for ch_name in channels_ordered:
                ch_row = win_data[win_data['channel'] == ch_name]
                if len(ch_row) != 1:
                    logger.warning(
                        f"Window {win_idx} channel {ch_name}: "
                        f"expected 1 row, got {len(ch_row)}. Skipping."
                    )
                    break
                channel_arrays.append(ch_row[amp_cols].values.flatten())
            else:
                # All channels found - shape: (sequence_length, num_channels)
                window_tensor = np.stack(channel_arrays, axis=1)
                X_list.append(window_tensor)

                # Target from first channel row (same across channels for a window)
                target_val = win_data[self.target_col].iloc[0]
                y_list.append(target_val)

        X = np.array(X_list)  # (num_windows, sequence_length, num_channels)
        y = np.array(y_list)  # (num_windows,)

        logger.info(f"Built tensors - X: {X.shape}, y: {y.shape}")
        logger.info(f"Target stats - min: {y.min():.2f}, max: {y.max():.2f}, "
                     f"mean: {y.mean():.2f}, std: {y.std():.2f}")

        return X, y

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize amplitude values per channel.

        Args:
            X: Shape (num_windows, sequence_length, num_channels)

        Returns:
            Normalized X with same shape
        """
        if not self.config['dataset']['normalize']:
            return X

        method = self.config['dataset']['normalization_method']
        num_channels = X.shape[2]

        for ch_idx in range(num_channels):
            # Reshape channel data to 2D for scaler: (num_windows * seq_len, 1)
            ch_data = X[:, :, ch_idx].reshape(-1, 1)

            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown normalization: {method}")

            ch_data_norm = scaler.fit_transform(ch_data)
            X[:, :, ch_idx] = ch_data_norm.reshape(X.shape[0], X.shape[1])
            self.scalers[ch_idx] = scaler

        logger.info(f"Normalized {num_channels} channels using {method}")
        return X

    def remove_outliers(self, X: np.ndarray) -> np.ndarray:
        """Remove outliers per channel using z-score thresholding."""
        if not self.config['preprocessing']['remove_outliers']:
            return X

        threshold = self.config['preprocessing']['outlier_threshold']
        num_channels = X.shape[2]

        for ch_idx in range(num_channels):
            ch_data = X[:, :, ch_idx]
            mean = np.mean(ch_data)
            std = np.std(ch_data) + 1e-8
            z_scores = np.abs((ch_data - mean) / std)
            X[:, :, ch_idx] = np.where(z_scores < threshold, ch_data, mean)

        logger.info("Outliers removed")
        return X

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load, preprocess, split, and return DataLoaders.

        Returns:
            train_loader, val_loader, test_loader
        """
        # Load and build multi-channel tensors
        df = self.load_data()
        X, y = self.build_multichannel_tensors(df)

        # Preprocess
        X = self.remove_outliers(X)
        X = self.normalize(X)

        # Update config with actual dimensions detected from data
        actual_channels = X.shape[2]
        actual_seq_len = X.shape[1]
        if actual_channels != self.config['model']['input_channels']:
            logger.warning(
                f"Config input_channels={self.config['model']['input_channels']} "
                f"but data has {actual_channels} channels. Updating config."
            )
            self.config['model']['input_channels'] = actual_channels
        if actual_seq_len != self.config['model']['sequence_length']:
            logger.warning(
                f"Config sequence_length={self.config['model']['sequence_length']} "
                f"but data has {actual_seq_len} samples. Updating config."
            )
            self.config['model']['sequence_length'] = actual_seq_len

        # Split data
        test_split = self.config['dataset']['test_split']
        val_split = self.config['dataset']['validation_split']
        seed = self.config['dataset']['random_seed']

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=seed
        )

        val_ratio = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=seed
        )

        logger.info(f"Split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Create datasets
        train_dataset = MultiChannelDataset(X_train, y_train)
        val_dataset = MultiChannelDataset(X_val, y_val)
        test_dataset = MultiChannelDataset(X_test, y_test)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['validation']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['validation']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )

        return train_loader, val_loader, test_loader
