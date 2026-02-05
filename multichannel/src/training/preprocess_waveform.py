import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WaveformBoundaryDetector:
    """
    Detect start and end points of force waveform based on gradient analysis.
    """
    
    def __init__(self, xlsx_path: str, output_dir: str = '.'):
        """
        Args:
            xlsx_path: Path to input Excel file
            output_dir: Directory to save output CSV
        """
        self.xlsx_path = xlsx_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """Load Excel file"""
        df = pd.read_excel(self.xlsx_path)
        logger.info(f"Loaded data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        return df
    
    def analyze_gradient(self, df: pd.DataFrame, 
                        gradient_threshold: float = 0.5,
                        window_size: int = 5) -> tuple:
        """
        Analyze gradient of force waveform to find start and end points.
        
        Strategy:
        1. Calculate gradient (derivative) of force signal
        2. Smooth gradient with moving average
        3. Find where gradient exceeds threshold (start of ascent)
        4. Find where gradient drops below threshold again (end of ascent)
        
        Args:
            df: DataFrame with force waveform
            gradient_threshold: Threshold for detecting steep gradient
            window_size: Window size for smoothing gradient
            
        Returns:
            (start_idx, end_idx, start_timestamp, end_timestamp)
        """
        # Get force signal
        force_col = None
        for col in df.columns:
            if 'force' in col.lower() or 'pulse' in col.lower():
                force_col = col
                break
        
        if force_col is None:
            raise ValueError(f"No force column found. Available: {df.columns.tolist()}")
        
        logger.info(f"Analyzing force column: {force_col}")
        
        force = df[force_col].values.astype(float)
        
        # Calculate gradient (difference between consecutive points)
        gradient = np.gradient(force)
        
        # Smooth gradient with moving average
        smoothed_gradient = pd.Series(gradient).rolling(
            window=window_size, center=True, min_periods=1
        ).mean().values
        
        # Normalize gradient for threshold detection
        grad_mean = np.mean(smoothed_gradient)
        grad_std = np.std(smoothed_gradient)
        normalized_grad = (smoothed_gradient - grad_mean) / (grad_std + 1e-8)
        
        logger.info(f"Gradient stats - Mean: {grad_mean:.4f}, Std: {grad_std:.4f}")
        
        # Find start point: where gradient significantly increases
        # (crosses threshold from below)
        start_idx = None
        for i in range(len(normalized_grad)):
            if normalized_grad[i] > gradient_threshold:
                start_idx = i
                break
        
        # Find end point: where gradient returns to low values after being high
        end_idx = None
        if start_idx is not None:
            for i in range(start_idx + window_size, len(normalized_grad)):
                if normalized_grad[i] < gradient_threshold / 2:
                    end_idx = i
                    break
        
        if start_idx is None or end_idx is None:
            logger.warning(f"Could not find clear boundaries. start_idx={start_idx}, end_idx={end_idx}")
            # Fallback: use middle portion
            if start_idx is None:
                start_idx = len(force) // 4
            if end_idx is None:
                end_idx = 3 * len(force) // 4
        
        # Get timestamps
        time_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'timestamp' in col.lower():
                time_col = col
                break
        
        if time_col:
            start_time = df[time_col].iloc[start_idx]
            end_time = df[time_col].iloc[end_idx]
        else:
            start_time = start_idx
            end_time = end_idx
        
        logger.info(f"Start index: {start_idx}, timestamp: {start_time}")
        logger.info(f"End index: {end_idx}, timestamp: {end_time}")
        
        return start_idx, end_idx, start_time, end_time, force_col, time_col, gradient, smoothed_gradient
    
    def filter_data(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        """
        Filter data to boundaries.
        
        Args:
            df: Original DataFrame
            start_idx: Start index
            end_idx: End index
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.iloc[start_idx:end_idx + 1].reset_index(drop=True)
        logger.info(f"Filtered data shape: {filtered_df.shape}")
        return filtered_df
    
    def save_output(self, filtered_df: pd.DataFrame, 
                   start_time, end_time, filename: str = None) -> str:
        """
        Save filtered data to CSV.
        
        Args:
            filtered_df: Filtered DataFrame
            start_time: Start timestamp
            end_time: End timestamp
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to output file
        """
        if filename is None:
            filename = f"Multi-channel-output_filtered_{start_time}_{end_time}.csv"
        
        output_path = self.output_dir / filename
        filtered_df.to_csv(output_path, index=False)
        logger.info(f"Saved filtered data to: {output_path}")
        
        return str(output_path)
    
    def visualize_analysis(self, df: pd.DataFrame, 
                          start_idx: int, end_idx: int,
                          force_col: str, time_col: str,
                          gradient: np.ndarray, smoothed_gradient: np.ndarray,
                          save_path: str = None):
        """
        Visualize gradient analysis and boundaries.
        
        Args:
            df: Original DataFrame
            start_idx: Start index
            end_idx: End index
            force_col: Force column name
            time_col: Time column name
            gradient: Raw gradient
            smoothed_gradient: Smoothed gradient
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Get time for x-axis
        time_data = df[time_col].values if time_col else np.arange(len(df))
        
        # Plot 1: Force waveform with boundaries
        ax = axes[0]
        ax.plot(time_data, df[force_col].values, 'b-', linewidth=2, label='Force Signal')
        ax.axvline(time_data[start_idx], color='g', linestyle='--', linewidth=2, label=f'Start: {time_data[start_idx]}')
        ax.axvline(time_data[end_idx], color='r', linestyle='--', linewidth=2, label=f'End: {time_data[end_idx]}')
        ax.fill_between(time_data[start_idx:end_idx + 1], 
                        df[force_col].iloc[start_idx:end_idx + 1].min(),
                        df[force_col].iloc[start_idx:end_idx + 1].max(),
                        alpha=0.2, color='green', label='Extracted Region')
        ax.set_xlabel('Time')
        ax.set_ylabel('Force Signal')
        ax.set_title('Force Waveform with Detected Boundaries')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Raw gradient
        ax = axes[1]
        ax.plot(time_data, gradient, 'b-', linewidth=1, alpha=0.7, label='Raw Gradient')
        ax.axvline(time_data[start_idx], color='g', linestyle='--', linewidth=2)
        ax.axvline(time_data[end_idx], color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Gradient (Force Derivative)')
        ax.set_title('Raw Gradient of Force Signal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Smoothed gradient
        ax = axes[2]
        ax.plot(time_data, smoothed_gradient, 'b-', linewidth=2, label='Smoothed Gradient')
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(time_data[start_idx], color='g', linestyle='--', linewidth=2, label='Start')
        ax.axvline(time_data[end_idx], color='r', linestyle='--', linewidth=2, label='End')
        ax.fill_between(time_data[start_idx:end_idx + 1],
                        smoothed_gradient[start_idx:end_idx + 1].min(),
                        smoothed_gradient[start_idx:end_idx + 1].max(),
                        alpha=0.2, color='green')
        ax.set_xlabel('Time')
        ax.set_ylabel('Smoothed Gradient')
        ax.set_title('Smoothed Gradient (Moving Average)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'waveform_analysis.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to: {save_path}")
        plt.close()
    
    def process(self, gradient_threshold: float = 0.5, 
               window_size: int = 5,
               visualize: bool = True) -> str:
        """
        Complete processing pipeline.
        
        Args:
            gradient_threshold: Threshold for gradient detection
            window_size: Window size for smoothing
            visualize: Whether to create visualization
            
        Returns:
            Path to output CSV file
        """
        logger.info("=" * 60)
        logger.info("Starting Waveform Boundary Detection")
        logger.info("=" * 60)
        
        # Load data
        df = self.load_data()
        
        # Analyze gradient
        start_idx, end_idx, start_time, end_time, force_col, time_col, gradient, smoothed_gradient = \
            self.analyze_gradient(df, gradient_threshold, window_size)
        
        # Filter data
        filtered_df = self.filter_data(df, start_idx, end_idx)
        
        # Save output
        output_path = self.save_output(filtered_df, start_time, end_time)
        
        # Visualize
        if visualize:
            self.visualize_analysis(df, start_idx, end_idx, force_col, time_col,
                                   gradient, smoothed_gradient)
        
        logger.info("=" * 60)
        logger.info("Processing Complete")
        logger.info("=" * 60)
        
        return output_path


def main():
    """Main execution"""
    input_file = r"C:\SenzrTech\Multi-channel\Multi-channel-input.xlsx"
    output_dir = r"C:\SenzrTech\Multi-channel"
    
    detector = WaveformBoundaryDetector(input_file, output_dir)
    
    # Process with gradient threshold and smoothing window
    # Adjust these parameters if needed:
    # - gradient_threshold: Higher = more strict (default 0.5)
    # - window_size: Larger = smoother (default 5)
    output_csv = detector.process(
        gradient_threshold=0.5,
        window_size=5,
        visualize=True
    )
    
    logger.info(f"\n✓ Output CSV: {output_csv}")
    logger.info("You can adjust gradient_threshold and window_size if boundaries are not optimal")


if __name__ == '__main__':
    main()
