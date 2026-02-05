#!/usr/bin/env python
"""
Multi-Channel PPG Analysis Web Application
===========================================
Visualizes intermediate processing steps and features from multi-channel pipeline.

Usage:
    python run_multichannel_web_app.py --data ./output

Then open: http://localhost:5001
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import web app
from src.multichannel_web_app import create_app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Channel PPG Analysis Web Interface')
    parser.add_argument('--data', type=str, default='./output',
                       help='Output directory containing processed files (default: ./output)')
    parser.add_argument('--port', type=int, default=5001,
                       help='Port to run server on (default: 5001)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')

    args = parser.parse_args()

    # Create Flask app with data directory
    app = create_app(args.data)

    print("=" * 70)
    print("Multi-Channel PPG Analysis Web Interface")
    print("=" * 70)
    print(f"\nData directory: {args.data}")
    print(f"\nStarting server on {args.host}:{args.port}...")
    print(f"Open your browser and navigate to: http://localhost:{args.port}")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)

    app.run(debug=True, host=args.host, port=args.port)
