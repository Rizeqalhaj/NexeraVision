#!/usr/bin/env python3
"""
Violence Detection MVP - Main Runner
Easy-to-use entry point for the Violence Detection system.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run main module
if __name__ == "__main__":
    from src.main import main
    main()