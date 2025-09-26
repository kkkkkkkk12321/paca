#!/usr/bin/env python3
# PACA Environment Setup Script
import os
import sys
from pathlib import Path

def setup_paca_environment():
    project_root = Path(__file__).parent.absolute()

    # Environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONPATH'] = str(project_root)
    os.environ['PACA_LOG_LEVEL'] = 'INFO'

    # Add to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("PACA environment configured successfully!")
    return True

if __name__ == "__main__":
    setup_paca_environment()
