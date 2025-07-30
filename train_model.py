#!/usr/bin/env python3
"""
Simple script to train the improved churn prediction model
Run this script to train and save the model before using the Streamlit app
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main function from model.py
from model import main

if __name__ == "__main__":
    print("Starting model training...")
    main()
    print("\nModel training completed! You can now run the Streamlit app with:")
    print("streamlit run app.py") 