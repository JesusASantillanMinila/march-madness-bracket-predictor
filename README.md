March Madness Bracket Predictor
A high-performance Streamlit dashboard for simulating the NCAA March Madness tournament. This tool allows users to predict tournament outcomes for both Men's and Women's leagues using either automated model predictions (powered by XGBoost) or custom manual picks.

Features
-Dual Simulation Modes:
--Auto-Simulate: Uses a pre-trained XGBoost model to predict winners based on historical win probabilities.
--Manual Picks: Allows users to act as the Selection Committee, making their own choices while viewing model-calculated win probabilities for guidance.
-Comprehensive Data Integration: Fetches real-time prediction data from Google Sheets and team metadata from Google Drive.
-Men's and Women's Leagues: Full support for both NCAA tournaments with independent bracket states.
-Intelligent Gating: In manual mode, future rounds (Sweet 16, Elite 8, etc.) remain locked until the preceding matchups are decided.
-Visual Analytics:
--Color-coded regional themes.
--Win probability badges showing the percentage chance of victory.
--Detailed summary table highlighting Upset picks where the user deviates from the model.

Tech Stack
-Frontend and App Framework: Streamlit
-Data Manipulation: pandas, numpy
-Integration: gspread (Google Sheets API), gdown (Drive downloads)
-Machine Learning Backend: Predictions generated via XGBoost (sourced from a private repository).

Getting Started
-Prerequisites
--Python 3.8 or higher
--A secrets.toml file (or Streamlit Cloud secrets) containing:
---gsheet_service_account: Google Service Account JSON credentials.
---KAGGLE_SUBMISSION_ID: The ID of your Google Sheet containing predictions.
---M_TEAM_NAMES_ID and W_TEAM_NAMES_ID: Google Drive IDs for team CSV files.
