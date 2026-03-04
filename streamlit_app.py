import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.title("🏀 March Madness Bracket Predictor")

# --- Read CSVs from Google Drive ---
def load_drive_csv(file_id):
    # Construct the direct download URL for Google Drive
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    return pd.read_csv(url)

# --- Read Documents ---
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    df_gsheet = conn.read(
        spreadsheet=st.secrets["ids"]["KAGGLE_SUBMISSION_ID"],
        worksheet="data"
    )

    df_m_teams = load_drive_csv(st.secrets["ids"]["M_TEAM_NAMES_ID"])
    df_w_teams = load_drive_csv(st.secrets["ids"]["W_TEAM_NAMES_ID"])
    
    
except Exception as e:
    st.error(f"Error loading document: {e}")

st.dataframe(df_gsheet)

st.dataframe(df_m_teams)

st.dataframe(df_w_teams)
