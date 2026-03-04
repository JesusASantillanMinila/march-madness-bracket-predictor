
import streamlit as st
import pandas as pd
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import gdown
import numpy as np

st.set_page_config(page_title="🏀 March Madness Bracket Predictor", layout="wide")

# 1. Setup Credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = st.secrets["gsheet_service_account"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
gc = gspread.authorize(creds)

def load_all_data():
    # --- 1. Read Google Sheet (Your requested shape) ---
    spreadsheet_key = st.secrets["KAGGLE_SUBMISSION_ID"]
    book = gc.open_by_key(spreadsheet_key)
    worksheet = book.worksheet("data") 
    table = worksheet.get_all_values()
    # Create DF using the first row as header and the rest as data
    df_gsheet = pd.DataFrame(table[1:], columns=table[0])

    # --- 2. Read Men's CSV ---
    m_csv_url = f'https://drive.google.com/uc?id={st.secrets["M_TEAM_NAMES_ID"]}'
    m_path = gdown.download(m_csv_url, quiet=True, fuzzy=True)
    df_m_teams = pd.read_csv(m_path)

    # --- 3. Read Women's CSV ---
    w_csv_url = f'https://drive.google.com/uc?id={st.secrets["W_TEAM_NAMES_ID"]}'
    w_path = gdown.download(w_csv_url, quiet=True, fuzzy=True)
    df_w_teams = pd.read_csv(w_path)

    return df_gsheet, df_m_teams, df_w_teams

st.title("🏀 March Madness Bracket Predictor")

df_gsheet, df_m_teams, df_w_teams = load_all_data()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Kaggle Submission (GSheet)")
    st.dataframe(df_gsheet, use_container_width=True)

with col2:
    st.subheader("👨 Men's Team Names (CSV)")
    st.dataframe(df_m_teams, use_container_width=True)

st.divider()

st.subheader("👩 Women's Team Names (CSV)")
st.dataframe(df_w_teams, use_container_width=True)
