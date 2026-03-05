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
    # --- 1. Read Google Sheet ---
    spreadsheet_key = st.secrets["KAGGLE_SUBMISSION_ID"]
    book = gc.open_by_key(spreadsheet_key)
    worksheet = book.worksheet("data") 
    table = worksheet.get_all_values()
    # Create DF using the first row as header and the rest as data
    df_gsheet = pd.DataFrame(table[1:], columns=table[0])
    df_gsheet[['Team 1', 'Team 2']] = (df_gsheet['ID'].str.split('_', expand=True).iloc[:, [1, 2]]).astype(int)

    # --- 2. Read Men's CSV ---
    m_csv_url = f'https://drive.google.com/uc?id={st.secrets["M_TEAM_NAMES_ID"]}'
    m_path = gdown.download(m_csv_url, quiet=True, fuzzy=True)
    df_m_teams = pd.read_csv(m_path)
    df_m_teams['League_M'] = """Men's League"""

    # --- 3. Read Women's CSV ---
    w_csv_url = f'https://drive.google.com/uc?id={st.secrets["W_TEAM_NAMES_ID"]}'
    w_path = gdown.download(w_csv_url, quiet=True, fuzzy=True)
    df_w_teams = pd.read_csv(w_path)
    df_w_teams['League_W'] = """Women's League"""

    return df_gsheet, df_m_teams, df_w_teams

st.title("🏀 March Madness Bracket Predictor")

df_gsheet, df_m_teams, df_w_teams = load_all_data()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Kaggle Submission (GSheet)")
    st.dataframe(df_gsheet, use_container_width=True)
    num_rows = len(df_gsheet)
    st.metric(label="Total Rows", value=num_rows)    

with col2:
    st.subheader("👨 Men's Team Names (CSV)")
    st.dataframe(df_m_teams, use_container_width=True)

st.divider()

st.subheader("👩 Women's Team Names (CSV)")
st.dataframe(df_w_teams, use_container_width=True)


st.divider()

st.subheader("JOIN Test")
df_team_results = pd.merge(df_gsheet, df_m_teams, left_on = 'Team 1', right_on = 'TeamID', how='left')
df_team_results.rename(columns={'TeamName':'TeamName_1M','League_M':'League_1M'}, inplace=True)
df_team_results = df_team_results[['ID','Pred','Team 1','Team 2','TeamName_1M','League_1M']]

df_team_results = pd.merge(df_team_results, df_m_teams, left_on = 'Team 2', right_on = 'TeamID', how='left')
df_team_results.rename(columns={'TeamName':'TeamName_2M'}, inplace=True)
df_team_results = df_team_results[['ID','Pred','Team 1','Team 2','TeamName_1M','TeamName_2M','League_1M']]

df_team_results = pd.merge(df_team_results, df_w_teams, left_on = 'Team 1', right_on = 'TeamID', how='left')
df_team_results.rename(columns={'TeamName':'TeamName_1W','League_W':'League_1W'}, inplace=True)
df_team_results = df_team_results[['ID','Pred','Team 1','Team 2','TeamName_1M','TeamName_2M','TeamName_1W','League_1M','League_1W']]

df_team_results = pd.merge(df_team_results, df_w_teams, left_on = 'Team 2', right_on = 'TeamID', how='left')
df_team_results.rename(columns={'TeamName':'TeamName_2W'}, inplace=True)
df_team_results = df_team_results[['ID','Pred','Team 1','Team 2','TeamName_1M','TeamName_2M','TeamName_1W','TeamName_2W','League_1M','League_1W']]

# Coalesce the columns
df_team_results['Team Name 1'] = df_team_results['TeamName_1M'].combine_first(df_team_results['TeamName_1W'])
df_team_results['Team Name 2'] = df_team_results['TeamName_2M'].combine_first(df_team_results['TeamName_2W'])
df_team_results['League'] = df_team_results['League_1M'].combine_first(df_team_results['League_1W'])

# Filter out intermediate columns
df_team_results = df_team_results[['ID', 'Pred', 'Team Name 1', 'Team Name 2', 'League']]

st.dataframe(df_team_results, use_container_width=True)
