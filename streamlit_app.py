import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import gdown
import numpy as np

st.set_page_config(page_title="🏀 March Madness Bracket Predictor", layout="wide")

# 1. Setup Credentials & Data Loading
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

@st.cache_data
def load_all_data():
    # --- 1. Read Google Sheet ---
    creds_dict = st.secrets["gsheet_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    gc = gspread.authorize(creds)
    
    spreadsheet_key = st.secrets["KAGGLE_SUBMISSION_ID"]
    book = gc.open_by_key(spreadsheet_key)
    worksheet = book.worksheet("data") 
    table = worksheet.get_all_values()
    
    df_gsheet = pd.DataFrame(table[1:], columns=table[0])
    df_gsheet[['Team 1', 'Team 2']] = (df_gsheet['ID'].str.split('_', expand=True).iloc[:, [1, 2]]).astype(int)
    df_gsheet['Pred'] = pd.to_numeric(df_gsheet['Pred'], errors='coerce')

    # --- 2. Read Men's CSV ---
    m_csv_url = f'https://drive.google.com/uc?id={st.secrets["M_TEAM_NAMES_ID"]}'
    m_path = gdown.download(m_csv_url, quiet=True, fuzzy=True)
    df_m_teams = pd.read_csv(m_path)
    df_m_teams['League'] = "Men's"

    # --- 3. Read Women's CSV ---
    w_csv_url = f'https://drive.google.com/uc?id={st.secrets["W_TEAM_NAMES_ID"]}'
    w_path = gdown.download(w_csv_url, quiet=True, fuzzy=True)
    df_w_teams = pd.read_csv(w_path)
    df_w_teams['League'] = "Women's"

    return df_gsheet, df_m_teams, df_w_teams

# --- STATE MANAGEMENT ---
if 'data_loaded' not in st.session_state:
    df_gsheet, df_m_teams, df_w_teams = load_all_data()
    st.session_state.df_gsheet = df_gsheet
    st.session_state.df_m_teams = df_m_teams
    st.session_state.df_w_teams = df_w_teams
    st.session_state.data_loaded = True

# Mapping for predictions
def build_lookup_table():
    m_map = st.session_state.df_m_teams.set_index('TeamID')['TeamName'].to_dict()
    w_map = st.session_state.df_w_teams.set_index('TeamID')['TeamName'].to_dict()
    full_map = {**m_map, **w_map}
    
    df = st.session_state.df_gsheet.copy()
    df['TeamName_1'] = df['Team 1'].map(full_map)
    df['TeamName_2'] = df['Team 2'].map(full_map)
    return df, sorted(list(full_map.values()))

df_lookup, all_team_names = build_lookup_table()

# Prediction Engine
def predict_winner(t1, t2):
    # Check T1 vs T2
    match = df_lookup[(df_lookup['TeamName_1'] == t1) & (df_lookup['TeamName_2'] == t2)]
    if not match.empty:
        return t1 if match.iloc[0]['Pred'] >= 0.5 else t2
    
    # Check T2 vs T1 (Inverse)
    match = df_lookup[(df_lookup['TeamName_1'] == t2) & (df_lookup['TeamName_2'] == t1)]
    if not match.empty:
        # If Pred is for T2 winning over T1
        return t2 if match.iloc[0]['Pred'] >= 0.5 else t1
    
    return t1 # Fallback

# --- UI INTERFACE ---
st.title("🏀 March Madness Bracket Simulator")
st.markdown("Select your 64 teams and let the model predict the path to the championship.")

# Region Setup
regions = ["East", "West", "South", "Midwest"]
bracket_inputs = {}

st.divider()
st.header("1. Set the Seeds")
reg_cols = st.columns(4)

for i, region in enumerate(regions):
    with reg_cols[i]:
        st.subheader(f"Region: {region}")
        region_teams = []
        for seed in range(1, 17):
            # Default selection shifts slightly so they aren't all the same team on load
            t = st.selectbox(f"Seed {seed}", all_team_names, index=(seed + i*5) % len(all_team_names), key=f"{region}_{seed}")
            region_teams.append(t)
        bracket_inputs[region] = region_teams

st.divider()

if st.button("🔥 Run Simulation", type="primary", use_container_width=True):
    final_four = []
    
    [Image of NCAA basketball tournament bracket structure]
    
    results_cols = st.columns(4)
    for i, region in enumerate(regions):
        with results_cols[i]:
            st.markdown(f"### {region} Bracket")
            teams = bracket_inputs[region]
            
            # Round 1: 1v16, 2v15, 3v14, 4v13, 5v12, 6v11, 7v10, 8v9
            r1_matchups = [(0,15), (7,8), (4,11), (3,12), (5,10), (2,13), (6,9), (1,14)]
            r32_teams = [predict_winner(teams[m[0]], teams[m[1]]) for m in r1_matchups]
            
            with st.expander("Round of 32"):
                st.write(r32_teams)
            
            # Round 2 -> Sweet 16
            s16_teams = [predict_winner(r32_teams[0], r32_teams[1]), 
                         predict_winner(r32_teams[2], r32_teams[3]),
                         predict_winner(r32_teams[4], r32_teams[5]),
                         predict_winner(r32_teams[6], r32_teams[7])]
            
            with st.expander("Sweet 16"):
                st.write(s16_teams)
                
            # Elite 8
            e8_teams = [predict_winner(s16_teams[0], s16_teams[1]),
                        predict_winner(s16_teams[2], s16_teams[3])]
            
            # Regional Final
            regional_
