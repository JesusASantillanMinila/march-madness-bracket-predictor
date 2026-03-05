import streamlit as st
import pandas as pd
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

# --- STATE MANAGEMENT LOGIC ---
# This checks if the keys already exist in session_state. 
# If not, it runs the loader once and stores them.
if 'df_gsheet' not in st.session_state:
    df_gsheet, df_m_teams, df_w_teams = load_all_data()
    st.session_state.df_gsheet = df_gsheet
    st.session_state.df_m_teams = df_m_teams
    st.session_state.df_w_teams = df_w_teams
else:
    # If they already exist, we just pull them from memory
    df_gsheet = st.session_state.df_gsheet
    df_m_teams = st.session_state.df_m_teams
    df_w_teams = st.session_state.df_w_teams

# --- UI AND JOIN LOGIC (Unchanged) ---
st.title("🏀 March Madness Bracket Predictor")

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



# --- EXISTING DATA LOADING (Simplified for snippet) ---
# (Keep your existing load_all_data() and session_state logic here)

def get_prediction(team1, team2, df_probs):
    """Finds the probability of team1 beating team2 from the gsheet data."""
    # Kaggle IDs are usually formatted as 'Year_LowerID_HigherID'
    # For this simulation, we'll look for the match in your joined df_team_results
    match = df_probs[((df_probs['Team Name 1'] == team1) & (df_probs['Team Name 2'] == team2)) |
                     ((df_probs['Team Name 1'] == team2) & (df_probs['Team Name 2'] == team1))]
    
    if not match.empty:
        pred = float(match.iloc[0]['Pred'])
        # If team1 is actually 'Team 2' in the CSV, the probability is 1 - Pred
        if match.iloc[0]['Team Name 1'] == team2:
            return 1 - pred
        return pred
    return 0.5  # Default if match not found

# --- UI SETTINGS ---
st.sidebar.header("Tournament Settings")
league = st.sidebar.selectbox("Select League", ["Men's League", "Women's League"])

# Filter data based on league
df_filtered = df_team_results[df_team_results['League'] == league].reset_index(drop=True)

# 1. INITIAL TEAM SELECTION (Top 64 based on available data)
# In a real scenario, you'd provide a multiselect or a predefined list
initial_teams = list(pd.unique(df_filtered[['Team Name 1', 'Team Name 2']].values.ravel('K')))[:64]

st.subheader(f"🏀 {league} Bracket Simulation")

if len(initial_teams) < 64:
    st.warning(f"Only {len(initial_teams)} teams found in data. Need 64 for a full bracket.")
    # For demo purposes, we'll proceed with what we have
    num_teams = 2**int(np.log2(len(initial_teams))) 
    current_round_teams = initial_teams[:num_teams]
else:
    current_round_teams = initial_teams[:64]

# --- SIMULATION LOGIC ---
bracket_results = {}
round_names = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]
temp_teams = current_round_teams.copy()

# Visualize rounds in columns
cols = st.columns(len(round_names))

round_idx = 0
while len(temp_teams) > 1:
    winners = []
    with cols[round_idx]:
        st.caption(f"**{round_names[round_idx]}**")
        for i in range(0, len(temp_teams), 2):
            t1 = temp_teams[i]
            t2 = temp_teams[i+1]
            
            # Simulate match
            prob = get_prediction(t1, t2, df_filtered)
            winner = t1 if prob >= 0.5 else t2
            winners.append(winner)
            
            # Display Matchup
            st.info(f"{t1} vs {t2}  \n**Winner: {winner}**")
            
    temp_teams = winners
    round_idx += 1

# Display Winner
if len(temp_teams) == 1:
    st.balloons()
    st.success(f"🏆 **Tournament Champion: {temp_teams[0]}**")

# --- DATA EDITOR FOR USERS ---
with st.expander("Edit Initial Matchups/Predictions"):
    edited_df = st.data_editor(df_filtered)
