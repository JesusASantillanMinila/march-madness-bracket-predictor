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

# --- (Keep your existing data loading logic here) ---
# Assuming df_team_results is already created from your join logic

st.title("🏀 March Madness Bracket Predictor")
st.markdown("""
    Welcome to the **Interactive Bracket Simulator**. 
    Select your league and simulate matches based on AI-driven win probabilities!
""")

# --- Filter by League ---
leagues = df_team_results['League'].unique()
selected_league = st.sidebar.selectbox("Select League", leagues)
df_filtered = df_team_results[df_team_results['League'] == selected_league].copy()

# Ensure Pred is numeric for simulation
df_filtered['Pred'] = pd.to_numeric(df_filtered['Pred'], errors='coerce')

# --- Simulation Logic ---
def simulate_match(team1, team2, prob_t1_wins):
    """Simulates a winner based on the probability."""
    if np.random.random() < prob_t1_wins:
        return team1, prob_t1_wins
    else:
        return team2, 1 - prob_t1_wins

# --- Bracket UI ---
tabs = st.tabs(["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"])

# We will use session state to keep track of winners across rounds
if 'bracket_winners' not in st.session_state:
    st.session_state.bracket_winners = {}

# Helper to render a "Matchup Card"
def render_matchup(t1_name, t2_name, prob, match_id):
    with st.container(border=True):
        col_a, col_vs, col_b = st.columns([4, 1, 4])
        with col_a:
            st.write(f"**{t1_name}**")
            st.caption(f"Win Prob: {prob:.1%}")
        with col_vs:
            st.write("VS")
        with col_b:
            st.write(f"**{t2_name}**")
            st.caption(f"Win Prob: {1-prob:.1%}")
        
        if st.button(f"Simulate Match {match_id}", key=f"btn_{match_id}"):
            winner, win_prob = simulate_match(t1_name, t2_name, prob)
            st.session_state.bracket_winners[match_id] = winner
            st.success(f"🏆 Winner: {winner}")

# --- ROUND 1: Initial 64 (Simplified for example) ---
with tabs[0]:
    st.header("Round of 64")
    # Displaying first 8 matchups as a sample
    matchups = df_filtered.head(8) 
    
    cols = st.columns(2)
    for idx, row in matchups.iterrows():
        with cols[idx % 2]:
            render_matchup(row['Team Name 1'], row['Team Name 2'], row['Pred'], f"R1_{idx}")

# --- CHAMPIONSHIP HUD ---
st.sidebar.divider()
st.sidebar.header("🏆 Your Champion")
if st.sidebar.button("Reset Bracket"):
    st.session_state.bracket_winners = {}
    st.rerun()

# Display current progress in Sidebar
for m_id, winner in st.session_state.bracket_winners.items():
    st.sidebar.write(f"{m_id}: **{winner}**")

# --- VISUAL STYLING ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #ff4b4b;
    }
    .stButton button {
        width: 100%;
        border-radius: 20px;
        border: 1px solid #ff4b4b;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
