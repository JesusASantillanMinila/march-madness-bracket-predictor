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


# --- NEW: Simulation Logic ---
def get_win_probability(team1, team2, df_probs):
    """Looks up the prediction value for two teams."""
    # Match names back to IDs or use the 'ID' string format: YYYY_ID1_ID2
    # For simplicity in this demo, we assume df_probs has 'Team Name 1', 'Team Name 2', and 'Pred'
    match = df_probs[((df_probs['Team Name 1'] == team1) & (df_probs['Team Name 2'] == team2)) |
                     ((df_probs['Team Name 1'] == team2) & (df_probs['Team Name 2'] == team1))]
    
    if match.empty:
        return 0.5  # Default if no data found
    
    prob = float(match.iloc[0]['Pred'])
    # If the order is swapped in the lookup, invert the probability
    if match.iloc[0]['Team Name 1'] == team2:
        prob = 1 - prob
    return prob

def simulate_round(teams, df_probs):
    winners = []
    matchups = []
    for i in range(0, len(teams), 2):
        t1, t2 = teams[i], teams[i+1]
        prob = get_win_probability(t1, t2, df_probs)
        # Simulation: If Pred > 0.5, Team 1 is favored
        winner = t1 if prob >= 0.5 else t2
        winners.append(winner)
        matchups.append(f"{t1} vs {t2}")
    return winners, matchups

# --- UI Setup ---
st.title("🏀 Interactive Bracket Simulator")

# 1. League Filter
league = st.selectbox("Select League", ["Men's League", "Women's League"])
filtered_results = df_team_results[df_team_results['League'] == league]

# 2. Team Selection (Initial 64)
# Get unique list of all possible teams in the data
all_teams = sorted(list(set(filtered_results['Team Name 1'].tolist() + filtered_results['Team Name 2'].tolist())))

st.subheader("📝 Seed the Tournament")
selected_64 = st.multiselect("Select 64 Teams (in seed order)", all_teams, max_selections=64)

if len(selected_64) == 64:
    if st.button("🚀 Run Simulation"):
        # We will store rounds in a list of lists
        bracket_rounds = [selected_64]
        current_teams = selected_64
        
        while len(current_teams) > 1:
            current_teams, _ = simulate_round(current_teams, filtered_results)
            bracket_rounds.append(current_teams)
        
        # 3. Bracket Visualization
        st.divider()
        cols = st.columns(len(bracket_rounds))
        titles = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final 4", "Championship", "Winner"]
        
        for i, round_teams in enumerate(bracket_rounds):
            with cols[i]:
                st.markdown(f"### {titles[i]}")
                for team in round_teams:
                    st.info(team)
                    # Add spacing to mimic bracket height
                    st.write("") 

else:
    st.warning(f"Please select exactly 64 teams to begin. Currently selected: {len(selected_64)}")
