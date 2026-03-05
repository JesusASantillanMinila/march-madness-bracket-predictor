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

# Prediction Engine: Returns (WinnerName, ProbForTeam1, WinnerSeed)
def get_matchup_result(t1, t2, s1, s2):
    # Check T1 vs T2
    match = df_lookup[(df_lookup['TeamName_1'] == t1) & (df_lookup['TeamName_2'] == t2)]
    if not match.empty:
        p1 = float(match.iloc[0]['Pred'])
        if p1 >= 0.5:
            return t1, p1, s1
        else:
            return t2, p1, s2
    
    # Check T2 vs T1 (Inverse)
    match = df_lookup[(df_lookup['TeamName_1'] == t2) & (df_lookup['TeamName_2'] == t1)]
    if not match.empty:
        p2 = float(match.iloc[0]['Pred'])
        p1 = 1 - p2 # Probability for T1
        if p1 >= 0.5:
            return t1, p1, s1
        else:
            return t2, p1, s2
            
    return t1, 0.5, s1 # Fallback

def format_matchup_html(t1, t2, p1, s1, s2):
    p2 = 1 - p1
    # Formatting: Bold and Underline if probability >= 50%
    line1 = f"<u>**Seed {s1}: {t1} {p1:.0%}**</u>" if p1 >= 0.5 else f"Seed {s1}: {t1} {p1:.0%}"
    line2 = f"<u>**Seed {s2}: {t2} {p2:.0%}**</u>" if p2 > 0.5 else f"Seed {s2}: {t2} {p2:.0%}"
    return f"{line1} vs {line2}"

# --- UI INTERFACE ---
st.title("🏀 March Madness Bracket Simulator")
st.markdown("Select your 64 teams and watch the model predict every round with probabilities.")

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
            t = st.selectbox(f"Seed {seed}", all_team_names, index=(seed + i*17) % len(all_team_names), key=f"{region}_{seed}")
            region_teams.append(t)
        bracket_inputs[region] = region_teams

st.divider()

if st.button("🔥 Run Simulation", type="primary", use_container_width=True):
    final_four_data = [] # List of (TeamName, Seed)
    
    res_cols = st.columns(4)
    for i, region in enumerate(regions):
        with res_cols[i]:
            st.header(f"{region} Region")
            teams = bracket_inputs[region]
            
            # Round 1 (64) -> Standard NCAA Matchups
            r1_idx = [(0,15), (7,8), (4,11), (3,12), (5,10), (2,13), (6,9), (1,14)]
            st.markdown("#### Round of 64")
            r32_results = []
            for m in r1_idx:
                res = get_matchup_result(teams[m[0]], teams[m[1]], m[0]+1, m[1]+1)
                st.markdown(format_matchup_html(teams[m[0]], teams[m[1]], res[1], m[0]+1, m[1]+1), unsafe_allow_html=True)
                r32_results.append(res) # (WinnerName, Prob, WinnerSeed)

            # Round 2 (32)
            st.markdown("#### Round of 32")
            s16_results = []
            for j in range(0, 8, 2):
                t1, p1_old, s1 = r32_results[j]
                t2, p2_old, s2 = r32_results[j+1]
                res = get_matchup_result(t1, t2, s1, s2)
                st.markdown(format_matchup_html(t1, t2, res[1], s1, s2), unsafe_allow_html=True)
                s16_results.append(res)

            # Sweet 16
            st.markdown("#### Sweet 16")
            e8_results = []
            for j in range(0, 4, 2):
                t1, p1_old, s1 = s16_results[j]
                t2, p2_old, s2 = s16_results[j+1]
                res = get_matchup_result(t1, t2, s1, s2)
                st.markdown(format_matchup_html(t1, t2, res[1], s1, s2), unsafe_allow_html=True)
                e8_results.append(res)

            # Elite 8
            st.markdown("#### Elite 8")
            t1, p1_old, s1 = e8_results[0]
            t2, p2_old, s2 = e8_results[1]
            reg_champ_res = get_matchup_result(t1, t2, s1, s2)
            st.markdown(format_matchup_html(t1, t2, reg_champ_res[1], s1, s2), unsafe_allow_html=True)
            
            st.success(f"**{region} Champ: {reg_champ_res[0]}**")
            final_four_data.append((reg_champ_res[0], reg_champ_res[2]))

    st.divider()
    st.header("🏆 The Final Four")
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        st.subheader("National Semifinals")
        # Game 1
        ga_t1, ga_s1 = final_four_data[0]
        ga_t2, ga_s2 = final_four_data[1]
        res1 = get_matchup_result(ga_t1, ga_t2, ga_s1, ga_s2)
        st.markdown(format_matchup_html(ga_t1, ga_t2, res1[1], f"{ga_s1}({regions[0]})", f"{ga_s2}({regions[1]})"), unsafe_allow_html=True)
        
        # Game 2
        gb_t1, gb_s1 = final_four_data[2]
        gb_t2, gb_s2 = final_four_data[3]
        res2 = get_matchup_result(gb_t1, gb_t2, gb_s1, gb_s2)
        st.markdown(format_matchup_html(gb_t1, gb_t2, res2[1], f"{gb_s1}({regions[2]})", f"{gb_s2}({regions[3]})"), unsafe_allow_html=True)

    with f_col2:
        st.subheader("National Championship")
        final_t1, final_s1 = res1[0], res1[2]
        final_t2, final_s2 = res2[0], res2[2]
        champ_res = get_matchup_result(final_t1, final_t2, final_s1, final_s2)
        st.markdown(format_matchup_html(final_t1, final_t2, champ_res[1], "Finalist", "Finalist"), unsafe_allow_html=True)
        
        st.balloons()
        st.title(f"👑 {champ_res[0]}")
