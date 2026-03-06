import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import gdown
import numpy as np
import os
import random

st.set_page_config(page_title="🏀 March Madness Bracket Predictor", layout="wide")

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
DIVISIONS      = ["East", "West", "South", "Midwest"]
SEEDS          = list(range(1, 17))
SEED_MATCHUPS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
ROUND_LABELS  = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]
FF_PAIRS      = [("East", "West"), ("South", "Midwest")]

DIVISION_COLORS = {"East": "#1a3a5c", "West": "#5c1a1a", "South": "#1a4a2a", "Midwest": "#4a3a1a"}

# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

@st.cache_resource
def load_all_data():
    creds_dict = st.secrets["gsheet_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    gc = gspread.authorize(creds)
    book = gc.open_by_key(st.secrets["KAGGLE_SUBMISSION_ID"])
    worksheet = book.worksheet("data")
    table = worksheet.get_all_values()
    df_gsheet = pd.DataFrame(table[1:], columns=table[0])
    df_gsheet[['Team 1', 'Team 2']] = (df_gsheet['ID'].str.split('_', expand=True).iloc[:, [1, 2]]).astype(int)
    df_gsheet['Pred'] = pd.to_numeric(df_gsheet['Pred'], errors='coerce')

    m_path = gdown.download(f'https://drive.google.com/uc?id={st.secrets["M_TEAM_NAMES_ID"]}', quiet=True, fuzzy=True)
    df_m = pd.read_csv(m_path)
    df_m['League'] = "Men's"

    w_path = gdown.download(f'https://drive.google.com/uc?id={st.secrets["W_TEAM_NAMES_ID"]}', quiet=True, fuzzy=True)
    df_w = pd.read_csv(w_path)
    df_w['League'] = "Women's"

    df_all_teams = pd.concat([df_m, df_w], ignore_index=True)
    
    # Merge names
    df = pd.merge(df_gsheet, df_m[['TeamID','TeamName']].rename(columns={'TeamName':'TN1M'}), left_on='Team 1', right_on='TeamID', how='left').drop(columns='TeamID')
    df = pd.merge(df, df_m[['TeamID','TeamName']].rename(columns={'TeamName':'TN2M'}), left_on='Team 2', right_on='TeamID', how='left').drop(columns='TeamID')
    df = pd.merge(df, df_w[['TeamID','TeamName']].rename(columns={'TeamName':'TN1W'}), left_on='Team 1', right_on='TeamID', how='left').drop(columns='TeamID')
    df = pd.merge(df, df_w[['TeamID','TeamName']].rename(columns={'TeamName':'TN2W'}), left_on='Team 2', right_on='TeamID', how='left').drop(columns='TeamID')

    df['Team Name 1'] = df['TN1M'].combine_first(df['TN1W'])
    df['Team Name 2'] = df['TN2M'].combine_first(df['TN2W'])
    df['League'] = df['TN1M'].notna().map({True: "Men's", False: "Women's"})
    return df[['ID','Pred','Team 1','Team 2','Team Name 1','Team Name 2','League']].copy(), df_all_teams

def load_bracket_csv():
    if os.path.exists("teams.csv"):
        df = pd.read_csv("teams.csv")
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    return None

df_results, df_all_teams = load_all_data()

# ══════════════════════════════════════════════════════════════
# 2. PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════
def get_pred(t1, t2):
    if t1 is None or t2 is None: return None
    row = df_results[(df_results['Team 1'] == t1) & (df_results['Team 2'] == t2)]
    if not row.empty: return float(row.iloc[0]['Pred'])
    row = df_results[(df_results['Team 1'] == t2) & (df_results['Team 2'] == t1)]
    if not row.empty: return 1.0 - float(row.iloc[0]['Pred'])
    return 0.5

def predict_winner(t1, t2):
    p = get_pred(t1, t2)
    return t1 if p >= 0.5 else t2

# ══════════════════════════════════════════════════════════════
# 3. BRACKET STATE & LOGIC
# ══════════════════════════════════════════════════════════════
def build_empty_bracket():
    b = {'FF': {0: None, 1: None}, 'Champion': None, 'overrides': {}}
    for div in DIVISIONS:
        b[div] = {rnd: {i: None for i in range(16 // (2 ** rnd))} for rnd in range(5)}
    return b

def get_default_seeds(league):
    defaults = {div: {s: None for s in SEEDS} for div in DIVISIONS}
    csv_df = load_bracket_csv()
    if csv_df is not None:
        filtered = csv_df[csv_df['league'].str.strip().str.lower() == league.lower()]
        for _, row in filtered.iterrows():
            div, seed, tid = str(row['division']).strip().title(), int(row['seed']), int(row['team_id'])
            if div in DIVISIONS and seed in SEEDS: defaults[div][seed] = tid
    return defaults

def init_bracket(league):
    key = f"bracket_{league}"
    if key not in st.session_state:
        b = build_empty_bracket()
        defaults = get_default_seeds(league)
        for div in DIVISIONS:
            for seed in SEEDS: b[div][0][seed] = defaults[div][seed]
        st.session_state[key] = b
    return st.session_state[key]

def run_simulation(bracket):
    """Cascading Logic: Manual Overrides > Model Predictions"""
    # 1. Divisional Rounds
    for div in DIVISIONS:
        for rnd in range(1, 5):
            for slot in range(16 // (2 ** rnd)):
                # Get teams from previous round
                if rnd == 1:
                    s1, s2 = SEED_MATCHUPS[slot]
                    t1, t2 = bracket[div][0][s1], bracket[div][0][s2]
                else:
                    t1, t2 = bracket[div][rnd-1].get(slot * 2), bracket[div][rnd-1].get(slot * 2 + 1)
                
                # Check manual override for this specific matchup
                override_key = f"{div}_{rnd}_{slot}"
                manual_winner = bracket['overrides'].get(override_key)
                
                if manual_winner and manual_winner in [t1, t2]:
                    bracket[div][rnd][slot] = manual_winner
                elif t1 and t2:
                    bracket[div][rnd][slot] = predict_winner(t1, t2)
                else:
                    bracket[div][rnd][slot] = t1 or t2

    # 2. Final Four
    for i, (div_a, div_b) in enumerate(FF_PAIRS):
        t1, t2 = bracket[div_a][4].get(0), bracket[div_b][4].get(0)
        override_key = f"FF_{i}"
        manual_winner = bracket['overrides'].get(override_key)
        
        if manual_winner and manual_winner in [t1, t2]:
            bracket['FF'][i] = manual_winner
        elif t1 and t2:
            bracket['FF'][i] = predict_winner(t1, t2)
        else:
            bracket['FF'][i] = t1 or t2

    # 3. Championship
    t1, t2 = bracket['FF'].get(0), bracket['FF'].get(1)
    manual_winner = bracket['overrides'].get("Champ")
    if manual_winner and manual_winner in [t1, t2]:
        bracket['Champion'] = manual_winner
    elif t1 and t2:
        bracket['Champion'] = predict_winner(t1, t2)
    else:
        bracket['Champion'] = t1 or t2
    return bracket

# ══════════════════════════════════════════════════════════════
# 4. UI COMPONENTS
# ══════════════════════════════════════════════════════════════
st.markdown("<h1 style='text-align:center;'>🏀 Interactive Bracket Predictor</h1>", unsafe_allow_html=True)
league = st.radio("🏆 Select League", ["Men's", "Women's"], horizontal=True)
league_teams = df_all_teams[df_all_teams['League'] == league].copy()
team_options = {int(r['TeamID']): r['TeamName'] for _, r in league_teams.iterrows()}
id_by_name = {v: k for k, v in team_options.items()}
names_sorted = ["— Select —"] + sorted(team_options.values())

bracket = init_bracket(league)

# Always run simulation to propagate overrides
bracket = run_simulation(bracket)

# Step 1: Assignment (Truncated for space, assume existing logic)
with st.expander("📝 Step 1: Assign Starting 64 Teams"):
    div_tabs = st.tabs([f"{d} Region" for d in DIVISIONS])
    for tab, div in zip(div_tabs, DIVISIONS):
        with tab:
            cols = st.columns(4)
            for idx, seed in enumerate(SEEDS):
                with cols[idx % 4]:
                    curr = bracket[div][0][seed]
                    cur_name = team_options.get(curr, "— Select —")
                    choice = st.selectbox(f"Seed {seed}", names_sorted, index=names_sorted.index(cur_name) if cur_name in names_sorted else 0, key=f"s_{league}_{div}_{seed}")
                    bracket[div][0][seed] = id_by_name.get(choice)

# Step 2: Interactive Results
st.markdown("---")
st.markdown("## 📊 Step 2: Interactive Results & Manual Picks")
st.info("Pick a winner in any card to override the model. Predictions update automatically.")

def interactive_matchup(t1, t2, current_winner, override_key, label=""):
    n1, n2 = team_options.get(t1, "TBD"), team_options.get(t2, "TBD")
    pred = get_pred(t1, t2)
    
    # UI Logic
    options = ["Model Prediction", n1, n2]
    # Check if we have an existing override
    saved_override = bracket['overrides'].get(override_key)
    default_idx = 0
    if saved_override == t1: default_idx = 1
    elif saved_override == t2: default_idx = 2

    with st.container():
        st.markdown(f"<div style='font-size:11px; color:#888'>{label}</div>", unsafe_allow_html=True)
        choice = st.selectbox("Pick Winner", options, index=default_idx, key=f"ov_{league}_{override_key}", label_visibility="collapsed")
        
        # Save Override
        if choice == "Model Prediction":
            if override_key in bracket['overrides']: del bracket['overrides'][override_key]
        elif choice == n1: bracket['overrides'][override_key] = t1
        elif choice == n2: bracket['overrides'][override_key] = t2

        # Styling result
        win_color = "#155724" if current_winner else "#333"
        st.markdown(f"""
            <div style='border:1px solid #444; border-radius:5px; padding:5px; background:{win_color}'>
                <div style='display:flex; justify-content:space-between; font-size:13px'>
                    <span>{'🏆 ' if current_winner==t1 else ''}{n1}</span>
                    <span style='opacity:0.6'>{pred*100:.1f}%</span>
                </div>
                <div style='display:flex; justify-content:space-between; font-size:13px'>
                    <span>{'🏆 ' if current_winner==t2 else ''}{n2}</span>
                    <span style='opacity:0.6'>{(1-pred)*100:.1f}%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Tabs for Rounds
res_tabs = st.tabs(["Divisional Rounds", "Final Four & Championship"])

with res_tabs[0]:
    div_sel = st.selectbox("View Region", DIVISIONS)
    for rnd in range(1, 5):
        st.markdown(f"#### {ROUND_LABELS[rnd-1]}")
        slots = 16 // (2 ** rnd)
        cols = st.columns(min(slots, 4))
        for s in range(slots):
            with cols[s % 4]:
                if rnd == 1:
                    s1, s2 = SEED_MATCHUPS[s]
                    t1, t2 = bracket[div_sel][0][s1], bracket[div_sel][0][s2]
                    lbl = f"#{s1} vs #{s2}"
                else:
                    t1, t2 = bracket[div_sel][rnd-1].get(s*2), bracket[div_sel][rnd-1].get(s*2+1)
                    lbl = f"Game {s+1}"
                interactive_matchup(t1, t2, bracket[div_sel][rnd].get(s), f"{div_sel}_{rnd}_{s}", lbl)

with res_tabs[1]:
    c1, c2 = st.columns(2)
    # Final Four
    for i, (da, db) in enumerate(FF_PAIRS):
        with c1 if i==0 else c2:
            st.markdown(f"#### Semifinal: {da} vs {db}")
            t1, t2 = bracket[da][4].get(0), bracket[db][4].get(0)
            interactive_matchup(t1, t2, bracket['FF'].get(i), f"FF_{i}", "Final Four")
    
    # Champ
    st.markdown("---")
    st.markdown("### 🥇 National Championship")
    t1, t2 = bracket['FF'].get(0), bracket['FF'].get(1)
    interactive_matchup(t1, t2, bracket['Champion'], "Champ", "Championship Game")
    
    if bracket['Champion']:
        st.balloons()
        st.success(f"Predicted Champion: {team_options.get(bracket['Champion'])}")

# Reset Button
if st.button("🗑️ Clear All Manual Overrides"):
    bracket['overrides'] = {}
    st.rerun()
