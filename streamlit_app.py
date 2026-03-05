import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import gdown
import numpy as np

st.set_page_config(page_title="🏀 March Madness Bracket Predictor", layout="wide")

# ─────────────────────────────────────────────
# 1.  Credentials & Data Loading
# ─────────────────────────────────────────────
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

def load_all_data():
    creds_dict = st.secrets["gsheet_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    gc = gspread.authorize(creds)

    spreadsheet_key = st.secrets["KAGGLE_SUBMISSION_ID"]
    book = gc.open_by_key(spreadsheet_key)
    worksheet = book.worksheet("data")
    table = worksheet.get_all_values()
    df_gsheet = pd.DataFrame(table[1:], columns=table[0])
    df_gsheet[['Team 1', 'Team 2']] = (
        df_gsheet['ID'].str.split('_', expand=True).iloc[:, [1, 2]]
    ).astype(int)
    df_gsheet['Pred'] = pd.to_numeric(df_gsheet['Pred'], errors='coerce')

    m_csv_url = f'https://drive.google.com/uc?id={st.secrets["M_TEAM_NAMES_ID"]}'
    m_path = gdown.download(m_csv_url, quiet=True, fuzzy=True)
    df_m_teams = pd.read_csv(m_path)
    df_m_teams['League'] = "Men's"

    w_csv_url = f'https://drive.google.com/uc?id={st.secrets["W_TEAM_NAMES_ID"]}'
    w_path = gdown.download(w_csv_url, quiet=True, fuzzy=True)
    df_w_teams = pd.read_csv(w_path)
    df_w_teams['League'] = "Women's"

    # Build combined lookup: TeamID -> TeamName, League
    df_all_teams = pd.concat([df_m_teams, df_w_teams], ignore_index=True)

    # Build df_team_results
    df = pd.merge(df_gsheet, df_m_teams[['TeamID','TeamName']].rename(columns={'TeamName':'TN1M'}),
                  left_on='Team 1', right_on='TeamID', how='left').drop(columns={'TeamID','LastD1Season','FirstD1Season'})
    df = pd.merge(df, df_m_teams[['TeamID','TeamName']].rename(columns={'TeamName':'TN2M'}),
                  left_on='Team 2', right_on='TeamID', how='left').drop(columns={'TeamID','LastD1Season','FirstD1Season'})
    df = pd.merge(df, df_w_teams[['TeamID','TeamName']].rename(columns={'TeamName':'TN1W','League':'Lg1W'}),
                  left_on='Team 1', right_on='TeamID', how='left').drop(columns={'TeamID','LastD1Season','FirstD1Season'})
    df = pd.merge(df, df_w_teams[['TeamID','TeamName']].rename(columns={'TeamName':'TN2W'}),
                  left_on='Team 2', right_on='TeamID', how='left').drop(columns={'TeamID','LastD1Season','FirstD1Season'})

    df['Team Name 1'] = df['TN1M'].combine_first(df['TN1W'])
    df['Team Name 2'] = df['TN2M'].combine_first(df['TN2W'])
    df['League'] = df['TN1M'].notna().map({True: "Men's", False: "Women's"})
    df_team_results = df[['ID','Pred','Team 1','Team 2','Team Name 1','Team Name 2','League']].copy()

    return df_team_results, df_all_teams

# ─────────────────────────────────────────────
# 2.  Session State Bootstrap
# ─────────────────────────────────────────────
if 'data_loaded' not in st.session_state:
    with st.spinner("Loading data…"):
        df_team_results, df_all_teams = load_all_data()
    st.session_state.df_team_results = df_team_results
    st.session_state.df_all_teams    = df_all_teams
    st.session_state.data_loaded     = True

df_team_results = st.session_state.df_team_results
df_all_teams    = st.session_state.df_all_teams

# ─────────────────────────────────────────────
# 3.  Helper: get prediction between two team IDs
# ─────────────────────────────────────────────
def get_pred(team1_id: int, team2_id: int) -> float | None:
    """Return P(team1 beats team2). Handles both orderings stored in df."""
    row = df_team_results[
        ((df_team_results['Team 1'] == team1_id) & (df_team_results['Team 2'] == team2_id))
    ]
    if not row.empty:
        return float(row.iloc[0]['Pred'])
    row = df_team_results[
        ((df_team_results['Team 1'] == team2_id) & (df_team_results['Team 2'] == team1_id))
    ]
    if not row.empty:
        return 1.0 - float(row.iloc[0]['Pred'])
    return None   # matchup not in model

# ─────────────────────────────────────────────
# 4.  Constants
# ─────────────────────────────────────────────
DIVISIONS   = ["East", "West", "South", "Midwest"]
SEEDS       = list(range(1, 17))          # 1-16
ROUND_NAMES = ["Round of 64", "Round of 32", "Sweet 16",
               "Elite 8", "Final Four", "Championship"]
# Seed matchups per March Madness bracket rules
SEED_MATCHUPS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]

# ─────────────────────────────────────────────
# 5.  Bracket State Initialiser
# ─────────────────────────────────────────────
def init_bracket(league: str):
    """Build empty bracket skeleton in session_state."""
    key = f"bracket_{league}"
    if key not in st.session_state:
        bracket = {}
        for div in DIVISIONS:
            bracket[div] = {}
            # Round 0: 16 seed slots  (None = TBD)
            bracket[div][0] = {seed: None for seed in SEEDS}
            # Rounds 1-4: 8,4,2,1 slots per division (indexed by slot 0-based)
            for rnd in range(1, 5):
                slots = 16 // (2 ** rnd)
                bracket[div][rnd] = {i: None for i in range(slots)}
        # Final Four: 2 games (East vs West, South vs Midwest)
        bracket['FF'] = {0: None, 1: None}   # winners of each semi
        bracket['Champion'] = None
        st.session_state[key] = bracket
    return st.session_state[key]

# ─────────────────────────────────────────────
# 6.  UI: League Filter
# ─────────────────────────────────────────────
st.title("🏀 March Madness Bracket Predictor")

league = st.radio("Select League", ["Men's", "Women's"],
                  horizontal=True, key="league_filter")

# Filter team list for dropdowns
league_teams = df_all_teams[df_all_teams['League'] == league].copy()
team_options = {int(row['TeamID']): row['TeamName']
                for _, row in league_teams.iterrows()}
team_names_sorted = ["— Select —"] + sorted(team_options.values())
id_by_name = {v: k for k, v in team_options.items()}

bracket = init_bracket(league)

# ─────────────────────────────────────────────
# 7.  Helper renderers
# ─────────────────────────────────────────────
def team_display(team_id, show_seed=None):
    if team_id is None:
        return "TBD"
    name = team_options.get(team_id, f"ID:{team_id}")
    if show_seed:
        return f"({show_seed}) {name}"
    return name

def matchup_widget(label, t1_id, t2_id, winner_id, pred, key_prefix):
    """Renders a single matchup card and returns the chosen winner_id."""
    t1_name = team_display(t1_id)
    t2_name = team_display(t2_id)

    if t1_id is None and t2_id is None:
        st.markdown(f"<div style='padding:6px;border:1px solid #444;border-radius:6px;"
                    f"background:#1a1a2e;color:#888;font-size:13px'>"
                    f"<b>{label}</b><br>TBD vs TBD</div>", unsafe_allow_html=True)
        return winner_id

    if pred is not None:
        pct1 = f"{pred*100:.1f}%"
        pct2 = f"{(1-pred)*100:.1f}%"
    else:
        pct1 = pct2 = "N/A"

    # Highlight winner
    c1_style = "background:#0d6e2e;color:white;font-weight:bold" if winner_id == t1_id else "background:#222;color:#ccc"
    c2_style = "background:#0d6e2e;color:white;font-weight:bold" if winner_id == t2_id else "background:#222;color:#ccc"

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"<div style='padding:8px;border-radius:6px;{c1_style};font-size:13px'>"
            f"🏀 {t1_name}<br><span style='font-size:11px'>Win Prob: {pct1}</span></div>",
            unsafe_allow_html=True)
        if t1_id is not None:
            if st.button("Pick ✔", key=f"{key_prefix}_t1"):
                return t1_id
    with col_b:
        st.markdown(
            f"<div style='padding:8px;border-radius:6px;{c2_style};font-size:13px'>"
            f"🏀 {t2_name}<br><span style='font-size:11px'>Win Prob: {pct2}</span></div>",
            unsafe_allow_html=True)
        if t2_id is not None:
            if st.button("Pick ✔", key=f"{key_prefix}_t2"):
                return t2_id
    return winner_id

# ─────────────────────────────────────────────
# 8.  Division Brackets (Rounds 0-3)
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📋 Divisional Brackets")

div_tabs = st.tabs(DIVISIONS)

for tab, div in zip(div_tabs, DIVISIONS):
    with tab:
        st.subheader(f"🗂 {div} Region")

        # ── Round 0: Seed Entry ──────────────────────────────
        with st.expander("🔢 Set Teams (Seed Assignments)", expanded=False):
            cols = st.columns(4)
            for idx, seed in enumerate(SEEDS):
                with cols[idx % 4]:
                    current = bracket[div][0][seed]
                    current_name = team_options.get(current, "— Select —") if current else "— Select —"
                    choice = st.selectbox(
                        f"Seed {seed}",
                        team_names_sorted,
                        index=team_names_sorted.index(current_name) if current_name in team_names_sorted else 0,
                        key=f"{league}_{div}_seed_{seed}"
                    )
                    bracket[div][0][seed] = id_by_name.get(choice) if choice != "— Select —" else None

        # ── Rounds 1-4 within division ───────────────────────
        round_labels = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]
        for rnd in range(1, 5):
            st.markdown(f"#### 🔄 {round_labels[rnd-1]}")
            matchups_this_round = SEED_MATCHUPS if rnd == 1 else []
            num_slots = 16 // (2 ** rnd)      # 8, 4, 2, 1
            prev_slots = 16 // (2 ** (rnd-1))  # 16, 8, 4, 2

            cols = st.columns(min(num_slots, 4))

            for slot in range(num_slots):
                with cols[slot % min(num_slots, 4)]:
                    if rnd == 1:
                        # Derive teams from seed matchups
                        s1, s2 = SEED_MATCHUPS[slot]
                        t1_id = bracket[div][0][s1]
                        t2_id = bracket[div][0][s2]
                        lbl = f"Seed {s1} vs {s2}"
                    else:
                        # Propagate winners from previous round
                        p1 = slot * 2
                        p2 = slot * 2 + 1
                        t1_id = bracket[div][rnd-1].get(p1)
                        t2_id = bracket[div][rnd-1].get(p2)
                        lbl = f"Game {slot+1}"

                    pred = get_pred(t1_id, t2_id) if t1_id and t2_id else None
                    current_winner = bracket[div][rnd].get(slot)

                    new_winner = matchup_widget(
                        lbl, t1_id, t2_id, current_winner, pred,
                        key_prefix=f"{league}_{div}_r{rnd}_s{slot}"
                    )
                    bracket[div][rnd][slot] = new_winner

        st.success(f"🏆 {div} Champion: **{team_display(bracket[div][4].get(0))}**")

# ─────────────────────────────────────────────
# 9.  Final Four
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🏟 Final Four")

# East vs West  |  South vs Midwest
ff_pairs = [("East", "West"), ("South", "Midwest")]
ff_cols  = st.columns(2)

for i, (div_a, div_b) in enumerate(ff_pairs):
    with ff_cols[i]:
        st.subheader(f"{div_a} vs {div_b}")
        t1_id = bracket[div_a][4].get(0)
        t2_id = bracket[div_b][4].get(0)
        pred  = get_pred(t1_id, t2_id) if t1_id and t2_id else None
        cw    = bracket['FF'].get(i)
        new_w = matchup_widget(
            f"Final Four Game {i+1}", t1_id, t2_id, cw, pred,
            key_prefix=f"{league}_FF_{i}"
        )
        bracket['FF'][i] = new_w

# ─────────────────────────────────────────────
# 10. Championship
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🥇 Championship Game")

t1_id = bracket['FF'].get(0)
t2_id = bracket['FF'].get(1)
pred  = get_pred(t1_id, t2_id) if t1_id and t2_id else None
cw    = bracket.get('Champion')

new_champ = matchup_widget(
    "🏆 National Championship", t1_id, t2_id, cw, pred,
    key_prefix=f"{league}_Champ"
)
bracket['Champion'] = new_champ

if bracket['Champion']:
    champ_name = team_display(bracket['Champion'])
    st.balloons()
    st.markdown(
        f"<div style='text-align:center;padding:24px;border-radius:12px;"
        f"background:linear-gradient(135deg,#f7971e,#ffd200);color:#111;"
        f"font-size:28px;font-weight:bold;margin-top:16px'>"
        f"🏆 {league} Champion: {champ_name} 🏆</div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# 11. Bracket Summary Table
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📊 Bracket Results Summary")

summary_rows = []
for div in DIVISIONS:
    for rnd in range(1, 5):
        num_slots = 16 // (2 ** rnd)
        for slot in range(num_slots):
            if rnd == 1:
                s1, s2 = SEED_MATCHUPS[slot]
                t1_id = bracket[div][0][s1]
                t2_id = bracket[div][0][s2]
            else:
                t1_id = bracket[div][rnd-1].get(slot*2)
                t2_id = bracket[div][rnd-1].get(slot*2+1)
            winner = bracket[div][rnd].get(slot)
            pred   = get_pred(t1_id, t2_id) if t1_id and t2_id else None

            summary_rows.append({
                "Division": div,
                "Round": ["Round of 64","Round of 32","Sweet 16","Elite 8"][rnd-1],
                "Team 1": team_display(t1_id),
                "Team 2": team_display(t2_id),
                "Pred (T1 Win%)": f"{pred*100:.1f}%" if pred is not None else "N/A",
                "Winner": team_display(winner) if winner else "—"
            })

# Final Four
for i, (div_a, div_b) in enumerate(ff_pairs):
    t1_id = bracket[div_a][4].get(0)
    t2_id = bracket[div_b][4].get(0)
    pred  = get_pred(t1_id, t2_id) if t1_id and t2_id else None
    winner = bracket['FF'].get(i)
    summary_rows.append({
        "Division": "Final Four",
        "Round": f"{div_a} vs {div_b}",
        "Team 1": team_display(t1_id),
        "Team 2": team_display(t2_id),
        "Pred (T1 Win%)": f"{pred*100:.1f}%" if pred is not None else "N/A",
        "Winner": team_display(winner) if winner else "—"
    })

# Championship
t1_id = bracket['FF'].get(0)
t2_id = bracket['FF'].get(1)
pred  = get_pred(t1_id, t2_id) if t1_id and t2_id else None
summary_rows.append({
    "Division": "Championship",
    "Round": "Final",
    "Team 1": team_display(t1_id),
    "Team 2": team_display(t2_id),
    "Pred (T1 Win%)": f"{pred*100:.1f}%" if pred is not None else "N/A",
    "Winner": team_display(bracket.get('Champion')) if bracket.get('Champion') else "—"
})

df_summary = pd.DataFrame(summary_rows)

def highlight_winner(row):
    styles = [''] * len(row)
    if row['Winner'] not in ('—', 'TBD', ''):
        winner_col = list(row.index)
        if 'Winner' in winner_col:
            styles[winner_col.index('Winner')] = 'background-color: #0d6e2e; color: white; font-weight: bold'
    return styles

st.dataframe(
    df_summary.style.apply(highlight_winner, axis=1),
    use_container_width=True,
    hide_index=True
)
