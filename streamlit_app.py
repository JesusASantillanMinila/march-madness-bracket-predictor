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
DIVISIONS     = ["East", "West", "South", "Midwest"]
SEEDS         = list(range(1, 17))
SEED_MATCHUPS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
ROUND_LABELS  = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]
FF_PAIRS      = [("East", "West"), ("South", "Midwest")]

DIVISION_COLORS = {
    "East":    "#1a3a5c",
    "West":    "#5c1a1a",
    "South":   "#1a4a2a",
    "Midwest": "#4a3a1a",
}

# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]


def load_all_data():
    creds_dict = st.secrets["gsheet_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    gc = gspread.authorize(creds)

    book      = gc.open_by_key(st.secrets["KAGGLE_SUBMISSION_ID"])
    worksheet = book.worksheet("data")
    table     = worksheet.get_all_values()
    df_gsheet = pd.DataFrame(table[1:], columns=table[0])
    df_gsheet[['Team 1', 'Team 2']] = (
        df_gsheet['ID'].str.split('_', expand=True).iloc[:, [1, 2]]
    ).astype(int)
    df_gsheet['Pred'] = pd.to_numeric(df_gsheet['Pred'], errors='coerce')

    m_path = gdown.download(
        f'https://drive.google.com/uc?id={st.secrets["M_TEAM_NAMES_ID"]}',
        quiet=True, fuzzy=True)
    df_m = pd.read_csv(m_path)
    df_m['League'] = "Men's"

    w_path = gdown.download(
        f'https://drive.google.com/uc?id={st.secrets["W_TEAM_NAMES_ID"]}',
        quiet=True, fuzzy=True)
    df_w = pd.read_csv(w_path)
    df_w['League'] = "Women's"

    df_all_teams = pd.concat([df_m, df_w], ignore_index=True)

    df = pd.merge(df_gsheet,
                  df_m[['TeamID','TeamName']].rename(columns={'TeamName':'TN1M'}),
                  left_on='Team 1', right_on='TeamID', how='left').drop(columns='TeamID')
    df = pd.merge(df,
                  df_m[['TeamID','TeamName']].rename(columns={'TeamName':'TN2M'}),
                  left_on='Team 2', right_on='TeamID', how='left').drop(columns='TeamID')
    df = pd.merge(df,
                  df_w[['TeamID','TeamName']].rename(columns={'TeamName':'TN1W'}),
                  left_on='Team 1', right_on='TeamID', how='left').drop(columns='TeamID')
    df = pd.merge(df,
                  df_w[['TeamID','TeamName']].rename(columns={'TeamName':'TN2W'}),
                  left_on='Team 2', right_on='TeamID', how='left').drop(columns='TeamID')

    df['Team Name 1'] = df['TN1M'].combine_first(df['TN1W'])
    df['Team Name 2'] = df['TN2M'].combine_first(df['TN2W'])
    df['League']      = df['TN1M'].notna().map({True: "Men's", False: "Women's"})
    df_results = df[['ID','Pred','Team 1','Team 2','Team Name 1','Team Name 2','League']].copy()

    return df_results, df_all_teams


def load_bracket_csv():
    """Load teams.csv from same folder as this script, if it exists."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "teams.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    return None


if 'data_loaded' not in st.session_state:
    with st.spinner("Loading data from Google Sheets & Drive…"):
        df_results, df_all_teams = load_all_data()
    st.session_state.df_results   = df_results
    st.session_state.df_all_teams = df_all_teams
    st.session_state.data_loaded  = True

df_results   = st.session_state.df_results
df_all_teams = st.session_state.df_all_teams

# ══════════════════════════════════════════════════════════════
# 2. PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════
def get_pred(t1, t2):
    """Return P(t1 beats t2), or None if not found."""
    if t1 is None or t2 is None:
        return None
    row = df_results[(df_results['Team 1'] == t1) & (df_results['Team 2'] == t2)]
    if not row.empty:
        return float(row.iloc[0]['Pred'])
    row = df_results[(df_results['Team 1'] == t2) & (df_results['Team 2'] == t1)]
    if not row.empty:
        return 1.0 - float(row.iloc[0]['Pred'])
    return None


def predict_winner(t1, t2):
    """Return predicted winner team_id based on Pred score."""
    p = get_pred(t1, t2)
    if p is None:
        return random.choice([t for t in [t1, t2] if t is not None])
    return t1 if p >= 0.5 else t2

# ══════════════════════════════════════════════════════════════
# 3. BRACKET STATE
# ══════════════════════════════════════════════════════════════
def build_empty_bracket():
    b = {}
    for div in DIVISIONS:
        b[div] = {0: {s: None for s in SEEDS}}
        for rnd in range(1, 5):
            b[div][rnd] = {i: None for i in range(16 // (2 ** rnd))}
    b['FF']       = {0: None, 1: None}
    b['Champion'] = None
    return b


def init_bracket(league):
    key = f"bracket_{league}"
    if key not in st.session_state:
        st.session_state[key] = build_empty_bracket()
    return st.session_state[key]


def get_default_seeds(league, team_opts_local):
    """
    Returns {div: {seed: team_id}} for 4 divisions × 16 seeds.
    Uses teams.csv if present; otherwise random-fills from matching league.
    """
    defaults = {div: {s: None for s in SEEDS} for div in DIVISIONS}
    csv_df   = load_bracket_csv()

    if csv_df is not None and {'team_id','seed','division','league'}.issubset(set(csv_df.columns)):
        filtered = csv_df[csv_df['league'].str.strip().str.lower() == league.lower()]
        for _, row in filtered.iterrows():
            div  = str(row['division']).strip().title()
            seed = int(row['seed'])
            tid  = int(row['team_id'])
            if div in DIVISIONS and seed in SEEDS:
                defaults[div][seed] = tid
        return defaults

    # Fallback: random fill from league team pool, no repeats
    pool = list(team_opts_local.keys())
    random.shuffle(pool)
    idx = 0
    for div in DIVISIONS:
        for seed in SEEDS:
            if idx < len(pool):
                defaults[div][seed] = pool[idx]
                idx += 1
    return defaults


def run_simulation(bracket):
    """Simulate all rounds automatically using Pred probabilities."""
    for div in DIVISIONS:
        for rnd in range(1, 5):
            for slot in range(16 // (2 ** rnd)):
                if rnd == 1:
                    s1, s2 = SEED_MATCHUPS[slot]
                    t1 = bracket[div][0][s1]
                    t2 = bracket[div][0][s2]
                else:
                    t1 = bracket[div][rnd-1].get(slot * 2)
                    t2 = bracket[div][rnd-1].get(slot * 2 + 1)

                if t1 and t2:
                    bracket[div][rnd][slot] = predict_winner(t1, t2)
                else:
                    bracket[div][rnd][slot] = t1 or t2

    for i, (div_a, div_b) in enumerate(FF_PAIRS):
        t1 = bracket[div_a][4].get(0)
        t2 = bracket[div_b][4].get(0)
        bracket['FF'][i] = predict_winner(t1, t2) if (t1 and t2) else (t1 or t2)

    t1 = bracket['FF'].get(0)
    t2 = bracket['FF'].get(1)
    bracket['Champion'] = predict_winner(t1, t2) if (t1 and t2) else (t1 or t2)
    return bracket

# ══════════════════════════════════════════════════════════════
# 4. PAGE HEADER & LEAGUE SELECTOR
# ══════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='text-align:center;margin-bottom:4px'>🏀 March Madness Bracket Predictor</h1>"
    "<p style='text-align:center;color:#aaa;margin-top:0'>64-team simulation across 4 divisions</p>",
    unsafe_allow_html=True)

st.markdown("---")
league = st.radio("🏆 Select League", ["Men's", "Women's"],
                  horizontal=True, key="league_filter")

league_teams = df_all_teams[df_all_teams['League'] == league].copy()
team_options = {int(r['TeamID']): r['TeamName'] for _, r in league_teams.iterrows()}
id_by_name   = {v: k for k, v in team_options.items()}
names_sorted = ["— Select —"] + sorted(team_options.values())

bracket = init_bracket(league)


def tdisplay(tid):
    if tid is None:
        return "TBD"
    return team_options.get(tid, f"ID:{tid}")

# ══════════════════════════════════════════════════════════════
# 5. TEAM SELECTION — 4 divisions × 16 seeds = 64 slots
# ══════════════════════════════════════════════════════════════
st.markdown(
    "<h2 style='color:#f7c948'>📝 Step 1 — Assign Your 64 Teams</h2>",
    unsafe_allow_html=True)

# Quick-load defaults row
load_col, info_col = st.columns([1, 3])
with load_col:
    if st.button("🔄 Load from CSV / Random Fill", use_container_width=True):
        defaults = get_default_seeds(league, team_options)
        for div in DIVISIONS:
            for seed in SEEDS:
                bracket[div][0][seed] = defaults[div][seed]
        csv_found = load_bracket_csv() is not None
        msg = "✅ Loaded from **teams.csv**" if csv_found else "✅ Randomly filled from league pool"
        st.success(msg)

with info_col:
    total_filled = sum(1 for d in DIVISIONS for s in SEEDS if bracket[d][0][s] is not None)
    pct = int(total_filled / 64 * 100)
    st.markdown(
        f"<div style='padding:10px;background:#1e2a1e;border-radius:6px;"
        f"border-left:4px solid #0d6e2e'>"
        f"<b style='color:#d4edda'>{total_filled}/64 teams assigned ({pct}%)</b><br>"
        f"<span style='color:#aaa;font-size:12px'>"
        f"teams.csv columns: <code>team_name, team_id, seed, division, league</code></span></div>",
        unsafe_allow_html=True)

# Four division columns side-by-side for compact entry
st.markdown("<br>", unsafe_allow_html=True)

div_tabs = st.tabs([f"🗂 {d} Region" for d in DIVISIONS])

for tab, div in zip(div_tabs, DIVISIONS):
    with tab:
        bg = DIVISION_COLORS[div]
        st.markdown(
            f"<div style='background:{bg};padding:10px 16px;border-radius:8px;"
            f"margin-bottom:14px'><b style='color:white;font-size:16px'>"
            f"🏟 {div} Region</b> &nbsp;"
            f"<span style='color:#ccc;font-size:12px'>"
            f"Seeds 1–16 · 1=strongest, 16=weakest</span></div>",
            unsafe_allow_html=True)

        # Show matchup preview alongside seed entry
        col_entry, col_preview = st.columns([2, 1])

        with col_entry:
            cols = st.columns(4)
            for idx, seed in enumerate(SEEDS):
                with cols[idx % 4]:
                    curr      = bracket[div][0][seed]
                    cur_name  = team_options.get(curr, "— Select —") if curr else "— Select —"
                    safe_idx  = names_sorted.index(cur_name) if cur_name in names_sorted else 0
                    choice = st.selectbox(
                        f"Seed {seed}",
                        names_sorted,
                        index=safe_idx,
                        key=f"{league}_{div}_seed_{seed}"
                    )
                    bracket[div][0][seed] = id_by_name.get(choice) if choice != "— Select —" else None

        with col_preview:
            st.markdown("**First Round Matchups Preview**")
            for s1, s2 in SEED_MATCHUPS:
                n1 = tdisplay(bracket[div][0][s1])
                n2 = tdisplay(bracket[div][0][s2])
                c1 = "#ccc" if n1 != "TBD" else "#666"
                c2 = "#ccc" if n2 != "TBD" else "#666"
                st.markdown(
                    f"<div style='font-size:12px;padding:3px 0;border-bottom:1px solid #333'>"
                    f"<span style='color:#aaa'>#{s1}</span> "
                    f"<span style='color:{c1}'>{n1}</span>"
                    f" <span style='color:#666'>vs</span> "
                    f"<span style='color:#aaa'>#{s2}</span> "
                    f"<span style='color:{c2}'>{n2}</span></div>",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 6. RUN SIMULATION BUTTON
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<h2 style='color:#f7c948'>🚀 Step 2 — Run Simulation</h2>",
    unsafe_allow_html=True)

c1, c2, c3 = st.columns([1.2, 1, 4])
with c1:
    run_btn = st.button("▶️ Run Full Bracket", type="primary", use_container_width=True)
with c2:
    reset_btn = st.button("🔁 Reset Results", use_container_width=True)

if reset_btn:
    for div in DIVISIONS:
        for rnd in range(1, 5):
            for slot in bracket[div][rnd]:
                bracket[div][rnd][slot] = None
    bracket['FF']       = {0: None, 1: None}
    bracket['Champion'] = None
    st.success("Bracket results cleared. Team assignments kept.")

if run_btn:
    total_filled = sum(1 for d in DIVISIONS for s in SEEDS if bracket[d][0][s] is not None)
    if total_filled < 64:
        st.warning(f"⚠️ Only {total_filled}/64 teams assigned. "
                   "Please fill remaining seeds or use 'Load from CSV / Random Fill' above.")
    else:
        bracket = run_simulation(bracket)
        st.session_state[f"bracket_{league}"] = bracket
        st.success("✅ Simulation complete! See results below.")

# ══════════════════════════════════════════════════════════════
# 7. MATCHUP CARD RENDERER
# ══════════════════════════════════════════════════════════════
def matchup_card(t1, t2, winner, seed_label=""):
    pred = get_pred(t1, t2)
    n1   = tdisplay(t1)
    n2   = tdisplay(t2)
    p1   = f"{pred*100:.1f}%" if pred is not None else "N/A"
    p2   = f"{(1-pred)*100:.1f}%" if pred is not None else "N/A"

    def row(tid, name, prob):
        is_win  = winner and winner == tid
        bg      = "#155724" if is_win else "#1e1e1e"
        color   = "#d4edda" if is_win else "#cccccc"
        trophy  = "🏆 " if is_win else ""
        bold    = "font-weight:700" if is_win else ""
        return (f"<div style='padding:6px 10px;background:{bg};color:{color};"
                f"{bold};display:flex;justify-content:space-between'>"
                f"<span>{trophy}{name}</span>"
                f"<span style='opacity:.8;font-size:12px'>{prob}</span></div>")

    label_html = (f"<div style='font-size:10px;color:#888;padding:3px 10px;"
                  f"background:#111'>{seed_label}</div>") if seed_label else ""

    st.markdown(
        f"<div style='border:1px solid #444;border-radius:8px;overflow:hidden;"
        f"margin-bottom:8px;font-size:13px'>"
        f"{label_html}"
        f"{row(t1, n1, p1)}"
        f"{row(t2, n2, p2)}"
        f"</div>",
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 8. BRACKET RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<h2 style='color:#f7c948'>📊 Step 3 — Bracket Results</h2>",
    unsafe_allow_html=True)

# ── 8a. Division overview (all 4 at once) ──
st.markdown("### 🗂 Divisional Results")
div_result_tabs = st.tabs([f"🏟 {d}" for d in DIVISIONS])

for tab, div in zip(div_result_tabs, DIVISIONS):
    with tab:
        bg = DIVISION_COLORS[div]
        st.markdown(
            f"<div style='background:{bg};padding:8px 14px;border-radius:8px;"
            f"margin-bottom:14px'><b style='color:white;font-size:15px'>"
            f"{div} Region</b></div>", unsafe_allow_html=True)

        for rnd in range(1, 5):
            st.markdown(f"**🔄 {ROUND_LABELS[rnd-1]}**")
            num_slots = 16 // (2 ** rnd)
            cols = st.columns(min(num_slots, 4))
            for slot in range(num_slots):
                with cols[slot % min(num_slots, 4)]:
                    if rnd == 1:
                        s1, s2 = SEED_MATCHUPS[slot]
                        t1 = bracket[div][0][s1]
                        t2 = bracket[div][0][s2]
                        slbl = f"#{s1} vs #{s2}"
                    else:
                        t1 = bracket[div][rnd-1].get(slot * 2)
                        t2 = bracket[div][rnd-1].get(slot * 2 + 1)
                        slbl = ""
                    matchup_card(t1, t2, bracket[div][rnd].get(slot), slbl)
            st.markdown("<br>", unsafe_allow_html=True)

        reg_champ = bracket[div][4].get(0)
        if reg_champ:
            st.markdown(
                f"<div style='background:linear-gradient(90deg,#0d6e2e,#1a9c46);"
                f"padding:12px 20px;border-radius:8px;text-align:center;"
                f"color:white;font-size:17px;font-weight:700'>"
                f"🏆 {div} Regional Champion: {tdisplay(reg_champ)}</div>",
                unsafe_allow_html=True)
        else:
            st.info("Run simulation to reveal the Regional Champion.")

# ── 8b. Final Four ────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🏟 Final Four")

ff_cols = st.columns(2)
for i, (div_a, div_b) in enumerate(FF_PAIRS):
    with ff_cols[i]:
        st.markdown(f"**{div_a} vs {div_b}**")
        t1 = bracket[div_a][4].get(0)
        t2 = bracket[div_b][4].get(0)
        matchup_card(t1, t2, bracket['FF'].get(i),
                     f"{div_a} Champion vs {div_b} Champion")

# ── 8c. Championship ──────────────────────────────────────────
st.markdown("---")
st.markdown("### 🥇 National Championship")

t1 = bracket['FF'].get(0)
t2 = bracket['FF'].get(1)
matchup_card(t1, t2, bracket.get('Champion'),
             "Final Four Winner vs Final Four Winner")

champ = bracket.get('Champion')
if champ:
    st.balloons()
    st.markdown(
        f"<div style='text-align:center;padding:30px;border-radius:14px;"
        f"background:linear-gradient(135deg,#f7971e,#ffd200);color:#111;"
        f"font-size:30px;font-weight:800;letter-spacing:1px;margin-top:16px'>"
        f"🏆 {league} CHAMPION<br><span style='font-size:36px'>{tdisplay(champ)}</span> 🏆"
        f"</div>",
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 9. FULL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📋 Full Bracket Summary Table")

rows = []
for div in DIVISIONS:
    for rnd in range(1, 5):
        for slot in range(16 // (2 ** rnd)):
            if rnd == 1:
                s1, s2 = SEED_MATCHUPS[slot]
                t1 = bracket[div][0][s1]
                t2 = bracket[div][0][s2]
                mlbl = f"#{s1} vs #{s2}"
            else:
                t1 = bracket[div][rnd-1].get(slot * 2)
                t2 = bracket[div][rnd-1].get(slot * 2 + 1)
                mlbl = f"Game {slot+1}"
            pred   = get_pred(t1, t2)
            winner = bracket[div][rnd].get(slot)
            rows.append({
                "Division":       div,
                "Round":          ROUND_LABELS[rnd-1],
                "Matchup":        mlbl,
                "Team 1":         tdisplay(t1),
                "Team 2":         tdisplay(t2),
                "Pred T1 Win%":   f"{pred*100:.1f}%" if pred is not None else "N/A",
                "Winner 🏆":      tdisplay(winner) if winner else "—",
            })

for i, (div_a, div_b) in enumerate(FF_PAIRS):
    t1, t2 = bracket[div_a][4].get(0), bracket[div_b][4].get(0)
    pred   = get_pred(t1, t2)
    rows.append({
        "Division": "Final Four", "Round": f"{div_a} vs {div_b}",
        "Matchup": "Semifinal",
        "Team 1": tdisplay(t1), "Team 2": tdisplay(t2),
        "Pred T1 Win%": f"{pred*100:.1f}%" if pred is not None else "N/A",
        "Winner 🏆": tdisplay(bracket['FF'].get(i)) if bracket['FF'].get(i) else "—",
    })

t1, t2 = bracket['FF'].get(0), bracket['FF'].get(1)
pred = get_pred(t1, t2)
rows.append({
    "Division": "Championship", "Round": "Final",
    "Matchup": "National Championship",
    "Team 1": tdisplay(t1), "Team 2": tdisplay(t2),
    "Pred T1 Win%": f"{pred*100:.1f}%" if pred is not None else "N/A",
    "Winner 🏆": tdisplay(bracket.get('Champion')) if bracket.get('Champion') else "—",
})

df_summary = pd.DataFrame(rows)


def hl(row):
    s = [''] * len(row)
    wi = list(row.index).index("Winner 🏆")
    if row["Winner 🏆"] not in ("—", "TBD", ""):
        s[wi] = "background-color:#155724;color:#d4edda;font-weight:700"
    return s


st.dataframe(
    df_summary.style.apply(hl, axis=1),
    use_container_width=True,
    hide_index=True
)
