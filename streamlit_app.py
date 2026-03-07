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
ROUND_LABELS  = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Elite 8 Winner"]
FF_PAIRS      = [("East", "West"), ("South", "Midwest")]

DIVISION_COLORS = {
    "East":    "#1a3a5c",
    "West":    "#5c1a1a",
    "South":    "#1a4a2a",
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
    table      = worksheet.get_all_values()
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
    """Load teams.csv from the root folder."""
    path = "teams.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    return None

if 'data_loaded' not in st.session_state:
    with st.spinner("Loading Data…"):
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

def get_default_seeds(league, team_opts_local):
    """Returns {div: {seed: team_id}} from teams.csv or random fallback."""
    defaults = {div: {s: None for s in SEEDS} for div in DIVISIONS}
    csv_df   = load_bracket_csv()

    if csv_df is not None and {'team_id','seed','division','league'}.issubset(set(csv_df.columns)):
        filtered = csv_df[csv_df['league'].str.strip().str.lower() == league.lower()]
        if not filtered.empty:
            for _, row in filtered.iterrows():
                div  = str(row['division']).strip().title()
                seed = int(row['seed'])
                tid  = int(row['team_id'])
                if div in DIVISIONS and seed in SEEDS:
                    defaults[div][seed] = tid
            return defaults

    # Fallback: random fill from league team pool
    pool = list(team_opts_local.keys())
    random.shuffle(pool)
    idx = 0
    for div in DIVISIONS:
        for seed in SEEDS:
            if idx < len(pool):
                defaults[div][seed] = pool[idx]
                idx += 1
    return defaults

def init_bracket(league, team_opts_local):
    """Initializes bracket and auto-populates teams from CSV by default."""
    key = f"bracket_{league}"
    if key not in st.session_state:
        b = build_empty_bracket()
        defaults = get_default_seeds(league, team_opts_local)
        for div in DIVISIONS:
            for seed in SEEDS:
                b[div][0][seed] = defaults[div][seed]
        st.session_state[key] = b
    return st.session_state[key]

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

bracket = init_bracket(league, team_options)

def tdisplay(tid):
    if tid is None:
        return "TBD"
    return team_options.get(tid, f"ID:{tid}")

# ══════════════════════════════════════════════════════════════
# 5. TEAM SELECTION — 4 divisions × 16 seeds = 64 slots
# ══════════════════════════════════════════════════════════════
st.markdown(
    "<h2 style='color:#f7c948'>📝 Step 1 — Review/Assign Your 64 Teams</h2>",
    unsafe_allow_html=True)

total_filled = sum(1 for d in DIVISIONS for s in SEEDS if bracket[d][0][s] is not None)
pct = int(total_filled / 64 * 100)
status_msg = "✅ Teams loaded from **teams.csv**" if load_bracket_csv() is not None else "⚠️ teams.csv not found (Random Fill used)"

st.markdown(
    f"<div style='padding:10px;background:#1e2a1e;border-radius:6px;"
    f"border-left:4px solid #0d6e2e'>"
    f"<b style='color:#d4edda'>{total_filled}/64 teams assigned ({pct}%)</b><br>"
    f"<span style='color:#aaa;font-size:12px'>{status_msg}</span></div>",
    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
div_tabs = st.tabs([f"🗂 {d} Region" for d in DIVISIONS])

for tab, div in zip(div_tabs, DIVISIONS):
    with tab:
        bg = DIVISION_COLORS[div]
        st.markdown(
            f"<div style='background:{bg};padding:10px 16px;border-radius:8px;"
            f"margin-bottom:14px'><b style='color:white;font-size:16px'>"
            f"🏟 {div} Region</b></div>",
            unsafe_allow_html=True)

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
            st.markdown("**Matchup Preview**")
            for s1, s2 in SEED_MATCHUPS:
                n1 = tdisplay(bracket[div][0][s1])
                n2 = tdisplay(bracket[div][0][s2])
                c1 = "#ccc" if n1 != "TBD" else "#666"
                c2 = "#ccc" if n2 != "TBD" else "#666"
                st.markdown(
                    f"<div style='font-size:12px;padding:3px 0;border-bottom:1px solid #333'>"
                    f"<span style='color:#aaa'>#{s1}</span> <span style='color:{c1}'>{n1}</span>"
                    f" <span style='color:#666'>vs</span> "
                    f"<span style='color:#aaa'>#{s2}</span> <span style='color:{c2}'>{n2}</span></div>",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 6. MODE SELECTION & CONTROLS
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<h2 style='color:#f7c948'>🚀 Step 2 — Build Your Bracket</h2>",
    unsafe_allow_html=True)

bracket_mode = st.radio(
    "🎯 Pick Mode",
    [ "✍️ Manual Picks (Your Choices)","🤖 Auto-Simulate (Model Picks)"],
    horizontal=True,
    key="bracket_mode"
)
is_manual = bracket_mode == "✍️ Manual Picks (Your Choices)"

if is_manual:
    st.info("👇 **Manual Mode**: Pick your winner for each matchup below. Win probabilities are shown to guide your choices.")

c1, c2, c3 = st.columns([1.2, 1, 4])
with c1:
    run_btn = st.button("▶️ Run Full Bracket", type="primary", use_container_width=True,
                        disabled=is_manual)
with c2:
    reset_btn = st.button("🔁 Reset Results", use_container_width=True)

if reset_btn:
    for div in DIVISIONS:
        for rnd in range(1, 5):
            for slot in bracket[div][rnd]:
                bracket[div][rnd][slot] = None
    bracket['FF']       = {0: None, 1: None}
    bracket['Champion'] = None
    st.session_state[f"bracket_{league}"] = bracket
    st.success("Bracket results cleared. Team assignments kept.")
    st.rerun()

if run_btn and not is_manual:
    total_filled = sum(1 for d in DIVISIONS for s in SEEDS if bracket[d][0][s] is not None)
    if total_filled < 64:
        st.warning(f"⚠️ Only {total_filled}/64 teams assigned. Please check your team assignments.")
    else:
        bracket = run_simulation(bracket)
        st.session_state[f"bracket_{league}"] = bracket
        st.success("✅ Simulation complete! See results below.")

# ══════════════════════════════════════════════════════════════
# 7. HELPER: WIN PROBABILITY BADGE
# ══════════════════════════════════════════════════════════════
def prob_badge(prob, is_pick=False):
    """Return an HTML badge showing win probability."""
    if prob is None:
        return "<span style='color:#888;font-size:11px'>N/A</span>"
    color = "#d4edda" if prob >= 0.5 else "#f8d7da"
    bg    = "#155724" if prob >= 0.5 else "#721c24"
    if is_pick:
        color = "#fff9c4"
        bg    = "#856404"
    return (f"<span style='background:{bg};color:{color};padding:2px 7px;"
            f"border-radius:10px;font-size:11px;font-weight:600'>{prob*100:.1f}%</span>")

# ══════════════════════════════════════════════════════════════
# 8. MATCHUP CARD RENDERER
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
        f"{label_html}{row(t1, n1, p1)}{row(t2, n2, p2)}</div>",
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 9. MANUAL PICK WIDGET
# ══════════════════════════════════════════════════════════════
def manual_pick_widget(t1, t2, widget_key, current_winner=None):
    """
    Show a pick widget for a matchup. Returns the selected winner team_id.
    Displays win probability for each team to guide the user.
    """
    n1   = tdisplay(t1)
    n2   = tdisplay(t2)
    pred = get_pred(t1, t2)

    if t1 is None and t2 is None:
        st.markdown("<span style='color:#666;font-size:12px'>No teams set</span>",
                    unsafe_allow_html=True)
        return None

    if t1 is None or t2 is None:
        only_team = t1 or t2
        st.markdown(
            f"<div style='padding:6px 10px;background:#1e2a1e;border-radius:6px;"
            f"color:#d4edda;font-size:12px'>✅ Advances: {tdisplay(only_team)}</div>",
            unsafe_allow_html=True)
        return only_team

    # Build option labels with probability
    if pred is not None:
        p1 = pred * 100
        p2 = (1 - pred) * 100
        opt1 = f"{n1}  ({p1:.1f}% win prob)"
        opt2 = f"{n2}  ({p2:.1f}% win prob)"
    else:
        opt1 = n1
        opt2 = n2

    options  = [opt1, opt2]
    # Map current winner back to option index
    if current_winner == t1:
        curr_idx = 0
    elif current_winner == t2:
        curr_idx = 1
    else:
        curr_idx = 0  # default to first

    choice_label = st.radio(
        "Pick winner:",
        options,
        index=curr_idx,
        key=widget_key,
        label_visibility="collapsed"
    )

    # Highlight the probability of the pick
    if pred is not None:
        pick_is_t1 = (choice_label == opt1)
        pick_prob  = pred if pick_is_t1 else (1 - pred)
        model_pick = t1 if pred >= 0.5 else t2
        pick_tid   = t1 if pick_is_t1 else t2

        if pick_tid == model_pick:
            badge_color = "#1e2a1e"
            badge_text_color = "#8bc34a"
            badge_label = f"✅ Agrees with model ({pick_prob*100:.1f}% win prob)"
        else:
            badge_color = "#2a1e1e"
            badge_text_color = "#ff8a80"
            badge_label = f"⚠️ Upset pick! Model gives only {pick_prob*100:.1f}% win prob"

        st.markdown(
            f"<div style='padding:4px 10px;background:{badge_color};"
            f"border-radius:4px;font-size:11px;color:{badge_text_color};"
            f"margin-top:-8px;margin-bottom:4px'>{badge_label}</div>",
            unsafe_allow_html=True)

    return t1 if choice_label == opt1 else t2

# ══════════════════════════════════════════════════════════════
# 10. BRACKET RESULTS / MANUAL PICK DISPLAY
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2 style='color:#f7c948'>📊 Step 3 — Bracket Results</h2>", unsafe_allow_html=True)

if is_manual:
    st.markdown(
        "<div style='padding:10px;background:#1e1e2e;border-radius:6px;"
        "border-left:4px solid #f7c948;margin-bottom:16px'>"
        "<b style='color:#f7c948'>✍️ Manual Pick Mode</b> — "
        "<span style='color:#ccc'>Select your winner for each matchup. "
        "Win probabilities from the model are shown to guide your picks. "
        "Rounds unlock as you complete prior rounds.</span></div>",
        unsafe_allow_html=True)

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
            rnd_label = ROUND_LABELS[rnd - 1]
            num_slots = 16 // (2 ** rnd)

            # Check if prior round is complete (for manual mode gate)
            prior_complete = True
            if rnd > 1:
                for slot in range(16 // (2 ** (rnd - 1))):
                    if bracket[div][rnd - 1].get(slot) is None:
                        prior_complete = False
                        break

            if is_manual and rnd > 1 and not prior_complete:
                st.markdown(
                    f"<div style='padding:8px 14px;background:#1a1a1a;border-radius:6px;"
                    f"color:#666;font-style:italic;margin-bottom:8px'>"
                    f"🔒 {rnd_label} — Complete Round of {16 // (2**(rnd-2))} first</div>",
                    unsafe_allow_html=True)
                continue

            st.markdown(f"**🔄 {rnd_label}**")
            cols = st.columns(min(num_slots, 4))

            for slot in range(num_slots):
                with cols[slot % min(num_slots, 4)]:
                    if rnd == 1:
                        s1, s2 = SEED_MATCHUPS[slot]
                        t1, t2 = bracket[div][0][s1], bracket[div][0][s2]
                        slbl = f"#{s1} vs #{s2}"
                    else:
                        t1 = bracket[div][rnd - 1].get(slot * 2)
                        t2 = bracket[div][rnd - 1].get(slot * 2 + 1)
                        slbl = ""

                    current_winner = bracket[div][rnd].get(slot)

                    if is_manual:
                        widget_key = f"manual_{league}_{div}_rnd{rnd}_slot{slot}"
                        if slbl:
                            st.markdown(
                                f"<div style='font-size:10px;color:#888;"
                                f"margin-bottom:2px'>{slbl}</div>",
                                unsafe_allow_html=True)
                        picked = manual_pick_widget(t1, t2, widget_key, current_winner)
                        bracket[div][rnd][slot] = picked
                    else:
                        matchup_card(t1, t2, current_winner, slbl)

            st.markdown("<br>", unsafe_allow_html=True)

        # Save bracket changes back to session state after each division
        st.session_state[f"bracket_{league}"] = bracket

        reg_champ = bracket[div][4].get(0)
        if reg_champ:
            st.markdown(
                f"<div style='background:linear-gradient(90deg,#0d6e2e,#1a9c46);"
                f"padding:12px 20px;border-radius:8px;text-align:center;"
                f"color:white;font-size:17px;font-weight:700'>"
                f"🏆 {div} Regional Champion: {tdisplay(reg_champ)}</div>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 11. FINAL FOUR
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🏟 Final Four")

# Check if all regional champs are set
all_reg_champs_set = all(bracket[div][4].get(0) is not None for div in DIVISIONS)

ff_cols = st.columns(2)
for i, (div_a, div_b) in enumerate(FF_PAIRS):
    with ff_cols[i]:
        st.markdown(f"**{div_a} vs {div_b}**")
        t1, t2 = bracket[div_a][4].get(0), bracket[div_b][4].get(0)

        if is_manual:
            if not all_reg_champs_set:
                st.markdown(
                    "<div style='color:#666;font-style:italic;font-size:13px'>"
                    "🔒 Complete all Regional rounds first</div>",
                    unsafe_allow_html=True)
            else:
                widget_key = f"manual_{league}_FF_{i}"
                current_ff_winner = bracket['FF'].get(i)
                picked = manual_pick_widget(t1, t2, widget_key, current_ff_winner)
                bracket['FF'][i] = picked
                st.session_state[f"bracket_{league}"] = bracket
        else:
            matchup_card(t1, t2, bracket['FF'].get(i), f"{div_a} vs {div_b}")

# ══════════════════════════════════════════════════════════════
# 12. NATIONAL CHAMPIONSHIP
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🥇 National Championship")

ff_complete = all(bracket['FF'].get(i) is not None for i in range(2))
t1, t2 = bracket['FF'].get(0), bracket['FF'].get(1)

if is_manual:
    if not ff_complete:
        st.markdown(
            "<div style='color:#666;font-style:italic;font-size:13px'>"
            "🔒 Complete both Final Four matchups first</div>",
            unsafe_allow_html=True)
    else:
        widget_key = f"manual_{league}_Championship"
        current_champ = bracket.get('Champion')
        picked_champ  = manual_pick_widget(t1, t2, widget_key, current_champ)
        bracket['Champion'] = picked_champ
        st.session_state[f"bracket_{league}"] = bracket
else:
    matchup_card(t1, t2, bracket.get('Champion'), "Championship")

champ = bracket.get('Champion')
if champ:
    
    st.markdown(
        f"<div style='text-align:center;padding:30px;border-radius:14px;"
        f"background:linear-gradient(135deg,#f7971e,#ffd200);color:#111;"
        f"font-size:30px;font-weight:800;letter-spacing:1px;margin-top:16px'>"
        f"🏆 {league} CHAMPION<br><span style='font-size:36px'>{tdisplay(champ)}</span> 🏆"
        f"</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 13. FULL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📋 Full Bracket Summary Table")

rows = []
for div in DIVISIONS:
    for rnd in range(1, 5):
        for slot in range(16 // (2 ** rnd)):
            if rnd == 1:
                s1, s2 = SEED_MATCHUPS[slot]
                t1, t2 = bracket[div][0][s1], bracket[div][0][s2]
                mlbl = f"#{s1} vs #{s2}"
            else:
                t1, t2 = bracket[div][rnd-1].get(slot * 2), bracket[div][rnd-1].get(slot * 2 + 1)
                mlbl = f"Game {slot+1}"
            pred   = get_pred(t1, t2)
            winner = bracket[div][rnd].get(slot)

            # Flag upset picks in manual mode
            model_pick = None
            is_upset   = False
            if pred is not None:
                model_pick = t1 if pred >= 0.5 else t2
                is_upset   = (winner is not None and winner != model_pick)

            rows.append({
                "Division": div,
                "Round": ROUND_LABELS[rnd - 1],
                "Matchup": mlbl,
                "Team 1": tdisplay(t1),
                "Team 2": tdisplay(t2),
                "Pred T1 Win%": f"{pred*100:.1f}%" if pred is not None else "N/A",
                "Model Pick": tdisplay(model_pick) if model_pick else "N/A",
                "Your Pick 🏆": tdisplay(winner) if winner else "—",
                "Upset?": "⚠️ Yes" if is_upset else ("" if winner else "—"),
            })

for i, (div_a, div_b) in enumerate(FF_PAIRS):
    t1, t2 = bracket[div_a][4].get(0), bracket[div_b][4].get(0)
    pred = get_pred(t1, t2)
    winner = bracket['FF'].get(i)
    model_pick = None
    is_upset   = False
    if pred is not None:
        model_pick = t1 if pred >= 0.5 else t2
        is_upset   = (winner is not None and winner != model_pick)
    rows.append({
        "Division": "Final Four",
        "Round": f"{div_a} vs {div_b}",
        "Matchup": "Semifinal",
        "Team 1": tdisplay(t1),
        "Team 2": tdisplay(t2),
        "Pred T1 Win%": f"{pred*100:.1f}%" if pred is not None else "N/A",
        "Model Pick": tdisplay(model_pick) if model_pick else "N/A",
        "Your Pick 🏆": tdisplay(winner) if winner else "—",
        "Upset?": "⚠️ Yes" if is_upset else ("" if winner else "—"),
    })

t1, t2 = bracket['FF'].get(0), bracket['FF'].get(1)
pred = get_pred(t1, t2)
winner = bracket.get('Champion')
model_pick = None
is_upset   = False
if pred is not None:
    model_pick = t1 if pred >= 0.5 else t2
    is_upset   = (winner is not None and winner != model_pick)
rows.append({
    "Division": "Championship",
    "Round": "Final",
    "Matchup": "National Championship",
    "Team 1": tdisplay(t1),
    "Team 2": tdisplay(t2),
    "Pred T1 Win%": f"{pred*100:.1f}%" if pred is not None else "N/A",
    "Model Pick": tdisplay(model_pick) if model_pick else "N/A",
    "Your Pick 🏆": tdisplay(winner) if winner else "—",
    "Upset?": "⚠️ Yes" if is_upset else ("" if winner else "—"),
})

df_summary = pd.DataFrame(rows)

def hl(row):
    s = [''] * len(row)
    pick_col = list(row.index).index("Your Pick 🏆")
    upset_col = list(row.index).index("Upset?")
    if row["Your Pick 🏆"] not in ("—", "TBD", ""):
        if row.get("Upset?", "") == "⚠️ Yes":
            s[pick_col]  = "background-color:#721c24;color:#f8d7da;font-weight:700"
            s[upset_col] = "background-color:#721c24;color:#f8d7da"
        else:
            s[pick_col] = "background-color:#155724;color:#d4edda;font-weight:700"
    return s

# Show upset summary if manual mode
if is_manual:
    upset_count = sum(1 for r in rows if r.get("Upset?") == "⚠️ Yes")
    total_picked = sum(1 for r in rows if r["Your Pick 🏆"] not in ("—", "TBD", ""))
    if total_picked > 0:
        st.markdown(
            f"<div style='padding:10px 16px;background:#2a1e1e;border-radius:6px;"
            f"border-left:4px solid #ff8a80;margin-bottom:12px'>"
            f"<b style='color:#ff8a80'>🎲 Your Bracket Summary</b><br>"
            f"<span style='color:#ccc'>{total_picked} picks made · "
            f"<b style='color:#f8d7da'>{upset_count} upset picks</b> · "
            f"{total_picked - upset_count} model-aligned picks</span></div>",
            unsafe_allow_html=True)

st.dataframe(df_summary.style.apply(hl, axis=1), use_container_width=True, hide_index=True)
