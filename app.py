import warnings, datetime, itertools
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import streamlit as st

# ============================================================
#  PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="🌙 Personal Aspect Scorer",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background-color: #0A0A1A; color: #E8E8F4; }
  section[data-testid="stSidebar"] { background-color: #0D0D28; }
  section[data-testid="stSidebar"] * { color: #E8E8F4 !important; }
  .stTextInput > div > div > input,
  .stNumberInput > div > div > input,
  .stDateInput > div > div > input {
      background-color: #1A1A38; color: #E8E8F4;
      border: 1px solid #2A2A4A;
  }
  .stSlider > div { color: #E8E8F4; }
  .stButton > button {
      background-color: #C8A84B; color: #0A0A1A;
      font-weight: bold; border: none; border-radius: 4px;
      padding: 0.5rem 2rem; width: 100%;
  }
  .stButton > button:hover { background-color: #E8C86B; }
  h1, h2, h3 { color: #C8A84B !important; }
  div[data-testid="stMetric"] {
      background-color: #0D0D28; border: 1px solid #2A2A4A;
      border-radius: 6px; padding: 0.5rem 1rem;
  }
  div[data-testid="stMetric"] label { color: #C8A84B !important; }
  .stTabs [data-baseweb="tab"] { color: #E8E8F4; }
  .stTabs [aria-selected="true"] { color: #C8A84B !important; border-bottom-color: #C8A84B !important; }
  .stCheckbox > label { color: #E8E8F4 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  EPHEMERIS — loaded from GitHub (cached)
# ============================================================

GITHUB_EPH_URL = (
    "https://raw.githubusercontent.com/"
    "goncuahm/kozmikharita/main/planet_degrees.csv"
)

@st.cache_data(show_spinner="Loading ephemeris from GitHub …")
def load_ephemeris(url):
    eph_raw = pd.read_csv(url, index_col='date', parse_dates=True)
    return eph_raw

# ============================================================
#  CONSTANTS
# ============================================================

EPH_PLANET_COLS = [
    'sun', 'moon', 'mercury', 'venus', 'mars',
    'jupiter', 'saturn', 'uranus', 'neptune',
    'pluto', 'true_node', 'mean_node',
]

ASPECTS   = [0, 60, 90, 120, 180]
ASP_NAMES = {0:'Conj', 60:'Sext', 90:'Sqr', 120:'Trine', 180:'Opp'}
SIGNS     = ['Aries','Taurus','Gemini','Cancer','Leo','Virgo',
             'Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces']

PLANET_POTENCY = {
    'jupiter':   3.0,
    'venus':     2.0,
    'sun':       1.5,
    'moon':      1.0,
    'mars':      2.0,
    'saturn':    2.5,
    'pluto':     1.5,
    'mercury':   0.5,
    'neptune':   0.5,
    'uranus':    0.5,
    'true_node': 0.5,
    'mean_node': 0.5,
    'asc':       1.5,
    'mc':        1.5,
}

PLANET_NATURE = {
    'jupiter':   +1,
    'venus':     +1,
    'sun':       +1,
    'moon':      +1,
    'neptune':   +1,
    'true_node': +1,
    'mean_node': +1,
    'mercury':    0,
    'mars':      -1,
    'saturn':    -1,
    'uranus':    -1,
    'pluto':     -1,
    'asc':        0,
    'mc':         0,
}

ASPECT_BASE = {
    0:   +1.0,
    60:  +1.5,
    90:  -1.8,
    120: +2.0,
    180: -1.5,
}

NATURE_MOD = 0.35

PHASE_FACTOR = {'apply': 1.0, 'sep': 0.6}

BG     = '#0A0A1A';  PANEL  = '#0D0D28';  GOLD   = '#C8A84B'
TEAL   = '#00D4B4';  WHITE  = '#E8E8F4';  GREY   = '#2A2A4A'
GREEN  = '#44DD88';  RED    = '#E84040';  ORANGE = '#FF8844'
PURPLE = '#CC44FF';  BLUE   = '#4488FF'

PLANET_SYMBOLS = {
    'sun':       '☉ Sun',
    'moon':      '☽ Moon',
    'mercury':   '☿ Mercury',
    'venus':     '♀ Venus',
    'mars':      '♂ Mars',
    'jupiter':   '♃ Jupiter',
    'saturn':    '♄ Saturn',
    'uranus':    '♅ Uranus',
    'neptune':   '♆ Neptune',
    'pluto':     '♇ Pluto',
    'true_node': '☊ True Node',
    'mean_node': '☊ Mean Node',
    'asc':       '↑ ASC',
    'mc':        '↑ MC',
}

# ============================================================
#  SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🌙 Personal Aspect Scorer")
    st.markdown("---")

    st.markdown("### 🎂 Birth Date")
    birth_date_input = st.text_input(
        "Birth date (YYYY-MM-DD)",
        value="1979-12-04",
        help="Enter the person's date of birth.")

    person_name = st.text_input(
        "Name / label (optional)",
        value="",
        help="Shown in chart titles. Leave blank to use birth date.")

    st.markdown("### 🔺 Natal Angles (optional)")
    st.caption(
        "Enter your Ascendant (ASC) and/or Midheaven (MC) to include them "
        "as natal points in the scoring. Leave sign as '—' to skip.")

    SIGN_OPTIONS = ['—'] + SIGNS

    asc_sign = st.selectbox(
        "ASC Sign", options=SIGN_OPTIONS, index=0,
        help="Zodiac sign of your Ascendant. Select '—' to ignore.")
    asc_deg = st.number_input(
        "ASC Degree (0–29)", min_value=0, max_value=29, value=0, step=1,
        help="Degree within the sign (0–29).")

    mc_sign = st.selectbox(
        "MC Sign", options=SIGN_OPTIONS, index=0,
        help="Zodiac sign of your Midheaven (MC). Select '—' to ignore.")
    mc_deg = st.number_input(
        "MC Degree (0–29)", min_value=0, max_value=29, value=0, step=1,
        help="Degree within the sign (0–29).")

    def angle_longitude(sign_str, deg):
        if sign_str == '—':
            return None
        sign_idx = SIGNS.index(sign_str)
        return float(sign_idx * 30 + deg)

    asc_longitude = angle_longitude(asc_sign, asc_deg)
    mc_longitude  = angle_longitude(mc_sign,  mc_deg)

    st.markdown("### 📅 Display Range")
    display_start = st.text_input(
        "Start date (YYYY-MM-DD)",
        value=(datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
        help="Beginning of the score plot window.")

    display_end = st.text_input(
        "End date / forecast to (YYYY-MM-DD)",
        value=(datetime.date.today() + datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
        help="End of the score plot window. Can be in the future.")

    st.markdown("### 🔭 Orb Settings")
    orb_apply = st.slider(
        "Applying orb (degrees)", min_value=1, max_value=6, value=4,
        help="Degrees before exact aspect to start scoring.")
    orb_sep = st.slider(
        "Separating orb (degrees)", min_value=1, max_value=3, value=1,
        help="Degrees after exact aspect to stop scoring.")

    st.markdown("### 📋 Table Horizon")
    table_past_days = st.slider(
        "Days back in tables", min_value=3, max_value=30, value=7)
    table_future_days = st.slider(
        "Days ahead in tables", min_value=7, max_value=90, value=30)

    st.markdown("### 📊 Chart Options")
    smooth_window = st.slider(
        "Smoothing window (days)", min_value=1, max_value=30, value=7,
        help="Rolling mean window for the smoothed overlay line.")

    st.markdown("### 🌟 Natal Planets to Score")
    st.caption(
        "Select which natal planets transits should be scored against. "
        "All transit × transit aspects are always computed regardless.")

    if 'natal_planet_selection' not in st.session_state:
        st.session_state['natal_planet_selection'] = {p: True for p in EPH_PLANET_COLS}

    natal_planet_checkboxes = {}
    for p in EPH_PLANET_COLS:
        label = PLANET_SYMBOLS.get(p, p.replace('_', ' ').title())
        natal_planet_checkboxes[p] = st.checkbox(
            label,
            value=st.session_state['natal_planet_selection'].get(p, True),
            key=f"natal_cb_{p}")

    for p in EPH_PLANET_COLS:
        st.session_state['natal_planet_selection'][p] = natal_planet_checkboxes[p]

    selected_natal_planets = [p for p in EPH_PLANET_COLS if natal_planet_checkboxes[p]]

    st.markdown("### 🔄 Transit Planets to Use")
    st.caption(
        "Select which transiting planets are included in aspect calculations. "
        "Affects both natal and transit × transit scoring.")

    if 'transit_planet_selection' not in st.session_state:
        st.session_state['transit_planet_selection'] = {p: True for p in EPH_PLANET_COLS}

    transit_planet_checkboxes = {}
    for p in EPH_PLANET_COLS:
        label = PLANET_SYMBOLS.get(p, p.replace('_', ' ').title())
        transit_planet_checkboxes[p] = st.checkbox(
            label,
            value=st.session_state['transit_planet_selection'].get(p, True),
            key=f"transit_cb_{p}")

    for p in EPH_PLANET_COLS:
        st.session_state['transit_planet_selection'][p] = transit_planet_checkboxes[p]

    selected_transit_planets = [p for p in EPH_PLANET_COLS if transit_planet_checkboxes[p]]

    st.markdown("---")
    run_btn = st.button("▶  Run Analysis", type="primary")

# ============================================================
#  HEADER
# ============================================================

st.markdown("# 🌙 Personal Natal Aspect Scorer")
st.markdown(
    "Scores daily planetary aspects relative to a person's natal chart. "
    "No stock prices — purely astrological energy over time. "
    "**Positive = harmonious / supportive conditions. Negative = challenging conditions.**"
)

if not run_btn:
    st.info("👈 Enter birth date and date range in the sidebar, then click **▶ Run Analysis**.")
    st.stop()

# ============================================================
#  VALIDATE INPUTS
# ============================================================

try:
    birth_ts = pd.Timestamp(birth_date_input)
except Exception:
    st.error("Invalid birth date format. Use YYYY-MM-DD.")
    st.stop()

try:
    start_ts = pd.Timestamp(display_start)
    end_ts   = pd.Timestamp(display_end)
except Exception:
    st.error("Invalid date range. Use YYYY-MM-DD for both start and end.")
    st.stop()

if end_ts <= start_ts:
    st.error("End date must be after start date.")
    st.stop()

if not selected_natal_planets:
    st.error("Please select at least one natal planet in the sidebar.")
    st.stop()

if not selected_transit_planets:
    st.error("Please select at least one transit planet in the sidebar.")
    st.stop()

label = person_name.strip() if person_name.strip() else birth_date_input
today_ts = pd.Timestamp(datetime.date.today())

# ============================================================
#  LOAD EPHEMERIS
# ============================================================

with st.spinner("Loading ephemeris …"):
    try:
        eph_raw = load_ephemeris(GITHUB_EPH_URL)
        avail_planets = [p for p in EPH_PLANET_COLS if p in eph_raw.columns]
        eph = eph_raw[avail_planets].copy()
        st.success(
            f"Ephemeris loaded: {len(eph):,} days  "
            f"({eph.index[0].date()} → {eph.index[-1].date()})"
        )
    except Exception as e:
        st.error(
            f"Failed to load ephemeris from GitHub.\n\n"
            f"**Update `GITHUB_EPH_URL`** at the top of this script "
            f"with your actual raw GitHub URL.\n\nError: {e}"
        )
        st.stop()

if start_ts < eph.index[0]:
    start_ts = eph.index[0]
    st.warning(f"Start date adjusted to ephemeris start: {start_ts.date()}")
if end_ts > eph.index[-1]:
    end_ts = eph.index[-1]
    st.warning(f"End date adjusted to ephemeris end: {end_ts.date()}")

selected_natal_planets = [p for p in selected_natal_planets if p in avail_planets]
if not selected_natal_planets:
    st.error("None of the selected natal planets are in the loaded ephemeris.")
    st.stop()

selected_transit_planets = [p for p in selected_transit_planets if p in avail_planets]
if not selected_transit_planets:
    st.error("None of the selected transit planets are in the loaded ephemeris.")
    st.stop()

# ============================================================
#  NATAL CHART
# ============================================================

if birth_ts not in eph.index:
    idx      = eph.index.get_indexer([birth_ts], method='nearest')[0]
    birth_ts = eph.index[idx]

natal_row = eph.loc[birth_ts]
natal = {p: float(natal_row[p]) % 360 for p in avail_planets}

natal_angles = {}
if asc_longitude is not None:
    natal_angles['asc'] = asc_longitude
if mc_longitude is not None:
    natal_angles['mc'] = mc_longitude

with st.expander(f"🌟 Natal Chart — {label} ({birth_date_input})", expanded=True):
    natal_data = []
    for p, lon in natal.items():
        sign    = SIGNS[int(lon // 30)]
        deg_in  = int(lon % 30)
        mins    = int((lon % 1) * 60)
        is_selected = p in selected_natal_planets
        natal_data.append({
            'Selected':  '✓' if is_selected else '',
            'Planet':    PLANET_SYMBOLS.get(p, p.replace('_',' ').title()),
            'Longitude': f"{lon:.3f}°",
            'Sign':      sign,
            'Position':  f"{deg_in:02d}° {mins:02d}′ {sign}",
            'Potency':   PLANET_POTENCY.get(p, 0.5),
            'Nature':    '+Benefic' if PLANET_NATURE.get(p, 0) > 0
                         else ('−Malefic' if PLANET_NATURE.get(p, 0) < 0
                               else 'Neutral'),
        })

    for angle_key, angle_lon in natal_angles.items():
        sign   = SIGNS[int(angle_lon // 30)]
        deg_in = int(angle_lon % 30)
        natal_data.append({
            'Selected':  '✓',
            'Planet':    PLANET_SYMBOLS.get(angle_key, angle_key.upper()),
            'Longitude': f"{angle_lon:.1f}°",
            'Sign':      sign,
            'Position':  f"{deg_in:02d}° 00′ {sign}",
            'Potency':   PLANET_POTENCY.get(angle_key, 1.5),
            'Nature':    'Neutral (angle)',
        })

    natal_df = pd.DataFrame(natal_data)
    st.dataframe(natal_df, use_container_width=True, hide_index=True)

    if len(selected_natal_planets) < len(avail_planets) or len(selected_transit_planets) < len(avail_planets):
        st.info(
            f"Natal planets scored: **{', '.join(PLANET_SYMBOLS.get(p, p) for p in selected_natal_planets)}**  \n"
            f"Transit planets used: **{', '.join(PLANET_SYMBOLS.get(p, p) for p in selected_transit_planets)}**")

    if natal_angles:
        angle_strs = []
        for k, v in natal_angles.items():
            sign = SIGNS[int(v // 30)]
            deg  = int(v % 30)
            angle_strs.append(f"**{PLANET_SYMBOLS[k]}** {deg}° {sign} ({v:.1f}°)")
        st.info("Natal angles included in scoring: " + "  |  ".join(angle_strs))

# ============================================================
#  CORE HELPERS
# ============================================================

def angular_diff(lon_a, lon_b):
    d = (lon_a - lon_b) % 360
    return np.where(d > 180, d - 360, d)

def orb_factor(abs_gap, orb_max):
    return np.clip(1.0 - abs_gap / orb_max, 0.0, 1.0)

def aspect_score_single(pot_a, pot_b, asp, orb_f, phase, nat_a=0, nat_b=0):
    avg_pot = (pot_a + pot_b) / 2.0
    avg_nat = (nat_a + nat_b) / 2.0

    if asp == 0:
        net_nature = nat_a + nat_b
        if net_nature == 0:
            return 0.0
        direction = float(np.sign(net_nature))
        magnitude = avg_pot * abs(net_nature) / 2.0
        return direction * magnitude * ASPECT_BASE[0] * orb_f * PHASE_FACTOR[phase]
    else:
        base      = ASPECT_BASE[asp]
        modulated = base * (1.0 + NATURE_MOD * avg_nat * float(np.sign(base)))
        return modulated * avg_pot * orb_f * PHASE_FACTOR[phase]

def compute_natal_score(date_index):
    eph_a  = eph.reindex(date_index, method='ffill')
    n      = len(date_index)
    scores = np.zeros(n)
    detail = []

    natal_targets = {}
    for np_ in selected_natal_planets:
        natal_targets[np_] = natal[np_]
    for angle_key, angle_lon in natal_angles.items():
        natal_targets[angle_key] = angle_lon

    for tp in selected_transit_planets:
        if tp not in eph_a.columns: continue
        t_lons  = eph_a[tp].values.astype(float) % 360
        motion  = np.gradient(np.unwrap(t_lons, period=360))
        pot_t   = PLANET_POTENCY.get(tp, 0.5)
        nat_t   = PLANET_NATURE.get(tp, 0)

        for np_, n_lon in natal_targets.items():
            pot_n = PLANET_POTENCY.get(np_, 0.5)
            nat_n = PLANET_NATURE.get(np_, 0)
            for asp in ASPECTS:
                target  = (n_lon + asp) % 360
                gap     = angular_diff(t_lons, target)
                abs_gap = np.abs(gap)
                applying = ((motion > 0) & (gap < 0)) | ((motion < 0) & (gap > 0))
                mask_a   = applying    & (abs_gap <= orb_apply)
                mask_s   = (~applying) & (abs_gap <= orb_sep)
                for i in np.where(mask_a)[0]:
                    of = float(orb_factor(abs_gap[i], orb_apply))
                    sc = aspect_score_single(pot_t, pot_n, asp, of, 'apply', nat_t, nat_n)
                    scores[i] += sc
                    detail.append({'date': date_index[i], 'transit': tp,
                                   'natal': np_, 'aspect': ASP_NAMES[asp],
                                   'phase': 'Applying',
                                   'orb': round(float(abs_gap[i]), 3),
                                   'score': round(sc, 4)})
                for i in np.where(mask_s)[0]:
                    of = float(orb_factor(abs_gap[i], orb_sep))
                    sc = aspect_score_single(pot_t, pot_n, asp, of, 'sep', nat_t, nat_n)
                    scores[i] += sc
                    detail.append({'date': date_index[i], 'transit': tp,
                                   'natal': np_, 'aspect': ASP_NAMES[asp],
                                   'phase': 'Separating',
                                   'orb': round(float(abs_gap[i]), 3),
                                   'score': round(sc, 4)})
    return pd.Series(scores, index=date_index), detail

def compute_transit_score(date_index):
    eph_a  = eph.reindex(date_index, method='ffill')
    n      = len(date_index)
    scores = np.zeros(n)
    detail = []
    pairs  = list(itertools.combinations(selected_transit_planets, 2))
    for (pA, pB) in pairs:
        if pA not in eph_a.columns or pB not in eph_a.columns: continue
        lon_A   = eph_a[pA].values.astype(float) % 360
        lon_B   = eph_a[pB].values.astype(float) % 360
        motion  = np.gradient(np.unwrap(lon_A, period=360))
        pot_A   = PLANET_POTENCY.get(pA, 0.5)
        pot_B   = PLANET_POTENCY.get(pB, 0.5)
        nat_A   = PLANET_NATURE.get(pA, 0)
        nat_B   = PLANET_NATURE.get(pB, 0)
        for asp in ASPECTS:
            target  = (lon_B + asp) % 360
            gap     = angular_diff(lon_A, target)
            abs_gap = np.abs(gap)
            applying = ((motion > 0) & (gap < 0)) | ((motion < 0) & (gap > 0))
            mask_a   = applying    & (abs_gap <= orb_apply)
            mask_s   = (~applying) & (abs_gap <= orb_sep)
            for i in np.where(mask_a)[0]:
                of = float(orb_factor(abs_gap[i], orb_apply))
                sc = aspect_score_single(pot_A, pot_B, asp, of, 'apply', nat_A, nat_B)
                scores[i] += sc
                detail.append({'date': date_index[i], 'planet_a': pA,
                               'planet_b': pB, 'aspect': ASP_NAMES[asp],
                               'phase': 'Applying',
                               'orb': round(float(abs_gap[i]), 3),
                               'score': round(sc, 4)})
            for i in np.where(mask_s)[0]:
                of = float(orb_factor(abs_gap[i], orb_sep))
                sc = aspect_score_single(pot_A, pot_B, asp, of, 'sep', nat_A, nat_B)
                scores[i] += sc
                detail.append({'date': date_index[i], 'planet_a': pA,
                               'planet_b': pB, 'aspect': ASP_NAMES[asp],
                               'phase': 'Separating',
                               'orb': round(float(abs_gap[i]), 3),
                               'score': round(sc, 4)})
    return pd.Series(scores, index=date_index), detail

# ============================================================
#  BUILD DATE INDEX & COMPUTE SCORES
# ============================================================

full_index = pd.date_range(start=start_ts, end=end_ts, freq='D')

table_end_ts = today_ts + pd.Timedelta(days=table_future_days + 3)
if table_end_ts > end_ts:
    table_ext = pd.date_range(
        start = end_ts + pd.Timedelta(days=1),
        end   = table_end_ts, freq='D')
    compute_index = full_index.append(table_ext)
else:
    compute_index = full_index

with st.spinner("Computing natal aspect scores …"):
    natal_scores_full, natal_detail_full = compute_natal_score(compute_index)

with st.spinner("Computing transit aspect scores …"):
    transit_scores_full, transit_detail_full = compute_transit_score(compute_index)

natal_scores   = natal_scores_full.reindex(full_index).fillna(0)
transit_scores = transit_scores_full.reindex(full_index).fillna(0)
combined_scores = natal_scores.add(transit_scores, fill_value=0)

hist_mask = full_index <= today_ts
fore_mask = full_index >  today_ts

dates_hist = full_index[hist_mask]
dates_fore = full_index[fore_mask]

natal_hist   = natal_scores[hist_mask]
natal_fore   = natal_scores[fore_mask]
transit_hist = transit_scores[hist_mask]
transit_fore = transit_scores[fore_mask]
combined_hist = combined_scores[hist_mask]
combined_fore = combined_scores[fore_mask]

today_natal    = float(natal_scores.get(today_ts, natal_scores.iloc[len(dates_hist)-1]))
today_transit  = float(transit_scores.get(today_ts, transit_scores.iloc[len(dates_hist)-1]))
today_combined = today_natal + today_transit

natal_planet_label = (
    "All planets" if len(selected_natal_planets) == len(avail_planets)
    else ", ".join(PLANET_SYMBOLS.get(p, p) for p in selected_natal_planets))

if natal_angles:
    angle_labels = [PLANET_SYMBOLS.get(k, k.upper()) for k in natal_angles]
    natal_planet_label += "  +  " + ", ".join(angle_labels)

transit_planet_label = (
    "All planets" if len(selected_transit_planets) == len(avail_planets)
    else ", ".join(PLANET_SYMBOLS.get(p, p) for p in selected_transit_planets))

st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Today's Natal Score",    f"{today_natal:.2f}",
          delta="▲ Positive" if today_natal > 0 else "▼ Negative")
c2.metric("Today's Transit Score",  f"{today_transit:.2f}",
          delta="▲ Positive" if today_transit > 0 else "▼ Negative")
c3.metric("Today's Combined Score", f"{today_combined:.2f}",
          delta="▲ Positive" if today_combined > 0 else "▼ Negative")
c4.metric("Planet filters",
          f"N:{len(selected_natal_planets) + len(natal_angles)} / T:{len(selected_transit_planets)}",
          delta=f"of {len(avail_planets)} ephem + {len(natal_angles)} angles")

# ============================================================
#  PLOT HELPERS
# ============================================================

def style_ax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(GREY)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.yaxis.label.set_color(WHITE)

def draw_zero(ax):
    ax.axhline(0, color=GREY, lw=1.0, zorder=2)

def draw_today_line(ax):
    if start_ts <= today_ts <= end_ts:
        ax.axvline(today_ts, color=GOLD, lw=1.8, ls='--', alpha=0.9, zorder=6)
        ylims = ax.get_ylim()
        ax.text(today_ts, ylims[1],
                ' Today', color=GOLD, fontsize=8,
                va='top', ha='left', fontweight='bold')

def format_xaxis(ax, n_days):
    """
    Tiered x-axis with denser, clearly readable date ticks.
    Rotation increased to 60° to prevent label overlap at higher densities.
    """
    if n_days <= 30:
        # ≤1 month: every day, e.g. "Mar 12"
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    elif n_days <= 90:
        # ≤3 months: every 3 days
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    elif n_days <= 180:
        # ≤6 months: every Monday (weekly)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
    elif n_days <= 365:
        # ≤1 year: every 2nd Monday (bi-weekly) with year on second line
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    elif n_days <= 730:
        # ≤2 years: every month with full date + year
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    else:
        # >2 years: every 2 months
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=60, ha='right', fontsize=7)
    ax.set_xlim(start_ts, end_ts + pd.Timedelta(days=2))

def draw_score_bars(ax, dates_h, scores_h, dates_f, scores_f, alpha=0.75):
    if len(dates_h):
        h_vals   = scores_h.values
        h_colors = [GREEN if v >= 0 else RED for v in h_vals]
        ax.bar(dates_h, h_vals, color=h_colors, alpha=alpha, width=1.0, zorder=3)
    if len(dates_f):
        f_vals   = scores_f.values
        f_colors = [GREEN if v >= 0 else RED for v in f_vals]
        ax.bar(dates_f, f_vals, color=f_colors, alpha=alpha, width=1.0, zorder=3)

def draw_shading(ax, dates_h, scores_h, dates_f, scores_f, alpha_cap, denom):
    for d, s in zip(dates_h, scores_h.values):
        if s == 0: continue
        ax.axvspan(d, d + pd.Timedelta(days=1),
                   alpha=min(alpha_cap, abs(s)/denom),
                   color=GREEN if s > 0 else RED, zorder=1)
    for d, s in zip(dates_f, scores_f.values):
        if s == 0: continue
        ax.axvspan(d, d + pd.Timedelta(days=1),
                   alpha=min(alpha_cap, abs(s)/denom),
                   color=GREEN if s > 0 else RED, zorder=1)

def draw_smooth_line(ax, dates_h, scores_h, dates_f, scores_f, color, lw=1.8):
    combined = pd.concat([scores_h, scores_f])
    sm = combined.rolling(window=smooth_window, center=True, min_periods=1).mean()
    if len(dates_h):
        ax.plot(dates_h, sm.reindex(dates_h).values,
                color=color, lw=lw, zorder=5, label=f'{smooth_window}d smoothed')
    if len(dates_f):
        ax.plot(dates_f, sm.reindex(dates_f).values,
                color=color, lw=lw, ls='--', zorder=5, alpha=0.8)

n_days = len(full_index)

# ============================================================
#  CHART 1: NATAL SCORE
# ============================================================

st.markdown("---")
st.markdown("## Chart 1 — Natal Aspect Score")
st.caption(
    f"Transit planets: **{transit_planet_label}** aspecting natal planets: **{natal_planet_label}**. "
    "Green = supportive. Red = challenging.")

fig1, ax1 = plt.subplots(figsize=(18, 5), facecolor=BG)
style_ax(ax1)
ax1.set_ylabel('Natal Score', fontsize=10)
draw_zero(ax1)
draw_shading(ax1, dates_hist, natal_hist, dates_fore, natal_fore, 0.18, 15)
draw_score_bars(ax1, dates_hist, natal_hist, dates_fore, natal_fore)
draw_smooth_line(ax1, dates_hist, natal_hist, dates_fore, natal_fore, GOLD)
if len(dates_hist):
    ax1.fill_between(dates_hist, natal_hist.values, 0,
                     where=(natal_hist.values >= 0), color=GREEN, alpha=0.08, zorder=1)
    ax1.fill_between(dates_hist, natal_hist.values, 0,
                     where=(natal_hist.values < 0),  color=RED,   alpha=0.08, zorder=1)
if len(dates_fore):
    ax1.fill_between(dates_fore, natal_fore.values, 0,
                     where=(natal_fore.values >= 0), color=GREEN, alpha=0.08, zorder=1)
    ax1.fill_between(dates_fore, natal_fore.values, 0,
                     where=(natal_fore.values < 0),  color=RED,   alpha=0.08, zorder=1)
draw_today_line(ax1)
format_xaxis(ax1, n_days)
ax1.legend(fontsize=8, facecolor='#1A1A38', labelcolor=WHITE, loc='upper left')
fig1.suptitle(
    f"{label}  |  Natal Aspect Score\n"
    f"Transit: {transit_planet_label}  →  Natal: {natal_planet_label}\n"
    f"Birth: {birth_date_input}  |  Apply≤{orb_apply}°  Sep≤{orb_sep}°  |  "
    f"Green=Supportive  Red=Challenging  |  Gold dashed = Today",
    color=GOLD, fontsize=10, fontweight='bold')
fig1.tight_layout()
st.pyplot(fig1, use_container_width=True)
plt.close(fig1)

# ============================================================
#  CHART 2: TRANSIT × TRANSIT SCORE
# ============================================================

st.markdown("---")
st.markdown("## Chart 2 — Transit × Transit Aspect Score")
st.caption(f"Transit pairs from: **{transit_planet_label}**. Independent of natal chart.")

fig2, ax2 = plt.subplots(figsize=(18, 5), facecolor=BG)
style_ax(ax2)
ax2.set_ylabel('Transit Score', fontsize=10)
draw_zero(ax2)
draw_shading(ax2, dates_hist, transit_hist, dates_fore, transit_fore, 0.18, 25)
draw_score_bars(ax2, dates_hist, transit_hist, dates_fore, transit_fore)
draw_smooth_line(ax2, dates_hist, transit_hist, dates_fore, transit_fore, GOLD)
if len(dates_hist):
    ax2.fill_between(dates_hist, transit_hist.values, 0,
                     where=(transit_hist.values >= 0), color=GREEN, alpha=0.08, zorder=1)
    ax2.fill_between(dates_hist, transit_hist.values, 0,
                     where=(transit_hist.values < 0),  color=RED,   alpha=0.08, zorder=1)
if len(dates_fore):
    ax2.fill_between(dates_fore, transit_fore.values, 0,
                     where=(transit_fore.values >= 0), color=GREEN, alpha=0.08, zorder=1)
    ax2.fill_between(dates_fore, transit_fore.values, 0,
                     where=(transit_fore.values < 0),  color=RED,   alpha=0.08, zorder=1)
draw_today_line(ax2)
format_xaxis(ax2, n_days)
ax2.legend(fontsize=8, facecolor='#1A1A38', labelcolor=WHITE, loc='upper left')
fig2.suptitle(
    f"{label}  |  Transit × Transit Aspect Score\n"
    f"Planets: {transit_planet_label}  |  "
    f"Apply≤{orb_apply}°  Sep≤{orb_sep}°  |  "
    f"Green=Supportive  Red=Challenging  |  Gold dashed = Today",
    color=GOLD, fontsize=11, fontweight='bold')
fig2.tight_layout()
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)

# ============================================================
#  CHART 3: CUMULATIVE SCORES
# ============================================================

st.markdown("---")
st.markdown("## Chart 3 — Cumulative Aspect Score")
st.caption(
    "Running total since display start. Rising = accumulating positive conditions. "
    "Falling = accumulating challenges. Slope matters more than absolute value.")

natal_cum    = natal_scores.cumsum()
transit_cum  = transit_scores.cumsum()
combined_cum = combined_scores.cumsum()

natal_cum_h    = natal_cum[hist_mask]
natal_cum_f    = natal_cum[fore_mask]
transit_cum_h  = transit_cum[hist_mask]
transit_cum_f  = transit_cum[fore_mask]
combined_cum_h = combined_cum[hist_mask]
combined_cum_f = combined_cum[fore_mask]

fig3, ax3 = plt.subplots(figsize=(18, 5), facecolor=BG)
style_ax(ax3)
ax3.set_ylabel('Cumulative Score', fontsize=10)
draw_zero(ax3)

if len(dates_hist):
    ax3.plot(dates_hist, natal_cum_h.values,
             color=ORANGE, lw=1.4, ls='--', alpha=0.85, zorder=3, label='Natal cum.')
    ax3.plot(dates_hist, transit_cum_h.values,
             color=PURPLE, lw=1.4, ls='--', alpha=0.85, zorder=3, label='Transit cum.')
    ax3.plot(dates_hist, combined_cum_h.values,
             color=TEAL, lw=2.4, zorder=4, label='Combined cum.')
    ax3.fill_between(dates_hist, combined_cum_h.values, 0,
                     where=(combined_cum_h.values >= 0), color=GREEN, alpha=0.12, zorder=1)
    ax3.fill_between(dates_hist, combined_cum_h.values, 0,
                     where=(combined_cum_h.values < 0),  color=RED,   alpha=0.12, zorder=1)

if len(dates_fore):
    if len(dates_hist):
        conn_dates = [dates_hist[-1], dates_fore[0]]
        ax3.plot(conn_dates, [natal_cum_h.iloc[-1],    natal_cum_f.iloc[0]],
                 color=ORANGE, lw=1.4, ls='--', alpha=0.5)
        ax3.plot(conn_dates, [transit_cum_h.iloc[-1],  transit_cum_f.iloc[0]],
                 color=PURPLE, lw=1.4, ls='--', alpha=0.5)
        ax3.plot(conn_dates, [combined_cum_h.iloc[-1], combined_cum_f.iloc[0]],
                 color=TEAL, lw=2.4, alpha=0.6)

    ax3.plot(dates_fore, natal_cum_f.values,
             color=ORANGE, lw=1.4, ls=':', alpha=0.75, zorder=3)
    ax3.plot(dates_fore, transit_cum_f.values,
             color=PURPLE, lw=1.4, ls=':', alpha=0.75, zorder=3)
    ax3.plot(dates_fore, combined_cum_f.values,
             color=TEAL, lw=2.4, ls='--', alpha=0.75, zorder=4)

    last_val = combined_cum_h.iloc[-1] if len(combined_cum_h) else 0
    ax3.fill_between(dates_fore, combined_cum_f.values, last_val,
                     where=(combined_cum_f.values >= last_val),
                     color=GREEN, alpha=0.10, zorder=1)
    ax3.fill_between(dates_fore, combined_cum_f.values, last_val,
                     where=(combined_cum_f.values < last_val),
                     color=RED, alpha=0.10, zorder=1)

draw_today_line(ax3)
format_xaxis(ax3, n_days)
ax3.legend(fontsize=8, facecolor='#1A1A38', labelcolor=WHITE, loc='upper left')
fig3.suptitle(
    f"{label}  |  Cumulative Aspect Score\n"
    f"Teal=Combined  Orange=Natal  Purple=Transit  |  "
    f"Solid/Dashed=History  Dotted=Forecast  |  Gold dashed = Today",
    color=GOLD, fontsize=11, fontweight='bold')
fig3.tight_layout()
st.pyplot(fig3, use_container_width=True)
plt.close(fig3)

# ============================================================
#  ASPECT TABLES
# ============================================================

table_start_ts = today_ts - pd.Timedelta(days=table_past_days)
table_end_ts2  = today_ts + pd.Timedelta(days=table_future_days)

def filter_window(detail_list):
    rows = []
    for r in detail_list:
        d = pd.Timestamp(r['date'])
        if table_start_ts <= d <= table_end_ts2:
            r2 = r.copy()
            r2['date']   = d.date()
            r2['period'] = 'Past' if d <= today_ts else 'Future'
            rows.append(r2)
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values(['date','score'], ascending=[True, False])
            .reset_index(drop=True))

natal_win   = filter_window(natal_detail_full)
transit_win = filter_window(transit_detail_full)

def score_style(val):
    if isinstance(val, (int, float)):
        if val > 0: return 'color: #44DD88; font-weight: bold'
        if val < 0: return 'color: #E84040; font-weight: bold'
    return ''

def build_daily_net(raw_win, period_filter):
    if raw_win.empty:
        return pd.DataFrame()
    grp = (raw_win.groupby(['date', 'period'])
           .agg(Aspects=('score', 'count'), Net_Score=('score', 'sum'))
           .reset_index().sort_values('date'))
    grp = grp[grp['period'] == period_filter].copy()
    if grp.empty:
        return pd.DataFrame()
    grp['Bias'] = grp['Net_Score'].apply(lambda x: '▲ Supportive' if x > 0 else '▼ Challenging')
    grp['date'] = grp['date'].astype(str)
    grp.columns = ['Date', 'Period', '# Aspects', 'Net Score', 'Bias']
    return grp[['Date', '# Aspects', 'Net Score', 'Bias']]

st.markdown("---")
st.markdown("## 📋 Aspect Tables")
st.caption(
    f"Last **{table_past_days}** days + next **{table_future_days}** days. "
    "Green = supportive · Red = challenging.")

tab1, tab2 = st.tabs(["🌟 Natal Aspects (Transit × Natal)", "🔄 Transit × Transit Aspects"])

with tab1:
    if natal_win.empty:
        st.info("No natal aspects active in the selected window.")
    else:
        def prep_natal(sub):
            if sub.empty: return sub
            df = sub.rename(columns={
                'date':'Date', 'transit':'Transit Planet',
                'natal':'Natal Planet', 'aspect':'Aspect',
                'phase':'Phase', 'orb':'Orb°', 'score':'Score'})
            df['Transit Planet'] = df['Transit Planet'].apply(
                lambda x: PLANET_SYMBOLS.get(x, x.replace('_',' ').title()))
            df['Natal Planet'] = df['Natal Planet'].apply(
                lambda x: PLANET_SYMBOLS.get(x, x.replace('_',' ').title()))
            cols = ['Date','Transit Planet','Natal Planet','Aspect','Phase','Orb°','Score']
            return df[[c for c in cols if c in df.columns]]

        st.markdown(f"### ⏪ Past {table_past_days} days")
        past_n = natal_win[natal_win['period']=='Past']
        if past_n.empty:
            st.info("No natal aspects in the past window.")
        else:
            st.dataframe(
                prep_natal(past_n).style.applymap(score_style, subset=['Score']),
                use_container_width=True, hide_index=True)

        st.markdown(f"### ⏩ Next {table_future_days} days")
        fut_n = natal_win[natal_win['period']=='Future']
        if fut_n.empty:
            st.info("No natal aspects in the forecast window.")
        else:
            st.dataframe(
                prep_natal(fut_n).style.applymap(score_style, subset=['Score']),
                use_container_width=True, hide_index=True)

        st.markdown("### 📆 Daily Net Natal Score")
        dn_past = build_daily_net(natal_win, 'Past')
        dn_fut  = build_daily_net(natal_win, 'Future')
        dn = pd.concat([dn_past, dn_fut]).reset_index(drop=True) if not dn_past.empty or not dn_fut.empty else pd.DataFrame()
        if dn.empty:
            st.info("No daily net data available.")
        else:
            st.dataframe(
                dn.style.applymap(score_style, subset=['Net Score']),
                use_container_width=True, hide_index=True)

with tab2:
    if transit_win.empty:
        st.info("No transit aspects active in the selected window.")
    else:
        def prep_transit(sub):
            if sub.empty: return sub
            df = sub.rename(columns={
                'date':'Date', 'planet_a':'Planet A', 'planet_b':'Planet B',
                'aspect':'Aspect', 'phase':'Phase', 'orb':'Orb°', 'score':'Score'})
            df['Planet A'] = df['Planet A'].apply(
                lambda x: PLANET_SYMBOLS.get(x, x.replace('_',' ').title()))
            df['Planet B'] = df['Planet B'].apply(
                lambda x: PLANET_SYMBOLS.get(x, x.replace('_',' ').title()))
            cols = ['Date','Planet A','Planet B','Aspect','Phase','Orb°','Score']
            return df[[c for c in cols if c in df.columns]]

        st.markdown(f"### ⏪ Past {table_past_days} days")
        past_t = transit_win[transit_win['period']=='Past']
        if past_t.empty:
            st.info("No transit aspects in the past window.")
        else:
            st.dataframe(
                prep_transit(past_t).style.applymap(score_style, subset=['Score']),
                use_container_width=True, hide_index=True)

        st.markdown(f"### ⏩ Next {table_future_days} days")
        fut_t = transit_win[transit_win['period']=='Future']
        if fut_t.empty:
            st.info("No transit aspects in the forecast window.")
        else:
            st.dataframe(
                prep_transit(fut_t).style.applymap(score_style, subset=['Score']),
                use_container_width=True, hide_index=True)

        st.markdown("### 📆 Daily Net Transit Score")
        dt_past = build_daily_net(transit_win, 'Past')
        dt_fut  = build_daily_net(transit_win, 'Future')
        dt = pd.concat([dt_past, dt_fut]).reset_index(drop=True) if not dt_past.empty or not dt_fut.empty else pd.DataFrame()
        if dt.empty:
            st.info("No daily net data available.")
        else:
            st.dataframe(
                dt.style.applymap(score_style, subset=['Net Score']),
                use_container_width=True, hide_index=True)

# ============================================================
#  SCORING LEGEND
# ============================================================

st.markdown("---")
with st.expander("📖 Scoring Methodology", expanded=False):
    angle_note = ""
    if natal_angles:
        angle_entries = []
        for k, v in natal_angles.items():
            sign = SIGNS[int(v // 30)]
            deg  = int(v % 30)
            angle_entries.append(f"{PLANET_SYMBOLS[k]}: {deg}° {sign} ({v:.1f}°) — Potency 1.5, Neutral")
        angle_note = "\n\n**Active Natal Angles:**\n" + "\n".join(f"- {e}" for e in angle_entries)

    st.markdown(f"""
**Score = modulated_base × avg_potency × orb_proximity × phase_factor**

| Component | Rule |
|---|---|
| **Conjunction** | Direction = sign of (nat_A + nat_B). Both benefic → +, both malefic → −, mixed → 0 |
| **Other aspects** | Base score × (1 + {NATURE_MOD} × avg_nature × sign(base)). Same-nature pairs amplify natural expression |
| **orb_proximity** | Linear: 1.0 at exact → 0.0 at orb edge |
| **phase_factor** | Applying = {PHASE_FACTOR['apply']} · Separating = {PHASE_FACTOR['sep']} |

**Planet Potency & Nature:**

| Planet | Potency | Nature |
|---|---|---|
| ♃ Jupiter | 3.0 | +Benefic |
| ♀ Venus | 2.0 | +Benefic |
| ☉ Sun | 1.5 | +Benefic |
| ☽ Moon | 1.0 | +Benefic |
| ♆ Neptune | 0.5 | +Benefic |
| ♄ Saturn | 2.5 | −Malefic |
| ♂ Mars | 2.0 | −Malefic |
| ♇ Pluto | 1.5 | −Malefic |
| ♅ Uranus | 0.5 | −Malefic |
| ☿ Mercury | 0.5 | Neutral |
| ☊ Nodes | 0.5 | +Benefic |
| ↑ ASC / MC | 1.5 | Neutral (angle) |

**Aspect Base Scores:** Trine +2.0 · Sextile +1.5 · Conjunction ±1.0 · Opposition −1.5 · Square −1.8

**Nature Modulation (NATURE_MOD = {NATURE_MOD}):**
- Two benefics making a trine → score amplified by {NATURE_MOD*100:.0f}%
- Two malefics making a square → tense score amplified by {NATURE_MOD*100:.0f}%
- Mixed pair → no amplification, base score unchanged

**Natal Angles (ASC / MC):**
- Treated as neutral natal points with potency 1.5
- Their conjunction score direction is determined solely by the transiting planet's nature
  (e.g. Jupiter conjunct ASC → positive; Saturn conjunct ASC → negative)
- Harmonious aspects (trine, sextile) to angles score positively; tense aspects (square, opposition) score negatively
- Leave sign selector as '—' in the sidebar to exclude angles from scoring{angle_note}

**Active filters:**
- Natal planets: **{natal_planet_label}**
- Transit planets: **{transit_planet_label}**

**How to read:**
- **Chart 1 (Natal)** — How current transits interact with your specific birth planets (and angles if set).
  Select fewer natal planets to isolate specific themes (e.g. only Jupiter + Saturn for luck/restriction cycle).
- **Chart 2 (Transit)** — General planetary weather affecting everyone.
- **Chart 3 (Cumulative)** — Running total. Rising slope = improving conditions; falling = accumulating challenges.
- **Orb settings:** Apply={orb_apply}°  Sep={orb_sep}° — Applying aspects score at full weight; separating at {PHASE_FACTOR['sep']:.0%}.
    """)


















# import warnings, datetime, itertools
# warnings.filterwarnings('ignore')

# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import matplotlib.patches as mpatches
# from matplotlib.lines import Line2D
# import streamlit as st

# # ============================================================
# #  PAGE CONFIG
# # ============================================================

# st.set_page_config(
#     page_title="🌙 Personal Aspect Scorer",
#     page_icon="🌙",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# st.markdown("""
# <style>
#   .stApp { background-color: #0A0A1A; color: #E8E8F4; }
#   section[data-testid="stSidebar"] { background-color: #0D0D28; }
#   section[data-testid="stSidebar"] * { color: #E8E8F4 !important; }
#   .stTextInput > div > div > input,
#   .stNumberInput > div > div > input,
#   .stDateInput > div > div > input {
#       background-color: #1A1A38; color: #E8E8F4;
#       border: 1px solid #2A2A4A;
#   }
#   .stSlider > div { color: #E8E8F4; }
#   .stButton > button {
#       background-color: #C8A84B; color: #0A0A1A;
#       font-weight: bold; border: none; border-radius: 4px;
#       padding: 0.5rem 2rem; width: 100%;
#   }
#   .stButton > button:hover { background-color: #E8C86B; }
#   h1, h2, h3 { color: #C8A84B !important; }
#   div[data-testid="stMetric"] {
#       background-color: #0D0D28; border: 1px solid #2A2A4A;
#       border-radius: 6px; padding: 0.5rem 1rem;
#   }
#   div[data-testid="stMetric"] label { color: #C8A84B !important; }
#   .stTabs [data-baseweb="tab"] { color: #E8E8F4; }
#   .stTabs [aria-selected="true"] { color: #C8A84B !important; border-bottom-color: #C8A84B !important; }
#   .stCheckbox > label { color: #E8E8F4 !important; }
# </style>
# """, unsafe_allow_html=True)

# # ============================================================
# #  EPHEMERIS — loaded from GitHub (cached)
# # ============================================================

# GITHUB_EPH_URL = (
#     "https://raw.githubusercontent.com/"
#     "goncuahm/kozmikharita/main/planet_degrees.csv"
# )

# @st.cache_data(show_spinner="Loading ephemeris from GitHub …")
# def load_ephemeris(url):
#     eph_raw = pd.read_csv(url, index_col='date', parse_dates=True)
#     return eph_raw

# # ============================================================
# #  CONSTANTS
# # ============================================================

# EPH_PLANET_COLS = [
#     'sun', 'moon', 'mercury', 'venus', 'mars',
#     'jupiter', 'saturn', 'uranus', 'neptune',
#     'pluto', 'true_node', 'mean_node',
# ]

# ASPECTS   = [0, 60, 90, 120, 180]
# ASP_NAMES = {0:'Conj', 60:'Sext', 90:'Sqr', 120:'Trine', 180:'Opp'}
# SIGNS     = ['Aries','Taurus','Gemini','Cancer','Leo','Virgo',
#              'Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces']

# # ── SCORING: consistent with trading app ────────────────────

# # Planet POTENCY: always positive — how strongly the planet expresses any aspect.
# PLANET_POTENCY = {
#     'jupiter':   3.0,
#     'venus':     2.0,
#     'sun':       1.5,
#     'moon':      1.0,
#     'mars':      2.0,
#     'saturn':    2.5,
#     'pluto':     1.5,
#     'mercury':   0.5,
#     'neptune':   0.5,
#     'uranus':    0.5,
#     'true_node': 0.5,
#     'mean_node': 0.5,
#     # Angles — treated as sensitive natal points
#     'asc':       1.5,
#     'mc':        1.5,
# }

# # Planet NATURE: +1 = benefic, -1 = malefic, 0 = neutral
# PLANET_NATURE = {
#     'jupiter':   +1,
#     'venus':     +1,
#     'sun':       +1,
#     'moon':      +1,
#     'neptune':   +1,
#     'true_node': +1,
#     'mean_node': +1,
#     'mercury':    0,
#     'mars':      -1,
#     'saturn':    -1,
#     'uranus':    -1,
#     'pluto':     -1,
#     # Angles — neutral receptors (direction comes from transiting planet nature)
#     'asc':        0,
#     'mc':         0,
# }

# # Base aspect polarity: positive = harmonious, negative = tense
# # Conjunction = 1.0 (neutral strength — direction decided entirely by planet nature)
# ASPECT_BASE = {
#     0:   +1.0,
#     60:  +1.5,
#     90:  -1.8,
#     120: +2.0,
#     180: -1.5,
# }

# # Nature modulation factor
# NATURE_MOD = 0.35

# PHASE_FACTOR = {'apply': 1.0, 'sep': 0.6}

# # Chart palette
# BG     = '#0A0A1A';  PANEL  = '#0D0D28';  GOLD   = '#C8A84B'
# TEAL   = '#00D4B4';  WHITE  = '#E8E8F4';  GREY   = '#2A2A4A'
# GREEN  = '#44DD88';  RED    = '#E84040';  ORANGE = '#FF8844'
# PURPLE = '#CC44FF';  BLUE   = '#4488FF'

# # Planet display symbols for sidebar
# PLANET_SYMBOLS = {
#     'sun':       '☉ Sun',
#     'moon':      '☽ Moon',
#     'mercury':   '☿ Mercury',
#     'venus':     '♀ Venus',
#     'mars':      '♂ Mars',
#     'jupiter':   '♃ Jupiter',
#     'saturn':    '♄ Saturn',
#     'uranus':    '♅ Uranus',
#     'neptune':   '♆ Neptune',
#     'pluto':     '♇ Pluto',
#     'true_node': '☊ True Node',
#     'mean_node': '☊ Mean Node',
#     'asc':       '↑ ASC',
#     'mc':        '↑ MC',
# }

# # ============================================================
# #  SIDEBAR
# # ============================================================

# with st.sidebar:
#     st.markdown("## 🌙 Personal Aspect Scorer")
#     st.markdown("---")

#     st.markdown("### 🎂 Birth Date")
#     birth_date_input = st.text_input(
#         "Birth date (YYYY-MM-DD)",
#         value="1979-12-04",
#         help="Enter the person's date of birth.")

#     person_name = st.text_input(
#         "Name / label (optional)",
#         value="",
#         help="Shown in chart titles. Leave blank to use birth date.")

#     # ── OPTIONAL ASC / MC ENTRY ───────────────────────────────
#     st.markdown("### 🔺 Natal Angles (optional)")
#     st.caption(
#         "Enter your Ascendant (ASC) and/or Midheaven (MC) to include them "
#         "as natal points in the scoring. Leave sign as '—' to skip.")

#     SIGN_OPTIONS = ['—'] + SIGNS  # '—' means not set

#     asc_sign = st.selectbox(
#         "ASC Sign", options=SIGN_OPTIONS, index=0,
#         help="Zodiac sign of your Ascendant. Select '—' to ignore.")
#     asc_deg = st.number_input(
#         "ASC Degree (0–29)", min_value=0, max_value=29, value=0, step=1,
#         help="Degree within the sign (0–29).")

#     mc_sign = st.selectbox(
#         "MC Sign", options=SIGN_OPTIONS, index=0,
#         help="Zodiac sign of your Midheaven (MC). Select '—' to ignore.")
#     mc_deg = st.number_input(
#         "MC Degree (0–29)", min_value=0, max_value=29, value=0, step=1,
#         help="Degree within the sign (0–29).")

#     # Compute absolute longitudes for angles (None if not set)
#     def angle_longitude(sign_str, deg):
#         if sign_str == '—':
#             return None
#         sign_idx = SIGNS.index(sign_str)
#         return float(sign_idx * 30 + deg)

#     asc_longitude = angle_longitude(asc_sign, asc_deg)
#     mc_longitude  = angle_longitude(mc_sign,  mc_deg)

#     st.markdown("### 📅 Display Range")
#     display_start = st.text_input(
#         "Start date (YYYY-MM-DD)",
#         value=(datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
#         help="Beginning of the score plot window.")

#     display_end = st.text_input(
#         "End date / forecast to (YYYY-MM-DD)",
#         value=(datetime.date.today() + datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
#         help="End of the score plot window. Can be in the future.")

#     st.markdown("### 🔭 Orb Settings")
#     orb_apply = st.slider(
#         "Applying orb (degrees)", min_value=1, max_value=6, value=4,
#         help="Degrees before exact aspect to start scoring.")
#     orb_sep = st.slider(
#         "Separating orb (degrees)", min_value=1, max_value=3, value=1,
#         help="Degrees after exact aspect to stop scoring.")

#     st.markdown("### 📋 Table Horizon")
#     table_past_days = st.slider(
#         "Days back in tables", min_value=3, max_value=30, value=7)
#     table_future_days = st.slider(
#         "Days ahead in tables", min_value=7, max_value=90, value=30)

#     st.markdown("### 📊 Chart Options")
#     smooth_window = st.slider(
#         "Smoothing window (days)", min_value=1, max_value=30, value=7,
#         help="Rolling mean window for the smoothed overlay line.")

#     # ── NATAL PLANET SELECTOR ─────────────────────────────────
#     st.markdown("### 🌟 Natal Planets to Score")
#     st.caption(
#         "Select which natal planets transits should be scored against. "
#         "All transit × transit aspects are always computed regardless.")

#     if 'natal_planet_selection' not in st.session_state:
#         st.session_state['natal_planet_selection'] = {p: True for p in EPH_PLANET_COLS}

#     natal_planet_checkboxes = {}
#     for p in EPH_PLANET_COLS:
#         label = PLANET_SYMBOLS.get(p, p.replace('_', ' ').title())
#         natal_planet_checkboxes[p] = st.checkbox(
#             label,
#             value=st.session_state['natal_planet_selection'].get(p, True),
#             key=f"natal_cb_{p}")

#     for p in EPH_PLANET_COLS:
#         st.session_state['natal_planet_selection'][p] = natal_planet_checkboxes[p]

#     selected_natal_planets = [p for p in EPH_PLANET_COLS if natal_planet_checkboxes[p]]

#     # ── TRANSIT PLANET SELECTOR ───────────────────────────────
#     st.markdown("### 🔄 Transit Planets to Use")
#     st.caption(
#         "Select which transiting planets are included in aspect calculations. "
#         "Affects both natal and transit × transit scoring.")

#     if 'transit_planet_selection' not in st.session_state:
#         st.session_state['transit_planet_selection'] = {p: True for p in EPH_PLANET_COLS}

#     transit_planet_checkboxes = {}
#     for p in EPH_PLANET_COLS:
#         label = PLANET_SYMBOLS.get(p, p.replace('_', ' ').title())
#         transit_planet_checkboxes[p] = st.checkbox(
#             label,
#             value=st.session_state['transit_planet_selection'].get(p, True),
#             key=f"transit_cb_{p}")

#     for p in EPH_PLANET_COLS:
#         st.session_state['transit_planet_selection'][p] = transit_planet_checkboxes[p]

#     selected_transit_planets = [p for p in EPH_PLANET_COLS if transit_planet_checkboxes[p]]

#     st.markdown("---")
#     run_btn = st.button("▶  Run Analysis", type="primary")

# # ============================================================
# #  HEADER
# # ============================================================

# st.markdown("# 🌙 Personal Natal Aspect Scorer")
# st.markdown(
#     "Scores daily planetary aspects relative to a person's natal chart. "
#     "No stock prices — purely astrological energy over time. "
#     "**Positive = harmonious / supportive conditions. Negative = challenging conditions.**"
# )

# if not run_btn:
#     st.info("👈 Enter birth date and date range in the sidebar, then click **▶ Run Analysis**.")
#     st.stop()

# # ============================================================
# #  VALIDATE INPUTS
# # ============================================================

# try:
#     birth_ts = pd.Timestamp(birth_date_input)
# except Exception:
#     st.error("Invalid birth date format. Use YYYY-MM-DD.")
#     st.stop()

# try:
#     start_ts = pd.Timestamp(display_start)
#     end_ts   = pd.Timestamp(display_end)
# except Exception:
#     st.error("Invalid date range. Use YYYY-MM-DD for both start and end.")
#     st.stop()

# if end_ts <= start_ts:
#     st.error("End date must be after start date.")
#     st.stop()

# if not selected_natal_planets:
#     st.error("Please select at least one natal planet in the sidebar.")
#     st.stop()

# if not selected_transit_planets:
#     st.error("Please select at least one transit planet in the sidebar.")
#     st.stop()

# label = person_name.strip() if person_name.strip() else birth_date_input
# today_ts = pd.Timestamp(datetime.date.today())

# # ============================================================
# #  LOAD EPHEMERIS
# # ============================================================

# with st.spinner("Loading ephemeris …"):
#     try:
#         eph_raw = load_ephemeris(GITHUB_EPH_URL)
#         avail_planets = [p for p in EPH_PLANET_COLS if p in eph_raw.columns]
#         eph = eph_raw[avail_planets].copy()
#         st.success(
#             f"Ephemeris loaded: {len(eph):,} days  "
#             f"({eph.index[0].date()} → {eph.index[-1].date()})"
#         )
#     except Exception as e:
#         st.error(
#             f"Failed to load ephemeris from GitHub.\n\n"
#             f"**Update `GITHUB_EPH_URL`** at the top of this script "
#             f"with your actual raw GitHub URL.\n\nError: {e}"
#         )
#         st.stop()

# # Check date coverage
# if start_ts < eph.index[0]:
#     start_ts = eph.index[0]
#     st.warning(f"Start date adjusted to ephemeris start: {start_ts.date()}")
# if end_ts > eph.index[-1]:
#     end_ts = eph.index[-1]
#     st.warning(f"End date adjusted to ephemeris end: {end_ts.date()}")

# # Filter selected natal planets to only those available in ephemeris
# selected_natal_planets = [p for p in selected_natal_planets if p in avail_planets]
# if not selected_natal_planets:
#     st.error("None of the selected natal planets are in the loaded ephemeris.")
#     st.stop()

# # Filter selected transit planets to only those available in ephemeris
# selected_transit_planets = [p for p in selected_transit_planets if p in avail_planets]
# if not selected_transit_planets:
#     st.error("None of the selected transit planets are in the loaded ephemeris.")
#     st.stop()

# # ============================================================
# #  NATAL CHART
# # ============================================================

# if birth_ts not in eph.index:
#     idx      = eph.index.get_indexer([birth_ts], method='nearest')[0]
#     birth_ts = eph.index[idx]

# natal_row = eph.loc[birth_ts]
# natal = {p: float(natal_row[p]) % 360 for p in avail_planets}

# # ── Add ASC / MC to natal dict if provided ───────────────────
# natal_angles = {}   # key → absolute longitude; used in scoring
# if asc_longitude is not None:
#     natal_angles['asc'] = asc_longitude
# if mc_longitude is not None:
#     natal_angles['mc'] = mc_longitude

# with st.expander(f"🌟 Natal Chart — {label} ({birth_date_input})", expanded=True):
#     natal_data = []
#     for p, lon in natal.items():
#         sign    = SIGNS[int(lon // 30)]
#         deg_in  = int(lon % 30)
#         mins    = int((lon % 1) * 60)
#         is_selected = p in selected_natal_planets
#         natal_data.append({
#             'Selected':  '✓' if is_selected else '',
#             'Planet':    PLANET_SYMBOLS.get(p, p.replace('_',' ').title()),
#             'Longitude': f"{lon:.3f}°",
#             'Sign':      sign,
#             'Position':  f"{deg_in:02d}° {mins:02d}′ {sign}",
#             'Potency':   PLANET_POTENCY.get(p, 0.5),
#             'Nature':    '+Benefic' if PLANET_NATURE.get(p, 0) > 0
#                          else ('−Malefic' if PLANET_NATURE.get(p, 0) < 0
#                                else 'Neutral'),
#         })

#     # Add angle rows if entered
#     for angle_key, angle_lon in natal_angles.items():
#         sign   = SIGNS[int(angle_lon // 30)]
#         deg_in = int(angle_lon % 30)
#         natal_data.append({
#             'Selected':  '✓',
#             'Planet':    PLANET_SYMBOLS.get(angle_key, angle_key.upper()),
#             'Longitude': f"{angle_lon:.1f}°",
#             'Sign':      sign,
#             'Position':  f"{deg_in:02d}° 00′ {sign}",
#             'Potency':   PLANET_POTENCY.get(angle_key, 1.5),
#             'Nature':    'Neutral (angle)',
#         })

#     natal_df = pd.DataFrame(natal_data)
#     st.dataframe(natal_df, use_container_width=True, hide_index=True)

#     if len(selected_natal_planets) < len(avail_planets) or len(selected_transit_planets) < len(avail_planets):
#         st.info(
#             f"Natal planets scored: **{', '.join(PLANET_SYMBOLS.get(p, p) for p in selected_natal_planets)}**  \n"
#             f"Transit planets used: **{', '.join(PLANET_SYMBOLS.get(p, p) for p in selected_transit_planets)}**")

#     if natal_angles:
#         angle_strs = []
#         for k, v in natal_angles.items():
#             sign = SIGNS[int(v // 30)]
#             deg  = int(v % 30)
#             angle_strs.append(f"**{PLANET_SYMBOLS[k]}** {deg}° {sign} ({v:.1f}°)")
#         st.info("Natal angles included in scoring: " + "  |  ".join(angle_strs))

# # ============================================================
# #  CORE HELPERS
# # ============================================================

# def angular_diff(lon_a, lon_b):
#     d = (lon_a - lon_b) % 360
#     return np.where(d > 180, d - 360, d)

# def orb_factor(abs_gap, orb_max):
#     return np.clip(1.0 - abs_gap / orb_max, 0.0, 1.0)

# def aspect_score_single(pot_a, pot_b, asp, orb_f, phase, nat_a=0, nat_b=0):
#     """
#     Consistent with trading app scoring:
#       pot_a / pot_b : planet potency (always > 0)
#       nat_a / nat_b : planet nature  (+1 benefic, -1 malefic, 0 neutral)
#       asp           : aspect angle (0, 60, 90, 120, 180)
#       orb_f         : orb proximity factor 0→1
#       phase         : 'apply' or 'sep'

#     Conjunction (asp=0):
#       - Direction = sign of (nat_a + nat_b)
#       - Magnitude scales with how strongly both planets share the same nature
#     Other aspects:
#       - Base score from ASPECT_BASE
#       - Nature modulates: same-nature pairs amplify the natural expression
#       - avg_nat > 0 (benefic pair)  → harmonious aspects grow, tense aspects shrink
#       - avg_nat < 0 (malefic pair)  → tense aspects grow, harmonious aspects shrink
#     """
#     avg_pot = (pot_a + pot_b) / 2.0
#     avg_nat = (nat_a + nat_b) / 2.0

#     if asp == 0:
#         net_nature = nat_a + nat_b
#         if net_nature == 0:
#             return 0.0
#         direction = float(np.sign(net_nature))
#         magnitude = avg_pot * abs(net_nature) / 2.0
#         return direction * magnitude * ASPECT_BASE[0] * orb_f * PHASE_FACTOR[phase]
#     else:
#         base      = ASPECT_BASE[asp]
#         modulated = base * (1.0 + NATURE_MOD * avg_nat * float(np.sign(base)))
#         return modulated * avg_pot * orb_f * PHASE_FACTOR[phase]

# def compute_natal_score(date_index):
#     """
#     Scores SELECTED transit planets against:
#       1. SELECTED natal planets (from ephemeris)
#       2. Any natal angles (ASC / MC) entered by the user
#     """
#     eph_a  = eph.reindex(date_index, method='ffill')
#     n      = len(date_index)
#     scores = np.zeros(n)
#     detail = []

#     # Build the full list of natal targets: ephemeris planets + user-entered angles
#     natal_targets = {}
#     for np_ in selected_natal_planets:
#         natal_targets[np_] = natal[np_]
#     for angle_key, angle_lon in natal_angles.items():
#         natal_targets[angle_key] = angle_lon

#     for tp in selected_transit_planets:
#         if tp not in eph_a.columns: continue
#         t_lons  = eph_a[tp].values.astype(float) % 360
#         motion  = np.gradient(np.unwrap(t_lons, period=360))
#         pot_t   = PLANET_POTENCY.get(tp, 0.5)
#         nat_t   = PLANET_NATURE.get(tp, 0)

#         for np_, n_lon in natal_targets.items():
#             pot_n = PLANET_POTENCY.get(np_, 0.5)
#             nat_n = PLANET_NATURE.get(np_, 0)
#             for asp in ASPECTS:
#                 target  = (n_lon + asp) % 360
#                 gap     = angular_diff(t_lons, target)
#                 abs_gap = np.abs(gap)
#                 applying = ((motion > 0) & (gap < 0)) | ((motion < 0) & (gap > 0))
#                 mask_a   = applying    & (abs_gap <= orb_apply)
#                 mask_s   = (~applying) & (abs_gap <= orb_sep)
#                 for i in np.where(mask_a)[0]:
#                     of = float(orb_factor(abs_gap[i], orb_apply))
#                     sc = aspect_score_single(pot_t, pot_n, asp, of, 'apply', nat_t, nat_n)
#                     scores[i] += sc
#                     detail.append({'date': date_index[i], 'transit': tp,
#                                    'natal': np_, 'aspect': ASP_NAMES[asp],
#                                    'phase': 'Applying',
#                                    'orb': round(float(abs_gap[i]), 3),
#                                    'score': round(sc, 4)})
#                 for i in np.where(mask_s)[0]:
#                     of = float(orb_factor(abs_gap[i], orb_sep))
#                     sc = aspect_score_single(pot_t, pot_n, asp, of, 'sep', nat_t, nat_n)
#                     scores[i] += sc
#                     detail.append({'date': date_index[i], 'transit': tp,
#                                    'natal': np_, 'aspect': ASP_NAMES[asp],
#                                    'phase': 'Separating',
#                                    'orb': round(float(abs_gap[i]), 3),
#                                    'score': round(sc, 4)})
#     return pd.Series(scores, index=date_index), detail

# def compute_transit_score(date_index):
#     eph_a  = eph.reindex(date_index, method='ffill')
#     n      = len(date_index)
#     scores = np.zeros(n)
#     detail = []
#     pairs  = list(itertools.combinations(selected_transit_planets, 2))
#     for (pA, pB) in pairs:
#         if pA not in eph_a.columns or pB not in eph_a.columns: continue
#         lon_A   = eph_a[pA].values.astype(float) % 360
#         lon_B   = eph_a[pB].values.astype(float) % 360
#         motion  = np.gradient(np.unwrap(lon_A, period=360))
#         pot_A   = PLANET_POTENCY.get(pA, 0.5)
#         pot_B   = PLANET_POTENCY.get(pB, 0.5)
#         nat_A   = PLANET_NATURE.get(pA, 0)
#         nat_B   = PLANET_NATURE.get(pB, 0)
#         for asp in ASPECTS:
#             target  = (lon_B + asp) % 360
#             gap     = angular_diff(lon_A, target)
#             abs_gap = np.abs(gap)
#             applying = ((motion > 0) & (gap < 0)) | ((motion < 0) & (gap > 0))
#             mask_a   = applying    & (abs_gap <= orb_apply)
#             mask_s   = (~applying) & (abs_gap <= orb_sep)
#             for i in np.where(mask_a)[0]:
#                 of = float(orb_factor(abs_gap[i], orb_apply))
#                 sc = aspect_score_single(pot_A, pot_B, asp, of, 'apply', nat_A, nat_B)
#                 scores[i] += sc
#                 detail.append({'date': date_index[i], 'planet_a': pA,
#                                'planet_b': pB, 'aspect': ASP_NAMES[asp],
#                                'phase': 'Applying',
#                                'orb': round(float(abs_gap[i]), 3),
#                                'score': round(sc, 4)})
#             for i in np.where(mask_s)[0]:
#                 of = float(orb_factor(abs_gap[i], orb_sep))
#                 sc = aspect_score_single(pot_A, pot_B, asp, of, 'sep', nat_A, nat_B)
#                 scores[i] += sc
#                 detail.append({'date': date_index[i], 'planet_a': pA,
#                                'planet_b': pB, 'aspect': ASP_NAMES[asp],
#                                'phase': 'Separating',
#                                'orb': round(float(abs_gap[i]), 3),
#                                'score': round(sc, 4)})
#     return pd.Series(scores, index=date_index), detail

# # ============================================================
# #  BUILD DATE INDEX & COMPUTE SCORES
# # ============================================================

# full_index = pd.date_range(start=start_ts, end=end_ts, freq='D')

# table_end_ts = today_ts + pd.Timedelta(days=table_future_days + 3)
# if table_end_ts > end_ts:
#     table_ext = pd.date_range(
#         start = end_ts + pd.Timedelta(days=1),
#         end   = table_end_ts, freq='D')
#     compute_index = full_index.append(table_ext)
# else:
#     compute_index = full_index

# with st.spinner("Computing natal aspect scores …"):
#     natal_scores_full, natal_detail_full = compute_natal_score(compute_index)

# with st.spinner("Computing transit aspect scores …"):
#     transit_scores_full, transit_detail_full = compute_transit_score(compute_index)

# # Slice to display window
# natal_scores   = natal_scores_full.reindex(full_index).fillna(0)
# transit_scores = transit_scores_full.reindex(full_index).fillna(0)
# combined_scores = natal_scores.add(transit_scores, fill_value=0)

# # Split history / forecast
# hist_mask = full_index <= today_ts
# fore_mask = full_index >  today_ts

# dates_hist = full_index[hist_mask]
# dates_fore = full_index[fore_mask]

# natal_hist   = natal_scores[hist_mask]
# natal_fore   = natal_scores[fore_mask]
# transit_hist = transit_scores[hist_mask]
# transit_fore = transit_scores[fore_mask]
# combined_hist = combined_scores[hist_mask]
# combined_fore = combined_scores[fore_mask]

# # Summary metrics
# today_natal    = float(natal_scores.get(today_ts, natal_scores.iloc[len(dates_hist)-1]))
# today_transit  = float(transit_scores.get(today_ts, transit_scores.iloc[len(dates_hist)-1]))
# today_combined = today_natal + today_transit

# natal_planet_label = (
#     "All planets" if len(selected_natal_planets) == len(avail_planets)
#     else ", ".join(PLANET_SYMBOLS.get(p, p) for p in selected_natal_planets))

# # Append angle labels if active
# if natal_angles:
#     angle_labels = [PLANET_SYMBOLS.get(k, k.upper()) for k in natal_angles]
#     natal_planet_label += "  +  " + ", ".join(angle_labels)

# transit_planet_label = (
#     "All planets" if len(selected_transit_planets) == len(avail_planets)
#     else ", ".join(PLANET_SYMBOLS.get(p, p) for p in selected_transit_planets))

# st.markdown("---")
# c1, c2, c3, c4 = st.columns(4)
# c1.metric("Today's Natal Score",    f"{today_natal:.2f}",
#           delta="▲ Positive" if today_natal > 0 else "▼ Negative")
# c2.metric("Today's Transit Score",  f"{today_transit:.2f}",
#           delta="▲ Positive" if today_transit > 0 else "▼ Negative")
# c3.metric("Today's Combined Score", f"{today_combined:.2f}",
#           delta="▲ Positive" if today_combined > 0 else "▼ Negative")
# c4.metric("Planet filters",
#           f"N:{len(selected_natal_planets) + len(natal_angles)} / T:{len(selected_transit_planets)}",
#           delta=f"of {len(avail_planets)} ephem + {len(natal_angles)} angles")

# # ============================================================
# #  PLOT HELPERS
# # ============================================================

# def style_ax(ax):
#     ax.set_facecolor(PANEL)
#     for sp in ax.spines.values(): sp.set_color(GREY)
#     ax.tick_params(colors=WHITE, labelsize=8)
#     ax.yaxis.label.set_color(WHITE)

# def draw_zero(ax):
#     ax.axhline(0, color=GREY, lw=1.0, zorder=2)

# def draw_today_line(ax):
#     if start_ts <= today_ts <= end_ts:
#         ax.axvline(today_ts, color=GOLD, lw=1.8, ls='--', alpha=0.9, zorder=6)
#         ylims = ax.get_ylim()
#         ax.text(today_ts, ylims[1],
#                 ' Today', color=GOLD, fontsize=8,
#                 va='top', ha='left', fontweight='bold')

# def format_xaxis(ax, n_days):
#     if n_days <= 90:
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#         ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
#     elif n_days <= 400:
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#         ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
#     else:
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#         ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
#     ax.set_xlim(start_ts, end_ts + pd.Timedelta(days=2))

# def draw_score_bars(ax, dates_h, scores_h, dates_f, scores_f, alpha=0.75):
#     if len(dates_h):
#         h_vals   = scores_h.values
#         h_colors = [GREEN if v >= 0 else RED for v in h_vals]
#         ax.bar(dates_h, h_vals, color=h_colors, alpha=alpha, width=1.0, zorder=3)
#     if len(dates_f):
#         f_vals   = scores_f.values
#         f_colors = [GREEN if v >= 0 else RED for v in f_vals]
#         ax.bar(dates_f, f_vals, color=f_colors, alpha=alpha, width=1.0, zorder=3)

# def draw_shading(ax, dates_h, scores_h, dates_f, scores_f, alpha_cap, denom):
#     for d, s in zip(dates_h, scores_h.values):
#         if s == 0: continue
#         ax.axvspan(d, d + pd.Timedelta(days=1),
#                    alpha=min(alpha_cap, abs(s)/denom),
#                    color=GREEN if s > 0 else RED, zorder=1)
#     for d, s in zip(dates_f, scores_f.values):
#         if s == 0: continue
#         ax.axvspan(d, d + pd.Timedelta(days=1),
#                    alpha=min(alpha_cap, abs(s)/denom),
#                    color=GREEN if s > 0 else RED, zorder=1)

# def draw_smooth_line(ax, dates_h, scores_h, dates_f, scores_f, color, lw=1.8):
#     combined = pd.concat([scores_h, scores_f])
#     sm = combined.rolling(window=smooth_window, center=True, min_periods=1).mean()
#     if len(dates_h):
#         ax.plot(dates_h, sm.reindex(dates_h).values,
#                 color=color, lw=lw, zorder=5, label=f'{smooth_window}d smoothed')
#     if len(dates_f):
#         ax.plot(dates_f, sm.reindex(dates_f).values,
#                 color=color, lw=lw, ls='--', zorder=5, alpha=0.8)

# n_days = len(full_index)

# # ============================================================
# #  CHART 1: NATAL SCORE
# # ============================================================

# st.markdown("---")
# st.markdown("## Chart 1 — Natal Aspect Score")
# st.caption(
#     f"Transit planets: **{transit_planet_label}** aspecting natal planets: **{natal_planet_label}**. "
#     "Green = supportive. Red = challenging.")

# fig1, ax1 = plt.subplots(figsize=(18, 5), facecolor=BG)
# style_ax(ax1)
# ax1.set_ylabel('Natal Score', fontsize=10)
# draw_zero(ax1)
# draw_shading(ax1, dates_hist, natal_hist, dates_fore, natal_fore, 0.18, 15)
# draw_score_bars(ax1, dates_hist, natal_hist, dates_fore, natal_fore)
# draw_smooth_line(ax1, dates_hist, natal_hist, dates_fore, natal_fore, GOLD)
# if len(dates_hist):
#     ax1.fill_between(dates_hist, natal_hist.values, 0,
#                      where=(natal_hist.values >= 0), color=GREEN, alpha=0.08, zorder=1)
#     ax1.fill_between(dates_hist, natal_hist.values, 0,
#                      where=(natal_hist.values < 0),  color=RED,   alpha=0.08, zorder=1)
# if len(dates_fore):
#     ax1.fill_between(dates_fore, natal_fore.values, 0,
#                      where=(natal_fore.values >= 0), color=GREEN, alpha=0.08, zorder=1)
#     ax1.fill_between(dates_fore, natal_fore.values, 0,
#                      where=(natal_fore.values < 0),  color=RED,   alpha=0.08, zorder=1)
# draw_today_line(ax1)
# format_xaxis(ax1, n_days)
# ax1.legend(fontsize=8, facecolor='#1A1A38', labelcolor=WHITE, loc='upper left')
# fig1.suptitle(
#     f"{label}  |  Natal Aspect Score\n"
#     f"Transit: {transit_planet_label}  →  Natal: {natal_planet_label}\n"
#     f"Birth: {birth_date_input}  |  Apply≤{orb_apply}°  Sep≤{orb_sep}°  |  "
#     f"Green=Supportive  Red=Challenging  |  Gold dashed = Today",
#     color=GOLD, fontsize=10, fontweight='bold')
# fig1.tight_layout()
# st.pyplot(fig1, use_container_width=True)
# plt.close(fig1)

# # ============================================================
# #  CHART 2: TRANSIT × TRANSIT SCORE
# # ============================================================

# st.markdown("---")
# st.markdown("## Chart 2 — Transit × Transit Aspect Score")
# st.caption(f"Transit pairs from: **{transit_planet_label}**. Independent of natal chart.")

# fig2, ax2 = plt.subplots(figsize=(18, 5), facecolor=BG)
# style_ax(ax2)
# ax2.set_ylabel('Transit Score', fontsize=10)
# draw_zero(ax2)
# draw_shading(ax2, dates_hist, transit_hist, dates_fore, transit_fore, 0.18, 25)
# draw_score_bars(ax2, dates_hist, transit_hist, dates_fore, transit_fore)
# draw_smooth_line(ax2, dates_hist, transit_hist, dates_fore, transit_fore, GOLD)
# if len(dates_hist):
#     ax2.fill_between(dates_hist, transit_hist.values, 0,
#                      where=(transit_hist.values >= 0), color=GREEN, alpha=0.08, zorder=1)
#     ax2.fill_between(dates_hist, transit_hist.values, 0,
#                      where=(transit_hist.values < 0),  color=RED,   alpha=0.08, zorder=1)
# if len(dates_fore):
#     ax2.fill_between(dates_fore, transit_fore.values, 0,
#                      where=(transit_fore.values >= 0), color=GREEN, alpha=0.08, zorder=1)
#     ax2.fill_between(dates_fore, transit_fore.values, 0,
#                      where=(transit_fore.values < 0),  color=RED,   alpha=0.08, zorder=1)
# draw_today_line(ax2)
# format_xaxis(ax2, n_days)
# ax2.legend(fontsize=8, facecolor='#1A1A38', labelcolor=WHITE, loc='upper left')
# fig2.suptitle(
#     f"{label}  |  Transit × Transit Aspect Score\n"
#     f"Planets: {transit_planet_label}  |  "
#     f"Apply≤{orb_apply}°  Sep≤{orb_sep}°  |  "
#     f"Green=Supportive  Red=Challenging  |  Gold dashed = Today",
#     color=GOLD, fontsize=11, fontweight='bold')
# fig2.tight_layout()
# st.pyplot(fig2, use_container_width=True)
# plt.close(fig2)

# # ============================================================
# #  CHART 3: CUMULATIVE SCORES
# # ============================================================

# st.markdown("---")
# st.markdown("## Chart 3 — Cumulative Aspect Score")
# st.caption(
#     "Running total since display start. Rising = accumulating positive conditions. "
#     "Falling = accumulating challenges. Slope matters more than absolute value.")

# natal_cum    = natal_scores.cumsum()
# transit_cum  = transit_scores.cumsum()
# combined_cum = combined_scores.cumsum()

# natal_cum_h    = natal_cum[hist_mask]
# natal_cum_f    = natal_cum[fore_mask]
# transit_cum_h  = transit_cum[hist_mask]
# transit_cum_f  = transit_cum[fore_mask]
# combined_cum_h = combined_cum[hist_mask]
# combined_cum_f = combined_cum[fore_mask]

# fig3, ax3 = plt.subplots(figsize=(18, 5), facecolor=BG)
# style_ax(ax3)
# ax3.set_ylabel('Cumulative Score', fontsize=10)
# draw_zero(ax3)

# if len(dates_hist):
#     ax3.plot(dates_hist, natal_cum_h.values,
#              color=ORANGE, lw=1.4, ls='--', alpha=0.85, zorder=3, label='Natal cum.')
#     ax3.plot(dates_hist, transit_cum_h.values,
#              color=PURPLE, lw=1.4, ls='--', alpha=0.85, zorder=3, label='Transit cum.')
#     ax3.plot(dates_hist, combined_cum_h.values,
#              color=TEAL, lw=2.4, zorder=4, label='Combined cum.')
#     ax3.fill_between(dates_hist, combined_cum_h.values, 0,
#                      where=(combined_cum_h.values >= 0), color=GREEN, alpha=0.12, zorder=1)
#     ax3.fill_between(dates_hist, combined_cum_h.values, 0,
#                      where=(combined_cum_h.values < 0),  color=RED,   alpha=0.12, zorder=1)

# if len(dates_fore):
#     if len(dates_hist):
#         conn_dates = [dates_hist[-1], dates_fore[0]]
#         ax3.plot(conn_dates, [natal_cum_h.iloc[-1],    natal_cum_f.iloc[0]],
#                  color=ORANGE, lw=1.4, ls='--', alpha=0.5)
#         ax3.plot(conn_dates, [transit_cum_h.iloc[-1],  transit_cum_f.iloc[0]],
#                  color=PURPLE, lw=1.4, ls='--', alpha=0.5)
#         ax3.plot(conn_dates, [combined_cum_h.iloc[-1], combined_cum_f.iloc[0]],
#                  color=TEAL, lw=2.4, alpha=0.6)

#     ax3.plot(dates_fore, natal_cum_f.values,
#              color=ORANGE, lw=1.4, ls=':', alpha=0.75, zorder=3)
#     ax3.plot(dates_fore, transit_cum_f.values,
#              color=PURPLE, lw=1.4, ls=':', alpha=0.75, zorder=3)
#     ax3.plot(dates_fore, combined_cum_f.values,
#              color=TEAL, lw=2.4, ls='--', alpha=0.75, zorder=4)

#     last_val = combined_cum_h.iloc[-1] if len(combined_cum_h) else 0
#     ax3.fill_between(dates_fore, combined_cum_f.values, last_val,
#                      where=(combined_cum_f.values >= last_val),
#                      color=GREEN, alpha=0.10, zorder=1)
#     ax3.fill_between(dates_fore, combined_cum_f.values, last_val,
#                      where=(combined_cum_f.values < last_val),
#                      color=RED, alpha=0.10, zorder=1)

# draw_today_line(ax3)
# format_xaxis(ax3, n_days)
# ax3.legend(fontsize=8, facecolor='#1A1A38', labelcolor=WHITE, loc='upper left')
# fig3.suptitle(
#     f"{label}  |  Cumulative Aspect Score\n"
#     f"Teal=Combined  Orange=Natal  Purple=Transit  |  "
#     f"Solid/Dashed=History  Dotted=Forecast  |  Gold dashed = Today",
#     color=GOLD, fontsize=11, fontweight='bold')
# fig3.tight_layout()
# st.pyplot(fig3, use_container_width=True)
# plt.close(fig3)

# # ============================================================
# #  ASPECT TABLES
# # ============================================================

# table_start_ts = today_ts - pd.Timedelta(days=table_past_days)
# table_end_ts2  = today_ts + pd.Timedelta(days=table_future_days)

# def filter_window(detail_list):
#     rows = []
#     for r in detail_list:
#         d = pd.Timestamp(r['date'])
#         if table_start_ts <= d <= table_end_ts2:
#             r2 = r.copy()
#             r2['date']   = d.date()
#             r2['period'] = 'Past' if d <= today_ts else 'Future'
#             rows.append(r2)
#     if not rows:
#         return pd.DataFrame()
#     return (pd.DataFrame(rows)
#             .sort_values(['date','score'], ascending=[True, False])
#             .reset_index(drop=True))

# natal_win   = filter_window(natal_detail_full)
# transit_win = filter_window(transit_detail_full)

# def score_style(val):
#     if isinstance(val, (int, float)):
#         if val > 0: return 'color: #44DD88; font-weight: bold'
#         if val < 0: return 'color: #E84040; font-weight: bold'
#     return ''

# def build_daily_net(raw_win, period_filter):
#     if raw_win.empty:
#         return pd.DataFrame()
#     grp = (raw_win.groupby(['date', 'period'])
#            .agg(Aspects=('score', 'count'), Net_Score=('score', 'sum'))
#            .reset_index().sort_values('date'))
#     grp = grp[grp['period'] == period_filter].copy()
#     if grp.empty:
#         return pd.DataFrame()
#     grp['Bias'] = grp['Net_Score'].apply(lambda x: '▲ Supportive' if x > 0 else '▼ Challenging')
#     grp['date'] = grp['date'].astype(str)
#     grp.columns = ['Date', 'Period', '# Aspects', 'Net Score', 'Bias']
#     return grp[['Date', '# Aspects', 'Net Score', 'Bias']]

# st.markdown("---")
# st.markdown("## 📋 Aspect Tables")
# st.caption(
#     f"Last **{table_past_days}** days + next **{table_future_days}** days. "
#     "Green = supportive · Red = challenging.")

# tab1, tab2 = st.tabs(["🌟 Natal Aspects (Transit × Natal)", "🔄 Transit × Transit Aspects"])

# # ── TABLE 1: NATAL ASPECTS ────────────────────────────────────
# with tab1:
#     if natal_win.empty:
#         st.info("No natal aspects active in the selected window.")
#     else:
#         def prep_natal(sub):
#             if sub.empty: return sub
#             df = sub.rename(columns={
#                 'date':'Date', 'transit':'Transit Planet',
#                 'natal':'Natal Planet', 'aspect':'Aspect',
#                 'phase':'Phase', 'orb':'Orb°', 'score':'Score'})
#             df['Transit Planet'] = df['Transit Planet'].apply(
#                 lambda x: PLANET_SYMBOLS.get(x, x.replace('_',' ').title()))
#             df['Natal Planet'] = df['Natal Planet'].apply(
#                 lambda x: PLANET_SYMBOLS.get(x, x.replace('_',' ').title()))
#             cols = ['Date','Transit Planet','Natal Planet','Aspect','Phase','Orb°','Score']
#             return df[[c for c in cols if c in df.columns]]

#         st.markdown(f"### ⏪ Past {table_past_days} days")
#         past_n = natal_win[natal_win['period']=='Past']
#         if past_n.empty:
#             st.info("No natal aspects in the past window.")
#         else:
#             st.dataframe(
#                 prep_natal(past_n).style.applymap(score_style, subset=['Score']),
#                 use_container_width=True, hide_index=True)

#         st.markdown(f"### ⏩ Next {table_future_days} days")
#         fut_n = natal_win[natal_win['period']=='Future']
#         if fut_n.empty:
#             st.info("No natal aspects in the forecast window.")
#         else:
#             st.dataframe(
#                 prep_natal(fut_n).style.applymap(score_style, subset=['Score']),
#                 use_container_width=True, hide_index=True)

#         st.markdown("### 📆 Daily Net Natal Score")
#         dn_past = build_daily_net(natal_win, 'Past')
#         dn_fut  = build_daily_net(natal_win, 'Future')
#         dn = pd.concat([dn_past, dn_fut]).reset_index(drop=True) if not dn_past.empty or not dn_fut.empty else pd.DataFrame()
#         if dn.empty:
#             st.info("No daily net data available.")
#         else:
#             st.dataframe(
#                 dn.style.applymap(score_style, subset=['Net Score']),
#                 use_container_width=True, hide_index=True)

# # ── TABLE 2: TRANSIT ASPECTS ──────────────────────────────────
# with tab2:
#     if transit_win.empty:
#         st.info("No transit aspects active in the selected window.")
#     else:
#         def prep_transit(sub):
#             if sub.empty: return sub
#             df = sub.rename(columns={
#                 'date':'Date', 'planet_a':'Planet A', 'planet_b':'Planet B',
#                 'aspect':'Aspect', 'phase':'Phase', 'orb':'Orb°', 'score':'Score'})
#             df['Planet A'] = df['Planet A'].apply(
#                 lambda x: PLANET_SYMBOLS.get(x, x.replace('_',' ').title()))
#             df['Planet B'] = df['Planet B'].apply(
#                 lambda x: PLANET_SYMBOLS.get(x, x.replace('_',' ').title()))
#             cols = ['Date','Planet A','Planet B','Aspect','Phase','Orb°','Score']
#             return df[[c for c in cols if c in df.columns]]

#         st.markdown(f"### ⏪ Past {table_past_days} days")
#         past_t = transit_win[transit_win['period']=='Past']
#         if past_t.empty:
#             st.info("No transit aspects in the past window.")
#         else:
#             st.dataframe(
#                 prep_transit(past_t).style.applymap(score_style, subset=['Score']),
#                 use_container_width=True, hide_index=True)

#         st.markdown(f"### ⏩ Next {table_future_days} days")
#         fut_t = transit_win[transit_win['period']=='Future']
#         if fut_t.empty:
#             st.info("No transit aspects in the forecast window.")
#         else:
#             st.dataframe(
#                 prep_transit(fut_t).style.applymap(score_style, subset=['Score']),
#                 use_container_width=True, hide_index=True)

#         st.markdown("### 📆 Daily Net Transit Score")
#         dt_past = build_daily_net(transit_win, 'Past')
#         dt_fut  = build_daily_net(transit_win, 'Future')
#         dt = pd.concat([dt_past, dt_fut]).reset_index(drop=True) if not dt_past.empty or not dt_fut.empty else pd.DataFrame()
#         if dt.empty:
#             st.info("No daily net data available.")
#         else:
#             st.dataframe(
#                 dt.style.applymap(score_style, subset=['Net Score']),
#                 use_container_width=True, hide_index=True)

# # ============================================================
# #  SCORING LEGEND
# # ============================================================

# st.markdown("---")
# with st.expander("📖 Scoring Methodology", expanded=False):
#     angle_note = ""
#     if natal_angles:
#         angle_entries = []
#         for k, v in natal_angles.items():
#             sign = SIGNS[int(v // 30)]
#             deg  = int(v % 30)
#             angle_entries.append(f"{PLANET_SYMBOLS[k]}: {deg}° {sign} ({v:.1f}°) — Potency 1.5, Neutral")
#         angle_note = "\n\n**Active Natal Angles:**\n" + "\n".join(f"- {e}" for e in angle_entries)

#     st.markdown(f"""
# **Score = modulated_base × avg_potency × orb_proximity × phase_factor**

# | Component | Rule |
# |---|---|
# | **Conjunction** | Direction = sign of (nat_A + nat_B). Both benefic → +, both malefic → −, mixed → 0 |
# | **Other aspects** | Base score × (1 + {NATURE_MOD} × avg_nature × sign(base)). Same-nature pairs amplify natural expression |
# | **orb_proximity** | Linear: 1.0 at exact → 0.0 at orb edge |
# | **phase_factor** | Applying = {PHASE_FACTOR['apply']} · Separating = {PHASE_FACTOR['sep']} |

# **Planet Potency & Nature:**

# | Planet | Potency | Nature |
# |---|---|---|
# | ♃ Jupiter | 3.0 | +Benefic |
# | ♀ Venus | 2.0 | +Benefic |
# | ☉ Sun | 1.5 | +Benefic |
# | ☽ Moon | 1.0 | +Benefic |
# | ♆ Neptune | 0.5 | +Benefic |
# | ♄ Saturn | 2.5 | −Malefic |
# | ♂ Mars | 2.0 | −Malefic |
# | ♇ Pluto | 1.5 | −Malefic |
# | ♅ Uranus | 0.5 | −Malefic |
# | ☿ Mercury | 0.5 | Neutral |
# | ☊ Nodes | 0.5 | +Benefic |
# | ↑ ASC / MC | 1.5 | Neutral (angle) |

# **Aspect Base Scores:** Trine +2.0 · Sextile +1.5 · Conjunction ±1.0 · Opposition −1.5 · Square −1.8

# **Nature Modulation (NATURE_MOD = {NATURE_MOD}):**
# - Two benefics making a trine → score amplified by {NATURE_MOD*100:.0f}%
# - Two malefics making a square → tense score amplified by {NATURE_MOD*100:.0f}%
# - Mixed pair → no amplification, base score unchanged

# **Natal Angles (ASC / MC):**
# - Treated as neutral natal points with potency 1.5
# - Their conjunction score direction is determined solely by the transiting planet's nature
#   (e.g. Jupiter conjunct ASC → positive; Saturn conjunct ASC → negative)
# - Harmonious aspects (trine, sextile) to angles score positively; tense aspects (square, opposition) score negatively
# - Leave sign selector as '—' in the sidebar to exclude angles from scoring{angle_note}

# **Active filters:**
# - Natal planets: **{natal_planet_label}**
# - Transit planets: **{transit_planet_label}**

# **How to read:**
# - **Chart 1 (Natal)** — How current transits interact with your specific birth planets (and angles if set).
#   Select fewer natal planets to isolate specific themes (e.g. only Jupiter + Saturn for luck/restriction cycle).
# - **Chart 2 (Transit)** — General planetary weather affecting everyone.
# - **Chart 3 (Cumulative)** — Running total. Rising slope = improving conditions; falling = accumulating challenges.
# - **Orb settings:** Apply={orb_apply}°  Sep={orb_sep}° — Applying aspects score at full weight; separating at {PHASE_FACTOR['sep']:.0%}.
#     """)


    




# import warnings, datetime, itertools
# warnings.filterwarnings('ignore')

# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import matplotlib.patches as mpatches
# from matplotlib.lines import Line2D
# import streamlit as st

# # ============================================================
# #  PAGE CONFIG
# # ============================================================

# st.set_page_config(
#     page_title="🌙 Personal Aspect Scorer",
#     page_icon="🌙",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# st.markdown("""
# <style>
#   .stApp { background-color: #0A0A1A; color: #E8E8F4; }
#   section[data-testid="stSidebar"] { background-color: #0D0D28; }
#   section[data-testid="stSidebar"] * { color: #E8E8F4 !important; }
#   .stTextInput > div > div > input,
#   .stNumberInput > div > div > input,
#   .stDateInput > div > div > input {
#       background-color: #1A1A38; color: #E8E8F4;
#       border: 1px solid #2A2A4A;
#   }
#   .stSlider > div { color: #E8E8F4; }
#   .stButton > button {
#       background-color: #C8A84B; color: #0A0A1A;
#       font-weight: bold; border: none; border-radius: 4px;
#       padding: 0.5rem 2rem; width: 100%;
#   }
#   .stButton > button:hover { background-color: #E8C86B; }
#   h1, h2, h3 { color: #C8A84B !important; }
#   div[data-testid="stMetric"] {
#       background-color: #0D0D28; border: 1px solid #2A2A4A;
#       border-radius: 6px; padding: 0.5rem 1rem;
#   }
#   div[data-testid="stMetric"] label { color: #C8A84B !important; }
#   .stTabs [data-baseweb="tab"] { color: #E8E8F4; }
#   .stTabs [aria-selected="true"] { color: #C8A84B !important; border-bottom-color: #C8A84B !important; }
#   .stCheckbox > label { color: #E8E8F4 !important; }
# </style>
# """, unsafe_allow_html=True)

# # ============================================================
# #  EPHEMERIS — loaded from GitHub (cached)
# # ============================================================

# GITHUB_EPH_URL = (
#     "https://raw.githubusercontent.com/"
#     "goncuahm/kozmikharita/main/planet_degrees.csv"
# )

# @st.cache_data(show_spinner="Loading ephemeris from GitHub …")
# def load_ephemeris(url):
#     eph_raw = pd.read_csv(url, index_col='date', parse_dates=True)
#     return eph_raw

# # ============================================================
# #  CONSTANTS
# # ============================================================

# EPH_PLANET_COLS = [
#     'sun', 'moon', 'mercury', 'venus', 'mars',
#     'jupiter', 'saturn', 'uranus', 'neptune',
#     'pluto', 'true_node', 'mean_node',
# ]

# ASPECTS   = [0, 60, 90, 120, 180]
# ASP_NAMES = {0:'Conj', 60:'Sext', 90:'Sqr', 120:'Trine', 180:'Opp'}
# SIGNS     = ['Aries','Taurus','Gemini','Cancer','Leo','Virgo',
#              'Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces']

# # ── SCORING: consistent with trading app ────────────────────

# # Planet POTENCY: always positive — how strongly the planet expresses any aspect.
# PLANET_POTENCY = {
#     'jupiter':   3.0,
#     'venus':     2.0,
#     'sun':       1.5,
#     'moon':      1.0,
#     'mars':      2.0,
#     'saturn':    2.5,
#     'pluto':     1.5,
#     'mercury':   0.5,
#     'neptune':   0.5,
#     'uranus':    0.5,
#     'true_node': 0.5,
#     'mean_node': 0.5,
# }

# # Planet NATURE: +1 = benefic, -1 = malefic, 0 = neutral
# PLANET_NATURE = {
#     'jupiter':   +1,
#     'venus':     +1,
#     'sun':       +1,
#     'moon':      +1,
#     'neptune':   +1,
#     'true_node': +1,
#     'mean_node': +1,
#     'mercury':    0,
#     'mars':      -1,
#     'saturn':    -1,
#     'uranus':    -1,
#     'pluto':     -1,
# }

# # Base aspect polarity: positive = harmonious, negative = tense
# # Conjunction = 1.0 (neutral strength — direction decided entirely by planet nature)
# ASPECT_BASE = {
#     0:   +1.0,
#     60:  +1.5,
#     90:  -1.8,
#     120: +2.0,
#     180: -1.5,
# }

# # Nature modulation factor
# NATURE_MOD = 0.35

# PHASE_FACTOR = {'apply': 1.0, 'sep': 0.6}

# # Chart palette
# BG     = '#0A0A1A';  PANEL  = '#0D0D28';  GOLD   = '#C8A84B'
# TEAL   = '#00D4B4';  WHITE  = '#E8E8F4';  GREY   = '#2A2A4A'
# GREEN  = '#44DD88';  RED    = '#E84040';  ORANGE = '#FF8844'
# PURPLE = '#CC44FF';  BLUE   = '#4488FF'

# # Planet display symbols for sidebar
# PLANET_SYMBOLS = {
#     'sun':       '☉ Sun',
#     'moon':      '☽ Moon',
#     'mercury':   '☿ Mercury',
#     'venus':     '♀ Venus',
#     'mars':      '♂ Mars',
#     'jupiter':   '♃ Jupiter',
#     'saturn':    '♄ Saturn',
#     'uranus':    '♅ Uranus',
#     'neptune':   '♆ Neptune',
#     'pluto':     '♇ Pluto',
#     'true_node': '☊ True Node',
#     'mean_node': '☊ Mean Node',
# }

# # ============================================================
# #  SIDEBAR
# # ============================================================

# with st.sidebar:
#     st.markdown("## 🌙 Personal Aspect Scorer")
#     st.markdown("---")

#     st.markdown("### 🎂 Birth Date")
#     birth_date_input = st.text_input(
#         "Birth date (YYYY-MM-DD)",
#         value="1979-12-04",
#         help="Enter the person's date of birth.")

#     person_name = st.text_input(
#         "Name / label (optional)",
#         value="",
#         help="Shown in chart titles. Leave blank to use birth date.")

#     st.markdown("### 📅 Display Range")
#     display_start = st.text_input(
#         "Start date (YYYY-MM-DD)",
#         value=(datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
#         help="Beginning of the score plot window.")

#     display_end = st.text_input(
#         "End date / forecast to (YYYY-MM-DD)",
#         value=(datetime.date.today() + datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
#         help="End of the score plot window. Can be in the future.")

#     st.markdown("### 🔭 Orb Settings")
#     orb_apply = st.slider(
#         "Applying orb (degrees)", min_value=1, max_value=6, value=4,
#         help="Degrees before exact aspect to start scoring.")
#     orb_sep = st.slider(
#         "Separating orb (degrees)", min_value=1, max_value=3, value=1,
#         help="Degrees after exact aspect to stop scoring.")

#     st.markdown("### 📋 Table Horizon")
#     table_past_days = st.slider(
#         "Days back in tables", min_value=3, max_value=30, value=7)
#     table_future_days = st.slider(
#         "Days ahead in tables", min_value=7, max_value=90, value=30)

#     st.markdown("### 📊 Chart Options")
#     smooth_window = st.slider(
#         "Smoothing window (days)", min_value=1, max_value=30, value=7,
#         help="Rolling mean window for the smoothed overlay line.")

#     # ── NATAL PLANET SELECTOR ─────────────────────────────────
#     st.markdown("### 🌟 Natal Planets to Score")
#     st.caption(
#         "Select which natal planets transits should be scored against. "
#         "All transit × transit aspects are always computed regardless.")

#     # We need ephemeris to show positions, so we use a placeholder until loaded.
#     # Store selections in session state.
#     if 'natal_planet_selection' not in st.session_state:
#         st.session_state['natal_planet_selection'] = {p: True for p in EPH_PLANET_COLS}

#     natal_planet_checkboxes = {}
#     for p in EPH_PLANET_COLS:
#         label = PLANET_SYMBOLS.get(p, p.replace('_', ' ').title())
#         natal_planet_checkboxes[p] = st.checkbox(
#             label,
#             value=st.session_state['natal_planet_selection'].get(p, True),
#             key=f"natal_cb_{p}")

#     # Update session state
#     for p in EPH_PLANET_COLS:
#         st.session_state['natal_planet_selection'][p] = natal_planet_checkboxes[p]

#     selected_natal_planets = [p for p in EPH_PLANET_COLS if natal_planet_checkboxes[p]]

#     # ── TRANSIT PLANET SELECTOR ───────────────────────────────
#     st.markdown("### 🔄 Transit Planets to Use")
#     st.caption(
#         "Select which transiting planets are included in aspect calculations. "
#         "Affects both natal and transit × transit scoring.")

#     if 'transit_planet_selection' not in st.session_state:
#         st.session_state['transit_planet_selection'] = {p: True for p in EPH_PLANET_COLS}

#     transit_planet_checkboxes = {}
#     for p in EPH_PLANET_COLS:
#         label = PLANET_SYMBOLS.get(p, p.replace('_', ' ').title())
#         transit_planet_checkboxes[p] = st.checkbox(
#             label,
#             value=st.session_state['transit_planet_selection'].get(p, True),
#             key=f"transit_cb_{p}")

#     for p in EPH_PLANET_COLS:
#         st.session_state['transit_planet_selection'][p] = transit_planet_checkboxes[p]

#     selected_transit_planets = [p for p in EPH_PLANET_COLS if transit_planet_checkboxes[p]]

#     st.markdown("---")
#     run_btn = st.button("▶  Run Analysis", type="primary")

# # ============================================================
# #  HEADER
# # ============================================================

# st.markdown("# 🌙 Personal Natal Aspect Scorer")
# st.markdown(
#     "Scores daily planetary aspects relative to a person's natal chart. "
#     "No stock prices — purely astrological energy over time. "
#     "**Positive = harmonious / supportive conditions. Negative = challenging conditions.**"
# )

# if not run_btn:
#     st.info("👈 Enter birth date and date range in the sidebar, then click **▶ Run Analysis**.")
#     st.stop()

# # ============================================================
# #  VALIDATE INPUTS
# # ============================================================

# try:
#     birth_ts = pd.Timestamp(birth_date_input)
# except Exception:
#     st.error("Invalid birth date format. Use YYYY-MM-DD.")
#     st.stop()

# try:
#     start_ts = pd.Timestamp(display_start)
#     end_ts   = pd.Timestamp(display_end)
# except Exception:
#     st.error("Invalid date range. Use YYYY-MM-DD for both start and end.")
#     st.stop()

# if end_ts <= start_ts:
#     st.error("End date must be after start date.")
#     st.stop()

# if not selected_natal_planets:
#     st.error("Please select at least one natal planet in the sidebar.")
#     st.stop()

# if not selected_transit_planets:
#     st.error("Please select at least one transit planet in the sidebar.")
#     st.stop()

# label = person_name.strip() if person_name.strip() else birth_date_input
# today_ts = pd.Timestamp(datetime.date.today())

# # ============================================================
# #  LOAD EPHEMERIS
# # ============================================================

# with st.spinner("Loading ephemeris …"):
#     try:
#         eph_raw = load_ephemeris(GITHUB_EPH_URL)
#         avail_planets = [p for p in EPH_PLANET_COLS if p in eph_raw.columns]
#         eph = eph_raw[avail_planets].copy()
#         st.success(
#             f"Ephemeris loaded: {len(eph):,} days  "
#             f"({eph.index[0].date()} → {eph.index[-1].date()})"
#         )
#     except Exception as e:
#         st.error(
#             f"Failed to load ephemeris from GitHub.\n\n"
#             f"**Update `GITHUB_EPH_URL`** at the top of this script "
#             f"with your actual raw GitHub URL.\n\nError: {e}"
#         )
#         st.stop()

# # Check date coverage
# if start_ts < eph.index[0]:
#     start_ts = eph.index[0]
#     st.warning(f"Start date adjusted to ephemeris start: {start_ts.date()}")
# if end_ts > eph.index[-1]:
#     end_ts = eph.index[-1]
#     st.warning(f"End date adjusted to ephemeris end: {end_ts.date()}")

# # Filter selected natal planets to only those available in ephemeris
# selected_natal_planets = [p for p in selected_natal_planets if p in avail_planets]
# if not selected_natal_planets:
#     st.error("None of the selected natal planets are in the loaded ephemeris.")
#     st.stop()

# # Filter selected transit planets to only those available in ephemeris
# selected_transit_planets = [p for p in selected_transit_planets if p in avail_planets]
# if not selected_transit_planets:
#     st.error("None of the selected transit planets are in the loaded ephemeris.")
#     st.stop()

# # ============================================================
# #  NATAL CHART
# # ============================================================

# if birth_ts not in eph.index:
#     idx      = eph.index.get_indexer([birth_ts], method='nearest')[0]
#     birth_ts = eph.index[idx]

# natal_row = eph.loc[birth_ts]
# natal = {p: float(natal_row[p]) % 360 for p in avail_planets}

# with st.expander(f"🌟 Natal Chart — {label} ({birth_date_input})", expanded=True):
#     natal_data = []
#     for p, lon in natal.items():
#         sign    = SIGNS[int(lon // 30)]
#         deg_in  = int(lon % 30)
#         mins    = int((lon % 1) * 60)
#         is_selected = p in selected_natal_planets
#         natal_data.append({
#             'Selected':  '✓' if is_selected else '',
#             'Planet':    PLANET_SYMBOLS.get(p, p.replace('_',' ').title()),
#             'Longitude': f"{lon:.3f}°",
#             'Sign':      sign,
#             'Position':  f"{deg_in:02d}° {mins:02d}′ {sign}",
#             'Potency':   PLANET_POTENCY.get(p, 0.5),
#             'Nature':    '+Benefic' if PLANET_NATURE.get(p, 0) > 0
#                          else ('−Malefic' if PLANET_NATURE.get(p, 0) < 0
#                                else 'Neutral'),
#         })
#     natal_df = pd.DataFrame(natal_data)
#     st.dataframe(natal_df, use_container_width=True, hide_index=True)

#     if len(selected_natal_planets) < len(avail_planets) or len(selected_transit_planets) < len(avail_planets):
#         st.info(
#             f"Natal planets scored: **{', '.join(PLANET_SYMBOLS.get(p, p) for p in selected_natal_planets)}**  \n"
#             f"Transit planets used: **{', '.join(PLANET_SYMBOLS.get(p, p) for p in selected_transit_planets)}**")

# # ============================================================
# #  CORE HELPERS
# # ============================================================

# def angular_diff(lon_a, lon_b):
#     d = (lon_a - lon_b) % 360
#     return np.where(d > 180, d - 360, d)

# def orb_factor(abs_gap, orb_max):
#     return np.clip(1.0 - abs_gap / orb_max, 0.0, 1.0)

# def aspect_score_single(pot_a, pot_b, asp, orb_f, phase, nat_a=0, nat_b=0):
#     """
#     Consistent with trading app scoring:
#       pot_a / pot_b : planet potency (always > 0)
#       nat_a / nat_b : planet nature  (+1 benefic, -1 malefic, 0 neutral)
#       asp           : aspect angle (0, 60, 90, 120, 180)
#       orb_f         : orb proximity factor 0→1
#       phase         : 'apply' or 'sep'

#     Conjunction (asp=0):
#       - Direction = sign of (nat_a + nat_b)
#       - Magnitude scales with how strongly both planets share the same nature
#     Other aspects:
#       - Base score from ASPECT_BASE
#       - Nature modulates: same-nature pairs amplify the natural expression
#       - avg_nat > 0 (benefic pair)  → harmonious aspects grow, tense aspects shrink
#       - avg_nat < 0 (malefic pair)  → tense aspects grow, harmonious aspects shrink
#     """
#     avg_pot = (pot_a + pot_b) / 2.0
#     avg_nat = (nat_a + nat_b) / 2.0

#     if asp == 0:
#         net_nature = nat_a + nat_b
#         if net_nature == 0:
#             return 0.0
#         direction = float(np.sign(net_nature))
#         magnitude = avg_pot * abs(net_nature) / 2.0
#         return direction * magnitude * ASPECT_BASE[0] * orb_f * PHASE_FACTOR[phase]
#     else:
#         base      = ASPECT_BASE[asp]
#         modulated = base * (1.0 + NATURE_MOD * avg_nat * float(np.sign(base)))
#         return modulated * avg_pot * orb_f * PHASE_FACTOR[phase]

# def compute_natal_score(date_index):
#     """
#     Scores SELECTED transit planets against SELECTED natal planets only.
#     """
#     eph_a  = eph.reindex(date_index, method='ffill')
#     n      = len(date_index)
#     scores = np.zeros(n)
#     detail = []
#     for tp in selected_transit_planets:          # ← filtered
#         if tp not in eph_a.columns: continue
#         t_lons  = eph_a[tp].values.astype(float) % 360
#         motion  = np.gradient(np.unwrap(t_lons, period=360))
#         pot_t   = PLANET_POTENCY.get(tp, 0.5)
#         nat_t   = PLANET_NATURE.get(tp, 0)
#         # Only score against selected natal planets
#         for np_ in selected_natal_planets:
#             n_lon = natal[np_]
#             pot_n = PLANET_POTENCY.get(np_, 0.5)
#             nat_n = PLANET_NATURE.get(np_, 0)
#             for asp in ASPECTS:
#                 target  = (n_lon + asp) % 360
#                 gap     = angular_diff(t_lons, target)
#                 abs_gap = np.abs(gap)
#                 applying = ((motion > 0) & (gap < 0)) | ((motion < 0) & (gap > 0))
#                 mask_a   = applying    & (abs_gap <= orb_apply)
#                 mask_s   = (~applying) & (abs_gap <= orb_sep)
#                 for i in np.where(mask_a)[0]:
#                     of = float(orb_factor(abs_gap[i], orb_apply))
#                     sc = aspect_score_single(pot_t, pot_n, asp, of, 'apply', nat_t, nat_n)
#                     scores[i] += sc
#                     detail.append({'date': date_index[i], 'transit': tp,
#                                    'natal': np_, 'aspect': ASP_NAMES[asp],
#                                    'phase': 'Applying',
#                                    'orb': round(float(abs_gap[i]), 3),
#                                    'score': round(sc, 4)})
#                 for i in np.where(mask_s)[0]:
#                     of = float(orb_factor(abs_gap[i], orb_sep))
#                     sc = aspect_score_single(pot_t, pot_n, asp, of, 'sep', nat_t, nat_n)
#                     scores[i] += sc
#                     detail.append({'date': date_index[i], 'transit': tp,
#                                    'natal': np_, 'aspect': ASP_NAMES[asp],
#                                    'phase': 'Separating',
#                                    'orb': round(float(abs_gap[i]), 3),
#                                    'score': round(sc, 4)})
#     return pd.Series(scores, index=date_index), detail

# def compute_transit_score(date_index):
#     eph_a  = eph.reindex(date_index, method='ffill')
#     n      = len(date_index)
#     scores = np.zeros(n)
#     detail = []
#     pairs  = list(itertools.combinations(selected_transit_planets, 2))  # ← filtered
#     for (pA, pB) in pairs:
#         if pA not in eph_a.columns or pB not in eph_a.columns: continue
#         lon_A   = eph_a[pA].values.astype(float) % 360
#         lon_B   = eph_a[pB].values.astype(float) % 360
#         motion  = np.gradient(np.unwrap(lon_A, period=360))
#         pot_A   = PLANET_POTENCY.get(pA, 0.5)
#         pot_B   = PLANET_POTENCY.get(pB, 0.5)
#         nat_A   = PLANET_NATURE.get(pA, 0)
#         nat_B   = PLANET_NATURE.get(pB, 0)
#         for asp in ASPECTS:
#             target  = (lon_B + asp) % 360
#             gap     = angular_diff(lon_A, target)
#             abs_gap = np.abs(gap)
#             applying = ((motion > 0) & (gap < 0)) | ((motion < 0) & (gap > 0))
#             mask_a   = applying    & (abs_gap <= orb_apply)
#             mask_s   = (~applying) & (abs_gap <= orb_sep)
#             for i in np.where(mask_a)[0]:
#                 of = float(orb_factor(abs_gap[i], orb_apply))
#                 sc = aspect_score_single(pot_A, pot_B, asp, of, 'apply', nat_A, nat_B)
#                 scores[i] += sc
#                 detail.append({'date': date_index[i], 'planet_a': pA,
#                                'planet_b': pB, 'aspect': ASP_NAMES[asp],
#                                'phase': 'Applying',
#                                'orb': round(float(abs_gap[i]), 3),
#                                'score': round(sc, 4)})
#             for i in np.where(mask_s)[0]:
#                 of = float(orb_factor(abs_gap[i], orb_sep))
#                 sc = aspect_score_single(pot_A, pot_B, asp, of, 'sep', nat_A, nat_B)
#                 scores[i] += sc
#                 detail.append({'date': date_index[i], 'planet_a': pA,
#                                'planet_b': pB, 'aspect': ASP_NAMES[asp],
#                                'phase': 'Separating',
#                                'orb': round(float(abs_gap[i]), 3),
#                                'score': round(sc, 4)})
#     return pd.Series(scores, index=date_index), detail

# # ============================================================
# #  BUILD DATE INDEX & COMPUTE SCORES
# # ============================================================

# full_index = pd.date_range(start=start_ts, end=end_ts, freq='D')

# table_end_ts = today_ts + pd.Timedelta(days=table_future_days + 3)
# if table_end_ts > end_ts:
#     table_ext = pd.date_range(
#         start = end_ts + pd.Timedelta(days=1),
#         end   = table_end_ts, freq='D')
#     compute_index = full_index.append(table_ext)
# else:
#     compute_index = full_index

# with st.spinner("Computing natal aspect scores …"):
#     natal_scores_full, natal_detail_full = compute_natal_score(compute_index)

# with st.spinner("Computing transit aspect scores …"):
#     transit_scores_full, transit_detail_full = compute_transit_score(compute_index)

# # Slice to display window
# natal_scores   = natal_scores_full.reindex(full_index).fillna(0)
# transit_scores = transit_scores_full.reindex(full_index).fillna(0)
# combined_scores = natal_scores.add(transit_scores, fill_value=0)

# # Split history / forecast
# hist_mask = full_index <= today_ts
# fore_mask = full_index >  today_ts

# dates_hist = full_index[hist_mask]
# dates_fore = full_index[fore_mask]

# natal_hist   = natal_scores[hist_mask]
# natal_fore   = natal_scores[fore_mask]
# transit_hist = transit_scores[hist_mask]
# transit_fore = transit_scores[fore_mask]
# combined_hist = combined_scores[hist_mask]
# combined_fore = combined_scores[fore_mask]

# # Summary metrics
# today_natal    = float(natal_scores.get(today_ts, natal_scores.iloc[len(dates_hist)-1]))
# today_transit  = float(transit_scores.get(today_ts, transit_scores.iloc[len(dates_hist)-1]))
# today_combined = today_natal + today_transit

# natal_planet_label = (
#     "All planets" if len(selected_natal_planets) == len(avail_planets)
#     else ", ".join(PLANET_SYMBOLS.get(p, p) for p in selected_natal_planets))

# transit_planet_label = (
#     "All planets" if len(selected_transit_planets) == len(avail_planets)
#     else ", ".join(PLANET_SYMBOLS.get(p, p) for p in selected_transit_planets))

# st.markdown("---")
# c1, c2, c3, c4 = st.columns(4)
# c1.metric("Today's Natal Score",    f"{today_natal:.2f}",
#           delta="▲ Positive" if today_natal > 0 else "▼ Negative")
# c2.metric("Today's Transit Score",  f"{today_transit:.2f}",
#           delta="▲ Positive" if today_transit > 0 else "▼ Negative")
# c3.metric("Today's Combined Score", f"{today_combined:.2f}",
#           delta="▲ Positive" if today_combined > 0 else "▼ Negative")
# c4.metric("Planet filters",
#           f"N:{len(selected_natal_planets)} / T:{len(selected_transit_planets)}",
#           delta=f"of {len(avail_planets)} available")

# # ============================================================
# #  PLOT HELPERS
# # ============================================================

# def style_ax(ax):
#     ax.set_facecolor(PANEL)
#     for sp in ax.spines.values(): sp.set_color(GREY)
#     ax.tick_params(colors=WHITE, labelsize=8)
#     ax.yaxis.label.set_color(WHITE)

# def draw_zero(ax):
#     ax.axhline(0, color=GREY, lw=1.0, zorder=2)

# def draw_today_line(ax):
#     if start_ts <= today_ts <= end_ts:
#         ax.axvline(today_ts, color=GOLD, lw=1.8, ls='--', alpha=0.9, zorder=6)
#         ylims = ax.get_ylim()
#         ax.text(today_ts, ylims[1],
#                 ' Today', color=GOLD, fontsize=8,
#                 va='top', ha='left', fontweight='bold')

# def format_xaxis(ax, n_days):
#     if n_days <= 90:
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#         ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
#     elif n_days <= 400:
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#         ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
#     else:
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#         ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
#     ax.set_xlim(start_ts, end_ts + pd.Timedelta(days=2))

# def draw_score_bars(ax, dates_h, scores_h, dates_f, scores_f, alpha=0.75):
#     if len(dates_h):
#         h_vals   = scores_h.values
#         h_colors = [GREEN if v >= 0 else RED for v in h_vals]
#         ax.bar(dates_h, h_vals, color=h_colors, alpha=alpha, width=1.0, zorder=3)
#     if len(dates_f):
#         f_vals   = scores_f.values
#         f_colors = [GREEN if v >= 0 else RED for v in f_vals]
#         ax.bar(dates_f, f_vals, color=f_colors, alpha=alpha, width=1.0, zorder=3)

# def draw_shading(ax, dates_h, scores_h, dates_f, scores_f, alpha_cap, denom):
#     for d, s in zip(dates_h, scores_h.values):
#         if s == 0: continue
#         ax.axvspan(d, d + pd.Timedelta(days=1),
#                    alpha=min(alpha_cap, abs(s)/denom),
#                    color=GREEN if s > 0 else RED, zorder=1)
#     for d, s in zip(dates_f, scores_f.values):
#         if s == 0: continue
#         ax.axvspan(d, d + pd.Timedelta(days=1),
#                    alpha=min(alpha_cap, abs(s)/denom),
#                    color=GREEN if s > 0 else RED, zorder=1)

# def draw_smooth_line(ax, dates_h, scores_h, dates_f, scores_f, color, lw=1.8):
#     combined = pd.concat([scores_h, scores_f])
#     sm = combined.rolling(window=smooth_window, center=True, min_periods=1).mean()
#     if len(dates_h):
#         ax.plot(dates_h, sm.reindex(dates_h).values,
#                 color=color, lw=lw, zorder=5, label=f'{smooth_window}d smoothed')
#     if len(dates_f):
#         ax.plot(dates_f, sm.reindex(dates_f).values,
#                 color=color, lw=lw, ls='--', zorder=5, alpha=0.8)

# n_days = len(full_index)

# # ============================================================
# #  CHART 1: NATAL SCORE
# # ============================================================

# st.markdown("---")
# st.markdown("## Chart 1 — Natal Aspect Score")
# st.caption(
#     f"Transit planets: **{transit_planet_label}** aspecting natal planets: **{natal_planet_label}**. "
#     "Green = supportive. Red = challenging.")

# fig1, ax1 = plt.subplots(figsize=(18, 5), facecolor=BG)
# style_ax(ax1)
# ax1.set_ylabel('Natal Score', fontsize=10)
# draw_zero(ax1)
# draw_shading(ax1, dates_hist, natal_hist, dates_fore, natal_fore, 0.18, 15)
# draw_score_bars(ax1, dates_hist, natal_hist, dates_fore, natal_fore)
# draw_smooth_line(ax1, dates_hist, natal_hist, dates_fore, natal_fore, GOLD)
# if len(dates_hist):
#     ax1.fill_between(dates_hist, natal_hist.values, 0,
#                      where=(natal_hist.values >= 0), color=GREEN, alpha=0.08, zorder=1)
#     ax1.fill_between(dates_hist, natal_hist.values, 0,
#                      where=(natal_hist.values < 0),  color=RED,   alpha=0.08, zorder=1)
# if len(dates_fore):
#     ax1.fill_between(dates_fore, natal_fore.values, 0,
#                      where=(natal_fore.values >= 0), color=GREEN, alpha=0.08, zorder=1)
#     ax1.fill_between(dates_fore, natal_fore.values, 0,
#                      where=(natal_fore.values < 0),  color=RED,   alpha=0.08, zorder=1)
# draw_today_line(ax1)
# format_xaxis(ax1, n_days)
# ax1.legend(fontsize=8, facecolor='#1A1A38', labelcolor=WHITE, loc='upper left')
# fig1.suptitle(
#     f"{label}  |  Natal Aspect Score\n"
#     f"Transit: {transit_planet_label}  →  Natal: {natal_planet_label}\n"
#     f"Birth: {birth_date_input}  |  Apply≤{orb_apply}°  Sep≤{orb_sep}°  |  "
#     f"Green=Supportive  Red=Challenging  |  Gold dashed = Today",
#     color=GOLD, fontsize=10, fontweight='bold')
# fig1.tight_layout()
# st.pyplot(fig1, use_container_width=True)
# plt.close(fig1)

# # ============================================================
# #  CHART 2: TRANSIT × TRANSIT SCORE
# # ============================================================

# st.markdown("---")
# st.markdown("## Chart 2 — Transit × Transit Aspect Score")
# st.caption(f"Transit pairs from: **{transit_planet_label}**. Independent of natal chart.")

# fig2, ax2 = plt.subplots(figsize=(18, 5), facecolor=BG)
# style_ax(ax2)
# ax2.set_ylabel('Transit Score', fontsize=10)
# draw_zero(ax2)
# draw_shading(ax2, dates_hist, transit_hist, dates_fore, transit_fore, 0.18, 25)
# draw_score_bars(ax2, dates_hist, transit_hist, dates_fore, transit_fore)
# draw_smooth_line(ax2, dates_hist, transit_hist, dates_fore, transit_fore, GOLD)
# if len(dates_hist):
#     ax2.fill_between(dates_hist, transit_hist.values, 0,
#                      where=(transit_hist.values >= 0), color=GREEN, alpha=0.08, zorder=1)
#     ax2.fill_between(dates_hist, transit_hist.values, 0,
#                      where=(transit_hist.values < 0),  color=RED,   alpha=0.08, zorder=1)
# if len(dates_fore):
#     ax2.fill_between(dates_fore, transit_fore.values, 0,
#                      where=(transit_fore.values >= 0), color=GREEN, alpha=0.08, zorder=1)
#     ax2.fill_between(dates_fore, transit_fore.values, 0,
#                      where=(transit_fore.values < 0),  color=RED,   alpha=0.08, zorder=1)
# draw_today_line(ax2)
# format_xaxis(ax2, n_days)
# ax2.legend(fontsize=8, facecolor='#1A1A38', labelcolor=WHITE, loc='upper left')
# fig2.suptitle(
#     f"{label}  |  Transit × Transit Aspect Score\n"
#     f"Planets: {transit_planet_label}  |  "
#     f"Apply≤{orb_apply}°  Sep≤{orb_sep}°  |  "
#     f"Green=Supportive  Red=Challenging  |  Gold dashed = Today",
#     color=GOLD, fontsize=11, fontweight='bold')
# fig2.tight_layout()
# st.pyplot(fig2, use_container_width=True)
# plt.close(fig2)

# # ============================================================
# #  CHART 3: CUMULATIVE SCORES
# # ============================================================

# st.markdown("---")
# st.markdown("## Chart 3 — Cumulative Aspect Score")
# st.caption(
#     "Running total since display start. Rising = accumulating positive conditions. "
#     "Falling = accumulating challenges. Slope matters more than absolute value.")

# natal_cum    = natal_scores.cumsum()
# transit_cum  = transit_scores.cumsum()
# combined_cum = combined_scores.cumsum()

# natal_cum_h    = natal_cum[hist_mask]
# natal_cum_f    = natal_cum[fore_mask]
# transit_cum_h  = transit_cum[hist_mask]
# transit_cum_f  = transit_cum[fore_mask]
# combined_cum_h = combined_cum[hist_mask]
# combined_cum_f = combined_cum[fore_mask]

# fig3, ax3 = plt.subplots(figsize=(18, 5), facecolor=BG)
# style_ax(ax3)
# ax3.set_ylabel('Cumulative Score', fontsize=10)
# draw_zero(ax3)

# if len(dates_hist):
#     ax3.plot(dates_hist, natal_cum_h.values,
#              color=ORANGE, lw=1.4, ls='--', alpha=0.85, zorder=3, label='Natal cum.')
#     ax3.plot(dates_hist, transit_cum_h.values,
#              color=PURPLE, lw=1.4, ls='--', alpha=0.85, zorder=3, label='Transit cum.')
#     ax3.plot(dates_hist, combined_cum_h.values,
#              color=TEAL, lw=2.4, zorder=4, label='Combined cum.')
#     ax3.fill_between(dates_hist, combined_cum_h.values, 0,
#                      where=(combined_cum_h.values >= 0), color=GREEN, alpha=0.12, zorder=1)
#     ax3.fill_between(dates_hist, combined_cum_h.values, 0,
#                      where=(combined_cum_h.values < 0),  color=RED,   alpha=0.12, zorder=1)

# if len(dates_fore):
#     if len(dates_hist):
#         conn_dates = [dates_hist[-1], dates_fore[0]]
#         ax3.plot(conn_dates, [natal_cum_h.iloc[-1],    natal_cum_f.iloc[0]],
#                  color=ORANGE, lw=1.4, ls='--', alpha=0.5)
#         ax3.plot(conn_dates, [transit_cum_h.iloc[-1],  transit_cum_f.iloc[0]],
#                  color=PURPLE, lw=1.4, ls='--', alpha=0.5)
#         ax3.plot(conn_dates, [combined_cum_h.iloc[-1], combined_cum_f.iloc[0]],
#                  color=TEAL, lw=2.4, alpha=0.6)

#     ax3.plot(dates_fore, natal_cum_f.values,
#              color=ORANGE, lw=1.4, ls=':', alpha=0.75, zorder=3)
#     ax3.plot(dates_fore, transit_cum_f.values,
#              color=PURPLE, lw=1.4, ls=':', alpha=0.75, zorder=3)
#     ax3.plot(dates_fore, combined_cum_f.values,
#              color=TEAL, lw=2.4, ls='--', alpha=0.75, zorder=4)

#     last_val = combined_cum_h.iloc[-1] if len(combined_cum_h) else 0
#     ax3.fill_between(dates_fore, combined_cum_f.values, last_val,
#                      where=(combined_cum_f.values >= last_val),
#                      color=GREEN, alpha=0.10, zorder=1)
#     ax3.fill_between(dates_fore, combined_cum_f.values, last_val,
#                      where=(combined_cum_f.values < last_val),
#                      color=RED, alpha=0.10, zorder=1)

# draw_today_line(ax3)
# format_xaxis(ax3, n_days)
# ax3.legend(fontsize=8, facecolor='#1A1A38', labelcolor=WHITE, loc='upper left')
# fig3.suptitle(
#     f"{label}  |  Cumulative Aspect Score\n"
#     f"Teal=Combined  Orange=Natal  Purple=Transit  |  "
#     f"Solid/Dashed=History  Dotted=Forecast  |  Gold dashed = Today",
#     color=GOLD, fontsize=11, fontweight='bold')
# fig3.tight_layout()
# st.pyplot(fig3, use_container_width=True)
# plt.close(fig3)

# # ============================================================
# #  ASPECT TABLES
# # ============================================================

# table_start_ts = today_ts - pd.Timedelta(days=table_past_days)
# table_end_ts2  = today_ts + pd.Timedelta(days=table_future_days)

# def filter_window(detail_list):
#     rows = []
#     for r in detail_list:
#         d = pd.Timestamp(r['date'])
#         if table_start_ts <= d <= table_end_ts2:
#             r2 = r.copy()
#             r2['date']   = d.date()
#             r2['period'] = 'Past' if d <= today_ts else 'Future'
#             rows.append(r2)
#     if not rows:
#         return pd.DataFrame()
#     return (pd.DataFrame(rows)
#             .sort_values(['date','score'], ascending=[True, False])
#             .reset_index(drop=True))

# natal_win   = filter_window(natal_detail_full)
# transit_win = filter_window(transit_detail_full)

# def score_style(val):
#     if isinstance(val, (int, float)):
#         if val > 0: return 'color: #44DD88; font-weight: bold'
#         if val < 0: return 'color: #E84040; font-weight: bold'
#     return ''

# def build_daily_net(raw_win, period_filter):
#     if raw_win.empty:
#         return pd.DataFrame()
#     grp = (raw_win.groupby(['date', 'period'])
#            .agg(Aspects=('score', 'count'), Net_Score=('score', 'sum'))
#            .reset_index().sort_values('date'))
#     grp = grp[grp['period'] == period_filter].copy()
#     if grp.empty:
#         return pd.DataFrame()
#     grp['Bias'] = grp['Net_Score'].apply(lambda x: '▲ Supportive' if x > 0 else '▼ Challenging')
#     grp['date'] = grp['date'].astype(str)
#     grp.columns = ['Date', 'Period', '# Aspects', 'Net Score', 'Bias']
#     return grp[['Date', '# Aspects', 'Net Score', 'Bias']]

# st.markdown("---")
# st.markdown("## 📋 Aspect Tables")
# st.caption(
#     f"Last **{table_past_days}** days + next **{table_future_days}** days. "
#     "Green = supportive · Red = challenging.")

# tab1, tab2 = st.tabs(["🌟 Natal Aspects (Transit × Natal)", "🔄 Transit × Transit Aspects"])

# # ── TABLE 1: NATAL ASPECTS ────────────────────────────────────
# with tab1:
#     if natal_win.empty:
#         st.info("No natal aspects active in the selected window.")
#     else:
#         def prep_natal(sub):
#             if sub.empty: return sub
#             df = sub.rename(columns={
#                 'date':'Date', 'transit':'Transit Planet',
#                 'natal':'Natal Planet', 'aspect':'Aspect',
#                 'phase':'Phase', 'orb':'Orb°', 'score':'Score'})
#             df['Transit Planet'] = df['Transit Planet'].apply(
#                 lambda x: PLANET_SYMBOLS.get(x, x.replace('_',' ').title()))
#             df['Natal Planet'] = df['Natal Planet'].apply(
#                 lambda x: PLANET_SYMBOLS.get(x, x.replace('_',' ').title()))
#             cols = ['Date','Transit Planet','Natal Planet','Aspect','Phase','Orb°','Score']
#             return df[[c for c in cols if c in df.columns]]

#         st.markdown(f"### ⏪ Past {table_past_days} days")
#         past_n = natal_win[natal_win['period']=='Past']
#         if past_n.empty:
#             st.info("No natal aspects in the past window.")
#         else:
#             st.dataframe(
#                 prep_natal(past_n).style.applymap(score_style, subset=['Score']),
#                 use_container_width=True, hide_index=True)

#         st.markdown(f"### ⏩ Next {table_future_days} days")
#         fut_n = natal_win[natal_win['period']=='Future']
#         if fut_n.empty:
#             st.info("No natal aspects in the forecast window.")
#         else:
#             st.dataframe(
#                 prep_natal(fut_n).style.applymap(score_style, subset=['Score']),
#                 use_container_width=True, hide_index=True)

#         st.markdown("### 📆 Daily Net Natal Score")
#         dn_past = build_daily_net(natal_win, 'Past')
#         dn_fut  = build_daily_net(natal_win, 'Future')
#         dn = pd.concat([dn_past, dn_fut]).reset_index(drop=True) if not dn_past.empty or not dn_fut.empty else pd.DataFrame()
#         if dn.empty:
#             st.info("No daily net data available.")
#         else:
#             st.dataframe(
#                 dn.style.applymap(score_style, subset=['Net Score']),
#                 use_container_width=True, hide_index=True)

# # ── TABLE 2: TRANSIT ASPECTS ──────────────────────────────────
# with tab2:
#     if transit_win.empty:
#         st.info("No transit aspects active in the selected window.")
#     else:
#         def prep_transit(sub):
#             if sub.empty: return sub
#             df = sub.rename(columns={
#                 'date':'Date', 'planet_a':'Planet A', 'planet_b':'Planet B',
#                 'aspect':'Aspect', 'phase':'Phase', 'orb':'Orb°', 'score':'Score'})
#             df['Planet A'] = df['Planet A'].apply(
#                 lambda x: PLANET_SYMBOLS.get(x, x.replace('_',' ').title()))
#             df['Planet B'] = df['Planet B'].apply(
#                 lambda x: PLANET_SYMBOLS.get(x, x.replace('_',' ').title()))
#             cols = ['Date','Planet A','Planet B','Aspect','Phase','Orb°','Score']
#             return df[[c for c in cols if c in df.columns]]

#         st.markdown(f"### ⏪ Past {table_past_days} days")
#         past_t = transit_win[transit_win['period']=='Past']
#         if past_t.empty:
#             st.info("No transit aspects in the past window.")
#         else:
#             st.dataframe(
#                 prep_transit(past_t).style.applymap(score_style, subset=['Score']),
#                 use_container_width=True, hide_index=True)

#         st.markdown(f"### ⏩ Next {table_future_days} days")
#         fut_t = transit_win[transit_win['period']=='Future']
#         if fut_t.empty:
#             st.info("No transit aspects in the forecast window.")
#         else:
#             st.dataframe(
#                 prep_transit(fut_t).style.applymap(score_style, subset=['Score']),
#                 use_container_width=True, hide_index=True)

#         st.markdown("### 📆 Daily Net Transit Score")
#         dt_past = build_daily_net(transit_win, 'Past')
#         dt_fut  = build_daily_net(transit_win, 'Future')
#         dt = pd.concat([dt_past, dt_fut]).reset_index(drop=True) if not dt_past.empty or not dt_fut.empty else pd.DataFrame()
#         if dt.empty:
#             st.info("No daily net data available.")
#         else:
#             st.dataframe(
#                 dt.style.applymap(score_style, subset=['Net Score']),
#                 use_container_width=True, hide_index=True)

# # ============================================================
# #  SCORING LEGEND
# # ============================================================

# st.markdown("---")
# with st.expander("📖 Scoring Methodology", expanded=False):
#     st.markdown(f"""
# **Score = modulated_base × avg_potency × orb_proximity × phase_factor**

# | Component | Rule |
# |---|---|
# | **Conjunction** | Direction = sign of (nat_A + nat_B). Both benefic → +, both malefic → −, mixed → 0 |
# | **Other aspects** | Base score × (1 + {NATURE_MOD} × avg_nature × sign(base)). Same-nature pairs amplify natural expression |
# | **orb_proximity** | Linear: 1.0 at exact → 0.0 at orb edge |
# | **phase_factor** | Applying = {PHASE_FACTOR['apply']} · Separating = {PHASE_FACTOR['sep']} |

# **Planet Potency & Nature:**

# | Planet | Potency | Nature |
# |---|---|---|
# | ♃ Jupiter | 3.0 | +Benefic |
# | ♀ Venus | 2.0 | +Benefic |
# | ☉ Sun | 1.5 | +Benefic |
# | ☽ Moon | 1.0 | +Benefic |
# | ♆ Neptune | 0.5 | +Benefic |
# | ♄ Saturn | 2.5 | −Malefic |
# | ♂ Mars | 2.0 | −Malefic |
# | ♇ Pluto | 1.5 | −Malefic |
# | ♅ Uranus | 0.5 | −Malefic |
# | ☿ Mercury | 0.5 | Neutral |
# | ☊ Nodes | 0.5 | +Benefic |

# **Aspect Base Scores:** Trine +2.0 · Sextile +1.5 · Conjunction ±1.0 · Opposition −1.5 · Square −1.8

# **Nature Modulation (NATURE_MOD = {NATURE_MOD}):**
# - Two benefics making a trine → score amplified by {NATURE_MOD*100:.0f}%
# - Two malefics making a square → tense score amplified by {NATURE_MOD*100:.0f}%
# - Mixed pair → no amplification, base score unchanged

# **Active filters:**
# - Natal planets: **{natal_planet_label}**
# - Transit planets: **{transit_planet_label}**

# **How to read:**
# - **Chart 1 (Natal)** — How current transits interact with your specific birth planets.
#   Select fewer natal planets to isolate specific themes (e.g. only Jupiter + Saturn for luck/restriction cycle).
# - **Chart 2 (Transit)** — General planetary weather affecting everyone.
# - **Chart 3 (Cumulative)** — Running total. Rising slope = improving conditions; falling = accumulating challenges.
# - **Orb settings:** Apply={orb_apply}°  Sep={orb_sep}° — Applying aspects score at full weight; separating at {PHASE_FACTOR['sep']:.0%}.
#     """)
