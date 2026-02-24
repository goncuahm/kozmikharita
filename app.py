# ============================================================
#  PERSONAL NATAL ASPECT SCORER — Streamlit App
#
#  For individuals rather than stocks.
#  No price data — only planetary aspect scores plotted over time.
#
#  Ephemeris loaded from GitHub (planet_degrees.csv)
#  User inputs: birth date, date range to display, forecast days, orbs
#  Outputs:
#    Chart 1 — Natal score (transit vs natal chart) over time
#    Chart 2 — Transit score (transit vs transit) over time
#    Chart 3 — Cumulative scores over time
#    Table 1 — Recent + upcoming natal aspects with scores
#    Table 2 — Recent + upcoming transit aspects with scores
# ============================================================

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
</style>
""", unsafe_allow_html=True)

# ============================================================
#  EPHEMERIS — loaded from GitHub (cached)
# ============================================================

GITHUB_EPH_URL = (
    "https://raw.githubusercontent.com/"
    "YOUR_USERNAME/YOUR_REPO/main/planet_degrees.csv"
    # ↑ Replace with your actual GitHub raw URL
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

PLANET_WEIGHT = {
    'sun':       +1.5,
    'moon':      +1.0,
    'mercury':   +0.5,
    'venus':     +2.0,
    'mars':      -1.5,
    'jupiter':   +3.0,
    'saturn':    -2.5,
    'uranus':    -0.5,
    'neptune':   +0.5,
    'pluto':     -1.0,
    'true_node': +0.5,
    'mean_node': +0.5,
}

ASPECT_MULT = {
    0:   +1.0,
    60:  +1.5,
    90:  -1.8,
    120: +2.0,
    180: -1.5,
}

PHASE_FACTOR = {'apply': 1.0, 'sep': 0.6}

# Chart palette
BG     = '#0A0A1A';  PANEL  = '#0D0D28';  GOLD   = '#C8A84B'
TEAL   = '#00D4B4';  WHITE  = '#E8E8F4';  GREY   = '#2A2A4A'
GREEN  = '#44DD88';  RED    = '#E84040';  ORANGE = '#FF8844'
PURPLE = '#CC44FF';  BLUE   = '#4488FF'

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

# Check date coverage
if start_ts < eph.index[0]:
    start_ts = eph.index[0]
    st.warning(f"Start date adjusted to ephemeris start: {start_ts.date()}")
if end_ts > eph.index[-1]:
    end_ts = eph.index[-1]
    st.warning(f"End date adjusted to ephemeris end: {end_ts.date()}")

# ============================================================
#  NATAL CHART
# ============================================================

if birth_ts not in eph.index:
    idx      = eph.index.get_indexer([birth_ts], method='nearest')[0]
    birth_ts = eph.index[idx]

natal_row = eph.loc[birth_ts]
natal = {p: float(natal_row[p]) % 360 for p in avail_planets}

with st.expander(f"🌟 Natal Chart — {label} ({birth_date_input})", expanded=True):
    natal_data = []
    for p, lon in natal.items():
        sign    = SIGNS[int(lon // 30)]
        deg_in  = int(lon % 30)
        mins    = int((lon % 1) * 60)
        natal_data.append({
            'Planet':    p.replace('_',' ').title(),
            'Longitude': f"{lon:.3f}°",
            'Sign':      sign,
            'Position':  f"{deg_in:02d}° {mins:02d}′ {sign}",
            'Weight':    PLANET_WEIGHT.get(p, 0),
        })
    natal_df = pd.DataFrame(natal_data)
    st.dataframe(natal_df, use_container_width=True, hide_index=True)

# ============================================================
#  CORE HELPERS
# ============================================================

def angular_diff(lon_a, lon_b):
    d = (lon_a - lon_b) % 360
    return np.where(d > 180, d - 360, d)

def orb_factor(abs_gap, orb_max):
    return np.clip(1.0 - abs_gap / orb_max, 0.0, 1.0)

def aspect_score_single(pw_a, pw_b, asp, orb_f, phase):
    magnitude = (abs(pw_a) + abs(pw_b)) / 2.0
    if asp == 0:
        net_polarity = pw_a + pw_b
        direction    = float(np.sign(net_polarity)) if net_polarity != 0 else 1.0
        asp_strength = abs(ASPECT_MULT[0])
    else:
        direction    = float(np.sign(ASPECT_MULT[asp]))
        asp_strength = abs(ASPECT_MULT[asp])
    return direction * magnitude * asp_strength * orb_f * PHASE_FACTOR[phase]

def compute_natal_score(date_index):
    eph_a  = eph.reindex(date_index, method='ffill')
    n      = len(date_index)
    scores = np.zeros(n)
    detail = []
    for tp in avail_planets:
        if tp not in eph_a.columns: continue
        t_lons  = eph_a[tp].values.astype(float) % 360
        motion  = np.gradient(np.unwrap(t_lons, period=360))
        pw_t    = PLANET_WEIGHT.get(tp, 0.0)
        for np_ in avail_planets:
            n_lon = natal[np_]
            pw_n  = PLANET_WEIGHT.get(np_, 0.0)
            for asp in ASPECTS:
                target  = (n_lon + asp) % 360
                gap     = angular_diff(t_lons, target)
                abs_gap = np.abs(gap)
                applying = ((motion > 0) & (gap < 0)) | ((motion < 0) & (gap > 0))
                mask_a   = applying   & (abs_gap <= orb_apply)
                mask_s   = (~applying) & (abs_gap <= orb_sep)
                for i in np.where(mask_a)[0]:
                    of = float(orb_factor(abs_gap[i], orb_apply))
                    sc = aspect_score_single(pw_t, pw_n, asp, of, 'apply')
                    scores[i] += sc
                    detail.append({'date': date_index[i], 'transit': tp,
                                   'natal': np_, 'aspect': ASP_NAMES[asp],
                                   'phase': 'Applying',
                                   'orb': round(float(abs_gap[i]), 3),
                                   'score': round(sc, 4)})
                for i in np.where(mask_s)[0]:
                    of = float(orb_factor(abs_gap[i], orb_sep))
                    sc = aspect_score_single(pw_t, pw_n, asp, of, 'sep')
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
    pairs  = list(itertools.combinations(avail_planets, 2))
    for (pA, pB) in pairs:
        if pA not in eph_a.columns or pB not in eph_a.columns: continue
        lon_A   = eph_a[pA].values.astype(float) % 360
        lon_B   = eph_a[pB].values.astype(float) % 360
        motion  = np.gradient(np.unwrap(lon_A, period=360))
        pw_A    = PLANET_WEIGHT.get(pA, 0.0)
        pw_B    = PLANET_WEIGHT.get(pB, 0.0)
        for asp in ASPECTS:
            target  = (lon_B + asp) % 360
            gap     = angular_diff(lon_A, target)
            abs_gap = np.abs(gap)
            applying = ((motion > 0) & (gap < 0)) | ((motion < 0) & (gap > 0))
            mask_a   = applying    & (abs_gap <= orb_apply)
            mask_s   = (~applying) & (abs_gap <= orb_sep)
            for i in np.where(mask_a)[0]:
                of = float(orb_factor(abs_gap[i], orb_apply))
                sc = aspect_score_single(pw_A, pw_B, asp, of, 'apply')
                scores[i] += sc
                detail.append({'date': date_index[i], 'planet_a': pA,
                               'planet_b': pB, 'aspect': ASP_NAMES[asp],
                               'phase': 'Applying',
                               'orb': round(float(abs_gap[i]), 3),
                               'score': round(sc, 4)})
            for i in np.where(mask_s)[0]:
                of = float(orb_factor(abs_gap[i], orb_sep))
                sc = aspect_score_single(pw_A, pw_B, asp, of, 'sep')
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

# Full daily date index covering the entire display window
full_index = pd.date_range(start=start_ts, end=end_ts, freq='D')

# Also extend slightly for table (in case end_ts < today + table_future_days)
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

# Slice to display window
natal_scores   = natal_scores_full.reindex(full_index).fillna(0)
transit_scores = transit_scores_full.reindex(full_index).fillna(0)
combined_scores = natal_scores.add(transit_scores, fill_value=0)

# Split into history (up to today) and forecast (after today)
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

# ── Summary metrics ───────────────────────────────────────────
today_natal   = float(natal_scores.get(today_ts, natal_scores.iloc[-1 if len(dates_fore)==0 else len(dates_hist)-1]))
today_transit = float(transit_scores.get(today_ts, transit_scores.iloc[-1 if len(dates_fore)==0 else len(dates_hist)-1]))
today_combined = today_natal + today_transit

st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Today's Natal Score",    f"{today_natal:.2f}",
          delta="▲ Positive" if today_natal > 0 else "▼ Negative")
c2.metric("Today's Transit Score",  f"{today_transit:.2f}",
          delta="▲ Positive" if today_transit > 0 else "▼ Negative")
c3.metric("Today's Combined Score", f"{today_combined:.2f}",
          delta="▲ Positive" if today_combined > 0 else "▼ Negative")
c4.metric("Display window",
          f"{len(full_index)} days",
          delta=f"{start_ts.date()} → {end_ts.date()}")

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
    if n_days <= 90:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    elif n_days <= 400:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
    ax.set_xlim(start_ts, end_ts + pd.Timedelta(days=2))

def draw_score_bars(ax, dates_h, scores_h, dates_f, scores_f, alpha=0.75):
    """Draw history bars solid, forecast bars same style (no candles here)."""
    if len(dates_h):
        h_vals   = scores_h.values
        h_colors = [GREEN if v >= 0 else RED for v in h_vals]
        ax.bar(dates_h, h_vals, color=h_colors, alpha=alpha, width=1.0, zorder=3)
    if len(dates_f):
        f_vals   = scores_f.values
        f_colors = [GREEN if v >= 0 else RED for v in f_vals]
        ax.bar(dates_f, f_vals, color=f_colors, alpha=alpha, width=1.0, zorder=3)

def draw_shading(ax, dates_h, scores_h, dates_f, scores_f, alpha_cap, denom):
    """Background shading — same colour logic, variable opacity by score magnitude."""
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
    "Daily score from transit planets aspecting your natal chart. "
    "Green bars = supportive transits. Red bars = challenging transits.")

fig1, ax1 = plt.subplots(figsize=(18, 5), facecolor=BG)
style_ax(ax1)
ax1.set_ylabel('Natal Score', fontsize=10)
draw_zero(ax1)

draw_shading(ax1, dates_hist, natal_hist, dates_fore, natal_fore, 0.18, 15)
draw_score_bars(ax1, dates_hist, natal_hist, dates_fore, natal_fore)
draw_smooth_line(ax1, dates_hist, natal_hist, dates_fore, natal_fore, GOLD)

# Fill above/below zero
if len(dates_hist):
    ax1.fill_between(dates_hist, natal_hist.values, 0,
                     where=(natal_hist.values >= 0),
                     color=GREEN, alpha=0.08, zorder=1)
    ax1.fill_between(dates_hist, natal_hist.values, 0,
                     where=(natal_hist.values < 0),
                     color=RED, alpha=0.08, zorder=1)
if len(dates_fore):
    ax1.fill_between(dates_fore, natal_fore.values, 0,
                     where=(natal_fore.values >= 0),
                     color=GREEN, alpha=0.08, zorder=1)
    ax1.fill_between(dates_fore, natal_fore.values, 0,
                     where=(natal_fore.values < 0),
                     color=RED, alpha=0.08, zorder=1)

draw_today_line(ax1)
format_xaxis(ax1, n_days)

ax1.legend(fontsize=8, facecolor='#1A1A38', labelcolor=WHITE, loc='upper left')
fig1.suptitle(
    f"{label}  |  Natal Aspect Score  (Transit × Natal)\n"
    f"Birth: {birth_date_input}  |  Apply≤{orb_apply}°  Sep≤{orb_sep}°  |  "
    f"Green=Supportive  Red=Challenging  |  Gold dashed = Today",
    color=GOLD, fontsize=11, fontweight='bold')
fig1.tight_layout()
st.pyplot(fig1, use_container_width=True)
plt.close(fig1)

# ============================================================
#  CHART 2: TRANSIT × TRANSIT SCORE
# ============================================================

st.markdown("---")
st.markdown("## Chart 2 — Transit × Transit Aspect Score")
st.caption(
    "Daily score from transit planets aspecting each other. "
    "This reflects the general planetary weather — independent of natal chart.")

fig2, ax2 = plt.subplots(figsize=(18, 5), facecolor=BG)
style_ax(ax2)
ax2.set_ylabel('Transit Score', fontsize=10)
draw_zero(ax2)

draw_shading(ax2, dates_hist, transit_hist, dates_fore, transit_fore, 0.18, 25)
draw_score_bars(ax2, dates_hist, transit_hist, dates_fore, transit_fore)
draw_smooth_line(ax2, dates_hist, transit_hist, dates_fore, transit_fore, GOLD)

if len(dates_hist):
    ax2.fill_between(dates_hist, transit_hist.values, 0,
                     where=(transit_hist.values >= 0),
                     color=GREEN, alpha=0.08, zorder=1)
    ax2.fill_between(dates_hist, transit_hist.values, 0,
                     where=(transit_hist.values < 0),
                     color=RED, alpha=0.08, zorder=1)
if len(dates_fore):
    ax2.fill_between(dates_fore, transit_fore.values, 0,
                     where=(transit_fore.values >= 0),
                     color=GREEN, alpha=0.08, zorder=1)
    ax2.fill_between(dates_fore, transit_fore.values, 0,
                     where=(transit_fore.values < 0),
                     color=RED, alpha=0.08, zorder=1)

draw_today_line(ax2)
format_xaxis(ax2, n_days)

ax2.legend(fontsize=8, facecolor='#1A1A38', labelcolor=WHITE, loc='upper left')
fig2.suptitle(
    f"{label}  |  Transit × Transit Aspect Score  (General Planetary Weather)\n"
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
    "Running total since the display start date. "
    "Rising = accumulating positive conditions. Falling = accumulating challenges. "
    "The absolute value is less meaningful than the direction of the trend.")

# Build cumulative series across the full window (history + forecast seamlessly)
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

# History lines
if len(dates_hist):
    ax3.plot(dates_hist, natal_cum_h.values,
             color=ORANGE, lw=1.4, ls='--', alpha=0.85, zorder=3, label='Natal cum.')
    ax3.plot(dates_hist, transit_cum_h.values,
             color=PURPLE, lw=1.4, ls='--', alpha=0.85, zorder=3, label='Transit cum.')
    ax3.plot(dates_hist, combined_cum_h.values,
             color=TEAL, lw=2.4, zorder=4, label='Combined cum.')
    ax3.fill_between(dates_hist, combined_cum_h.values, 0,
                     where=(combined_cum_h.values >= 0),
                     color=GREEN, alpha=0.12, zorder=1)
    ax3.fill_between(dates_hist, combined_cum_h.values, 0,
                     where=(combined_cum_h.values < 0),
                     color=RED, alpha=0.12, zorder=1)

# Forecast lines (continue seamlessly from history)
if len(dates_fore):
    # Connector dots at today boundary
    if len(dates_hist):
        conn_dates = [dates_hist[-1], dates_fore[0]]
        ax3.plot(conn_dates, [natal_cum_h.iloc[-1],   natal_cum_f.iloc[0]],
                 color=ORANGE, lw=1.4, ls='--', alpha=0.5)
        ax3.plot(conn_dates, [transit_cum_h.iloc[-1], transit_cum_f.iloc[0]],
                 color=PURPLE, lw=1.4, ls='--', alpha=0.5)
        ax3.plot(conn_dates, [combined_cum_h.iloc[-1],combined_cum_f.iloc[0]],
                 color=TEAL, lw=2.4, alpha=0.6)

    ax3.plot(dates_fore, natal_cum_f.values,
             color=ORANGE, lw=1.4, ls=':', alpha=0.75, zorder=3)
    ax3.plot(dates_fore, transit_cum_f.values,
             color=PURPLE, lw=1.4, ls=':', alpha=0.75, zorder=3)
    ax3.plot(dates_fore, combined_cum_f.values,
             color=TEAL, lw=2.4, ls='--', alpha=0.75, zorder=4)

    # Fill forecast zone relative to today's level
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

st.markdown("---")
st.markdown("## 📋 Aspect Tables")
st.caption(
    f"Last **{table_past_days}** days + next **{table_future_days}** days. "
    "Green = supportive · Red = challenging.")

tab1, tab2 = st.tabs(["🌟 Natal Aspects (Transit × Natal)", "🔄 Transit × Transit Aspects"])

# ── TABLE 1: NATAL ASPECTS ────────────────────────────────────
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
            df['Transit Planet'] = df['Transit Planet'].str.replace('_',' ').str.title()
            df['Natal Planet']   = df['Natal Planet'].str.replace('_',' ').str.title()
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

        # Daily net
        st.markdown("### 📆 Daily Net Natal Score")
        dn = (natal_win.groupby(['date','period'])
              .agg(N=('score','count'), Net=('score','sum'))
              .reset_index().sort_values('date'))
        dn['Bias']   = dn['Net'].apply(lambda x: '▲ Supportive' if x > 0 else '▼ Challenging')
        dn['date']   = dn['date'].astype(str)
        dn.columns   = ['Date','Period','# Aspects','Net Score','Bias']
        st.dataframe(
            dn.style.applymap(score_style, subset=['Net Score']),
            use_container_width=True, hide_index=True)

# ── TABLE 2: TRANSIT ASPECTS ──────────────────────────────────
with tab2:
    if transit_win.empty:
        st.info("No transit aspects active in the selected window.")
    else:
        def prep_transit(sub):
            if sub.empty: return sub
            df = sub.rename(columns={
                'date':'Date', 'planet_a':'Planet A', 'planet_b':'Planet B',
                'aspect':'Aspect', 'phase':'Phase', 'orb':'Orb°', 'score':'Score'})
            df['Planet A'] = df['Planet A'].str.replace('_',' ').str.title()
            df['Planet B'] = df['Planet B'].str.replace('_',' ').str.title()
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
        dt = (transit_win.groupby(['date','period'])
              .agg(N=('score','count'), Net=('score','sum'))
              .reset_index().sort_values('date'))
        dt['Bias']   = dt['Net'].apply(lambda x: '▲ Supportive' if x > 0 else '▼ Challenging')
        dt['date']   = dt['date'].astype(str)
        dt.columns   = ['Date','Period','# Aspects','Net Score','Bias']
        st.dataframe(
            dt.style.applymap(score_style, subset=['Net Score']),
            use_container_width=True, hide_index=True)

# ============================================================
#  SCORING LEGEND
# ============================================================

st.markdown("---")
with st.expander("📖 Scoring Methodology", expanded=False):
    st.markdown(f"""
**Score = direction × magnitude × aspect_strength × orb_proximity × phase_factor**

| Component | Rule |
|---|---|
| **direction** | Trine/Sext = +1 (supportive) · Sq/Opp = −1 (challenging) · Conj = sign of planet weight sum |
| **magnitude** | (\\|Planet A weight\\| + \\|Planet B weight\\|) / 2 |
| **aspect_strength** | \\|aspect multiplier\\| |
| **orb_proximity** | Linear: 1.0 at exact → 0.0 at orb edge |
| **phase_factor** | Applying = 1.0 · Separating = {PHASE_FACTOR['sep']} |

**Planet Weights:**

| Supportive | Weight | Challenging | Weight |
|---|---|---|---|
| Jupiter | +3.0 | Saturn | −2.5 |
| Venus | +2.0 | Mars | −1.5 |
| Sun | +1.5 | Pluto | −1.0 |
| Moon | +1.0 | Uranus | −0.5 |
| Neptune | +0.5 | | |
| Mercury | +0.5 | | |
| North Node | +0.5 | | |

**Aspect Multipliers:** Trine +2.0 · Sextile +1.5 · Conjunction ±1.0 · Opposition −1.5 · Square −1.8

**How to read the charts:**
- **Chart 1 (Natal)** — Reflects how current planetary movements interact with *your* unique birth chart.
  High positive score = transiting benefics making harmonious angles to your natal planets.
- **Chart 2 (Transit)** — The general planetary weather affecting everyone.
  Peaks and troughs here are shared across all people born in any era.
- **Chart 3 (Cumulative)** — The running total. A rising teal line means conditions are improving;
  falling means accumulating challenges. The *slope* matters more than the absolute value.
- **Orb settings:** Apply={orb_apply}°  Sep={orb_sep}°  — Applying aspects (planet approaching exact)
  score at full weight; separating aspects score at {PHASE_FACTOR['sep']:.0%} weight.
    """)
