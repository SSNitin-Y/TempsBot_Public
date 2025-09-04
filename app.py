import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
import hashlib, json, re
from streamlit.components.v1 import html as st_html

from weather_api import get_weather
from summarizer import summarize_weather
from wardrobe_advisor import get_outfit_recommendation, add_accessory_tips   # NEW
from forecast_api import get_forecast, get_hourly_today                      # NEW
from outfit_forecast import generate_outfit_recommendations
from gpt_summarizer import (
    generate_gpt_summary,
    generate_outfit_summary,
    generate_daily_outfit_summary,
)
from sunscreen_advisor import get_sunscreen_recommendation
from bs4 import BeautifulSoup

# Optional: color name helper for PDF labels (only used there)
try:
    from color_utils import closest_color_name
except Exception:
    def closest_color_name(x): return x  # fallback

# Optional PDF support (ReportLab). If not installed, we’ll just hide the PDF download.
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor, black
    from reportlab.lib.units import inch
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

from datetime import datetime

# ⭐ NEW: We'll import pandas & date formatter only where needed to minimize global changes
# (kept as-is at top level for your original style)

# --- Plotly (interactive hourly graphs) ---  ✅ NEW
import plotly.graph_objects as go
import numpy as np

# --- UI tweak: keep long select values inside the borders ---
st.markdown("""
<style>
/* Make the BaseWeb Select (used by st.selectbox) not overflow its column */
div[data-baseweb="select"] > div { max-width: 100%; }

/* Truncate the selected value instead of letting it stretch the control */
div[data-baseweb="select"] span {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
""", unsafe_allow_html=True)



def render_alerts(alerts, condensed=False):
    """
    Renders OpenWeather 'alerts' list with expanders and no forced truncation.
    If condensed=True, shows the first paragraph only, with a 'Show full text' expander.
    """
    if not alerts:
        return

    st.markdown("### ⚠️ Weather Alerts")

    for idx, a in enumerate(alerts[:6]):  # show up to 6 alerts
        title = a.get("event", f"Weather alert #{idx+1}")
        sender = a.get("sender_name") or ""
        start_ts = a.get("start")
        end_ts   = a.get("end")
        # Convert epoch -> local human time if present
        def _fmt(ts):
            try:
                return datetime.fromtimestamp(int(ts)).strftime("%a, %b %d • %I:%M %p")
            except Exception:
                return None
        when_bits = []
        s = _fmt(start_ts); e = _fmt(end_ts)
        if s: when_bits.append(f"**Starts:** {s}")
        if e: when_bits.append(f"**Ends:** {e}")
        when_line = "  •  ".join(when_bits)

        desc = (a.get("description") or "").strip()

        # Optional condensed display (first paragraph only)
        first_para, sep, rest = desc.partition("\n\n")
        header_line = f"**{title}**" + (f" — {sender}" if sender else "")
        if when_line:
            header_line += f"\n\n{when_line}"

        with st.expander(header_line, expanded=True):
            if condensed and rest:
                # Show a short preview + expandable full text
                st.markdown(first_para)
                with st.expander("Show full text"):
                    st.markdown(desc.replace("\n", "  \n"))  # preserve newlines
            else:
                # Full text by default
                st.markdown(desc.replace("\n", "  \n"))

            # Optional: quick download for this alert text
            fname = f"alert_{idx+1}_{title.replace(' ', '_')}.txt"
            st.download_button(
                label="Save this alert (.txt)",
                data=desc,
                file_name=fname,
                mime="text/plain",
                key=f"dl_alert_{idx+1}"
            )


# =========================
# CACHING HELPERS
# =========================
@st.cache_data(ttl=600)
def fetch_weather(city, api_key):
    return get_weather(city, api_key)

@st.cache_data(ttl=600)
def fetch_forecast(city, api_key, units="metric"):
    return get_forecast(city, api_key, units=units)

@st.cache_data(ttl=600)
def fetch_hourly(lat, lon, api_key, units="metric"):
    return get_hourly_today(lat, lon, api_key, units=units)

@st.cache_data(ttl=600)
def cached_gpt_summary(payload: dict):
    weather = payload.get("weather", {})
    outfit_text = payload.get("outfit_text", "")
    return {
        "gpt_weather": generate_gpt_summary(weather),
        "gpt_outfit":  generate_outfit_summary(outfit_text),
    }

@st.cache_data(ttl=600)
def cached_daily_tip(date_text: str, outfit_text: str, color_labels: list[str]) -> str:
    return generate_daily_outfit_summary(date_text, outfit_text, color_labels)

# ⭐ NEW: small helper to get city timezone offset (seconds) from OpenWeather
@st.cache_data(ttl=3600)
def get_tz_offset_seconds(lat: float, lon: float, api_key: str) -> int:
    """
    Ask OpenWeather One Call for the timezone_offset (seconds) for a lat/lon.
    Returns 0 on failure (i.e., keeps UTC).
    """
    import requests
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/3.0/onecall",
            params={"lat": lat, "lon": lon, "appid": api_key},
            timeout=10,
        )
        r.raise_for_status()
        j = r.json()
        return int(j.get("timezone_offset", 0) or 0)
    except Exception:
        return 0


@st.cache_data
def build_outfit_plan_pdf_cached(sections, city_name=""):
    if not REPORTLAB_OK:
        return None
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    left, top, y = 50, height - 50, height - 50

    # Header
    c.setFont("Helvetica-Bold", 16)
    title = "5-Day Outfit Plan" + (f" — {city_name}" if city_name else "")
    c.drawString(left, y, title)
    y -= 24
    c.setFont("Helvetica", 10)

    page_num = 1
    def draw_header_footer():
        c.setFont("Helvetica", 8)
        c.setFillColor(black)
        c.drawRightString(8.0*inch - 40, 20, f"Page {page_num}")
        c.drawString(50, 20, "Climbot — weather • outfits • UV")

    def new_page():
        nonlocal y, page_num
        draw_header_footer()
        c.showPage()
        page_num += 1
        y = top
        c.setFont("Helvetica", 10)

    for sec in sections:
        date = sec.get("date", "")
        outfit = sec.get("outfit", "")
        pairs = sec.get("pairs", [])

        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, date)
        y -= 16
        c.setFont("Helvetica", 10)

        if y < 80:
            new_page()
        c.drawString(left, y, f"Outfit: {outfit}")
        y -= 14

        if pairs:
            if y < 100:
                new_page()
            c.drawString(left, y, "Colors:")
            y -= 12
            sw, gap = 18, 6
            for (h1, h2, lbl) in pairs:
                if y < 80:
                    new_page()
                try:
                    col1, col2 = HexColor(h1), HexColor(h2)
                except Exception:
                    col1 = col2 = HexColor("#000000")
                c.setFillColor(col1); c.setStrokeColor(black)
                c.rect(left, y - sw + 2, sw, sw, fill=1, stroke=1)
                c.setFillColor(col2)
                c.rect(left + sw + gap, y - sw + 2, sw, sw, fill=1, stroke=1)
                c.setFillColor(black)
                # names + hex
                try: n1 = closest_color_name(h1)
                except: n1 = h1
                try: n2 = closest_color_name(h2)
                except: n2 = h2
                pretty = f"{n1} ({h1}) & {n2} ({h2})"
                c.drawString(left + (sw * 2) + (gap * 2) + 6, y + 2, pretty)
                y -= (sw + 8)

        y -= 8
        if y < 60:
            new_page()

    draw_header_footer()
    c.save()
    buf.seek(0)
    return buf.getvalue()


def _inputs_key(city, skin_type_number):
    return hashlib.md5(json.dumps({"city": city, "skin": skin_type_number}).encode("utf-8")).hexdigest()

# =========================
# APP UI
# =========================
st.title("Welcome I'm TempsBot")
st.caption("v2.5.0 • Powered by OpenWeatherMap & GPT")

intro = (
    "Hello! I gather weather for the next five days, suggest **what to wear today**, "
    "offer **UV/sunscreen advice** based on your Fitzpatrick skin type, and propose "
    "**color pairs** for stylish photos. Download your plan as Markdown or PDF/PNG."
)
with st.expander("👋 What I do (tap to read)", expanded=False):
    st.write(intro)

api_key = st.secrets["OPENWEATHER_API_KEY"]

# Sidebar simple units setting (optional)
with st.sidebar.expander("⚙️ Settings", expanded=False):
    unit_choice = st.radio("Units", ["Metric (°C, m/s)", "Imperial (°F, mph)"], index=0)
units_mode = "metric" if unit_choice.startswith("Metric") else "imperial"

# ---- Inputs form ----
with st.form("inputs"):
    # CHANGED: give the selectbox column a bit more width ([2, 3])
    col1, col2 = st.columns([1, 1.6], gap="medium")
    with col1:
        city = st.text_input("Enter your city:", key="city",help="Try: Boston, London, Mumbai …")
        
    with col2:
        skin_display = st.selectbox(
            "Fitzpatrick skin type:",
            [
                "Type I - Pale, always burns, never tans",
                "Type II - Fair, usually burns, tans minimally",
                "Type III - Medium, sometimes mild burn, tans uniformly",
                "Type IV - Olive, rarely burns, tans easily",
                "Type V - Brown, very rarely burns, tans very easily",
                "Type VI - Dark brown to black, never burns"
            ],
            index=st.session_state.get("skin_idx", 2),
            help="Used to tailor UV & sunscreen tips."
        )

    colA, colB = st.columns([1,1])
    with colA:
        include_daily_tips = st.checkbox("Include GPT tip for each forecast day", value=True)
    with colB:
        show_raw_blocks = st.checkbox("Developer: show raw HTML blocks", value=False)

    submitted = st.form_submit_button("Generate / Refresh")

try:
    skin_type_number = int(skin_display.split(" ")[1])  # Extract 1–6
except (IndexError, ValueError):
    skin_type_number = 3

# =========================
# COMPUTE ON SUBMIT
# =========================
if submitted:
    city = st.session_state.get("city", "").strip()
    if not city:
        st.warning("Please enter a city.")
    else:
        st.session_state.skin_idx = ["Type I","Type II","Type III","Type IV","Type V","Type VI"].index(
            skin_display.split(" - ")[0]
        )
        st.session_state.inputs_key = _inputs_key(city, skin_type_number)
        st.session_state.include_daily_tips = include_daily_tips

    # 1) Fetch data (cached)
    data = fetch_weather(city, api_key) or {}
    df_forecast = fetch_forecast(city, api_key, units=units_mode)

    # 2) Outfit + GPT summaries
    outfit_today = get_outfit_recommendation(data) if data else "⚠️ Unable to suggest clothing due to invalid weather data."
    # Accessory tips based on UV (wind gust not available from current call; pass None or wire from forecast)
    outfit_today = add_accessory_tips(outfit_today, uv=data.get("uv_index"), wind_gust_ms=None)  # NEW
    gpt_pack = cached_gpt_summary({"weather": data, "outfit_text": outfit_today})

    # 3) Outfit forecast HTML + export structures + daily tips
    try:
        outfit_recs = generate_outfit_recommendations(df_forecast, uv_hint=data.get("uv_index"))
    except TypeError:
        outfit_recs = generate_outfit_recommendations(df_forecast)

    plan_md_sections, plan_pdf_sections, daily_gpt_tips = [], [], []

    for rec in outfit_recs:
        soup = BeautifulSoup(rec, 'html.parser')

        date_tag = soup.find('b')
        if not date_tag:
            continue
        date = date_tag.text.replace('📅 ', '').strip()

        outfit_line = soup.find(string=lambda t: isinstance(t, str) and "Outfit:" in t)
        if not outfit_line:
            continue
        outfit_text = outfit_line.split("Outfit:")[-1].strip()

        labels = [
            span.text.strip()
            for span in soup.find_all("span", style=lambda x: x and "font-weight: bold" in x)
        ]

        section_md = f"### {date}\n- **Outfit:** {outfit_text}\n"
        if labels:
            bullets = "\n".join([f"  - {lbl}" for lbl in labels])
            section_md += f"- **Color pairs:**\n{bullets}\n"
        plan_md_sections.append(section_md)

        # hexes + labels for PDF/PNG
        pdf_pairs = []
        for ddiv in soup.find_all("div", style=lambda x: x and "margin-top" in x):
            spans = ddiv.find_all("span")
            hexes = []
            for sp in spans[:2]:
                m = re.search(r'background-color:\s*(#[0-9a-fA-F]{6})', sp.get("style", ""))
                if m:
                    hexes.append(m.group(1))
            label_span = next((sp for sp in spans[2:] if "font-weight: bold" in (sp.get("style") or "")), None)
            label_txt = (label_span.text.strip() if label_span else (hexes[0] + " & " + hexes[1]) if len(hexes) == 2 else "")
            if len(hexes) == 2:
                try:
                    n1, n2 = closest_color_name(hexes[0]), closest_color_name(hexes[1])
                    label_txt = f"{n1} ({hexes[0]}) & {n2} ({hexes[1]})"
                except Exception:
                    pass
                pdf_pairs.append((hexes[0], hexes[1], label_txt))
        plan_pdf_sections.append({"date": date, "outfit": outfit_text, "pairs": pdf_pairs})

        if include_daily_tips:
            try:
                tip = cached_daily_tip(date, outfit_text, labels)
            except Exception:
                tip = "Style tip: pair thoughtfully and stay weather-smart."
        else:
            tip = ""
        daily_gpt_tips.append(tip)

    # 4) Weather alerts
    alerts = data.get("alerts") or []            

    # 5) Store for render
    st.session_state.data = data
    st.session_state.alerts = alerts
    st.session_state.df_forecast_json = (
        df_forecast.to_json(date_format="iso") if hasattr(df_forecast, "empty") and not df_forecast.empty else ""
    )
    st.session_state.outfit_today = outfit_today
    st.session_state.gpt_weather = gpt_pack["gpt_weather"]
    st.session_state.gpt_outfit = gpt_pack["gpt_outfit"]
    st.session_state.outfit_html_blocks = outfit_recs
    st.session_state.outfit_gpt_tips = daily_gpt_tips
    st.session_state.plan_md = "# 5-Day Outfit Plan\n\n" + "\n\n".join(plan_md_sections) if plan_md_sections else ""
    st.session_state.plan_pdf_bytes = build_outfit_plan_pdf_cached(plan_pdf_sections, city_name=city) if plan_pdf_sections else None
    # Build PNG color grid
    try:
        from export_utils import color_grid_image
        st.session_state.plan_png_bytes = color_grid_image(plan_pdf_sections)
    except Exception:
        st.session_state.plan_png_bytes = None

    st.session_state.plan_ready = True

# =========================
# RENDER FROM SESSION STATE
# =========================
if st.session_state.get("plan_ready"):
    data = st.session_state.get("data", {})
    alerts = st.session_state.get("alerts", [])
    df_forecast_json = st.session_state.get("df_forecast_json", "")
    outfit_today = st.session_state.get("outfit_today", "")
    gpt_weather = st.session_state.get("gpt_weather", "")
    gpt_outfit = st.session_state.get("gpt_outfit", "")
    outfit_html_blocks = st.session_state.get("outfit_html_blocks", [])
    daily_gpt_tips = st.session_state.get("outfit_gpt_tips", [])
    include_daily_tips = st.session_state.get("include_daily_tips", True)
    full_md = st.session_state.get("plan_md", "")
    pdf_bytes = st.session_state.get("plan_pdf_bytes", None)
    png_bytes = st.session_state.get("plan_png_bytes", None)

    # Summary & Today
    with st.expander("🧾 Climate Summary", expanded=True):
        st.write(summarize_weather(data) if data else "⚠️ Unable to fetch weather data.")
        st.markdown("**💬 GPT Says:**")
        st.info(gpt_weather or "—")

    with st.expander("👕 What to Wear Today", expanded=True):
        st.write(outfit_today or "—")
        st.markdown("**💬 GPT on Your Look:**")
        st.success(gpt_outfit or "—")

    # UV & Sunscreen (+ next 5 days UV)
    uv = data.get("uv_index")
    uv_source = data.get("uv_source")
    with st.expander("☀️ UV & Sunscreen Advice", expanded=True):
        st.markdown(f"**UV Index:** {uv if uv is not None else '—'}")
        try:
            parsed = st.session_state.get("skin_idx", 2) + 1
            skin_type_number = parsed
        except Exception:
            skin_type_number = 3
        st.warning(get_sunscreen_recommendation(uv, skin_type_number))
        uv5 = data.get("uv_next5") or []
        if uv5:
            st.markdown("**Next 5 days (UV index):** " + " · ".join(f"{u:.0f}" if u is not None else "—" for u in uv5))
        st.caption(f"UV data source: One Call {uv_source}" if uv_source else "UV data source: unavailable")

    # Weather Alerts
    if alerts:
        condensed_mode = st.toggle("Show alerts in condensed mode", value=False, key="alerts_condensed")
        render_alerts(alerts, condensed=condensed_mode)

    # Forecast toggle: Today (hourly) / Next 5 days
    with st.expander("📊 Forecast", expanded=True):
        view = st.radio("View", ["Today (hourly)", "Next 5 days"], horizontal=True)
        if view == "Today (hourly)":
            coord = data.get("coord") or {}
            lat, lon = coord.get("lat"), coord.get("lon")
            if lat is None or lon is None:
                st.warning("No coordinates for hourly data.")
            else:
                df_hourly = fetch_hourly(lat, lon, st.secrets["OPENWEATHER_API_KEY"], units=units_mode)
                if df_hourly.empty:
                    st.info("No hourly data available.")
                else:
                    # ⭐ CHANGED ONLY HERE: robust UTC parse → shift → drop tz; separate label column floored to hour
                    import pandas as pd
                    import matplotlib.dates as mdates
                    try:
                        tz_offset = get_tz_offset_seconds(lat, lon, st.secrets["OPENWEATHER_API_KEY"])
                    except Exception:
                        tz_offset = 0

                    df_hourly = df_hourly.copy()
                    # Parse as UTC, add city offset to get local, drop tz for plotting
                    dt_h = pd.to_datetime(df_hourly["datetime"], utc=True, errors="coerce")
                    local_dt = (dt_h + pd.to_timedelta(tz_offset, unit="s")).dt.tz_localize(None)

                    # Keep true local timestamps but make a floored label column for neat :00 display
                    df_hourly["datetime"] = local_dt
                    df_hourly["datetime_label"] = df_hourly["datetime"].dt.floor("H")

                    # Table uses the neat labels
                    st.dataframe(
                        df_hourly[["datetime_label","temp","uvi","pop","wind_speed","wind_gust"]]
                          .rename(columns={"datetime_label":"datetime"}),
                        use_container_width=True, hide_index=True
                    )

                    # ====== ✅ NEW: INTERACTIVE PLOTLY HOURLY CHART (Temp + PoP% + UVI) ======
                    dfp = df_hourly.copy()

                    # Ensure numeric
                    for col in ["temp", "pop", "uvi"]:
                        if col in dfp.columns:
                            dfp[col] = pd.to_numeric(dfp[col], errors="coerce")

                    # PoP to 0–100%
                    if "pop" in dfp.columns:
                        dfp["pop_pct"] = (dfp["pop"].clip(lower=0, upper=1) * 100.0)
                    else:
                        dfp["pop_pct"] = np.nan

                    # Scale UVI to 0–100 so it can share the right axis with PoP%
                    uvi_max = float(np.nanmax(dfp["uvi"])) if "uvi" in dfp.columns else 0.0
                    baseline = max(11.0, uvi_max if np.isfinite(uvi_max) else 0.0)
                    scale = (100.0 / baseline) if baseline > 0 else 0.0
                    dfp["uvi_pct"] = (dfp["uvi"] * scale) if scale > 0 else np.nan

                    fig = go.Figure()

                    # Temp (left axis)
                    temp_label = "Temp (°C)" if units_mode == "metric" else "Temp (°F)"
                    if "temp" in dfp.columns:
                        tmin = float(np.nanmin(dfp["temp"])) if np.isfinite(np.nanmin(dfp["temp"])) else None
                        tmax = float(np.nanmax(dfp["temp"])) if np.isfinite(np.nanmax(dfp["temp"])) else None
                        pad = (tmax - tmin) * 0.1 if (tmin is not None and tmax is not None and tmax > tmin) else 2.0
                        yrange_left = [tmin - pad, tmax + pad] if (tmin is not None and tmax is not None) else None

                        fig.add_trace(
                            go.Scatter(
                                x=dfp["datetime_label"],
                                y=dfp["temp"],
                                name=temp_label,
                                mode="lines+markers",
                                line=dict(width=2),
                                marker=dict(size=5),
                                hovertemplate="<b>%{x|%a %b %d • %H:%M}</b><br>" + temp_label + ": %{y:.1f}<extra></extra>",
                                yaxis="y",
                            )
                        )
                    else:
                        yrange_left = None

                    # PoP% (right axis)
                    if "pop_pct" in dfp.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=dfp["datetime_label"],
                                y=dfp["pop_pct"],
                                name="Precip prob (%)",
                                mode="lines+markers",
                                line=dict(width=2, dash="dash"),
                                marker=dict(size=5),
                                hovertemplate="<b>%{x|%a %b %d • %H:%M}</b><br>Precip prob: %{y:.0f}%<extra></extra>",
                                yaxis="y2",
                            )
                        )

                    # UVI (right axis, scaled)
                    if "uvi_pct" in dfp.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=dfp["datetime_label"],
                                y=dfp["uvi_pct"],
                                name=f"UVI (scaled to %; max={int(max(11, uvi_max))})",
                                mode="lines+markers",
                                line=dict(width=2, dash="dot"),
                                marker=dict(size=5),
                                hovertemplate="<b>%{x|%a %b %d • %H:%M}</b><br>UVI: %{customdata:.1f}<extra></extra>",
                                customdata=dfp["uvi"],
                                yaxis="y2",
                            )
                        )

                    # ► Styling: black axes/labels + legend; grid stays light gray
                    fig.update_layout(
                        height=420,
                        margin=dict(l=10, r=10, t=40, b=35),
                        hovermode="x unified",
                        xaxis=dict(
                            title=dict(font=dict(color="black")),
                            showgrid=True,
                            gridcolor="rgba(0,0,0,0.12)",
                            tickformat="%a %H:%M",
                            rangeslider=dict(visible=True, thickness=0.08),
                            rangeselector=dict(
                                buttons=[
                                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                                    dict(count=24, label="24h", step="hour", stepmode="backward"),
                                    dict(step="all"),
                                ]
                            ),
                            spikemode="across",
                            showspikes=True,
                            spikedash="dot",
                            spikecolor="rgba(0,0,0,0.3)",
                            spikethickness=1,
                            showline=True, linecolor="black", mirror=True,
                            tickfont=dict(color="black"),
                            ticks="outside", tickcolor="black",
                        ),
                        yaxis=dict(
                            title=dict(text=(temp_label if "temp" in dfp.columns else "Temperature"),
                                       font=dict(color="black")),
                            rangemode="tozero" if yrange_left is None else None,
                            range=yrange_left,
                            zeroline=True,
                            zerolinecolor="rgba(0,0,0,0.2)",
                            showgrid=True,
                            gridcolor="rgba(0,0,0,0.1)",
                            showline=True, linecolor="black", mirror=True,
                            tickfont=dict(color="black"),
                            ticks="outside", tickcolor="black",
                        ),
                        yaxis2=dict(
                            title=dict(text="Precip / UVI (%)", font=dict(color="black")),
                            range=[0, 100],
                            ticksuffix="%",
                            overlaying="y",
                            side="right",
                            zeroline=True,
                            zerolinecolor="rgba(0,0,0,0.2)",
                            showgrid=False,
                            showline=True, linecolor="black", mirror=True,
                            tickfont=dict(color="black"),
                            ticks="outside", tickcolor="black",
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom", y=1.02,
                            xanchor="right", x=1,
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=0,
                            font=dict(color="black"),
                        ),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                    )

                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                    # ====== ✅ END PLOTLY HOURLY CHART ======

        else:
            # Your existing 5-day graph table (UNCHANGED)
            if df_forecast_json:
                import pandas as pd
                import matplotlib.dates as mdates
                df_forecast = pd.read_json(df_forecast_json)

                # ⭐ keep as you had it previously for 5-day (no functional change requested for the table)
                try:
                    coord = data.get("coord") or {}
                    lat, lon = coord.get("lat"), coord.get("lon")
                    tz_offset = get_tz_offset_seconds(lat, lon, st.secrets["OPENWEATHER_API_KEY"]) if (lat is not None and lon is not None) else 0
                except Exception:
                    tz_offset = 0
                df_forecast = df_forecast.copy()
                dt_f = pd.to_datetime(df_forecast["datetime"], utc=False, errors="coerce")
                df_forecast["datetime"] = (dt_f + pd.to_timedelta(tz_offset, unit="s")).dt.floor("H")

                st.dataframe(
                    df_forecast[['datetime','temperature','humidity','condition']].rename(
                        columns={'datetime':'Date/Time','temperature':'Temp °C','humidity':'Humidity %','condition':'Condition'}
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                # ====== ✅ NEW: INTERACTIVE PLOTLY 5-DAY CHART (Temp left; Humidity right) ======
                dfp5 = df_forecast.copy()
                dfp5["temperature"] = pd.to_numeric(dfp5["temperature"], errors="coerce")
                dfp5["humidity"] = pd.to_numeric(dfp5["humidity"], errors="coerce")

                fig5 = go.Figure()
                temp_label_5 = "Temp (°C)" if units_mode == "metric" else "Temp (°F)"

                # Temperature (left)
                fig5.add_trace(
                    go.Scatter(
                        x=dfp5["datetime"],
                        y=dfp5["temperature"],
                        name=temp_label_5,
                        mode="lines+markers",
                        line=dict(width=2),
                        marker=dict(size=5),
                        hovertemplate="<b>%{x|%a %b %d • %H:%M}</b><br>" + temp_label_5 + ": %{y:.1f}<extra></extra>",
                        yaxis="y",
                    )
                )

                # Humidity (right)
                fig5.add_trace(
                    go.Scatter(
                        x=dfp5["datetime"],
                        y=dfp5["humidity"],
                        name="Humidity (%)",
                        mode="lines+markers",
                        line=dict(width=2, dash="dash"),
                        marker=dict(size=5),
                        hovertemplate="<b>%{x|%a %b %d • %H:%M}</b><br>Humidity: %{y:.0f}%<extra></extra>",
                        yaxis="y2",
                    )
                )

                # Axis ranges for temperature padding
                try:
                    tmin5 = float(np.nanmin(dfp5["temperature"]))
                except Exception:
                    tmin5 = None
                try:
                    tmax5 = float(np.nanmax(dfp5["temperature"]))
                except Exception:
                    tmax5 = None
                pad5 = (tmax5 - tmin5) * 0.1 if (tmin5 is not None and tmax5 is not None and tmax5 > tmin5) else 2.0
                yrange_left_5 = [tmin5 - pad5, tmax5 + pad5] if (tmin5 is not None and tmax5 is not None) else None

                fig5.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=40, b=35),
                    hovermode="x unified",
                    xaxis=dict(
                        title=dict(font=dict(color="black")),
                        showgrid=True,
                        gridcolor="rgba(0,0,0,0.12)",
                        tickformat="%a %H:%M",
                        showline=True, linecolor="black", mirror=True,
                        tickfont=dict(color="black"),
                        ticks="outside", tickcolor="black",
                    ),
                    yaxis=dict(
                        title=dict(text=temp_label_5, font=dict(color="black")),
                        range=yrange_left_5,
                        zeroline=True,
                        zerolinecolor="rgba(0,0,0,0.2)",
                        showgrid=True,
                        gridcolor="rgba(0,0,0,0.1)",
                        showline=True, linecolor="black", mirror=True,
                        tickfont=dict(color="black"),
                        ticks="outside", tickcolor="black",
                    ),
                    yaxis2=dict(
                        title=dict(text="Humidity (%)", font=dict(color="black")),
                        range=[0, 100],
                        ticksuffix="%",
                        overlaying="y",
                        side="right",
                        zeroline=True,
                        zerolinecolor="rgba(0,0,0,0.2)",
                        showgrid=False,
                        showline=True, linecolor="black", mirror=True,
                        tickfont=dict(color="black"),
                        ticks="outside", tickcolor="black",
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom", y=1.02,
                        xanchor="right", x=1,
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=0,
                        font=dict(color="black"),
                    ),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )

                st.plotly_chart(fig5, use_container_width=True, theme="streamlit")
                # ====== ✅ END PLOTLY 5-DAY CHART ======
            else:
                st.warning("⚠️ Forecast data unavailable.")

    # Outfit & Color Suggestions
    st.markdown("### 👗 5-Day Outfit & Color Suggestions")
    for i, rec in enumerate(outfit_html_blocks):
        st.markdown(rec, unsafe_allow_html=True)
        if include_daily_tips and i < len(daily_gpt_tips) and daily_gpt_tips[i]:
            st.markdown(f"<i>🧠 {daily_gpt_tips[i]}</i>", unsafe_allow_html=True)

    # Click-to-copy HEX (unchanged)
    st.markdown(
        """
        <script>
        (function(){
          if (window.__climbot_swatches_bound__) return;
          window.__climbot_swatches_bound__ = true;
          function copy(text){ navigator.clipboard?.writeText(text).then(()=>{}).catch(()=>{}); }
          const spans = Array.from(document.querySelectorAll('span'))
            .filter(s => (s.getAttribute('style')||'').includes('background-color'));
          spans.forEach(s=>{
            const m = /background-color:\\s*(#[0-9a-fA-F]{6})/i.exec(s.getAttribute('style')||'');
            if (m){ s.style.cursor = 'pointer'; s.title = 'Click to copy ' + m[1]; s.addEventListener('click', ()=>copy(m[1])); }
          });
        })();
        </script>
        """,
        unsafe_allow_html=True
    )

    # Downloads
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    if full_md:
        with col_dl1:
            key_md = "download_plan_" + (st.session_state.get("inputs_key") or "md")
            st.download_button(
                "📥 Download 5-day plan (.md)",
                full_md,
                file_name="outfit_plan.md",
                mime="text/markdown",
                key=key_md
            )
    if pdf_bytes:
        with col_dl2:
            key_pdf = "download_pdf_" + (st.session_state.get("inputs_key") or "pdf")
            st.download_button(
                "📄 Download 5-day plan (PDF)",
                pdf_bytes,
                file_name="outfit_plan.pdf",
                mime="application/pdf",
                key=key_pdf
            )
    if png_bytes:
        with col_dl3:
            key_png = "download_png_" + (st.session_state.get("inputs_key") or "png")
            st.download_button(
                "🖼️ Download color grid (PNG)",
                png_bytes,
                file_name="color_grid.png",
                mime="image/png",
                key=key_png
            )
    elif st.session_state.get("plan_md") and not REPORTLAB_OK:
        st.info("Install ReportLab to enable PDF download: `pip install reportlab`")



# Footer
st.write("---")
st.markdown("""
<style>
.footer-container { text-align:center; font-size:0.9em; line-height:1.6; }
.social-link { text-decoration:none; color:inherit; }
.social-link:hover { color:#00A37A !important; }
social-icon { vertical-align:middle; margin-right:6px; }
.social-row { display:flex; justify-content:center; gap:24px; flex-wrap:wrap; margin:6px 0 12px 0; }
</style>

<div class="footer-container">
  Made out of curiosity to learn • In collaboration between <b>Ashraiy</b> and <b>Nitin</b>
  <br><br>
  <div class="social-row">
    <a href="https://www.linkedin.com/in/ssny15" target="_blank" class="social-link">
      <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="16" class="social-icon">
      LinkedIn - <b>Nitin</b>
    </a>
    <a href="https://www.linkedin.com/in/ashraiy-manohar" target="_blank" class="social-link">
      <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="16" class="social-icon">
      LinkedIn - <b>Ashraiy</b>
    </a>
  </div>
  <div class="social-row">
    <a href="https://github.com/SSNitin-Y" target="_blank" class="social-link">
      <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="16" class="social-icon">
      GitHub - <b>Nitin</b>
    </a>
    <a href="https://github.com/ashraiymanohar-maker" target="_blank" class="social-link">
      <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="16" class="social-icon">
      GitHub - <b>Ashraiy</b>
    </a>
  </div>
</div>
""", unsafe_allow_html=True)









