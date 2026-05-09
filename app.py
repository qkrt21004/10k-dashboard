import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import anthropic
import json
import os

st.set_page_config(page_title="10-K Financial Dashboard", layout="wide")

HEADERS = {"User-Agent": "financial-dashboard research@example.com"}
COLORS  = ["#4C9BE8", "#56C596", "#F4845F", "#A78BFA", "#FACC15"]

def _secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")

# ── SEC EDGAR ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def load_all_tickers() -> list[str]:
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS)
    items = r.json().values()
    return sorted([f"{v['ticker'].upper()} – {v['title']}" for v in items])


@st.cache_data(ttl=86400)
def get_cik(ticker: str) -> tuple:
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS)
    for item in r.json().values():
        if item["ticker"].upper() == ticker.upper():
            return str(item["cik_str"]).zfill(10), item["title"]
    return None, None


@st.cache_data(ttl=86400)
def get_xbrl_facts(cik: str) -> dict:
    r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=HEADERS, timeout=30)
    return r.json().get("facts", {}).get("us-gaap", {})


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────

def _find_fy_end(quarter_end: pd.Timestamp, fy_ends: list) -> pd.Timestamp | None:
    """분기 종료일 이후 가장 가까운 FY 종료일 반환 (최대 13개월 이내)"""
    for fy_end in sorted(fy_ends):
        if fy_end >= quarter_end and (fy_end - quarter_end).days <= 400:
            return fy_end
    return None


def _sort_quarters(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    def key(lbl):
        p = lbl.split("-Q")
        return (int(p[0]), int(p[1])) if len(p) == 2 else (0, 0)
    return s.iloc[sorted(range(len(s)), key=lambda i: key(s.index[i]))]


# ── 연간 데이터 ───────────────────────────────────────────────────────────────

def extract_annual(facts: dict, concepts: list, unit: str = "USD") -> pd.Series:
    combined = pd.Series(dtype=float)
    for concept in concepts:
        rows = [d for d in facts.get(concept, {}).get("units", {}).get(unit, [])
                if d.get("form") == "10-K" and d.get("fp") == "FY" and d.get("val") is not None]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["end"] = pd.to_datetime(df["end"])
        if "start" in df.columns:
            df["start"] = pd.to_datetime(df["start"])
            mask = df["start"].isna() | ((df["end"] - df["start"]).dt.days >= 300)
            df = df[mask]
        df["year"] = df["end"].dt.year
        df = df.sort_values("filed").drop_duplicates("year", keep="last")
        combined = df.set_index("year")["val"].sort_index().combine_first(combined)
    return combined.sort_index()


# ── 분기 데이터 ───────────────────────────────────────────────────────────────

def extract_quarterly_flow(facts: dict, concepts: list, unit: str = "USD") -> pd.Series:
    """
    흐름 항목 분기 추출.
    - FY 종료일 기준으로 분기 귀속
    - 기간 일수로 단일 분기(~90일) vs YTD(~180/270일) 자동 판별
    - 회사마다 다른 보고 방식(NVDA 등) 자동 대응
    """
    combined = pd.Series(dtype=float)

    for concept in concepts:
        all_rows = facts.get(concept, {}).get("units", {}).get(unit, [])
        if not all_rows:
            continue

        df = pd.DataFrame(all_rows)
        df["end"]   = pd.to_datetime(df["end"])
        df["start"] = pd.to_datetime(df["start"]) if "start" in df.columns else pd.NaT
        df["fp"]    = df["fp"].fillna("")
        df["days"]  = (df["end"] - df["start"]).dt.days

        # FY 데이터 (10-K, 300일 이상)
        fy_df = df[(df["form"] == "10-K") & (df["fp"] == "FY") & (df["days"] >= 300)]
        fy_df = fy_df.sort_values("filed").drop_duplicates("end", keep="last")
        fy_map  = fy_df.set_index("end")["val"].to_dict()
        fy_ends = list(fy_map.keys())
        if not fy_ends:
            continue

        # 분기 데이터 (10-Q, fp=Q1/Q2/Q3) → FY별로 수집
        q_df = df[(df["form"] == "10-Q") & (df["fp"].isin(["Q1", "Q2", "Q3"]))]
        q_df = q_df.sort_values("filed").drop_duplicates(["end", "fp"], keep="last")

        fy_qtrs = {}  # {fy_end: {Q1, Q2_single, Q2_ytd, Q3_single, Q3_ytd}}
        for _, row in q_df.iterrows():
            fy_end = _find_fy_end(row["end"], fy_ends)
            if not fy_end:
                continue
            d = fy_qtrs.setdefault(fy_end, {})
            fp, days, val = row["fp"], row["days"], row["val"]

            if fp == "Q1":
                d["Q1"] = val                          # Q1은 항상 단일
            elif fp == "Q2":
                if days <= 110:
                    d["Q2_single"] = val               # 단일 분기
                else:
                    d["Q2_ytd"] = val                  # 6개월 누적
            elif fp == "Q3":
                if days <= 110:
                    d["Q3_single"] = val               # 단일 분기
                else:
                    d["Q3_ytd"] = val                  # 9개월 누적

        # FY별 실제 분기 계산
        records = {}
        for fy_end, d in fy_qtrs.items():
            fy_year = str(fy_end.year)
            vfy = fy_map[fy_end]

            q1 = d.get("Q1")

            # Q2 실제값
            if "Q2_single" in d:
                q2 = d["Q2_single"]
            elif "Q2_ytd" in d:
                q2 = d["Q2_ytd"] - q1 if q1 is not None else d["Q2_ytd"]
            else:
                q2 = None

            # Q3 실제값
            if "Q3_single" in d:
                q3 = d["Q3_single"]
            elif "Q3_ytd" in d:
                q2_ytd = d.get("Q2_ytd", (q1 or 0) + (q2 or 0))
                q3 = d["Q3_ytd"] - q2_ytd
            else:
                q3 = None

            if q1 is not None: records[f"{fy_year}-Q1"] = q1
            if q2 is not None: records[f"{fy_year}-Q2"] = q2
            if q3 is not None: records[f"{fy_year}-Q3"] = q3

            # Q4 = FY - (Q1+Q2+Q3)
            known = [v for v in [q1, q2, q3] if v is not None]
            if len(known) == 3:
                records[f"{fy_year}-Q4"] = vfy - sum(known)

        if records:
            combined = pd.Series(records).combine_first(combined)

    return _sort_quarters(combined)


def extract_quarterly_balance(facts: dict, concepts: list, unit: str = "USD") -> pd.Series:
    """
    잔액 항목 분기 추출.
    FY 종료일 기준으로 각 분기를 FY에 귀속시켜 레이블링.
    """
    combined = pd.Series(dtype=float)

    for concept in concepts:
        all_rows = facts.get(concept, {}).get("units", {}).get(unit, [])
        if not all_rows:
            continue

        df = pd.DataFrame(all_rows)
        df["end"] = pd.to_datetime(df["end"])
        df["fp"]  = df["fp"].fillna("")

        # FY 종료일 목록
        fy_df = df[(df["form"] == "10-K") & (df["fp"] == "FY")].copy()
        fy_df = fy_df.sort_values("filed").drop_duplicates("end", keep="last")
        fy_ends = fy_df["end"].tolist()

        if not fy_ends:
            continue

        # Q1/Q2/Q3 잔액 (10-Q)
        q_rows = df[(df["form"] == "10-Q") & (df["fp"].isin(["Q1","Q2","Q3"]))]
        q_rows = q_rows.sort_values("filed").drop_duplicates(["end","fp"], keep="last")

        records = {}
        for _, row in q_rows.iterrows():
            fy_end = _find_fy_end(row["end"], fy_ends)
            if fy_end:
                lbl = f"{fy_end.year}-{row['fp']}"
                records[lbl] = row["val"]

        if records:
            combined = pd.Series(records).combine_first(combined)

    return _sort_quarters(combined)


# ── CONCEPTS ─────────────────────────────────────────────────────────────────

CONCEPTS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet"],
    "gross":   ["GrossProfit"],
    "op_inc":  ["OperatingIncomeLoss"],
    "net_inc": ["NetIncomeLoss", "ProfitLoss"],
    "assets":  ["Assets"],
    "liab":    ["Liabilities"],
    "equity":  ["StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash":    ["CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsAndShortTermInvestments"],
    "debt":    ["LongTermDebt", "LongTermDebtNoncurrent",
                "LongTermDebtAndCapitalLeaseObligations",
                "DebtLongtermAndShorttermCombinedAmount"],
    "op_cf":   ["NetCashProvidedByUsedInOperatingActivities"],
    "inv_cf":  ["NetCashProvidedByUsedInInvestingActivities"],
    "fin_cf":  ["NetCashProvidedByUsedInFinancingActivities"],
    "capex":   ["PaymentsToAcquirePropertyPlantAndEquipment",
                "PaymentsToAcquireProductiveAssets",
                "PaymentsForCapitalImprovements",
                "PaymentsToAcquireOtherPropertyPlantAndEquipment"],
    "eps":     ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
}


# ── 데이터 빌드 ───────────────────────────────────────────────────────────────

def build_annual(facts: dict, num_years: int) -> dict:
    def get(key, unit="USD"):
        return extract_annual(facts, CONCEPTS[key], unit).tail(num_years) / 1e6

    op_cf = get("op_cf")
    capex = get("capex") * -1
    return {
        "income": pd.DataFrame({
            "Revenue": get("revenue"), "Gross Profit": get("gross"),
            "Operating Income": get("op_inc"), "Net Income": get("net_inc"),
        }).sort_index().tail(num_years),
        "eps": extract_annual(facts, CONCEPTS["eps"], "USD/shares").tail(num_years),
        "balance": pd.DataFrame({
            "Total Assets": get("assets"), "Total Liabilities": get("liab"),
            "Total Equity": get("equity"), "Cash & Equiv.": get("cash"),
            "Total Debt": get("debt"),
        }).sort_index().tail(num_years),
        "cashflow": pd.DataFrame({
            "Operating CF": op_cf, "Investing CF": get("inv_cf"),
            "Financing CF": get("fin_cf"), "CapEx": capex,
            "Free Cash Flow": op_cf.add(capex, fill_value=0),
        }).sort_index().tail(num_years),
    }


def build_quarterly(facts: dict, num_quarters: int) -> dict:
    def flow(key):
        return extract_quarterly_flow(facts, CONCEPTS[key]).tail(num_quarters) / 1e6

    def bal(key):
        return extract_quarterly_balance(facts, CONCEPTS[key]).tail(num_quarters) / 1e6

    def finalize(df):
        return df.sort_index().dropna(how="all").tail(num_quarters)

    op_cf  = flow("op_cf")
    inv_cf = flow("inv_cf")
    fin_cf = flow("fin_cf")
    capex  = flow("capex") * -1
    fcf    = op_cf.add(capex, fill_value=0)

    return {
        "income": finalize(pd.DataFrame({
            "Revenue": flow("revenue"), "Gross Profit": flow("gross"),
            "Operating Income": flow("op_inc"), "Net Income": flow("net_inc"),
        })),
        "balance": finalize(pd.DataFrame({
            "Total Assets": bal("assets"), "Total Liabilities": bal("liab"),
            "Total Equity": bal("equity"), "Cash & Equiv.": bal("cash"),
            "Total Debt": bal("debt"),
        })),
        "cashflow": finalize(pd.DataFrame({
            "Operating CF": op_cf, "Investing CF": inv_cf,
            "Financing CF": fin_cf, "CapEx": capex,
            "Free Cash Flow": fcf,
        })),
    }


# ── 주가 데이터 ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_price(ticker: str, range_: str = "max") -> pd.DataFrame:
    period_map = {"max": "max", "10y": "10y", "5y": "5y", "1y": "1y", "1mo": "1mo"}
    interval   = "1d" if range_ in ("1mo", "1y") else "1wk"
    try:
        df = yf.download(ticker, period=period_map[range_], interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = df[["Close"]].reset_index()
        df.columns = ["date", "close"]
        df["close"] = df["close"].squeeze()
        return df.dropna().reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def price_chart(df: pd.DataFrame, ticker: str, company_name: str) -> go.Figure:
    if df.empty:
        return None
    first, last = df["close"].iloc[0], df["close"].iloc[-1]
    change_pct  = (last - first) / first * 100
    color       = "#22c55e" if change_pct >= 0 else "#ef4444"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"],
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=color.replace(")", ", 0.08)").replace("rgb", "rgba") if "rgb" in color
                  else f"{'rgba(34,197,94' if change_pct >= 0 else 'rgba(239,68,68'}, 0.08)",
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
    ))
    sign = "+" if change_pct >= 0 else ""
    fig.update_layout(
        title=dict(
            text=f"{company_name} ({ticker})  "
                 f"<span style='font-size:14px; color:{color}'>{sign}{change_pct:.1f}%</span>  "
                 f"<span style='font-size:14px; color:gray'>${last:,.2f}</span>",
            font_size=18,
        ),
        height=380,
        xaxis=dict(showgrid=False, rangeslider_visible=False),
        yaxis=dict(title="Price (USD)", tickprefix="$", showgrid=True,
                   gridcolor="rgba(128,128,128,0.1)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=20),
        hovermode="x unified",
    )
    return fig


# ── 밸류에이션 데이터 ─────────────────────────────────────────────────────────

VALUATION_METRICS = {
    "P/E":          ("trailingPE",                    "defaultKeyStatistics", False),
    "Forward P/E":  ("forwardPE",                     "defaultKeyStatistics", False),
    "PEG":          ("pegRatio",                      "defaultKeyStatistics", False),
    "P/S":          ("priceToSalesTrailing12Months",  "summaryDetail",        False),
    "P/B":          ("priceToBook",                   "defaultKeyStatistics", False),
    "EV/EBITDA":    ("enterpriseToEbitda",            "defaultKeyStatistics", False),
    "EV/Revenue":   ("enterpriseToRevenue",           "defaultKeyStatistics", False),
    "EV/FCF":       (None,                            None,                   False),
    "Div Yield (%)":("dividendYield",                 "summaryDetail",        True),
    "ROE (%)":      ("returnOnEquity",                "financialData",        True),
    "ROA (%)":      ("returnOnAssets",                "financialData",        True),
    "Profit Margin (%)": ("profitMargins",            "financialData",        True),
}

@st.cache_data(ttl=3600)
def fetch_valuation(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        mkt = info.get("marketCap")
        ev  = info.get("enterpriseValue")
        fcf = info.get("freeCashflow")

        def pct(v):
            return round(v * 100, 2) if v is not None else None

        def val(v):
            return round(v, 2) if v is not None else None

        return {
            "P/E":               val(info.get("trailingPE")),
            "Forward P/E":       val(info.get("forwardPE")),
            "PEG":               val(info.get("pegRatio")),
            "P/S":               val(info.get("priceToSalesTrailing12Months")),
            "P/B":               val(info.get("priceToBook")),
            "EV/EBITDA":         val(info.get("enterpriseToEbitda")),
            "EV/Revenue":        val(info.get("enterpriseToRevenue")),
            "EV/FCF":            round(ev / fcf, 2) if ev and fcf and fcf > 0 else None,
            "Div Yield (%)":     pct(info.get("dividendYield")),
            "ROE (%)":           pct(info.get("returnOnEquity")),
            "ROA (%)":           pct(info.get("returnOnAssets")),
            "Profit Margin (%)": pct(info.get("profitMargins")),
            "_market_cap":       mkt,
            "_ev":               ev,
        }
    except Exception:
        return {}


def valuation_table(tickers_data: dict):
    """tickers_data = {ticker: {metric: value}}"""
    metrics = [m for m in VALUATION_METRICS.keys()]
    rows = []
    for tkr, vals in tickers_data.items():
        row = {"Ticker": tkr}
        for m in metrics:
            row[m] = vals.get(m)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Ticker")

    # 높을수록 좋은 지표 (역방향 색상)
    higher_better = {"Div Yield (%)", "ROE (%)", "ROA (%)", "Profit Margin (%)"}

    def color_col(col_data, reverse=False):
        valid = col_data.dropna()
        if len(valid) < 2:
            return [""] * len(col_data)
        mn, mx = valid.min(), valid.max()
        styles = []
        for v in col_data:
            if pd.isna(v) or mx == mn:
                styles.append("color: gray")
                continue
            ratio = (v - mn) / (mx - mn)
            if reverse:
                ratio = 1 - ratio
            r = int(239 * ratio + 34 * (1 - ratio))
            g = int(68  * ratio + 197 * (1 - ratio))
            b = int(68  * ratio + 94  * (1 - ratio))
            styles.append(f"color: rgb({r},{g},{b}); font-weight: bold")
        return styles

    styled = df.style
    for col in df.columns:
        reverse = col in higher_better
        styled = styled.apply(lambda s, rv=reverse: color_col(s, rv), subset=[col])

    styled = styled.format(lambda v: f"{v:,.2f}" if pd.notna(v) else "—")
    st.dataframe(styled, use_container_width=True)


# ── 차트 & 테이블 ─────────────────────────────────────────────────────────────

def fmt(val):
    if pd.isna(val): return "—"
    if abs(val) >= 1000: return f"${val/1000:,.1f}B"
    return f"${val:,.0f}M"


def fmt_pct(val):
    if pd.isna(val): return "—"
    return f"+{val:.1f}%" if val > 0 else f"{val:.1f}%"


def bar_chart(df, title, cols=None, x_label=""):
    cols = [c for c in (cols or df.columns.tolist()) if c in df.columns]
    fig = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Bar(
            name=col, x=df.index.astype(str), y=df[col],
            marker_color=COLORS[i % len(COLORS)],
            text=[f"${v:,.0f}M" if pd.notna(v) else "" for v in df[col]],
            textposition="outside", textfont_size=10,
        ))
    fig.update_layout(
        title=dict(text=title, font_size=15), barmode="group", height=400,
        yaxis=dict(title="USD Millions", tickformat="$,.0f"), xaxis_title=x_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def line_chart(df, title, cols=None, x_label=""):
    cols = [c for c in (cols or df.columns.tolist()) if c in df.columns]
    fig = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Scatter(
            name=col, x=df.index.astype(str), y=df[col],
            mode="lines+markers+text",
            line=dict(color=COLORS[i % len(COLORS)], width=2), marker=dict(size=7),
            text=[f"${v:,.0f}M" if pd.notna(v) else "" for v in df[col]],
            textposition="top center", textfont_size=10,
        ))
    fig.update_layout(
        title=dict(text=title, font_size=15), height=400,
        yaxis=dict(title="USD Millions", tickformat="$,.0f"), xaxis_title=x_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def margin_chart(inc, x_label=""):
    mdf = pd.DataFrame(index=inc.index)
    if "Gross Profit" in inc and "Revenue" in inc:
        mdf["Gross Margin %"]     = (inc["Gross Profit"] / inc["Revenue"] * 100).round(1)
    if "Operating Income" in inc and "Revenue" in inc:
        mdf["Operating Margin %"] = (inc["Operating Income"] / inc["Revenue"] * 100).round(1)
    if "Net Income" in inc and "Revenue" in inc:
        mdf["Net Margin %"]       = (inc["Net Income"] / inc["Revenue"] * 100).round(1)
    if mdf.empty:
        return
    fig = go.Figure()
    for i, col in enumerate(mdf.columns):
        fig.add_trace(go.Scatter(
            name=col, x=mdf.index.astype(str), y=mdf[col],
            mode="lines+markers+text",
            line=dict(color=COLORS[i], width=2), marker=dict(size=7),
            text=[f"{v:.1f}%" if pd.notna(v) else "" for v in mdf[col]],
            textposition="top center", textfont_size=10,
        ))
    fig.update_layout(
        title="Margin Trends (%)", height=380,
        yaxis=dict(title="%", ticksuffix="%"), xaxis_title=x_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_table(df, chg_label, eps=None):
    disp = df.copy().astype(object)
    for col in df.columns:
        disp[col] = df[col].apply(fmt)
    if eps is not None and not eps.empty:
        disp["EPS (Diluted)"] = eps.reindex(df.index).apply(
            lambda v: f"${v:.2f}" if pd.notna(v) else "—")
    disp.index = disp.index.astype(str)
    st.dataframe(disp, use_container_width=True)

    pct = df.pct_change() * 100
    pct_disp = pct.copy().astype(object)
    for col in pct.columns:
        pct_disp[col] = pct[col].apply(fmt_pct)
    pct_disp.index = pct_disp.index.astype(str)
    with st.expander(f"{chg_label} 변화율 (%)"):
        def color(v):
            if isinstance(v, str) and v.startswith("+"): return "color: #22c55e"
            if isinstance(v, str) and v.startswith("-"): return "color: #ef4444"
            return ""
        st.dataframe(pct_disp.style.map(color), use_container_width=True)


def render_section(data, key, title_bar, cols_bar, title_line, cols_line, chg_label, x_label, eps=None):
    df = data[key]
    if df.empty or df.dropna(how="all").empty:
        st.warning(f"{key} 데이터를 찾을 수 없습니다.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(bar_chart(df, title_bar, cols_bar, x_label), use_container_width=True)
    with c2:
        st.plotly_chart(line_chart(df, title_line, cols_line, x_label), use_container_width=True)
    if key == "income":
        margin_chart(df, x_label)
    st.subheader("수치")
    show_table(df, chg_label, eps)


# ── UI ────────────────────────────────────────────────────────────────────────

st.sidebar.title("📊 Dashboard")
page = st.sidebar.radio(
    "페이지",
    ["10-K 분석", "SOTP 밸류에이션", "Consumer & Retail"],
    label_visibility="collapsed",
)
st.sidebar.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SOTP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400)
def sotp_get_segments(ticker: str, company_name: str) -> list:
    api_key = _secret("ANTHROPIC_API_KEY")
    if not api_key:
        return []
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""You are a financial analyst. Perform a Sum-of-the-Parts (SOTP) breakdown for {company_name} ({ticker}).

Return a JSON array (no markdown, no explanation — raw JSON only) where each element is a business segment with these fields:
- "name": segment name (string)
- "revenue_ttm_b": TTM revenue in billions USD (number, based on most recent annual report)
- "ebitda_ttm_b": TTM EBITDA in billions USD (number or null if not meaningful)
- "metric": "EV/Revenue" or "EV/EBITDA" — whichever is standard for this type of business
- "peers": array of 3-5 ticker symbols of publicly traded pure-play comparable companies
- "rationale": one sentence explaining the peer selection

Use only publicly available data from the company's most recent annual report."""

    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    # strip markdown code fences if present
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
    return json.loads(raw)


@st.cache_data(ttl=3600)
def sotp_peer_multiples(peers: tuple) -> dict:
    result = {}
    for ticker in peers:
        try:
            info = yf.Ticker(ticker).info
            result[ticker] = {
                "EV/Revenue": info.get("enterpriseToRevenue"),
                "EV/EBITDA":  info.get("enterpriseToEbitda"),
            }
        except Exception:
            result[ticker] = {"EV/Revenue": None, "EV/EBITDA": None}
    return result


def sotp_median_multiple(peers: list, metric: str) -> float | None:
    data = sotp_peer_multiples(tuple(peers))
    vals = [v[metric] for v in data.values() if v.get(metric) is not None]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return round((vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2), 2)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SOTP 밸류에이션
# ═══════════════════════════════════════════════════════════════════════════════
if page == "SOTP 밸류에이션":
    st.title("⚖️ SOTP 밸류에이션")
    st.caption("AI가 사업부문을 자동 분류하고 pure-play peer 멀티플을 적용해 내재가치를 산출합니다.")

    all_tickers_sotp = load_all_tickers()
    with st.form("sotp_search"):
        col1, col2 = st.columns([4, 1])
        sel = col1.selectbox("티커 검색", options=all_tickers_sotp,
                             index=None, placeholder="티커 또는 회사명 입력 (예: GOOGL)")
        go_btn = col2.form_submit_button("분석 시작", type="primary", use_container_width=True)

    if go_btn and sel:
        st.session_state["sotp_ticker"] = sel.split(" – ")[0].strip()
        st.session_state["sotp_company"] = sel.split(" – ")[1].strip() if " – " in sel else sel
        st.session_state.pop("sotp_edited", None)

    if "sotp_ticker" not in st.session_state:
        st.stop()

    s_ticker  = st.session_state["sotp_ticker"]
    s_company = st.session_state["sotp_company"]
    st.markdown(f"### {s_company} ({s_ticker})")

    # ── 세그먼트 로드 ────────────────────────────────────────────────────────
    if "sotp_edited" not in st.session_state:
        with st.spinner("Claude가 사업부문을 분석 중입니다..."):
            try:
                segments = sotp_get_segments(s_ticker, s_company)
            except Exception as e:
                st.error(f"Claude API 오류: {e}")
                st.stop()

        if not segments:
            st.error("세그먼트 데이터를 가져오지 못했습니다. ANTHROPIC_API_KEY를 확인하세요.")
            st.stop()

        # peer 멀티플 자동 계산
        with st.spinner("Peer 멀티플 계산 중..."):
            rows = []
            for seg in segments:
                metric = seg.get("metric", "EV/Revenue")
                peers  = seg.get("peers", [])
                median = sotp_median_multiple(peers, metric)
                rows.append({
                    "세그먼트":    seg["name"],
                    "매출 (B$)":  seg.get("revenue_ttm_b"),
                    "EBITDA (B$)": seg.get("ebitda_ttm_b"),
                    "지표":        metric,
                    "배수":        median if median else 10.0,
                    "Peers":      ", ".join(peers),
                    "근거":        seg.get("rationale", ""),
                })
            st.session_state["sotp_edited"] = pd.DataFrame(rows)

    df_edit = st.session_state["sotp_edited"]

    # ── 편집 가능 테이블 ─────────────────────────────────────────────────────
    st.subheader("📋 세그먼트 설정 (직접 수정 가능)")
    edited = st.data_editor(
        df_edit,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "세그먼트":    st.column_config.TextColumn("세그먼트"),
            "매출 (B$)":  st.column_config.NumberColumn("매출 (B$)", format="%.2f"),
            "EBITDA (B$)": st.column_config.NumberColumn("EBITDA (B$)", format="%.2f"),
            "지표":        st.column_config.SelectboxColumn("지표", options=["EV/Revenue", "EV/EBITDA"]),
            "배수":        st.column_config.NumberColumn("배수 (x)", format="%.1fx"),
            "Peers":      st.column_config.TextColumn("Peers (쉼표 구분)"),
            "근거":        st.column_config.TextColumn("근거", width="large"),
        },
        key="sotp_editor",
    )
    st.session_state["sotp_edited"] = edited

    # peer 멀티플 새로고침
    if st.button("🔄 Peer 멀티플 재계산"):
        for i, row in edited.iterrows():
            peers  = [p.strip() for p in str(row["Peers"]).split(",") if p.strip()]
            metric = row["지표"]
            median = sotp_median_multiple(peers, metric)
            if median:
                edited.at[i, "배수"] = median
        st.session_state["sotp_edited"] = edited
        st.rerun()

    st.divider()

    # ── SOTP 계산 ────────────────────────────────────────────────────────────
    st.subheader("📐 SOTP 계산 결과")

    try:
        info      = yf.Ticker(s_ticker).info
        net_debt  = (info.get("totalDebt") or 0) - (info.get("totalCash") or 0)
        shares    = info.get("sharesOutstanding") or 0
        cur_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        net_debt_b = net_debt / 1e9
        shares_b   = shares / 1e9
    except Exception:
        net_debt_b = 0.0
        shares_b   = 0.0
        cur_price  = 0.0

    result_rows = []
    total_ev = 0.0
    for _, row in edited.iterrows():
        metric = row["지표"]
        if metric == "EV/Revenue":
            base = row["매출 (B$)"] or 0
        else:
            base = row["EBITDA (B$)"] or 0
        implied = round(base * (row["배수"] or 0), 2)
        total_ev += implied
        result_rows.append({
            "세그먼트":     row["세그먼트"],
            "기준값 (B$)": round(base, 2),
            "지표":         metric,
            "배수":         f"{row['배수']:.1f}x",
            "Implied EV (B$)": implied,
        })

    df_result = pd.DataFrame(result_rows)
    st.dataframe(df_result, use_container_width=True, hide_index=True)

    equity_val   = total_ev - net_debt_b
    intrinsic    = round((equity_val * 1e9) / shares, 2) if shares > 0 else 0
    upside       = round((intrinsic / cur_price - 1) * 100, 1) if cur_price > 0 else 0
    upside_color = "🟢" if upside > 0 else "🔴"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("합산 EV",        f"${total_ev:,.1f}B")
    c2.metric("Net Debt",       f"${net_debt_b:,.1f}B")
    c3.metric("Equity Value",   f"${equity_val:,.1f}B")
    c4.metric("내재가치 (주당)", f"${intrinsic:,.2f}")
    c5.metric("현재주가",        f"${cur_price:,.2f}",
              delta=f"{upside_color} {upside:+.1f}% 업사이드")

    # ── Peer 멀티플 상세 ─────────────────────────────────────────────────────
    with st.expander("🔍 Peer 멀티플 상세"):
        all_peers = []
        for _, row in edited.iterrows():
            all_peers += [p.strip() for p in str(row["Peers"]).split(",") if p.strip()]
        all_peers = list(dict.fromkeys(all_peers))
        if all_peers:
            peer_data = sotp_peer_multiples(tuple(all_peers))
            peer_rows = [{"Ticker": t,
                          "EV/Revenue": v.get("EV/Revenue"),
                          "EV/EBITDA":  v.get("EV/EBITDA")}
                         for t, v in peer_data.items()]
            st.dataframe(pd.DataFrame(peer_rows), use_container_width=True, hide_index=True)

    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Consumer & Retail
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Consumer & Retail":
    st.title("🛒 Consumer & Retail")
    st.caption("소비재·유통 업종 전용 분석 대시보드")
    st.info("🚧 준비 중입니다.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: 10-K 분석 (기존)
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📊 10-K Financial Dashboard")
st.caption("SEC EDGAR XBRL 기반 · 연간 / 분기 재무제표 비교")

all_tickers = load_all_tickers()

with st.form("search"):
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    selected    = c1.selectbox("티커 검색", options=all_tickers,
                               index=None, placeholder="티커 또는 회사명 입력 (예: NV, NVIDIA...)")
    ticker      = selected.split(" – ")[0].strip() if selected else ""
    period      = c2.radio("기간", ["연간", "분기"], horizontal=True)
    num_periods = c3.selectbox(
        "기간 수",
        list(range(1, 11)) if period == "연간" else [4, 8, 12, 16],
        index=4 if period == "연간" else 1,
    )
    submitted = c4.form_submit_button("분석", type="primary", use_container_width=True)

if submitted and ticker:
    with st.spinner("데이터 로딩 중..."):
        cik, company_name = get_cik(ticker)
        if not cik:
            st.error(f"'{ticker}' 티커를 찾을 수 없습니다.")
            st.stop()
        facts = get_xbrl_facts(cik)

    is_qtr    = period == "분기"
    x_label   = "Quarter (Fiscal Year)" if is_qtr else "Year"
    chg_label = "QoQ" if is_qtr else "YoY"
    data      = build_quarterly(facts, num_periods) if is_qtr else build_annual(facts, num_periods)

    st.session_state["result"] = {
        "ticker": ticker, "company_name": company_name,
        "data": data, "is_qtr": is_qtr,
        "x_label": x_label, "chg_label": chg_label,
    }

if "result" in st.session_state:
    r          = st.session_state["result"]
    ticker     = r["ticker"]
    company_name = r["company_name"]
    data       = r["data"]
    is_qtr     = r["is_qtr"]
    x_label    = r["x_label"]
    chg_label  = r["chg_label"]

    st.success(f"**{company_name}** ({ticker}) — {'분기' if is_qtr else '연간'} 로드 완료")

    # ── 주가 차트 ──────────────────────────────────────────────────────────────
    st.subheader("주가")
    RANGE_OPTIONS = {
        "상장 이후 전체": "max",
        "10년": "10y",
        "5년":  "5y",
        "1년":  "1y",
        "1개월": "1mo",
    }
    selected_range = st.radio(
        "기간", list(RANGE_OPTIONS.keys()), index=0, horizontal=True, label_visibility="collapsed"
    )
    with st.spinner("주가 데이터 로딩 중..."):
        price_df = fetch_price(ticker, RANGE_OPTIONS[selected_range])

    if price_df.empty:
        st.warning("주가 데이터를 가져올 수 없습니다.")
    else:
        fig = price_chart(price_df, ticker, company_name)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    tab1, tab2, tab3 = st.tabs(["📈 Income Statement", "🏦 Balance Sheet", "💵 Cash Flow"])

    with tab1:
        render_section(data, "income",
            "Revenue vs Gross Profit",        ["Revenue", "Gross Profit"],
            "Operating Income vs Net Income", ["Operating Income", "Net Income"],
            chg_label, x_label, data.get("eps"))

    with tab2:
        render_section(data, "balance",
            "Assets · Liabilities · Equity",  ["Total Assets", "Total Liabilities", "Total Equity"],
            "Cash & Total Debt",               ["Cash & Equiv.", "Total Debt"],
            chg_label, x_label)

    with tab3:
        render_section(data, "cashflow",
            "Operating · Investing · Financing CF", ["Operating CF", "Investing CF", "Financing CF"],
            "Free Cash Flow vs CapEx",              ["Free Cash Flow", "CapEx"],
            chg_label, x_label)

    # ── 밸류에이션 ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📐 Valuation Metrics")

    with st.spinner("밸류에이션 데이터 로딩 중..."):
        val_data = fetch_valuation(ticker)

    if not val_data:
        st.warning("밸류에이션 데이터를 가져올 수 없습니다.")
    else:
        # 시가총액 / 기업가치 요약
        mkt = val_data.get("_market_cap")
        ev  = val_data.get("_ev")
        c1, c2 = st.columns(2)
        if mkt:
            c1.metric("Market Cap", f"${mkt/1e9:,.1f}B")
        if ev:
            c2.metric("Enterprise Value", f"${ev/1e9:,.1f}B")

        valuation_table({ticker: val_data})
