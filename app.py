"""
app.py
------
Streamlit web app for the Personal Finance AI Agent.

Run locally:   streamlit run app.py

Tabs in the app:
  1. 🤖 AI Assistant      - ask questions, the LangChain agent picks the tool
  2. 🧮 Quick Calculators - EMI / SIP / Budget / Goal calculators with charts (no API key needed)
  3. 📜 History           - all questions asked in this session (downloadable)
  4. ℹ️ About             - how the project works, tools and formulas
"""

from datetime import datetime

import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from agent import MODEL_NAME, TEMPERATURE, build_agent, get_api_key, run_query
from tools import (calculate_budget, calculate_emi, calculate_goal_savings,
                   calculate_sip, format_inr)

# ===========================================================================
# CHART HELPERS (tables with pandas, charts with plotly)
# ===========================================================================
# One colour per meaning, used everywhere in the app
COLORS = {
    "principal": "#4F46E5",   # indigo
    "interest": "#F43F5E",    # rose
    "invested": "#94A3B8",    # slate
    "returns": "#14B8A6",     # teal
    "needs": "#4F46E5",
    "wants": "#F59E0B",       # amber
    "savings": "#14B8A6",
}


def _style(fig, height=360):
    """Common look for every chart."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def donut_chart(labels, values, colors, title):
    """A donut (ring) chart, e.g. Principal vs Interest."""
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.6,
        marker=dict(colors=colors),
        textinfo="percent", hovertemplate="%{label}: ₹%{value:,.0f}<extra></extra>",
    ))
    fig.update_layout(title=title)
    return _style(fig)


# ---------------------------------------------------------------------------
# EMI: year-by-year amortization (how the loan gets paid off)
# ---------------------------------------------------------------------------
def emi_schedule(loan_amount, annual_rate, years):
    """Return a table with principal paid, interest paid and balance for each year."""
    emi = calculate_emi(loan_amount, annual_rate, years)["emi"]
    r = annual_rate / 12 / 100
    balance = loan_amount
    rows = []

    for year in range(1, int(years) + 1):
        principal_paid = 0
        interest_paid = 0
        for _ in range(12):                    # 12 months in each year
            if balance <= 0:
                break
            interest = balance * r
            principal = min(emi - interest, balance)
            balance -= principal
            principal_paid += principal
            interest_paid += interest
        rows.append({
            "Year": year,
            "Principal Paid (₹)": round(principal_paid),
            "Interest Paid (₹)": round(interest_paid),
            "Balance Left (₹)": round(max(balance, 0)),
        })
    return pd.DataFrame(rows)


def emi_schedule_chart(df):
    fig = go.Figure()
    fig.add_bar(x=df["Year"], y=df["Principal Paid (₹)"], name="Principal",
                marker_color=COLORS["principal"])
    fig.add_bar(x=df["Year"], y=df["Interest Paid (₹)"], name="Interest",
                marker_color=COLORS["interest"])
    fig.add_scatter(x=df["Year"], y=df["Balance Left (₹)"], name="Balance Left",
                    mode="lines+markers", yaxis="y2", line=dict(color="#0F172A", width=2))
    fig.update_layout(
        barmode="stack", title="Year-wise Repayment",
        xaxis_title="Year", yaxis_title="Paid in the year (₹)",
        yaxis2=dict(title="Balance (₹)", overlaying="y", side="right", showgrid=False,
                    rangemode="tozero"),
    )
    return _style(fig, 400)


# ---------------------------------------------------------------------------
# SIP: growth of investment year by year
# ---------------------------------------------------------------------------
def sip_growth_table(monthly, annual_rate, years):
    """Value of the SIP at the end of every year (uses the same FV formula)."""
    r = annual_rate / 12 / 100
    rows = []
    for year in range(1, int(years) + 1):
        n = year * 12
        value = monthly * n if r == 0 else monthly * ((1 + r) ** n - 1) / r
        invested = monthly * n
        rows.append({
            "Year": year,
            "Invested (₹)": round(invested),
            "Value (₹)": round(value),
            "Returns (₹)": round(value - invested),
        })
    return pd.DataFrame(rows)


def sip_growth_chart(df, title="Investment Growth"):
    fig = go.Figure()
    fig.add_scatter(x=df["Year"], y=df["Invested (₹)"], name="Amount Invested",
                    fill="tozeroy", line=dict(color=COLORS["invested"]))
    fig.add_scatter(x=df["Year"], y=df["Value (₹)"], name="Total Value",
                    fill="tonexty", line=dict(color=COLORS["returns"], width=3))
    fig.update_layout(title=title, xaxis_title="Year", yaxis_title="₹")
    return _style(fig, 400)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
def budget_breakdown_table(income, needs_pct, wants_pct, savings_pct):
    """Suggested spending items inside each bucket (simple example split)."""
    # Each bucket is divided into example items (shares add up to 1.0 per bucket)
    plan = {
        "Needs": [("Rent / Home EMI", 0.50), ("Groceries", 0.20), ("Utilities & Bills", 0.10),
                  ("Transport", 0.10), ("Insurance", 0.10)],
        "Wants": [("Dining Out", 0.30), ("Shopping", 0.30), ("Travel", 0.20),
                  ("Entertainment & OTT", 0.20)],
        "Savings": [("SIP / Mutual Funds", 0.50), ("Emergency Fund", 0.30),
                    ("Other Investments (PPF/FD)", 0.20)],
    }
    pct = {"Needs": needs_pct, "Wants": wants_pct, "Savings": savings_pct}
    rows = []
    for bucket, items in plan.items():
        bucket_amount = income * pct[bucket] / 100
        for item, share in items:
            rows.append({"Category": bucket, "Item": item,
                         "Amount (₹)": round(bucket_amount * share)})
    return pd.DataFrame(rows)


def budget_sunburst(df):
    color_map = {"Needs": COLORS["needs"], "Wants": COLORS["wants"], "Savings": COLORS["savings"]}
    fig = px.sunburst(df, path=["Category", "Item"], values="Amount (₹)",
                      color="Category", color_discrete_map=color_map,
                      title="Where Your Money Goes")
    return _style(fig, 420)


# ---------------------------------------------------------------------------
# Goal savings
# ---------------------------------------------------------------------------
def goal_rate_comparison(target, years, rates=(6, 8, 10, 12, 15)):
    """How the required monthly saving changes with different return rates."""
    rows = []
    for rate in rates:
        g = calculate_goal_savings(target, years, rate)
        rows.append({"Return Rate (%)": f"{rate}%",
                     "Monthly Saving (₹)": round(g["monthly_saving"]),
                     "Total Invested (₹)": round(g["total_invested"])})
    return pd.DataFrame(rows)


def goal_rate_chart(df):
    fig = px.bar(df, x="Return Rate (%)", y="Monthly Saving (₹)", text="Monthly Saving (₹)",
                 title="Monthly Saving Needed at Different Returns",
                 color_discrete_sequence=[COLORS["principal"]])
    fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
    fig.update_xaxes(type="category")   # show "6%", "8%"... as labels, not numbers
    return _style(fig, 380)



# ---------------------------------------------------------------------------
# Page settings (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Personal Finance AI Agent", page_icon="💰", layout="wide",
                   initial_sidebar_state="collapsed")   # no left panel in this app

# ---------------------------------------------------------------------------
# Custom CSS to make the app look nicer
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #14B8A6 100%);
    padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 2.1rem; color: white; }
.hero p  { margin: 6px 0 0 0; opacity: 0.92; font-size: 1.05rem; }
.tool-badge {
    display: inline-block; padding: 4px 12px; margin: 2px 6px 2px 0; border-radius: 999px;
    background: #14B8A6; color: white; font-size: 0.85rem; font-weight: 600;
}
/* App accent colour (indigo) for buttons, sliders and tabs */
button[data-testid="stBaseButton-primaryFormSubmit"], button[data-testid="stBaseButton-primary"] {
    background-color: #4F46E5 !important; border-color: #4F46E5 !important; color: white !important;
}
button[data-testid="stBaseButton-primaryFormSubmit"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    background-color: #4338CA !important; border-color: #4338CA !important;
}
/* Sliders: turn Streamlit's default red into indigo (greys are not affected) */
div[data-testid="stSlider"] div[role="group"] { filter: hue-rotate(245deg) saturate(0.9); }
/* Tabs: selected tab text and underline in indigo */
div[data-testid="stTab"][aria-selected="true"] p { color: #4F46E5 !important; }
div[data-testid="stTab"] .react-aria-SelectionIndicator { background-color: #4F46E5 !important; }
.sample-title { font-weight: 600; margin: 10px 0 4px 0; opacity: 0.8; }
div[data-testid="stMetric"] {
    background: rgba(148,163,184,0.10); border: 1px solid rgba(148,163,184,0.25);
    padding: 12px 16px; border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sample queries shown below the search bar
# (the first 5 are the required test queries from the assignment)
# ---------------------------------------------------------------------------
SAMPLE_QUERIES = [
    ("🏠 EMI", "Calculate EMI for ₹10 lakh loan at 8% interest for 20 years"),
    ("📈 SIP", "If I invest ₹7000 monthly for 15 years at 10% return, how much will I have?"),
    ("🧾 Budget", "Create a budget plan for income ₹90,000"),
    ("➗ Maths", "What is 20% of ₹85,000?"),
    ("💹 SIP", "How much will I get if I invest ₹3000 per month for 5 years at 8% return?"),
    ("🎯 Goal", "How much should I save monthly to reach ₹20 lakh in 10 years?"),
    ("🚗 Car Loan", "What will be my EMI for a ₹6 lakh car loan at 9.5% for 5 years?"),
    ("👶 Goal", "I need ₹50 lakh for my child's education in 15 years. How much to invest monthly at 11%?"),
]

TOOL_LABELS = {
    "calculator": "🧮 Calculator",
    "emi_calculator": "🏠 EMI Calculator",
    "sip_calculator": "📈 SIP Calculator",
    "budget_planner": "🧾 Budget Planner",
    "goal_savings_planner": "🎯 Goal Savings Planner",
}

# ---------------------------------------------------------------------------
# Session state = memory of the app while the browser tab is open
# ---------------------------------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []          # list of {question, answer, tool_steps, time}
if "query_box" not in st.session_state:
    st.session_state.query_box = ""


def use_sample(query):
    """Called when a sample query button is clicked."""
    st.session_state.query_box = query
    st.session_state.run_now = True


@st.cache_resource(show_spinner=False)
def get_agent(api_key, model_name, temperature):
    """Build the agent once and reuse it (cached) to keep the app fast."""
    return build_agent(api_key, model_name, temperature)


# ===========================================================================
# FIXED SETTINGS (users cannot change these)
# The API key comes from .env (local) or Streamlit Secrets (cloud), never typed in the app.
# Model name and temperature are set in agent.py.
# ===========================================================================
api_key = get_api_key()
REMEMBER_CHAT = True         # lets users ask follow-ups like "what if the rate is 7%?"
SHOW_STEPS = True            # show which tool the agent called and with what inputs
SHOW_CHARTS = True           # draw a chart under each answer


# ===========================================================================
# HEADER
# ===========================================================================
st.markdown("""
<div class="hero">
  <h1>💰 Personal Finance AI Agent</h1>
  <p>Ask about EMIs, SIP returns, budgets or savings goals. The AI picks the right tool and calculates the answer exactly.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SESSION STATS on the main page
# ---------------------------------------------------------------------------
all_tools_used = [s["tool"] for c in st.session_state.chat for s in c["tool_steps"]]
most_used = (TOOL_LABELS.get(max(set(all_tools_used), key=all_tools_used.count), "-")
             if all_tools_used else "-")

s1, s2, s3, s4, s5 = st.columns([1, 1, 1.6, 1.2, 1])
s1.metric("💬 Questions", len(st.session_state.chat))
s2.metric("🛠️ Tool calls", len(all_tools_used))
s3.metric("🏆 Most used tool", most_used)
s4.metric("🧠 AI model", MODEL_NAME)
with s5:
    st.write("")   # small spacer so the button lines up with the cards
    if st.button("🗑️ Clear chat", width="stretch"):
        st.session_state.chat = []
        st.session_state.query_box = ""
        st.rerun()

# Warn the app owner if the key was not added to Streamlit Secrets
if not api_key:
    st.warning("🔑 OpenAI API key not found. Add `OPENAI_API_KEY` to your `.env` file "
               "(or Streamlit Secrets when deployed). Quick Calculators still work without it.")

tab_ai, tab_calc, tab_history, tab_about = st.tabs(
    ["🤖 AI Assistant", "🧮 Quick Calculators", "📜 History", "ℹ️ About"]
)


# ---------------------------------------------------------------------------
# Helper: draw a small chart based on the tool the agent used
# ---------------------------------------------------------------------------
def show_chart_for_step(step, key):
    """Use the same inputs the agent sent to the tool to draw a chart."""
    args = step["input"]
    try:
        if step["tool"] == "emi_calculator":
            r = calculate_emi(args["loan_amount"], args["annual_interest_rate"], args["years"])
            fig = donut_chart(["Principal", "Interest"],
                                     [args["loan_amount"], r["total_interest"]],
                                     [COLORS["principal"], COLORS["interest"]],
                                     "Principal vs Interest")
        elif step["tool"] == "sip_calculator":
            df = sip_growth_table(args["monthly_investment"], args["annual_return_rate"], args["years"])
            fig = sip_growth_chart(df)
        elif step["tool"] == "budget_planner":
            df = budget_breakdown_table(args["monthly_income"], 50, 30, 20)
            fig = budget_sunburst(df)
        elif step["tool"] == "goal_savings_planner":
            rate = args.get("annual_return_rate", 12.0)
            g = calculate_goal_savings(args["target_amount"], args["years"], rate)
            df = sip_growth_table(g["monthly_saving"], rate, args["years"])
            fig = sip_growth_chart(df, "Path to Your Goal")
        else:
            return   # calculator has no chart
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:
        pass   # if anything is missing, just skip the chart


def show_answer(item, idx):
    """Display one question + answer block."""
    st.markdown(f"**🙋 You asked:** {item['question']}")
    with st.container(border=True):     # a neat box around the answer
        st.markdown("**🤖 Agent's answer:**")
        st.markdown(item["answer"])

    if item["tool_steps"]:
        badges = "".join(f'<span class="tool-badge">{TOOL_LABELS.get(s["tool"], s["tool"])}</span>'
                         for s in item["tool_steps"])
        st.markdown(f"<br>**Tools used by the agent:** {badges}", unsafe_allow_html=True)
    else:
        st.caption("No tool was used for this answer.")

    if SHOW_STEPS and item["tool_steps"]:
        with st.expander("🔍 See how the agent solved it"):
            for n, step in enumerate(item["tool_steps"], start=1):
                st.markdown(f"**Step {n}: called `{step['tool']}`**")
                st.markdown("Input the LLM sent to the tool:")
                st.json(step["input"])
                st.markdown("Output returned by the tool:")
                st.code(step["output"], language=None)

    if SHOW_CHARTS:
        for n, step in enumerate(item["tool_steps"]):
            show_chart_for_step(step, key=f"chart_{idx}_{n}")


# ===========================================================================
# TAB 1: AI ASSISTANT
# ===========================================================================
with tab_ai:
    # --- Search bar ---
    with st.form("ask_form", clear_on_submit=False):
        col_input, col_btn = st.columns([6, 1])
        col_input.text_input("Ask a question", key="query_box", label_visibility="collapsed",
                             placeholder="e.g. Calculate EMI for ₹10 lakh loan at 9% for 20 years")
        submitted = col_btn.form_submit_button("🔍 Ask", width="stretch", type="primary")

    # --- Sample queries below the search bar ---
    st.markdown('<div class="sample-title">💡 Try a sample question (click to run):</div>',
                unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (label, query) in enumerate(SAMPLE_QUERIES):
        cols[i % 2].button(f"{label} · {query}", key=f"sample_{i}", width="stretch",
                           on_click=use_sample, args=(query,))

    # --- Run the agent ---
    run_now = st.session_state.pop("run_now", False)
    if submitted or run_now:
        question = st.session_state.query_box.strip()
        if not question:
            st.warning("Please type a question first.")
        elif not api_key:
            st.error("🔑 The OpenAI API key is missing. Add it to your .env file or Streamlit Secrets.")
        else:
            # Build earlier conversation for follow-up questions
            history = []
            if REMEMBER_CHAT:
                for old in st.session_state.chat[-5:]:   # last 5 Q&As are enough
                    history.append({"role": "user", "content": old["question"]})
                    history.append({"role": "assistant", "content": old["answer"]})

            success = False
            with st.spinner("🤖 The agent is choosing a tool and calculating..."):
                try:
                    agent = get_agent(api_key, MODEL_NAME, TEMPERATURE)
                    result = run_query(agent, question, history)
                    st.session_state.chat.append({
                        "question": question,
                        "answer": result["answer"],
                        "tool_steps": result["tool_steps"],
                        "time": datetime.now().strftime("%H:%M:%S"),
                    })
                    success = True
                except Exception as error:
                    st.error(f"Something went wrong: {error}")
                    st.info("Check that your API key is correct and has credit.")
            if success:
                st.rerun()   # refresh the page so the session stats update too

    # --- Show answers (newest first) ---
    if st.session_state.chat:
        st.markdown("---")
        latest_idx = len(st.session_state.chat) - 1
        show_answer(st.session_state.chat[latest_idx], latest_idx)

        if len(st.session_state.chat) > 1:
            st.markdown("---")
            st.markdown("#### 🕘 Earlier in this conversation")
            for idx in range(latest_idx - 1, -1, -1):
                item = st.session_state.chat[idx]
                with st.expander(f"{item['time']} · {item['question']}"):
                    show_answer(item, idx)
    else:
        st.info("👆 Type a question or click a sample question to get started.")


# ===========================================================================
# TAB 2: QUICK CALCULATORS (work without any API key)
# ===========================================================================
with tab_calc:
    st.caption("These calculators use the same tool functions the AI agent uses, with interactive ")
    emi_tab, sip_tab, budget_tab, goal_tab = st.tabs(
        ["🏠 EMI Calculator", "📈 SIP Calculator", "🧾 Budget Planner", "🎯 Goal Planner"]
    )

    # ---------------- EMI ----------------
    with emi_tab:
        c1, c2, c3 = st.columns(3)
        loan = c1.number_input("Loan amount (₹)", 10000, 100000000, 1000000, step=50000)
        rate = c2.slider("Interest rate (% per year)", 1.0, 20.0, 9.0, 0.1)
        years = c3.slider("Tenure (years)", 1, 30, 20)

        r = calculate_emi(loan, rate, years)
        m1, m2, m3 = st.columns(3)
        m1.metric("Monthly EMI", format_inr(r["emi"]))
        m2.metric("Total Interest", format_inr(r["total_interest"]))
        m3.metric("Total Payment", format_inr(r["total_payment"]))

        g1, g2 = st.columns([1, 2])
        g1.plotly_chart(donut_chart(["Principal", "Interest"], [loan, r["total_interest"]],
                                           [COLORS["principal"], COLORS["interest"]],
                                           "Principal vs Interest"), width="stretch")
        schedule = emi_schedule(loan, rate, years)
        g2.plotly_chart(emi_schedule_chart(schedule), width="stretch")

        with st.expander("📋 Year-wise repayment table"):
            st.dataframe(schedule, hide_index=True, width="stretch")
            st.download_button("⬇️ Download schedule (CSV)", schedule.to_csv(index=False),
                               "emi_schedule.csv", "text/csv")

    # ---------------- SIP ----------------
    with sip_tab:
        c1, c2, c3 = st.columns(3)
        monthly = c1.number_input("Monthly investment (₹)", 500, 1000000, 5000, step=500)
        sip_rate = c2.slider("Expected return (% per year)", 1.0, 30.0, 12.0, 0.5)
        sip_years = c3.slider("Investment period (years)", 1, 40, 10)

        s = calculate_sip(monthly, sip_rate, sip_years)
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Invested", format_inr(s["total_invested"]))
        m2.metric("Estimated Returns", format_inr(s["returns_earned"]))
        m3.metric("Future Value", format_inr(s["future_value"]))

        growth = sip_growth_table(monthly, sip_rate, sip_years)
        st.plotly_chart(sip_growth_chart(growth), width="stretch")
        with st.expander("📋 Year-wise growth table"):
            st.dataframe(growth, hide_index=True, width="stretch")
            st.download_button("⬇️ Download growth table (CSV)", growth.to_csv(index=False),
                               "sip_growth.csv", "text/csv")

    # ---------------- BUDGET ----------------
    with budget_tab:
        income = st.number_input("Monthly income (₹)", 5000, 10000000, 80000, step=5000)
        st.markdown("**Adjust the split** (default is the 50-30-20 rule):")
        c1, c2, c3 = st.columns(3)
        needs = c1.slider("Needs %", 0, 100, 50)
        wants = c2.slider("Wants %", 0, 100, 30)
        savings = c3.slider("Savings %", 0, 100, 20)

        if needs + wants + savings != 100:
            st.warning(f"Your percentages add up to {needs + wants + savings}%. Please make them total 100%.")
        else:
            b = calculate_budget(income, needs, wants, savings)
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Needs ({needs}%)", format_inr(b["needs"]))
            m2.metric(f"Wants ({wants}%)", format_inr(b["wants"]))
            m3.metric(f"Savings ({savings}%)", format_inr(b["savings"]))

            breakdown = budget_breakdown_table(income, needs, wants, savings)
            g1, g2 = st.columns([3, 2])
            g1.plotly_chart(budget_sunburst(breakdown), width="stretch")
            g2.markdown("**Suggested breakdown**")
            g2.dataframe(breakdown, hide_index=True, width="stretch", height=420)

            yearly_savings = b["savings"] * 12
            st.success(f"💡 Saving {format_inr(b['savings'])} every month = "
                       f"{format_inr(yearly_savings)} per year. Investing it as a SIP at 12% for 10 years "
                       f"could grow to about {format_inr(calculate_sip(b['savings'], 12, 10)['future_value'])}.")

    # ---------------- GOAL ----------------
    with goal_tab:
        c1, c2, c3 = st.columns(3)
        target = c1.number_input("Target amount (₹)", 10000, 1000000000, 2000000, step=100000)
        goal_years = c2.slider("Years to reach goal", 1, 40, 10)
        goal_rate = c3.slider("Expected return (% per year) ", 0.0, 30.0, 12.0, 0.5)

        g = calculate_goal_savings(target, goal_years, goal_rate)
        m1, m2, m3 = st.columns(3)
        m1.metric("Save every month", format_inr(g["monthly_saving"]))
        m2.metric("Total you invest", format_inr(g["total_invested"]))
        m3.metric("Returns earned", format_inr(g["returns_earned"]))

        g1, g2 = st.columns(2)
        path = sip_growth_table(g["monthly_saving"], goal_rate, goal_years)
        g1.plotly_chart(sip_growth_chart(path, "Path to Your Goal"), width="stretch")
        comparison = goal_rate_comparison(target, goal_years)
        g2.plotly_chart(goal_rate_chart(comparison), width="stretch")


# ===========================================================================
# TAB 3: HISTORY
# ===========================================================================
with tab_history:
    if not st.session_state.chat:
        st.info("No questions asked yet. Your questions and the tools used will appear here.")
    else:
        table = pd.DataFrame([{
            "Time": c["time"],
            "Question": c["question"],
            "Tools Used": ", ".join(s["tool"] for s in c["tool_steps"]) or "None",
            "Answer": c["answer"],
        } for c in st.session_state.chat])

        # Small chart: how often each tool was used
        tool_counts = pd.Series(
            [s["tool"] for c in st.session_state.chat for s in c["tool_steps"]]
        ).value_counts()
        c1, c2 = st.columns([2, 1])
        c1.dataframe(table, hide_index=True, width="stretch")
        if not tool_counts.empty:
            c2.markdown("**Tool usage**")
            c2.bar_chart(tool_counts, color="#14B8A6")

        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download history (CSV)", table.to_csv(index=False),
                           "finance_agent_history.csv", "text/csv", width="stretch")
        text_log = "\n\n".join(f"[{c['time']}] Q: {c['question']}\nA: {c['answer']}"
                               for c in st.session_state.chat)
        d2.download_button("⬇️ Download chat (TXT)", text_log, "finance_agent_chat.txt",
                           "text/plain", width="stretch")


# ===========================================================================
# TAB 4: ABOUT
# ===========================================================================
with tab_about:
    st.markdown("### 🧠 How the agent works")
    st.markdown("""
1. **You ask** a question in plain English.
2. **The LLM (GPT-4o-mini)** reads the question and the description of every tool.
3. **It decides** which tool fits best and extracts the numbers (e.g. 10 lakh → 1000000).
4. **The tool** does the exact calculation in Python, with no guessing by the LLM.
5. **The LLM** turns the tool's result into a friendly answer.
""")
    st.code("User Question  →  LLM (decides)  →  Tool (calculates)  →  LLM (explains)  →  Answer",
            language=None)

    st.markdown("### 🛠️ Tools available to the agent")
    st.dataframe(pd.DataFrame([
        {"Tool": "🧮 calculator", "Purpose": "General maths & percentages"},
        {"Tool": "🏠 emi_calculator", "Purpose": "Monthly EMI of a loan"},
        {"Tool": "📈 sip_calculator", "Purpose": "Future value of monthly investments"},
        {"Tool": "🧾 budget_planner", "Purpose": "50-30-20 budget split"},
        {"Tool": "🎯 goal_savings_planner", "Purpose": "Monthly saving needed for a target (bonus)"},
    ]), hide_index=True, width="stretch")

    st.markdown("### 📐 Formulas used")
    st.markdown("**EMI** (P = loan, r = monthly rate, n = months)")
    st.latex(r"EMI = \frac{P \times r \times (1+r)^n}{(1+r)^n - 1}")
    st.markdown("**SIP Future Value** (P = monthly investment, r = monthly return, n = months)")
    st.latex(r"FV = P \times \frac{(1+r)^n - 1}{r}")
    st.markdown("**Goal Savings** (reverse of the SIP formula)")
    st.latex(r"P = \frac{FV \times r}{(1+r)^n - 1}")
    st.markdown("**Budget**: 50% Needs · 30% Wants · 20% Savings")

    st.markdown("### 🧰 Tech stack")
    st.markdown("LangChain · OpenAI GPT-4o-mini · Streamlit · Plotly · Pandas")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("⚠️ For learning purposes only. Not financial advice. · Built with LangChain, GPT-4o-mini & Streamlit")
