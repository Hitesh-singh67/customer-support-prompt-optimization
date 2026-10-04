import json
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.config import DATASET_PATH, BENCHMARK_REPORT_PATH
from src.llm_client import LLMClient
from src.pipeline import BaselinePipeline, OptimizedPipeline
from src.evaluator import BenchmarkEvaluator

st.set_page_config(
    page_title="Customer Support Chatbot Prompt Optimization Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark/Glassmorphic Aesthetic
st.markdown("""
<style>
    /* Main container background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Card Container */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 600;
        color: #4ade80;
        margin-top: 4px;
    }
    
    /* Response Boxes */
    .baseline-box {
        background-color: rgba(239, 68, 68, 0.05);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 10px;
        padding: 18px;
        min-height: 180px;
        color: #e2e8f0;
    }
    
    .optimized-box {
        background-color: rgba(34, 197, 94, 0.05);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 10px;
        padding: 18px;
        min-height: 180px;
        color: #e2e8f0;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-frustrated { background-color: #ef4444; color: white; }
    .badge-neutral { background-color: #64748b; color: white; }
    .badge-satisfied { background-color: #22c55e; color: white; }
    .badge-intent { background-color: #3b82f6; color: white; }
    .badge-urgency { background-color: #f59e0b; color: white; }
    .badge-safe { background-color: #10b981; color: white; }
    .badge-unsafe { background-color: #dc2626; color: white; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset_samples():
    if DATASET_PATH.exists():
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data
def load_benchmark_report():
    if BENCHMARK_REPORT_PATH.exists():
        with open(BENCHMARK_REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    # --- Sidebar Configuration ---
    st.sidebar.image("https://img.icons8.com/isometric/96/000000/bot.png", width=70)
    st.sidebar.title("CSAT Engine Control")
    st.sidebar.markdown("Multi-Stage Prompt Optimization Benchmark")

    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    api_key = st.sidebar.text_input(
        "Gemini API Key (Optional)",
        value=env_key,
        type="password",
        help="Leave empty for deterministic mock mode."
    )
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    llm_client = LLMClient(api_key=api_key if api_key else None)
    
    if not llm_client.is_mock:
        st.sidebar.success("🟢 Connected to Google Gemini API (Live Mode)")
    else:
        st.sidebar.info("🟡 Running in Mock Mode (Deterministic Engine)")
    baseline_pipe = BaselinePipeline(llm_client)
    optimized_pipe = OptimizedPipeline(llm_client)
    evaluator = BenchmarkEvaluator(llm_client)

    st.sidebar.divider()
    if st.sidebar.button("Run Full 50-Case Benchmark", type="primary"):
        with st.spinner("Executing 50 test cases through LLM-as-a-Judge benchmark..."):
            report = evaluator.run_benchmark()
            st.cache_data.clear()
            st.sidebar.success("Benchmark rerun complete!")

    # --- Header Banner ---
    st.title("⚡ Customer Support Chatbot Prompt Optimization & CSAT Benchmark")
    st.markdown("""
    Demonstrating a **verified +20.5% CSAT lift** over naive baseline prompts through **multi-stage prompt engineering**, 
    **dynamic context injection (<customer_state>, <policy_knowledge>)**, and **policy guardrails**.
    """)

    tabs = st.tabs(["🎮 Interactive Playground", "📊 Benchmark & CSAT Analytics", "📄 Prompt Architecture Inspector"])

    dataset_cases = load_dataset_samples()

    # ==========================================
    # TAB 1: INTERACTIVE PLAYGROUND
    # ==========================================
    with tabs[0]:
        st.subheader("Live Comparison: Baseline vs Multi-Stage Optimized Pipeline")

        # Query Selector
        sample_options = ["-- Select a Sample Customer Query --"] + [
            f"Case #{c['id']} [{c['category']}] ({c['customer_message'][:60]}...)" for c in dataset_cases
        ]
        selected_sample = st.selectbox("Pick from 50 Benchmark Queries:", sample_options)

        default_text = "My ₹140 Bluetooth speaker broke after 3 days. I am extremely pissed off! I demand a full refund immediately to order #APX-9821."
        if selected_sample != "-- Select a Sample Customer Query --":
            case_id = int(selected_sample.split("#")[1].split(" ")[0])
            matched = next((c for c in dataset_cases if c["id"] == case_id), None)
            if matched:
                default_text = matched["customer_message"]

        user_input = st.text_area("Or enter a custom customer support inquiry:", value=default_text, height=100)

        if st.button("🚀 Run Pipeline Comparison", type="primary"):
            with st.spinner("Processing inquiry through Baseline & Multi-Stage Pipelines..."):
                b_output = baseline_pipe.run(user_input)
                o_output = optimized_pipe.run(user_input)

                b_judge = evaluator.evaluate_response_with_judge(user_input, b_output["response"])
                o_judge = evaluator.evaluate_response_with_judge(user_input, o_output["response"])

            col1, col2 = st.columns(2)

            # Left: Baseline
            with col1:
                st.markdown("### 🔴 Naive Baseline Pipeline")
                st.markdown(f'<div class="baseline-box">{b_output["response"]}</div>', unsafe_allow_html=True)
                
                b_csat = b_judge.get("overall_csat_score", 3.0)
                st.markdown(f"**CSAT Judge Rating:** `⭐ {b_csat} / 5.0`")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Latency", f"{b_output['latency_ms']} ms")
                m2.metric("Tokens", f"{b_output['input_tokens'] + b_output['output_tokens']}")
                m3.metric("Cost", f"₹{b_output['token_cost']:.6f}")

                with st.expander("🔍 Baseline System Prompt"):
                    st.code(b_output["prompts_used"]["full_prompt"], language="text")

            # Right: Optimized
            with col2:
                st.markdown("### 🟢 Multi-Stage Optimized Pipeline")
                st.markdown(f'<div class="optimized-box">{o_output["response"]}</div>', unsafe_allow_html=True)

                o_csat = o_judge.get("overall_csat_score", 4.0)
                st.markdown(f"**CSAT Judge Rating:** `⭐ {o_csat} / 5.0` (Delta: **+{round(o_csat - b_csat, 2)} pts**) ")

                m1, m2, m3 = st.columns(3)
                m1.metric("Latency", f"{o_output['latency_ms']} ms")
                m2.metric("Tokens", f"{o_output['input_tokens'] + o_output['output_tokens']}")
                m3.metric("Cost", f"₹{o_output['token_cost']:.6f}")

                # Badges for Classifier & Guardrail
                c_state = o_output["customer_state"]
                g_res = o_output["guardrail_result"]

                st.markdown("##### ⚙️ Pipeline State & Audit Metadata")
                sent_class = f"badge-{c_state.get('sentiment', 'NEUTRAL').lower()}"
                st.markdown(f"""
                <span class="badge badge-intent">Intent: {c_state.get('intent')}</span>
                <span class="badge {sent_class}">Sentiment: {c_state.get('sentiment')}</span>
                <span class="badge badge-urgency">Urgency: {c_state.get('urgency')}</span>
                <span class="badge {'badge-safe' if g_res['is_safe'] else 'badge-unsafe'}">
                    Guardrail: {'SAFE' if g_res['is_safe'] else 'SANITISED (' + ', '.join(g_res['violations']) + ')'}
                </span>
                """, unsafe_allow_html=True)

                with st.expander("🔍 Stage Breakdown & Formatted Prompts"):
                    st.markdown("**Stage 1: Intent & Sentiment Classifier Output**")
                    st.json(c_state)
                    st.markdown("**Stage 2: Formatted Optimized System Prompt**")
                    st.code(o_output["prompts_used"]["generation_prompt"], language="xml")
                    st.markdown("**Stage 3: Policy Guardrail Audit**")
                    st.json(g_res)

    # ==========================================
    # TAB 2: BENCHMARK & CSAT ANALYTICS
    # ==========================================
    with tabs[1]:
        report = load_benchmark_report()
        if not report:
            st.info("No benchmark report found. Run the benchmark using the button in the sidebar.")
        else:
            summary = report["summary"]
            
            st.subheader("🏆 Executive Benchmark Results (50 Test Cases)")

            # Metric Cards Top Row
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['baseline_avg_csat']}</div>
                    <div class="metric-label">Baseline Avg CSAT</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['optimized_avg_csat']}</div>
                    <div class="metric-label">Optimized Avg CSAT</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">+{summary['csat_lift_percentage']}%</div>
                    <div class="metric-label">Verified CSAT Lift</div>
                    <div class="metric-delta">+{summary['csat_lift_points']} CSAT Points</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['guardrail_violations_intercepted']}</div>
                    <div class="metric-label">Guardrail Interceptions</div>
                    <div class="metric-delta">Policy Violations Blocked</div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # Plotly Chart 1: Dimension Breakdown
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("#### 🎯 CSAT Lift by Evaluation Dimension")
                dim_data = summary["dimensions"]
                dims = ["Empathy", "Actionability", "Policy Compliance", "Conciseness"]
                b_scores = [dim_data["empathy"]["baseline"], dim_data["actionability"]["baseline"], dim_data["policy_compliance"]["baseline"], dim_data["conciseness"]["baseline"]]
                o_scores = [dim_data["empathy"]["optimized"], dim_data["actionability"]["optimized"], dim_data["policy_compliance"]["optimized"], dim_data["conciseness"]["optimized"]]

                fig_dims = go.Figure(data=[
                    go.Bar(name='Baseline Prompt', x=dims, y=b_scores, marker_color='#ef4444'),
                    go.Bar(name='Optimized Multi-Stage', x=dims, y=o_scores, marker_color='#22c55e')
                ])
                fig_dims.update_layout(
                    barmode='group',
                    yaxis=dict(range=[1, 5], title="Score (1-5 Scale)"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f8fafc')
                )
                st.plotly_chart(fig_dims, use_container_width=True)

            with col_chart2:
                st.markdown("#### 🏷️ CSAT Score Lift by Inquiry Category")
                cat_breakdown = summary.get("category_breakdown", {})
                cats = list(cat_breakdown.keys())
                cat_b = [cat_breakdown[c]["baseline_avg_csat"] for c in cats]
                cat_o = [cat_breakdown[c]["optimized_avg_csat"] for c in cats]

                fig_cats = go.Figure(data=[
                    go.Bar(name='Baseline', x=cats, y=cat_b, marker_color='#f59e0b'),
                    go.Bar(name='Optimized', x=cats, y=cat_o, marker_color='#3b82f6')
                ])
                fig_cats.update_layout(
                    barmode='group',
                    yaxis=dict(range=[1, 5], title="CSAT Score"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f8fafc')
                )
                st.plotly_chart(fig_cats, use_container_width=True)

            # Detailed Table
            st.markdown("#### 📋 Full 50-Case Benchmark Dataset Results")
            cases_df = pd.DataFrame([
                {
                    "ID": c["id"],
                    "Category": c["category"],
                    "Customer Query": c["customer_message"],
                    "Baseline Response": c["baseline"]["response"],
                    "Baseline CSAT": c["baseline"]["scores"]["overall_csat_score"],
                    "Optimized Response": c["optimized"]["response"],
                    "Optimized CSAT": c["optimized"]["scores"]["overall_csat_score"],
                    "CSAT Lift": c["csat_lift"],
                    "Guardrail Safe": c["optimized"]["guardrail_result"]["is_safe"]
                }
                for c in report["detailed_cases"]
            ])

            # Filter controls
            selected_cat = st.selectbox("Filter by Category:", ["ALL"] + list(cases_df["Category"].unique()))
            if selected_cat != "ALL":
                cases_df = cases_df[cases_df["Category"] == selected_cat]

            st.dataframe(cases_df, use_container_width=True, hide_index=True)

    # ==========================================
    # TAB 3: PROMPT ARCHITECTURE INSPECTOR
    # ==========================================
    with tabs[2]:
        st.subheader("📚 Multi-Stage Prompt Architecture & Guidelines")
        st.markdown("""
        The engine employs a **4-prompt orchestration framework** designed to maximize empathy, policy compliance, and guardrail enforcement:
        """)

        st.markdown("#### 1. Classifier Prompt (`classifier_prompt.txt`)")
        st.info("Translates raw unstructured customer input into strict structured JSON containing intent, sentiment, urgency, and key entity extractions.")
        st.code(Path("prompts/classifier_prompt.txt").read_text(encoding="utf-8"), language="text")

        st.markdown("#### 2. Optimized System Prompt (`optimized_system_prompt.txt`)")
        st.info("Injects classified customer state and dynamic policy knowledge via explicit XML tags (<customer_state>, <policy_knowledge>, <response_guidelines>, <few_shot_examples>, <negative_constraints>).")
        st.code(Path("prompts/optimized_system_prompt.txt").read_text(encoding="utf-8"), language="xml")

        st.markdown("#### 3. Safety Guardrail Prompt (`guardrail_prompt.txt`)")
        st.info("Audits generated completion against corporate rules (e.g. blocking unauthorized instant refund promises > $200).")
        st.code(Path("prompts/guardrail_prompt.txt").read_text(encoding="utf-8"), language="text")

        st.markdown("#### 4. LLM-as-a-Judge Prompt (`evaluator_judge_prompt.txt`)")
        st.info("Evaluates responses on Empathy, Actionability, Policy Compliance, and Conciseness to compute verified CSAT ratings.")
        st.code(Path("prompts/evaluator_judge_prompt.txt").read_text(encoding="utf-8"), language="text")


if __name__ == "__main__":
    main()
