"""
Robotic Chef Platform - Multi-Agent AI System
===============================================
Session 5: The Challenge - Agent-to-Agent (A2A) Integration

This Streamlit app integrates two AI agents:
- Agent 1: Food Analysis Agent (analyses dishes using Recipe MCP Server)
- Agent 2: Robotics Agent (designs robots using Robotics MCP Server)

All LLM calls go through llm_client (local LLM service via requests).

Run with:
    streamlit run app.py
"""

import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

from agents import run_robotic_chef_pipeline, run_budget_nutrition_agent, run_robotics_agent
import llm_client

load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Robotic Chef Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("How It Works")
    st.markdown(
        """
        This platform demonstrates **Agent-to-Agent (A2A)** communication
        between two specialised AI agents:

        **1. Food Analysis Agent**
        - Connects to the Recipe MCP Server
        - Analyses the dish: ingredients, techniques, equipment, safety
        - Produces a structured task specification

        **2. Robotics Designer Agent**
        - Receives the task specification from Agent 1
        - Connects to the Robotics MCP Server
        - Searches component databases
        - Designs a complete robotic cooking platform

        The output of Agent 1 flows directly into Agent 2 -- this is
        the A2A pattern in action.
        """
    )

    st.divider()
    st.header("Example Dishes to Try")
    st.markdown(
        """
        - Pasta Carbonara
        - Cheese Souffle
        - Sushi Rolls
        - Pizza Margherita
        - Beef Stir-Fry
        - Chocolate Cake
        - Fish and Chips
        - Pad Thai
        - French Omelette
        - Artisan Bread
        """
    )

    st.divider()
    st.caption(
        "AI Workshop - Session 5: The Challenge\n\n"
        "University of Hertfordshire"
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("Smart Budget RobotChef")
st.markdown("### The Challenge: Plan a Two-Person Meal within a Budget")
st.markdown(
    """
    Enter your budget, nutrition target, and the system will:
    1. **Pick a dish** that fits your budget and nutrition (Agent 1)
    2. **Design a robot** to cook it (Agent 2)
    3. **Show all reasoning and trade-offs**
    (P.s: I love Gold Color hence I want to win the gold prize :P)
    """
)

# ---------------------------------------------------------------------------
# Check for LLM service connectivity
# ---------------------------------------------------------------------------

llm_url = os.getenv("LLM_SERVICE_URL", "http://localhost:8000")
llm_token = os.getenv("LLM_API_TOKEN", "")

if not llm_token or llm_token == "your-token-here":
    st.warning(
        "**LLM API token not configured.** "
        "Please create a `.env` file in the session5 directory with:\n\n"
        "```\nLLM_SERVICE_URL=http://localhost:8000\nLLM_API_TOKEN=your-token\n```\n\n"
        "Or copy `.env.example` to `.env` and fill in your token."
    )

# ---------------------------------------------------------------------------


# Contest Prompt Input
st.subheader("Enter Full Prompt")
prompt_text = st.text_area(
    "Prompt",
    value="",
    height=80,
    help="Paste the full contest prompt here (budget, people, nutrition, etc.)"
)
run_button = st.button(
    "Run RobotChef Pipeline",
    type="primary",
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------



import re
def parse_prompt_for_fields(prompt):
    # Defaults
    budget = 15.0
    people = 2
    nutrition = "balanced"
    # Budget
    match = re.search(r"£\s?(\d+(?:\.\d{1,2})?)", prompt)
    if match:
        budget = float(match.group(1))
    # People
    match = re.search(r"(\d+)\s*people", prompt, re.IGNORECASE)
    if match:
        people = int(match.group(1))
    # Nutrition
    if re.search(r"high[- ]?protein", prompt, re.IGNORECASE):
        nutrition = "high protein"
    elif re.search(r"vegetarian", prompt, re.IGNORECASE):
        nutrition = "vegetarian"
    elif re.search(r"balanced", prompt, re.IGNORECASE):
        nutrition = "balanced"
    return budget, nutrition, people

if run_button:
    budget, nutrition, people = parse_prompt_for_fields(prompt_text)
    status_container = st.status(
        f"Prompt: {prompt_text}\n→ Parsed as: £{budget}, {people} people, nutrition: {nutrition}", expanded=True
    )
    status_lines = []

    def status_callback(msg: str):
        status_lines.append(msg)
        with status_container:
            st.text(msg)

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        # Step 1: Agent 1 (budget/nutrition)
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                food_analysis = pool.submit(
                    asyncio.run,
                    run_budget_nutrition_agent(budget, nutrition, people, status_callback),
                ).result()
        else:
            food_analysis = asyncio.run(
                run_budget_nutrition_agent(budget, nutrition, people, status_callback)
            )

        status_callback("=== Stage 2: Designing Robot ===")
        # Step 2: Agent 2 (robotics)
        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                robot_design = pool.submit(
                    asyncio.run,
                    run_robotics_agent(food_analysis, status_callback),
                ).result()
        else:
            robot_design = asyncio.run(
                run_robotics_agent(food_analysis, status_callback)
            )

        status_container.update(label="Pipeline complete!", state="complete", expanded=False)
        st.divider()
        with st.expander("Agent 1: Dish Selection & Analysis", expanded=True):
            st.markdown(food_analysis)
        with st.expander("Agent 2: Robot Design", expanded=True):
            st.markdown(robot_design)

    except Exception as e:
        status_container.update(label="Pipeline failed", state="error")
        st.error(f"An error occurred: {e}")
        st.exception(e)
