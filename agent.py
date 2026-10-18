import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from tools import ALL_TOOLS

load_dotenv()

# agent LLM setting
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0


def get_api_key():
    """Return the OpenAI API key from the environment ('' if it is missing)."""
    return os.getenv("OPENAI_API_KEY", "").strip()

# Instructions given to the LLM before every conversation
SYSTEM_PROMPT = """You are a friendly Personal Finance Assistant for Indian users.

Rules:
1. ALWAYS use a tool for any calculation. Never do maths in your head.
2. Pick the right tool:
   - calculator           -> general maths / percentages
   - emi_calculator       -> loan EMI
   - sip_calculator       -> future value of monthly investments
   - budget_planner       -> monthly budget from income (50-30-20 rule)
   - goal_savings_planner -> monthly saving needed to reach a target amount
3. Convert Indian units before calling tools: 1 lakh = 100000, 1 crore = 10000000.
4. If a value is missing (e.g. return rate), assume a sensible default
   (12% for investments) and clearly tell the user you assumed it.
5. Present the answer neatly with bullet points and ₹ amounts, then add one
   short, practical tip. Keep it concise.
6. If the question is not about personal finance, politely say you can only
   help with personal finance topics.
"""


def build_agent(api_key, model_name=MODEL_NAME, temperature=TEMPERATURE):
    """Create and return the finance agent."""
    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)
    agent = create_agent(model=llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
    return agent


def run_query(agent, question, chat_history=None):
    """
    Send a question to the agent.

    chat_history: optional list of earlier messages like
                  [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                  (lets the agent remember the conversation)

    Returns a dictionary:
        answer     -> final text answer from the agent
        tool_steps -> list of tools used, with their inputs and outputs
    """
    messages = list(chat_history or [])
    messages.append({"role": "user", "content": question})

    result = agent.invoke({"messages": messages})
    all_messages = result["messages"]

    # Only look at messages created for THIS question (skip old history)
    new_messages = all_messages[len(messages):]

    tool_steps = []
    tool_outputs = {}

    # Collect tool outputs, matched by their tool_call_id
    for msg in new_messages:
        if isinstance(msg, ToolMessage):
            tool_outputs[msg.tool_call_id] = msg.content

    # Collect which tools the LLM decided to call and with which inputs
    for msg in new_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                tool_steps.append({
                    "tool": call["name"],
                    "input": call["args"],
                    "output": tool_outputs.get(call["id"], ""),
                })

    final_answer = all_messages[-1].content
    return {"answer": final_answer, "tool_steps": tool_steps}
