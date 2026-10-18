import math
import re

from langchain_core.tools import tool


# Helper: convert amounts like "1 crore", "10 lakh", "₹90,000", "50k" into numbers.
# We do this in Python (not the LLM) because LLMs sometimes add or drop a zero when converting lakh/crore by themselves.

UNIT_VALUES = {
    "crore": 10000000, "crores": 10000000, "cr": 10000000,
    "lakh": 100000, "lakhs": 100000, "lac": 100000, "lacs": 100000, "l": 100000,
    "thousand": 1000, "k": 1000,
}


def parse_amount(value):
    """Turn '1 crore' -> 10000000, '10 lakh' -> 1000000, '₹90,000' -> 90000, 5000 -> 5000."""
    if isinstance(value, (int, float)):          # already a number
        return float(value)

    text = str(value).lower()
    for junk in ["₹", "rs.", "rs", "inr", ",", "rupees", "rupee"]:
        text = text.replace(junk, "")
    text = text.strip()

    # e.g. "1.5 crore"  ->  number = 1.5, unit = "crore"
    match = re.fullmatch(r"([\d.]+)\s*([a-z]*)", text)
    if not match:
        raise ValueError(f"Could not understand the amount '{value}'.")

    number = float(match.group(1))
    unit = match.group(2)
    if unit == "":
        return number
    if unit not in UNIT_VALUES:
        raise ValueError(f"Unknown unit '{unit}' in amount '{value}'.")
    return number * UNIT_VALUES[unit]


# Helper: format numbers the Indian way  ->  ₹10,00,000.00

def format_inr(amount):
    """Convert a number into Indian currency format, e.g. 1000000 -> ₹10,00,000.00"""
    negative = amount < 0
    amount = abs(round(amount, 2))

    rupees = int(amount)
    paise = round((amount - rupees) * 100)
    if paise == 100:          # handles rounding like 9.999 -> 10.00
        rupees += 1
        paise = 0

    text = str(rupees)
    # The last 3 digits stay together, the rest are grouped in 2s (Indian system)
    if len(text) > 3:
        last_three = text[-3:]
        rest = text[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        text = ",".join(groups) + "," + last_three

    return ("-" if negative else "") + f"₹{text}.{paise:02d}"



# PART 1: Plain Python functions (the actual maths)

def calculate_emi(loan_amount, annual_interest_rate, years):
    """
    EMI = P × r × (1+r)^n / ((1+r)^n - 1)
      P = loan amount, r = monthly interest rate, n = number of months
    """
    P = loan_amount
    r = annual_interest_rate / 12 / 100   # yearly % -> monthly decimal
    n = int(years * 12)                   # years -> months

    if r == 0:                            # 0% interest: just divide equally
        emi = P / n
    else:
        emi = P * r * (1 + r) ** n / ((1 + r) ** n - 1)

    total_payment = emi * n
    total_interest = total_payment - P

    return {
        "emi": emi,
        "months": n,
        "total_payment": total_payment,
        "total_interest": total_interest,
    }


def calculate_sip(monthly_investment, annual_return_rate, years):
    """
    FV = P × ((1+r)^n − 1) / r
      P = monthly investment, r = monthly return, n = total months
    """
    P = monthly_investment
    r = annual_return_rate / 12 / 100
    n = int(years * 12)

    if r == 0:
        future_value = P * n
    else:
        future_value = P * ((1 + r) ** n - 1) / r

    total_invested = P * n
    returns_earned = future_value - total_invested

    return {
        "future_value": future_value,
        "total_invested": total_invested,
        "returns_earned": returns_earned,
        "months": n,
    }


def calculate_budget(monthly_income, needs_pct=50, wants_pct=30, savings_pct=20):
    """Split income using the 50-30-20 rule (percentages can be changed)."""
    return {
        "needs": monthly_income * needs_pct / 100,
        "wants": monthly_income * wants_pct / 100,
        "savings": monthly_income * savings_pct / 100,
        "needs_pct": needs_pct,
        "wants_pct": wants_pct,
        "savings_pct": savings_pct,
    }


def calculate_goal_savings(target_amount, years, annual_return_rate=12.0):
    """
    Reverse of the SIP formula: how much to invest monthly to reach a target.
      P = FV × r / ((1+r)^n − 1)
    """
    FV = target_amount
    r = annual_return_rate / 12 / 100
    n = int(years * 12)

    if r == 0:
        monthly_saving = FV / n
    else:
        monthly_saving = FV * r / ((1 + r) ** n - 1)

    total_invested = monthly_saving * n
    returns_earned = FV - total_invested

    return {
        "monthly_saving": monthly_saving,
        "total_invested": total_invested,
        "returns_earned": returns_earned,
        "months": n,
    }


def safe_calculate(expression):
    """
    Safely evaluate a maths expression like '0.20 * 85000' or '(5+3)**2'.
    Only numbers, + - * / ( ) . % ** and a few maths functions are allowed,
    so nobody can run dangerous code through the calculator.
    """
    allowed_names = {
        "sqrt": math.sqrt, "pow": pow, "round": round, "abs": abs,
        "log": math.log, "exp": math.exp, "pi": math.pi,
    }
    # Remove things users commonly type that Python doesn't understand
    cleaned = expression.replace("₹", "").replace(",", "").replace("^", "**")
    cleaned = cleaned.replace("×", "*").replace("÷", "/")

    # Block anything that is not a number, operator or allowed function name
    check = cleaned
    for name in allowed_names:
        check = check.replace(name, "")
    for ch in check:
        if ch not in "0123456789+-*/().% ":
            raise ValueError(f"Character '{ch}' is not allowed in the calculator.")

    return eval(cleaned, {"__builtins__": {}}, allowed_names)



# LangChain tools (wrappers the AI agent can call)
# The docstring tells the LLM WHEN to use each tool.


@tool
def calculator(expression: str) -> str:
    """Use this for GENERAL maths calculations such as percentages, addition,
    multiplication or division. Input must be a valid maths expression.
    Example: 'What is 20% of 85000?' -> expression '0.20 * 85000'.
    Do NOT use this for EMI, SIP, budget or goal savings questions."""
    try:
        result = safe_calculate(expression)
        return f"Result of {expression} = {result:,.2f}"
    except Exception as error:
        return f"Could not calculate '{expression}'. Error: {error}"


@tool
def emi_calculator(loan_amount: float, annual_interest_rate: float, years: float) -> str:
    """Use this to calculate the monthly EMI of a LOAN (home loan, car loan,
    personal loan). Inputs: loan_amount in rupees (1 lakh = 100000),
    annual_interest_rate in percent (e.g. 9 for 9%), years = loan tenure."""
    result = calculate_emi(loan_amount, annual_interest_rate, years)
    return (
        f"Loan Amount: {format_inr(loan_amount)}\n"
        f"Interest Rate: {annual_interest_rate}% per year\n"
        f"Tenure: {years} years ({result['months']} months)\n"
        f"Monthly EMI: {format_inr(result['emi'])}\n"
        f"Total Interest Payable: {format_inr(result['total_interest'])}\n"
        f"Total Amount Payable: {format_inr(result['total_payment'])}"
    )


@tool
def sip_calculator(monthly_investment: float, annual_return_rate: float, years: float) -> str:
    """Use this to find the FUTURE VALUE of a fixed MONTHLY investment (SIP).
    Use when the user says 'If I invest X per month for Y years at Z% return'.
    Inputs: monthly_investment in rupees, annual_return_rate in percent,
    years = investment period."""
    result = calculate_sip(monthly_investment, annual_return_rate, years)
    return (
        f"Monthly Investment: {format_inr(monthly_investment)}\n"
        f"Expected Return: {annual_return_rate}% per year\n"
        f"Period: {years} years ({result['months']} months)\n"
        f"Total Amount Invested: {format_inr(result['total_invested'])}\n"
        f"Estimated Returns: {format_inr(result['returns_earned'])}\n"
        f"Future Value: {format_inr(result['future_value'])}"
    )


@tool
def budget_planner(monthly_income: float) -> str:
    """Use this to create a MONTHLY BUDGET plan from a person's income using
    the 50-30-20 rule (50% Needs, 30% Wants, 20% Savings).
    Input: monthly_income in rupees."""
    b = calculate_budget(monthly_income)
    return (
        f"Monthly Income: {format_inr(monthly_income)}\n"
        f"Needs (50%): {format_inr(b['needs'])} - rent, groceries, bills, EMIs, transport\n"
        f"Wants (30%): {format_inr(b['wants'])} - dining out, shopping, travel, entertainment\n"
        f"Savings (20%): {format_inr(b['savings'])} - SIPs, emergency fund, investments"
    )


@tool
def goal_savings_planner(target_amount: float, years: float, annual_return_rate: float = 12.0) -> str:
    """Use this when the user wants to know HOW MUCH TO SAVE / INVEST EVERY MONTH
    to reach a TARGET amount (a goal) in a given number of years.
    Example: 'How much should I save monthly to reach 20 lakh in 10 years?'
    Inputs: target_amount in rupees, years, annual_return_rate in percent
    (use 12 if the user does not mention a return rate)."""
    g = calculate_goal_savings(target_amount, years, annual_return_rate)
    return (
        f"Target Amount: {format_inr(target_amount)}\n"
        f"Time Period: {years} years ({g['months']} months)\n"
        f"Assumed Return: {annual_return_rate}% per year\n"
        f"Required Monthly Saving: {format_inr(g['monthly_saving'])}\n"
        f"Total You Will Invest: {format_inr(g['total_invested'])}\n"
        f"Returns Earned: {format_inr(g['returns_earned'])}"
    )


# A simple list so other files can import all tools at once
ALL_TOOLS = [calculator, emi_calculator, sip_calculator, budget_planner, goal_savings_planner]
