# credit_score.py
"""
Credit Score Module
--------------------------------------
This module provides a simple dummy credit score calculator
based on:
- Monthly income
- Sector (salaried / self employed / others)
- Age (calculated from date of birth)

The output score is mapped to a 300–900 range,
with additional metadata for explainability.

This is suitable for:
- Prototyping in Agentic AI loan workflows
- Mock scoring
- Pre-screening

Not suitable for real production lending decisions.
"""

from datetime import datetime


def dummy_credit_score(monthly_income: float, sector: str, dob: str) -> dict:
    """
    Calculate a dummy credit score using basic inputs.

    Args:
        monthly_income (float): Applicant's monthly salary/income.
        sector (str): "salaried", "self employed", or any other sector name.
        dob (str): Date of birth in "YYYY-MM-DD" format.

    Returns:
        dict: {
            "score": int,
            "pd": float,
            "age": int,
            "sector": str,
            "income_factor": float,
            "sector_factor": float,
            "age_factor": float,
            "rationale": str,
            "model_version": str
        }
    """

    # 1. Calculate age
    birth = datetime.strptime(dob, "%Y-%m-%d")
    today = datetime.today()
    age = today.year - birth.year - (
        (today.month, today.day) < (birth.month, birth.day)
    )

    # 2. Income Factor (0–100)
    if monthly_income <= 15000:
        income_factor = 30
    elif monthly_income <= 30000:
        income_factor = 50
    elif monthly_income <= 60000:
        income_factor = 70
    elif monthly_income <= 100000:
        income_factor = 85
    else:
        income_factor = 95

    # 3. Sector Factor (0–100)
    sector_clean = sector.lower().strip()

    if sector_clean == "salaried":
        sector_factor = 90
    elif sector_clean == "self employed":
        sector_factor = 70
    else:
        sector_factor = 50

    # 4. Age Factor (0–100)
    if age < 21:
        age_factor = 40
    elif 21 <= age <= 30:
        age_factor = 75
    elif 30 < age <= 45:
        age_factor = 90
    elif 45 < age <= 60:
        age_factor = 70
    else:
        age_factor = 50

    # 5. Weighted scoring (0–100)
    score_0_100 = (
        income_factor * 0.40 +
        sector_factor * 0.30 +
        age_factor * 0.30
    )

    # 6. Convert to final credit score (300–900)
    final_score = int(
        max(300, min(900, round(300 + score_0_100 * 6)))
    )

    # 7. Probability of Default (mock PD)
    pd = round(max(0.001, (900 - final_score) / 1000), 4)

    # 8. Return structured result
    return {
        "score": final_score,
        "pd": pd,
        "age": age,
        "sector": sector_clean,
        "income_factor": income_factor,
        "sector_factor": sector_factor,
        "age_factor": age_factor,
        "rationale": (
            f"Income={income_factor}, "
            f"Sector={sector_factor}, "
            f"Age={age_factor}"
        ),
        "model_version": "dummy-v1"
    }


def calculate_credit_score(data: dict) -> dict:
    """
    Wrapper function for future extensibility.
    Allows integration with your Agentic AI Orchestrator.

    Args:
        data (dict): {
            "monthly_income": float,
            "sector": str,
            "dob": str
        }

    Returns:
        dict: Result from dummy_credit_score()
    """

    monthly_income = data.get("monthly_income")
    sector = data.get("sector")
    dob = data.get("dob")

    if not monthly_income or not sector or not dob:
        raise ValueError("monthly_income, sector, and dob are required.")

    return dummy_credit_score(monthly_income, sector, dob)
