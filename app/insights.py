"""
app/insights.py

Single source of truth for the "why this score" reasons and suggestions
shown in the result panel. Previously this logic existed twice — an unused
generate_recommendations() in recommendation.py, and a separate inline
generate_insights() in main.py — with no connection between them, meaning
edits to one silently wouldn't affect the other. There is only one version
now.
"""

from app.schemas import DiabetesInput


def generate_insights(data: DiabetesInput) -> tuple[list[str], list[str]]:
    """Returns (reasons, suggestions) for the result panel's explanation card."""
    reasons: list[str] = []
    suggestions: list[str] = []

    if data.bmi < 30:
        reasons.append("BMI within recommended range")
    else:
        reasons.append("BMI may increase diabetes risk")
        suggestions.append("Consider gradual weight reduction through diet and activity")

    if data.phys_activity:
        reasons.append("Active lifestyle detected")
    else:
        suggestions.append("Aim for 150 minutes of moderate activity per week")

    if not data.smoker:
        reasons.append("No smoking history")
    else:
        suggestions.append("Consider a smoking cessation program")

    if data.fruits and data.veggies:
        reasons.append("Balanced fruit and vegetable intake")
    else:
        suggestions.append("Increase fruit and vegetable intake")

    if data.high_bp:
        suggestions.append("Monitor blood pressure regularly")
    if data.high_chol:
        suggestions.append("Discuss cholesterol management with a healthcare provider")

    if not suggestions:
        suggestions = [
            "Maintain a balanced diet",
            "Continue regular activity",
            "Monitor blood glucose periodically",
        ]

    return reasons, suggestions