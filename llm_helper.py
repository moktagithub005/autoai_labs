# llm_helper.py
import os
import json
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def _client() -> OpenAI:
    # Reads OPENAI_API_KEY from .env / env
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _safe_json(o: Any) -> str:
    try:
        return json.dumps(o, indent=2, ensure_ascii=False)
    except Exception:
        return str(o)

_SYSTEM_PROMPT_MENTOR = (
    "You are an expert ML mentor and career coach. "
    "Teach clearly, avoid jargon unless necessary, and give practical next steps. "
    "Be supportive, concise, and structured with bullet points when helpful."
)

def _chat(messages, model: str = "gpt-4o-mini") -> str:
    try:
        client = _client()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ LLM error: {e}"

def explain_dataset(eda: Dict[str, Any], sample_schema: Optional[Dict[str, str]] = None) -> str:
    prompt = (
        "Explain the dataset to a student. Summarize size, types, missing values, "
        "and any immediate data quality concerns. Offer 3–5 practical tips.\n\n"
        f"EDA JSON:\n{_safe_json(eda)}\n\n"
        f"Schema (optional):\n{_safe_json(sample_schema)}"
    )
    return _chat([
        {"role": "system", "content": _SYSTEM_PROMPT_MENTOR},
        {"role": "user", "content": prompt}
    ])

def explain_model_choice(summary: Dict[str, Any]) -> str:
    prompt = (
        "Given this AutoML summary, explain why the best model was likely chosen, "
        "what its strengths/weaknesses are for the detected task, and 3 next steps "
        "to try for improvement (feature engineering, tuning, or alt models).\n\n"
        f"AUTOML SUMMARY:\n{_safe_json(summary)}"
    )
    return _chat([
        {"role": "system", "content": _SYSTEM_PROMPT_MENTOR},
        {"role": "user", "content": prompt}
    ])

def explain_metrics(summary: Dict[str, Any]) -> str:
    prompt = (
        "Explain these evaluation metrics to a student in 6–10 bullet points. "
        "Clarify what each metric means and how to judge if it's good or not. "
        "Add 3 suggestions to improve metrics.\n\n"
        f"AUTOML SUMMARY:\n{_safe_json(summary)}"
    )
    return _chat([
        {"role": "system", "content": _SYSTEM_PROMPT_MENTOR},
        {"role": "user", "content": prompt}
    ])

def ask_ai_mentor(question: str, context: Dict[str, Any]) -> str:
    # Context can include EDA, summary, task, and user goals
    prompt = (
        "User is learning ML/DL. Answer like a mentor + career coach. "
        "If code helps, include short Python snippets. Use plain English and bullets.\n\n"
        f"CONTEXT:\n{_safe_json(context)}\n\n"
        f"QUESTION:\n{question}"
    )
    return _chat([
        {"role": "system", "content": _SYSTEM_PROMPT_MENTOR},
        {"role": "user", "content": prompt}
    ])
