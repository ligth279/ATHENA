from __future__ import annotations

import json
from typing import Any

from athena.amc.memory import ConversationMemory
from athena.amc.types import LlamaRole


def _dump_section(section: Any) -> str:
    if section is None:
        return "(no section loaded)"
    if isinstance(section, str):
        return section
    return json.dumps(section, ensure_ascii=False, indent=2)


def tutor_system_prompt(section: Any, history: ConversationMemory | None = None) -> str:
    """Event T — explain the current section simply. Keep prior doubts."""

    body = (
        "You are the xilo tutor (Llama 3.1, role TUTOR).\n"
        "The student is studying one section of a chapter. Explain doubts "
        "about that section in simple, easy-to-understand language.\n"
        "Stay with the section unless a short extra example is needed to "
        "make a concept click. Do not invent curriculum beyond it.\n"
        "Current section (JSON):\n"
        f"{_dump_section(section)}\n"
    )
    if history and len(history):
        body += (
            "\nThe GPU chat cache was cleared. Continue this conversation:\n"
            f"{history.render()}\n"
        )
    return body


def evaluator_system_prompt(
    behavioral_level: int,
    question: str,
    student_answer: str,
    allow_approximate: bool,
) -> str:
    """Event E — hints only. Never leak the answer.

    Behavioral gear (ALE) supplies `behavioral_level`. Higher = more
    scaffolding, still never the answer itself.
    """

    closeness = (
        "This question type may accept a close paraphrase (paragraph-style)."
        if allow_approximate
        else "This question type requires an exact match (math / one-word / spelling)."
    )
    return (
        "You are the xilo evaluator (Llama 3.1, role EVALUATOR).\n"
        "The student is in a quiz. You generate HINTS only.\n"
        "HARD RULES:\n"
        "- Never state the correct answer.\n"
        "- Never restate the answer in other words.\n"
        "- Never confirm or deny by quoting the key.\n"
        "- Give one short hint that makes the student think.\n"
        f"- Behavioral gear level: {behavioral_level} "
        "(1=tiny nudge, 3=worked-example-without-answer).\n"
        f"- {closeness}\n"
        f"Question: {question}\n"
        f"Student's answer: {student_answer}\n"
    )


def role_system_prompt(role: LlamaRole, **kwargs: Any) -> str:
    if role is LlamaRole.TUTOR:
        return tutor_system_prompt(
            kwargs.get("section"),
            kwargs.get("history"),
        )
    if role is LlamaRole.EVALUATOR:
        return evaluator_system_prompt(
            behavioral_level=int(kwargs.get("behavioral_level", 1)),
            question=str(kwargs.get("question", "")),
            student_answer=str(kwargs.get("student_answer", "")),
            allow_approximate=bool(kwargs.get("allow_approximate", False)),
        )
    raise ValueError(f"unknown role {role}")
