"""
סוכן שירות לקוחות ב-LangGraph:
  draft -> critic -> (לולאה חזרה ל-draft אם הביקורת נכשלה) -> human_review (interrupt) -> END
"""
import os
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from knowledge import POLICY

llm = init_chat_model(os.getenv("MODEL", "openai:gpt-4o-mini"), temperature=0)


# ---------- State ----------
class State(TypedDict, total=False):
    question: str
    draft: str
    feedback: str          # הערות מהמבקר או מהאדם
    critic_ok: bool
    iterations: int
    max_iterations: int    # 0 = בלי לולאת ביקורת (גרסת בסיס)
    human_review: bool     # False בזמן הערכה אוטומטית
    final_answer: str


# ---------- Prompts (v1 = לפני שיפור, v2 = אחרי) ----------
PROMPTS = {
    "v1": "אתה נציג שירות של חנות ספרים. ענה ללקוח.\n\nמידע:\n{policy}",
    "v2": (
        "אתה נציג שירות של חנות הספרים 'אור'.\n"
        "כללים:\n"
        "- ענה אך ורק לפי המדיניות שלהלן. אל תמציא מחירים, זמנים או הבטחות.\n"
        "- אם המידע לא מופיע במדיניות - אמור זאת והפנה לטלפון/דוא\"ל השירות.\n"
        "- ענה בעברית, בנימוס, בעד 3 משפטים.\n\n"
        "מדיניות:\n{policy}"
    ),
}


class Critique(BaseModel):
    ok: bool = Field(description="True אם התשובה נכונה לפי המדיניות, מלאה ותמציתית")
    feedback: str = Field(description="מה צריך לתקן, אם בכלל")


# ---------- Nodes ----------
def draft_node(state: State, config: RunnableConfig) -> State:
    version = config.get("configurable", {}).get("prompt_version", "v2")
    messages = [
        ("system", PROMPTS[version].format(policy=POLICY)),
        ("human", state["question"]),
    ]
    if state.get("feedback"):
        messages.append(("human", f"טיוטה קודמת:\n{state['draft']}\n\nתקן לפי ההערות:\n{state['feedback']}"))
    answer = llm.invoke(messages).content
    return {"draft": answer, "iterations": state.get("iterations", 0) + 1, "feedback": ""}


def critic_node(state: State) -> State:
    judge = llm.with_structured_output(Critique)
    result = judge.invoke([
        ("system", f"בדוק תשובת נציג שירות מול המדיניות הבאה בלבד:\n{POLICY}"),
        ("human", f"שאלה: {state['question']}\n\nתשובה: {state['draft']}"),
    ])
    return {"critic_ok": result.ok, "feedback": "" if result.ok else result.feedback}


def human_review_node(state: State) -> State:
    """Human-in-the-loop: הגרף נעצר כאן עד שאדם מחליט."""
    if not state.get("human_review", True):          # מצב הערכה אוטומטית
        return {"final_answer": state["draft"]}

    decision = interrupt({
        "question": state["question"],
        "draft": state["draft"],
        "options": "approve | edit (עם text) | reject (עם feedback)",
    })
    action = decision.get("action")
    if action == "approve":
        return {"final_answer": state["draft"]}
    if action == "edit":
        return {"final_answer": decision["text"]}
    # reject -> חוזר לכתיבה מחדש עם הערות האדם
    return {"feedback": decision.get("feedback", ""), "critic_ok": False}


# ---------- Conditional edges ----------
def after_draft(state: State) -> str:
    return "critic" if state.get("max_iterations", 3) > 0 else "human_review"


def after_critic(state: State) -> str:
    if state["critic_ok"] or state["iterations"] >= state.get("max_iterations", 3):
        return "human_review"
    return "draft"                                    # <-- הלולאה


def after_human(state: State) -> str:
    return END if state.get("final_answer") else "draft"


# ---------- Build ----------
def build_graph(checkpointer=None):
    g = StateGraph(State)
    g.add_node("draft", draft_node)
    g.add_node("critic", critic_node)
    g.add_node("human_review", human_review_node)

    g.add_edge(START, "draft")
    g.add_conditional_edges("draft", after_draft, ["critic", "human_review"])
    g.add_conditional_edges("critic", after_critic, ["draft", "human_review"])
    g.add_conditional_edges("human_review", after_human, ["draft", END])
    return g.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # מדפיס תרשים Mermaid של הגרף - אפשר להדביק ב-README
    print(build_graph().get_graph().draw_mermaid())
