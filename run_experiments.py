"""
הערכה אוטומטית ב-LangSmith: ניסוי לפני שיפור (v1) ואחרי שיפור (v2).
שימוש:
    python run_experiments.py before
    python run_experiments.py after
"""
import sys

from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from pydantic import BaseModel, Field

from graph import build_graph, llm
from create_dataset import DATASET_NAME

client = Client()
graph = build_graph()  # בלי checkpointer - בהערכה אין interrupt

SETUPS = {
    "before": {"prompt_version": "v1", "max_iterations": 0},   # פרומפט בסיסי, בלי לולאת ביקורת
    "after":  {"prompt_version": "v2", "max_iterations": 3},   # פרומפט משופר + לולאת ביקורת
}


def make_target(prompt_version: str, max_iterations: int):
    def target(inputs: dict) -> dict:
        out = graph.invoke(
            {"question": inputs["question"], "iterations": 0,
             "max_iterations": max_iterations, "human_review": False},
            config={"configurable": {"prompt_version": prompt_version}},
        )
        return {"answer": out["final_answer"], "iterations": out["iterations"]}
    return target


# ---------- Evaluators ----------
class Grade(BaseModel):
    correct: bool = Field(description="האם התשובה תואמת עובדתית את תשובת הייחוס")
    reasoning: str


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-judge: השוואה לתשובת הייחוס."""
    grade = llm.with_structured_output(Grade).invoke([
        ("system", "אתה בודק. קבע אם תשובת הסוכן נכונה עובדתית ביחס לתשובת הייחוס. "
                   "ניסוח שונה זה בסדר; עובדה שגויה או מומצאת = לא נכון."),
        ("human", f"שאלה: {inputs['question']}\n"
                  f"ייחוס: {reference_outputs['answer']}\n"
                  f"תשובת הסוכן: {outputs['answer']}"),
    ])
    return {"key": "correctness", "score": int(grade.correct), "comment": grade.reasoning}


def conciseness(outputs: dict) -> dict:
    """בדיקה דטרמיניסטית: תשובה עד 400 תווים."""
    return {"key": "concise", "score": int(len(outputs["answer"]) <= 400)}


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "before"
    setup = SETUPS[name]
    results = client.evaluate(
        make_target(**setup),
        data=DATASET_NAME,
        evaluators=[correctness, conciseness],
        experiment_prefix=f"{name}-{setup['prompt_version']}",
        metadata=setup,
        max_concurrency=2,
    )
    print(results)
