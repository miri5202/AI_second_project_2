"""הרצה אינטראקטיבית עם interrupt / human-in-the-loop."""
import uuid

from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from graph import build_graph

graph = build_graph(checkpointer=InMemorySaver())   # interrupt דורש checkpointer


def main():
    question = input("שאלת הלקוח: ")
    config = {"configurable": {"thread_id": str(uuid.uuid4()), "prompt_version": "v2"}}
    result = graph.invoke(
        {"question": question, "iterations": 0, "max_iterations": 3, "human_review": True},
        config,
    )

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n--- טיוטה לאישור ---\n" + payload["draft"])
        choice = input("\n[a] אשר  [e] ערוך  [r] דחה: ").strip().lower()
        if choice == "a":
            decision = {"action": "approve"}
        elif choice == "e":
            decision = {"action": "edit", "text": input("הטקסט המתוקן: ")}
        else:
            decision = {"action": "reject", "feedback": input("מה לתקן? ")}
        result = graph.invoke(Command(resume=decision), config)

    print("\n=== תשובה סופית ===\n" + result["final_answer"])
    print(f"(מספר סבבי כתיבה: {result['iterations']})")


if __name__ == "__main__":
    main()
