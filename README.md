# סוכן שירות לקוחות - חנות הספרים "אור" (LangGraph + LangSmith)

סוכן שעונה לשאלות לקוחות לפי מדיניות החנות. הוא כותב טיוטה, מבקר אותה בעצמו ומתקן אותה בלולאה, ולפני השליחה עוצר ומחכה לאישור של נציג אנושי.

## איך זה עומד בדרישות

| דרישה | איפה במימוש |
|---|---|
| ספריית LangGraph | `graph.py` - `StateGraph` |
| לולאה / ענף מותנה | `critic -> draft` כשהביקורת נכשלת (עד 3 סבבים), ו-`human_review -> draft` כשהאדם דוחה |
| interrupt / human-in-the-loop | `human_review_node` קורא ל-`interrupt()`, והחידוש נעשה עם `Command(resume=...)` ב-`run_interactive.py` |
| דאטהסט + הערכה אוטומטית | `create_dataset.py` (12 דוגמאות), `run_experiments.py` (LLM-as-judge + בדיקת אורך) |
| ניסוי לפני/אחרי | `before` = פרומפט v1 בלי ביקורת, `after` = פרומפט v2 עם לולאת ביקורת |

## מבנה הגרף

```mermaid
<!-- להדביק כאן את הפלט של: python graph.py -->
```

## הרצה

```bash
pip install -r requirements.txt
cp .env.example .env        # למלא מפתחות
python create_dataset.py
python run_experiments.py before
python run_experiments.py after
python run_interactive.py   # הדגמה של human-in-the-loop
```

## תוצאות

| ניסוי | correctness | concise |
|---|---|---|
| before (v1) | __ | __ |
| after (v2) | __ | __ |

![השוואת ניסויים](screenshots/compare.png)

## צילומי Trace

![trace של ריצה עם לולאת ביקורת](screenshots/trace_loop.png)
![trace של interrupt](screenshots/trace_interrupt.png)

## תובנה מה-traces

_לכתוב כאן מה שראית בפועל. לדוגמה:_
"ב-trace של השאלה על סניף בחיפה ראיתי שגרסה v1 המציאה כתובת. ב-v2 המבקר תפס את זה בסבב הראשון, אבל זה עלה סבב LLM נוסף (בערך פי 2 בזמן ובטוקנים). לכן החלטתי..."
