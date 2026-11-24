# סוכן ניקוי קידומי מכירות וספאם (Gmail + LangGraph)

סוכן שעובר על התוויות **Promotions** ו-**Spam** ב-Gmail, מסווג כל הודעה בעזרת Gemini ל-`junk` או `maybe_important` (עם נימוק), ומציג מסך אישור ב-Streamlit. **שום הודעה לא נמחקת בלי אישור אנושי**, ומחיקה היא תמיד העברה לאשפה (ניתן לשחזר 30 יום).

```
START → fetch → route_empty ─ "empty" ───────────────────────→ END
                     └─ "classify" → classify → review ⏸ → execute → route_more
                                                                  ├─ "more" → fetch   ← לולאה
                                                                  └─ "done" → END
```

---

## 1. צילומי trace מ-LangSmith

> TODO: להוסיף צילומי מסך לתיקייה `docs/` ולקשר אליהם כאן.

- **ריצה מלאה** – לולאה של כמה סבבים (`fetch → classify → review → execute`), כל סבב עד 30 הודעות.
  `![trace – ריצה מלאה](docs/trace_full_run.png)`
- **שלב `classify`** – 3 קריאות LLM לסבב (10 הודעות בכל קריאה) עם structured output.
  `![trace – classify](docs/trace_classify.png)`
- **interrupt / resume** – עצירה ב-`review` והמשך עם `Command(resume=...)`.
  `![trace – interrupt](docs/trace_interrupt.png)`

## 2. תובנה: ניסוי v1 מול v2

> TODO: למלא את התוצאות מ-LangSmith (Datasets → `spam-cleaner-v1` → Compare experiments).

| מדד | v1 (שולח, נושא, תקציר) | v2 (+ `List-Id` / `List-Unsubscribe`, הגדרות, few-shot) |
|---|---|---|
| `important_recall` | ? | ? |
| `false_alarm_rate` | ? | ? |
| `reason_quality` | ? | ? |

**התובנה הצפויה:** ב-v1 הודעות של Google Groups מסווגות כ-`junk` כי הן נראות כמו דיוור המוני. ב-v2 הכותרת `List-Id` מכריעה נכון ומסמנת אותן ⭐.

> TODO: לאשר או לתקן לפי ה-trace בפועל, ולצרף צילום של דוגמה אחת.

---

## 3. התקנה והרצה

### 3.1 התקנת חבילות
בתוך הסביבה הווירטואלית (`.venv`, Python 3.12):
```bash
pip install langgraph langchain langchain-google-genai langsmith langgraph-checkpoint-sqlite google-api-python-client google-auth-oauthlib streamlit pandas pytest truststore
```

### 3.2 הגדרת Gmail API
1. ב-[Google Cloud Console](https://console.cloud.google.com/) – ליצור פרויקט ולהפעיל את **Gmail API**.
2. ליצור **OAuth client ID** מסוג **Desktop app** ולהוריד את הקובץ.
3. לשמור אותו בשם `credentials.json` בתיקייה `spam_cleaner/`.
4. בהרצה הראשונה ייפתח דפדפן להתחברות, ונוצר הקובץ `token.json`.

ההרשאה היחידה שנדרשת היא `gmail.modify` (קריאה, העברה לאשפה ושינוי תוויות – בלי מחיקה לצמיתות).

### 3.3 משתני סביבה
```bash
GOOGLE_API_KEY=...          # מפתח Gemini מ-Google AI Studio
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=spam-cleaner
```

### 3.4 קבצים שאסור להעלות ל-git
להוסיף ל-`.gitignore`:
```
credentials.json
token.json
*.sqlite
```

### 3.5 הרצה (מתוך `spam_cleaner/`)
| פעולה | פקודה |
|---|---|
| האפליקציה (Streamlit) | `streamlit run app.py` |
| יצירת הדאטהסט ב-LangSmith (פעם אחת) | `python eval/dataset.py` |
| ניסוי v1 מול v2 | `python eval/run_eval.py` |
| בדיקות | `python -m pytest tests -q` |

הבדיקות רצות בלי Gmail ובלי LLM אמיתיים (`FakeGmail`, `FakeClassifier`, `InMemorySaver`), ולכן לא צריך מפתחות כדי להריץ אותן.

---

## מבנה הקבצים
```
spam_cleaner/
├── state.py        # CleanerState, Verdict, reducers
├── gmail_tools.py  # fetch / trash / move_to_inbox (+ @traceable)
├── nodes.py        # fetch, classify, review, execute, route_empty, route_more
├── graph.py        # בניית הגרף + compile
├── app.py          # Streamlit: הרצה, מסך אישור, Command(resume=...)
├── eval/
│   ├── dataset.py  # יצירת הדאטהסט ב-LangSmith
│   └── run_eval.py # evaluators + ניסוי v1 מול v2
├── tests/          # בדיקות ל-reducers, routers, מדדים והגרף המלא
└── README.md
```
