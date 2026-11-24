# תוכנית עבודה – סוכן ניקוי קידומי מכירות וספאם

כל שלב = קובץ אחד. כל תת־שלב = פונקציה או מחלקה אחת, שנכתבת אחרי אישור.
כל הקבצים נמצאים בתיקייה `spam_cleaner/`.

## שלב 1: `state.py` – ה-State והסכמות (4 תתי־שלבים)
- [x] 1.1 `merge_unique` – reducer עבור `seen_ids` (איחוד רשימות בלי כפילויות)
- [x] 1.2 `sum_counts` – reducer עבור `stats` (חיבור מונים לפי מפתח)
- [x] 1.3 `Verdict`, `VerdictList` – סכמות Pydantic לפלט המובנה של ה-LLM
- [x] 1.4 `CleanerState` – ה-State של הגרף (TypedDict + Annotated)

## שלב 2: `gmail_tools.py` – החיבור ל-Gmail (5 תתי־שלבים)
- [x] 2.1 `get_service` – התחברות OAuth (הרשאת `gmail.modify`) ויצירת לקוח
- [x] 2.2 `_parse_message` – הפיכת הודעה גולמית ל-dict (שולח, נושא, תקציר, כותרות נבחרות)
- [x] 2.3 `fetch_messages` – משיכת עד 30 הודעות מ-PROMOTIONS ו-SPAM, בסינון לפי `seen_ids`
- [x] 2.4 `trash_message` – העברה לאשפה (`@traceable`)
- [x] 2.5 `move_to_inbox` – העברה לתיבה הראשית (`@traceable`)

## שלב 3: `nodes.py` – צמתי הגרף (6 תתי־שלבים)
- [x] 3.1 `fetch`
- [x] 3.2 `classify` – סיווג בחלקים של 10 הודעות עם `with_structured_output`
- [x] 3.3 `review` – `interrupt()` לאישור אנושי
- [x] 3.4 `execute` – ביצוע ההחלטות
- [x] 3.5 `route_empty` – ענף מותנה: האם יש הודעות
- [x] 3.6 `route_more` – ענף מותנה: הלולאה חזרה ל-`fetch`

## שלב 4: `graph.py` (1 תתי־שלבים)
- [x] 4.1 `build_graph` – `add_node` / `add_edge` / `add_conditional_edges` + compile עם checkpointer

## שלב 5: `app.py` – מסך Streamlit (3 תתי־שלבים)
- [x] 5.1 `get_graph` – `@st.cache_resource` + `SqliteSaver`
- [x] 5.2 `render_review` – מסך אישור מתוך ה-interrupt
- [x] 5.3 `main` – הרצה, המשך עם `Command(resume=...)`, סיכום ויומן

## שלב 6: `eval/` – LangSmith (6 תתי־שלבים)
- [x] 6.1 `dataset.py` → `create_dataset` – יצירת דאטהסט (30 הודעות)
- [x] 6.2 `run_eval.py` → `make_target` – target שמסווג הודעה לפי פרומפט נתון (v1 / v2)
- [x] 6.3 `run_eval.py` → `important_recall` – summary evaluator
- [x] 6.4 `run_eval.py` → `false_alarm_rate` – summary evaluator
- [x] 6.5 `run_eval.py` → `reason_quality` – LLM-as-judge
- [x] 6.6 `run_eval.py` → `main` – ניסוי v1 מול v2 (`max_concurrency=2`)

## שלב 7: `tests/` – בדיקות pytest (9 תתי־שלבים)
- [x] 7.1 `conftest.py` → `FakeClassifier` – LLM מזויף שמחזיר `VerdictList`
- [x] 7.2 `conftest.py` → `FakeGmail` – תיבת דואר מזויפת (fetch / trash / move)
- [x] 7.3 `conftest.py` → `fake_env` – fixture: monkeypatch ל-`nodes` + גרף עם `InMemorySaver`
- [x] 7.4 `test_state.py` → `TestReducers` – `merge_unique`, `sum_counts`
- [x] 7.5 `test_routers.py` → `TestRouters` – `route_empty`, `route_more`
- [x] 7.6 `test_metrics.py` → `TestMetrics` – `important_recall`, `false_alarm_rate`
- [x] 7.7 `test_graph.py` → `test_full_run_loops_until_empty` – interrupt → `Command(resume=...)` → לולאה → END
- [x] 7.8 `test_graph.py` → `test_approve_and_stop` – "אישור ועצירה" + ברירת מחדל בטוחה להודעה שהמודל השמיט
- [x] 7.9 `test_graph.py` → `test_empty_mailbox_ends_without_review` – תיבה ריקה → END בלי סיווג

## שלב 8: `README.md` (1 תתי־שלבים)
- [ ] צילומי trace + תובנה
