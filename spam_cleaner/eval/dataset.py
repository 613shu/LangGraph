"""Create the LangSmith evaluation dataset for the spam-cleaner classifier.

One-time setup script: run `python eval/dataset.py` once. Re-running is safe
(an existing dataset with the same name is reused, not duplicated).
"""
# Use the Windows certificate store (needed behind filtered / inspected networks).
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from langsmith import Client

DATASET_NAME = "spam-cleaner-v1"

_UNSUB = {"List-Unsubscribe": "<mailto:unsubscribe@example.com>", "Precedence": "bulk"}


def _msg(i: int, label: str, sender: str, subject: str, snippet: str,
         headers: dict | None = None) -> dict:
    """Build one synthetic message in the same shape as gmail_tools._parse_message."""
    return {"id": f"ex{i:02d}", "label": label, "sender": sender,
            "subject": subject, "snippet": snippet, "headers": headers or {}}


P, S = "CATEGORY_PROMOTIONS", "SPAM"

# (message, expected verdict, kind)
EXAMPLES: list[tuple[dict, str, str]] = [
    # ---------- maybe_important (8) ----------
    (_msg(1, P, "Torah Study Group <torah-study@googlegroups.com>", "עדכון למשתתפי הקבוצה – שינוי שעה",
          "שלום לכולם, השיעור ביום שלישי יתקיים השבוע בשעה 20:30 במקום 20:00...",
          {"List-Id": "<torah-study.googlegroups.com>", "List-Unsubscribe": "<mailto:torah-study+unsubscribe@googlegroups.com>",
           "Precedence": "list"}), "maybe_important", "google_group"),
    (_msg(2, P, "Course Staff <ai-course-2026@googlegroups.com>", "הגשת תרגיל 3 – תזכורת",
          "תזכורת: מועד ההגשה של תרגיל 3 הוא יום חמישי. קישור לטופס ההגשה מצורף...",
          {"List-Id": "<ai-course-2026.googlegroups.com>", "Precedence": "list"}), "maybe_important", "google_group"),
    (_msg(3, S, "Neighborhood Committee <shchuna-bet@googlegroups.com>", "סיכום ישיבת ועד השכונה",
          "מצורף סיכום הישיבה מאתמול, כולל לוח הזמנים לשיפוץ הגינה הציבורית...",
          {"List-Id": "<shchuna-bet.googlegroups.com>", "Precedence": "list"}), "maybe_important", "google_group"),
    (_msg(4, P, "Shop Online <orders@shoponline.co.il>", "אישור הזמנה מס' 48213",
          "תודה על הזמנתך! סכום לתשלום: 249 ₪. ההזמנה תישלח תוך 3 ימי עסקים...",
          {"Reply-To": "support@shoponline.co.il"}), "maybe_important", "receipt"),
    (_msg(5, P, "Electric Company <billing@electric.co.il>", "החשבונית שלך לחודש אוקטובר",
          "החשבונית הדיגיטלית שלך מוכנה. סכום לתשלום: 412.30 ₪, מועד אחרון לתשלום..."), "maybe_important", "receipt"),
    (_msg(6, S, "Yossi Cohen <yossi.cohen1980@gmail.com>", "בקשר לפגישה מחר",
          "היי, רציתי לוודא שאנחנו עדיין נפגשים מחר בעשר. תעדכני אותי אם משהו השתנה."), "maybe_important", "personal_in_spam"),
    (_msg(7, S, "Miriam Levi <miriam.levi@walla.co.il>", "המסמכים שביקשת",
          "שלום, מצרפת את המסמכים ששאלת עליהם בשבוע שעבר. אם חסר משהו אשמח לשלוח."), "maybe_important", "personal_in_spam"),
    (_msg(8, P, "Bank Hapoalim <no-reply@bankhapoalim.co.il>", "הודעה חשובה על שינוי בתנאי החשבון",
          "לקוח/ה יקר/ה, בהתאם להוראות הפיקוח על הבנקים, החל מ-1.12 יחולו שינויים בעמלות..."),
     "maybe_important", "official_notice"),

    # ---------- junk: regular promotions (14) ----------
    (_msg(9, P, "Fashion Hub <deals@fashionhub.com>", "50% הנחה רק היום!", "קולקציית החורף בחצי מחיר. משלוח חינם מעל 199 ₪...", _UNSUB), "junk", "promo"),
    (_msg(10, P, "SkyAir <promo@skyair.com>", "יעדים חדשים לחורף", "טיסות לאירופה החל מ-99$. הזמינו עכשיו לפני שהמקומות ייגמרו...", _UNSUB), "junk", "promo"),
    (_msg(11, P, "Pizza Now <news@pizzanow.co.il>", "1+1 על כל הפיצות", "רק בסוף השבוע: קנו פיצה אחת וקבלו את השנייה במתנה...", _UNSUB), "junk", "promo"),
    (_msg(12, P, "TechStore <newsletter@techstore.com>", "המבצעים הכי חמים של השבוע", "אוזניות, מטענים ושעונים חכמים במחירים שלא תאמינו...", _UNSUB), "junk", "newsletter"),
    (_msg(13, P, "BookWorld <hello@bookworld.com>", "ספרים חדשים שחשבנו שתאהבי", "על סמך הרכישות הקודמות שלך, בחרנו עבורך את הספרים הבאים...", _UNSUB), "junk", "newsletter"),
    (_msg(14, P, "HomeDesign <offers@homedesign.co.il>", "סייל סוף עונה – עד 70%", "רהיטים, טקסטיל ואקססוריז לבית במחירי חיסול...", _UNSUB), "junk", "promo"),
    (_msg(15, P, "Gym Plus <team@gymplus.co.il>", "חודש ראשון חינם!", "הצטרפו עכשיו וקבלו חודש ראשון ללא עלות, כולל אימון אישי...", _UNSUB), "junk", "promo"),
    (_msg(16, P, "Supermarket Club <club@supermarket.co.il>", "הקופונים שלך לשבוע הזה", "חסכו בקניות עם הקופונים המיוחדים לחברי המועדון...", _UNSUB), "junk", "newsletter"),
    (_msg(17, P, "Travel Deals <deals@traveldeals.com>", "חופשה באילת ב-499 ₪ ללילה", "מלונות מובחרים, ארוחת בוקר כלולה. מספר המקומות מוגבל...", _UNSUB), "junk", "promo"),
    (_msg(18, P, "App Store Weekly <weekly@appnews.com>", "5 אפליקציות שחובה להכיר", "הניוזלטר השבועי שלנו עם ההמלצות החמות ביותר...", _UNSUB), "junk", "newsletter"),
    (_msg(19, P, "Shoes4U <sale@shoes4u.com>", "הזוג השני ב-1 ₪", "מבצע מטורף על כל הנעליים באתר, רק עד חצות...", _UNSUB), "junk", "promo"),
    (_msg(20, P, "Insurance Pro <marketing@insurancepro.co.il>", "חוסכים בביטוח הרכב?", "השוו מחירים וקבלו הצעה משתלמת תוך 2 דקות...", _UNSUB), "junk", "promo"),
    (_msg(21, P, "Online Courses <info@learnfast.com>", "קורס פייתון ב-90% הנחה", "הצטרפו ל-20,000 תלמידים. מבצע לזמן מוגבל...", _UNSUB), "junk", "promo"),
    (_msg(22, P, "Kitchen Master <news@kitchenmaster.com>", "מתכונים חדשים + מבצע על סירים", "גלו את המתכונים של החודש וקבלו 30% הנחה על סדרת הסירים...", _UNSUB), "junk", "newsletter"),

    # ---------- junk: spam (4) ----------
    (_msg(23, S, "Prize Center <winner@prize-center.biz>", "זכית ב-iPhone חדש!!!", "לחצי כאן כדי לממש את הפרס שלך לפני שיפוג תוקפו..."), "junk", "spam"),
    (_msg(24, S, "Security Alert <secure@acc0unt-verify.net>", "החשבון שלך יושעה", "זיהינו פעילות חשודה. אמתי את פרטייך מיד בקישור המצורף..."), "junk", "spam"),
    (_msg(25, S, "Crypto Profit <invest@crypto-profit.xyz>", "הרווחתי 10,000$ בשבוע", "גלי את השיטה הסודית שהבנקים לא רוצים שתדעי..."), "junk", "spam"),
    (_msg(26, S, "Delivery Service <track@deliv-ery.info>", "החבילה שלך מעוכבת", "יש לשלם דמי משלוח של 4.90 ₪ כדי לשחרר את החבילה. לחצי כאן..."), "junk", "spam"),

    # ---------- junk: ads disguised as personal (4) ----------
    (_msg(27, P, "Dana from SmartLoans <dana@smartloans.co.il>", "היי, רק רציתי לעדכן אותך",
          "היי! רציתי לספר לך שאושרה עבורך הלוואה בתנאים מיוחדים, רק לחצי לפרטים...", _UNSUB), "junk", "fake_personal_ad"),
    (_msg(28, P, "Avi <avi@realestate-deals.com>", "שאלה קטנה",
          "שלום, ראיתי שאת מתעניינת בנדל\"ן. יש לנו פרויקט חדש שיכול לעניין אותך...", _UNSUB), "junk", "fake_personal_ad"),
    (_msg(29, S, "Noa <noa.k@mail-promo.net>", "מזמן לא דיברנו!",
          "היי, מזמן לא דיברנו! תראי איזה מוצר מדהים גיליתי, יש עכשיו מבצע..."), "junk", "fake_personal_ad"),
    (_msg(30, P, "Michal | GlowSkin <michal@glowskin.com>", "הכנתי לך משהו מיוחד",
          "הכנתי במיוחד בשבילך ערכת טיפוח במחיר חברות. ההטבה בתוקף עד מחר...", _UNSUB), "junk", "fake_personal_ad"),
]


# ---------- 6.1 create_dataset ----------

def create_dataset(name: str = DATASET_NAME) -> str:
    """Create (or reuse) the LangSmith dataset and return its id."""
    client = Client()
    if client.has_dataset(dataset_name=name):
        return str(client.read_dataset(dataset_name=name).id)

    dataset = client.create_dataset(
        dataset_name=name,
        description="30 synthetic Promotions/Spam messages: 22 junk, 8 maybe_important.",
    )
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"message": msg} for msg, _, _ in EXAMPLES],
        outputs=[{"verdict": verdict, "kind": kind} for _, verdict, kind in EXAMPLES],
    )
    return str(dataset.id)


if __name__ == "__main__":
    print("Dataset id:", create_dataset())
