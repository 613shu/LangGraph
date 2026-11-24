"""Gmail access for the spam-cleaner agent (gmail.modify scope only)."""
from functools import lru_cache
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langsmith import traceable

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
BASE_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"   # downloaded from Google Cloud Console
TOKEN_FILE = BASE_DIR / "token.json"               # created on first login

LABELS = ["CATEGORY_PROMOTIONS", "SPAM"]
KEEP_HEADERS = {"From", "Subject", "List-Id", "List-Unsubscribe", "Reply-To", "Precedence"}


@lru_cache(maxsize=1)
def get_service():
    """Return an authenticated Gmail API client (cached for the process)."""
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def _parse_message(raw: dict, label: str) -> dict:
    """Turn a Gmail API message (format='metadata') into the agent's message dict.

    Only selected headers are kept; the message body is never included.
    """
    headers = {
        h["name"]: h["value"]
        for h in raw.get("payload", {}).get("headers", [])
        if h["name"] in KEEP_HEADERS
    }
    return {
        "id": raw["id"],
        "label": label,
        "sender": headers.pop("From", ""),
        "subject": headers.pop("Subject", ""),
        "snippet": raw.get("snippet", ""),
        "headers": headers,  # List-Id, List-Unsubscribe, Reply-To, Precedence (if present)
    }


def _unseen_ids(service, label: str, seen_ids: set[str], needed: int) -> list[str]:
    """Scan a label from the top and collect up to `needed` ids not in seen_ids."""
    found: list[str] = []
    page_token = None
    while len(found) < needed:
        resp = service.users().messages().list(
            userId="me",
            labelIds=[label],
            includeSpamTrash=(label == "SPAM"),
            maxResults=100,
            pageToken=page_token,
        ).execute()
        for m in resp.get("messages", []):
            if m["id"] not in seen_ids:
                found.append(m["id"])
                if len(found) == needed:
                    break
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return found


@traceable(run_type="tool", name="gmail_fetch")
def fetch_messages(seen_ids: set[str], limit: int = 30) -> tuple[list[dict], bool]:
    """Fetch up to `limit` unseen messages from PROMOTIONS and SPAM.

    Always scans from the top of each label (no nextPageToken kept between rounds),
    because trashing/moving messages shifts the results.
    Returns (batch, has_more).
    """
    service = get_service()
    candidates: list[tuple[str, str]] = []  # (id, label)
    for label in LABELS:
        remaining = limit + 1 - len(candidates)  # +1 to detect has_more
        if remaining <= 0:
            break
        for msg_id in _unseen_ids(service, label, seen_ids, remaining):
            candidates.append((msg_id, label))

    has_more = len(candidates) > limit
    batch = []
    for msg_id, label in candidates[:limit]:
        raw = service.users().messages().get(
            userId="me",
            id=msg_id,
            format="metadata",
            metadataHeaders=sorted(KEEP_HEADERS),
        ).execute()
        batch.append(_parse_message(raw, label))
    return batch, has_more


@traceable(run_type="tool", name="gmail_trash")
def trash_message(msg_id: str) -> None:
    """Move a message to Trash (recoverable for 30 days). Never deletes permanently."""
    get_service().users().messages().trash(userId="me", id=msg_id).execute()


_MOVE_RULES = {
    "CATEGORY_PROMOTIONS": {"removeLabelIds": ["CATEGORY_PROMOTIONS"], "addLabelIds": ["CATEGORY_PERSONAL"]},
    "SPAM": {"removeLabelIds": ["SPAM"], "addLabelIds": ["INBOX"]},
}


@traceable(run_type="tool", name="gmail_move_to_inbox")
def move_to_inbox(msg_id: str, label: str) -> None:
    """Move a message from PROMOTIONS or SPAM to the primary inbox."""
    if label not in _MOVE_RULES:
        raise ValueError(f"Unknown label: {label!r}")
    get_service().users().messages().modify(
        userId="me", id=msg_id, body=_MOVE_RULES[label]
    ).execute()
