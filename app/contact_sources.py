"""Contacts from wherever the user keeps them, in one shape.

Each source (a .vcf file from an iPhone, a .csv from Google or Outlook, Google Contacts
through the People API, the phone's contact picker, or a form) becomes a list of
{"name", "phone", "email", "company", "title", "notes"}. key() tells duplicates apart, so
importing the same people twice updates them instead of adding copies.
"""

import csv
import io
import re

FIELDS = ("name", "phone", "email", "company", "title", "notes")
PUBLIC_EMAIL = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "me.com", "live.com",
                "aol.com", "proton.me", "protonmail.com", "ymail.com", "googlemail.com"}


def clean(contact: dict) -> dict | None:
    """Trim every field; None if there's no name to show."""
    out = {f: (str(contact.get(f) or "").strip() or None) for f in FIELDS}
    if out["email"]:
        out["email"] = out["email"].lower()
    if not out["name"]:
        out["name"] = out["email"] or out["phone"]
    return out if out["name"] else None


def key(contact: dict) -> str:
    """Same phone (last 10 digits), else same email, else same name: the same person."""
    digits = re.sub(r"\D", "", contact.get("phone") or "")
    if len(digits) >= 7:
        return "tel:" + digits[-10:]
    if contact.get("email"):
        return "email:" + contact["email"].lower()
    return "name:" + " ".join(contact["name"].lower().split())


def full_name(name: str) -> bool:
    """Two or more words: worth a web search. "Mum" or "Tunde" alone matches too many people."""
    return len([w for w in re.split(r"\s+", name.strip()) if len(w) > 1]) >= 2


def work_domain(email: str | None) -> str | None:
    domain = (email or "").rsplit("@", 1)[-1].lower() if email and "@" in email else None
    return domain if domain and domain not in PUBLIC_EMAIL else None


def country_from_phone(phone: str | None) -> str | None:
    digits = re.sub(r"[^\d+]", "", phone or "")
    if digits.startswith("+234") or digits.startswith("234") or re.match(r"^0[789][01]\d{8}$", digits):
        return "Nigeria"
    if digits.startswith("+44"):
        return "United Kingdom"
    if digits.startswith("+1"):
        return "United States"
    return None


# --- files ----------------------------------------------------------------------

def parse_file(filename: str, text: str) -> list[dict]:
    if filename.lower().endswith((".vcf", ".vcard")) or "BEGIN:VCARD" in text[:2000].upper():
        return parse_vcard(text)
    return parse_csv(text)


def parse_vcard(text: str) -> list[dict]:
    """vCard 2.1/3.0/4.0, as exported by iPhone, Android and Outlook."""
    lines = re.sub(r"\r?\n[ \t]", "", text).splitlines()  # unfold continued lines
    contacts, card = [], None
    for line in lines:
        upper = line.strip().upper()
        if upper == "BEGIN:VCARD":
            card = {}
        elif upper == "END:VCARD":
            if card is not None and (c := clean(card)):
                contacts.append(c)
            card = None
        elif card is not None and ":" in line:
            head, value = line.split(":", 1)
            prop = head.split(";")[0].split(".")[-1].upper()  # "item1.TEL;type=CELL" -> TEL
            value = _vcard_text(value)
            if prop == "FN":
                card["name"] = value
            elif prop == "N" and not card.get("name"):
                parts = value.split(";")
                card["name"] = " ".join(p for p in (parts[1:2] + parts[0:1]) if p)  # given + family
            elif prop == "TEL":
                card.setdefault("phone", value)
            elif prop == "EMAIL":
                card.setdefault("email", value)
            elif prop == "ORG":
                card.setdefault("company", value.split(";")[0])
            elif prop == "TITLE":
                card.setdefault("title", value)
            elif prop == "NOTE":
                card.setdefault("notes", value)
    return contacts


def _vcard_text(value: str) -> str:
    return value.replace("\\n", " ").replace("\\,", ",").replace("\\;", ";").strip()


# Header names used by Google Contacts, Outlook and most other exports.
CSV_COLUMNS = {
    "name": ["name", "full name", "display name"],
    "first": ["first name", "given name"],
    "last": ["last name", "family name", "surname"],
    "phone": ["phone 1 - value", "mobile phone", "phone", "phone number", "mobile", "primary phone", "home phone",
              "business phone"],
    "email": ["e-mail 1 - value", "email", "e-mail address", "email address", "e-mail"],
    "company": ["organization 1 - name", "organization name", "company", "organization"],
    "title": ["organization 1 - title", "organization title", "job title", "title"],
    "notes": ["notes", "note"],
}


def parse_csv(text: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))
    if not rows:
        return []
    headers = {h.strip().lower(): h for h in rows[0].keys() if h}
    column = {field: next((headers[n] for n in names if n in headers), None) for field, names in CSV_COLUMNS.items()}
    contacts = []
    for row in rows:
        get = lambda field: (row.get(column[field]) or "").strip() if column[field] else ""
        name = get("name") or " ".join(p for p in (get("first"), get("last")) if p)
        phone = get("phone").split(":::")[0].strip()  # Google joins several numbers with :::
        if c := clean({"name": name, "phone": phone, "email": get("email").split(":::")[0], "company": get("company"),
                       "title": get("title"), "notes": get("notes")}):
            contacts.append(c)
    return contacts


# --- Google Contacts (People API) -------------------------------------------------

def from_google(person: dict) -> dict | None:
    first = lambda field, key: next((x.get(key) for x in person.get(field, []) if x.get(key)), None)
    org = (person.get("organizations") or [{}])[0]
    return clean({
        "name": first("names", "displayName"),
        "phone": first("phoneNumbers", "value"),
        "email": first("emailAddresses", "value"),
        "company": org.get("name"),
        "title": org.get("title"),
        "notes": first("biographies", "value"),
    })
