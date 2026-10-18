#!/usr/bin/env python3
"""
Test how well a local Mistral model (running in Ollama) recognizes package emails,
and what kind of package email each one is.

It reads your Gmail Takeout files:
  - every .mbox with "Jijikat" in its name     -> emails that ARE about packages
    (including nested carrier labels like Jijikat/postnl)
  - every .mbox with "not-package" in its name -> emails that are NOT about packages
An email that shows up in several files is only tested once.

For every email the model answers:
  is_package: true / false
  type:       "upcoming"  (on its way to you)
              "delivered" (delivered to your address or a neighbour)
              "pickup"    (waiting at a service point, incl. pick-up reminders)
              "return"    (a parcel you sent back, e.g. received by the shop)
              "pending"  (shipped or about to ship, but no tracking number yet)

Usage (in Terminal, from the folder that contains your Takeout folders):
  python3 evaluate.py --limit 10   # quick test: first 10 emails of each file
  python3 evaluate.py              # all emails
  python3 evaluate.py --sender-filter   # only ask the model about known carrier senders

Outputs:
  mistakes.csv  - emails where is_package was wrong or unclear
  results.csv   - every email with the model's answer, to spot-check the types

Everything runs on your Mac. No email leaves your computer.
"""
import argparse
import csv
import glob
import html
import json
import mailbox
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from email.header import decode_header, make_header

MODEL = "mistral"
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_CHARS = 1500  # only send the start of each email; the rest is mostly boilerplate

# Every .mbox file whose name contains these words is used.
# (Nested Gmail labels can produce several files, e.g. one per carrier.)
GROUPS = [
    ("package emails", "jijikat", True),          # True  = these are package emails
    ("not-package emails", "not-package", False), # False = these are not
]

TYPES = {"upcoming", "delivered", "pickup", "return", "pending"}

# Sender filter (only used with --sender-filter): emails from other domains are
# treated as "not package" without asking the model. Subdomains count too,
# so "postnl.nl" also matches "edm.postnl.nl". Add domains here if needed.
KNOWN_SENDER_DOMAINS = [
    "postnl.nl",
    "dhl.nl", "dhl.com", "dhlparcel.nl", "dhl.de",
    "ups.com",
    "dpd.nl", "dpd.com", "pickup-services.com",  # DPD pick-up network
]

# Carrier labels you added in Gmail (matched case-insensitively)
CARRIERS = ["PostNL", "DHL", "UPS", "DPD"]

PROMPT = """You are classifying emails about parcels.
Answer with JSON only, in exactly this format:
{{"is_package": true or false, "type": "upcoming" or "delivered" or "pickup" or "return" or "pending" or null}}

Set "is_package" to true if the email is about one of these, and choose the matching "type":
- "upcoming": a parcel is on its way to the recipient
  (shipped, on its way, expected delivery time, out for delivery).
- "delivered": the parcel has been delivered to the recipient's address or to a neighbour.
- "pickup": the parcel is waiting at a service point, pick-up point or parcel locker,
  including reminders to collect it before a deadline.
- "return": a parcel the recipient sent back to a shop
  (return registered, return on its way, return received by the shop, refund after a return).
- "pending": the shop says the parcel has been shipped or handed to a carrier, or will arrive
  soon, but there is no tracking number yet. International shipments announced without
  tracking information belong here.

Set "is_package" to false and "type" to null for everything else, including plain order
confirmations ("we received your order"), newsletters, invoices and personal emails.

Also set "is_package" to false for these, even when they come from a carrier and mention
parcels or delivery:
- surveys, review or rating requests ("how did the delivery go?", "rate us and win a voucher");
- general service announcements that are not about one specific parcel (weather, holidays,
  general delays, new prices or rules);
- account or membership emails, and advertisements that use shipping words
  ("fast shipping", "free delivery", "order now").
An email counts if it is about a specific parcel of the recipient: a status update, or a code
needed to receive or collect it (for example an "ontvangstcode" or pick-up code), even when
the tracking number is partly hidden or missing.
If an email gives a parcel status AND asks for feedback (for example "your parcel has been
delivered, tell us how it went"), it still counts: classify it by the parcel status.

The email may be in Dutch or English.

From: {sender}
Subject: {subject}

{body}
"""


# ---------- reading emails ----------

def get_header(msg, name):
    raw = msg.get(name, "")
    try:
        return str(make_header(decode_header(raw)))
    except Exception:
        return str(raw)


def strip_html(text):
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return html.unescape(text)


def get_body(msg):
    plain, html_part = None, None
    for part in msg.walk():
        if part.is_multipart():
            continue
        if part.get_content_disposition() == "attachment":
            continue
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        charset = part.get_content_charset() or "utf-8"
        try:
            text = payload.decode(charset, errors="replace")
        except LookupError:
            text = payload.decode("utf-8", errors="replace")
        if ctype == "text/plain" and plain is None:
            plain = text
        elif ctype == "text/html" and html_part is None:
            html_part = text
    text = plain if plain and plain.strip() else strip_html(html_part or "")
    return re.sub(r"\s+", " ", text).strip()[:MAX_CHARS]


def get_carrier(msg):
    """Reads the Gmail labels Takeout stores in each email and returns the carrier label, if any."""
    labels = set()
    for label in get_header(msg, "X-Gmail-Labels").split(","):
        for part in label.split("/"):  # nested labels look like "Jijikat/postnl"
            labels.add(part.strip().strip('"').lower())
    for carrier in CARRIERS:
        if carrier.lower() in labels:
            return carrier
    return "no carrier label"


def is_known_sender(sender):
    match = re.search(r"@([\w.-]+)", sender)
    if not match:
        return False
    domain = match.group(1).lower().rstrip(".>")
    return any(domain == d or domain.endswith("." + d) for d in KNOWN_SENDER_DOMAINS)


def find_files(word):
    matches = sorted(f for f in glob.glob("**/*.mbox", recursive=True)
                     if word in os.path.basename(f).lower())
    if not matches:
        sys.exit(f"Could not find any .mbox file with '{word}' in its name. "
                 "Run this script from the folder that contains your Takeout folders.")
    return matches


# ---------- asking the model ----------

def ask_model(sender, subject, body):
    """Returns (is_package, type, raw_answer). is_package is None if the answer was unusable."""
    request_data = {
        "model": MODEL,
        "prompt": PROMPT.format(sender=sender, subject=subject, body=body),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},  # same email -> same answer
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(request_data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            answer = json.loads(response.read())["response"]
    except urllib.error.URLError:
        sys.exit("Could not reach Ollama. Is the Ollama app running?")

    try:
        parsed = json.loads(answer)
        value = parsed.get("is_package")
        kind = parsed.get("type")
    except (json.JSONDecodeError, AttributeError):
        return None, None, answer

    if isinstance(value, str):
        value = value.strip().lower() == "true"
    if not isinstance(value, bool):
        return None, None, answer

    if value:
        kind = kind.strip().lower() if isinstance(kind, str) else None
        if kind not in TYPES:
            kind = "unknown"
    else:
        kind = None
    return value, kind, answer


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only test the first N emails of each group (quick test)")
    parser.add_argument("--sender-filter", action="store_true",
                        help="skip the model for senders not in KNOWN_SENDER_DOMAINS")
    args = parser.parse_args()

    results = {}      # per group: correct, wrong, unclear, total
    type_counts = {}  # per group: how many of each type the model chose
    per_carrier = {}  # per group: {carrier: Counter(correct=.., total=..)}
    seen = {}         # Message-ID -> group, so an email in several files is tested once
    conflicts = []    # emails that appear both as package and as not-package
    all_rows = []
    mistakes = []
    start = time.time()

    for group, word, expected in GROUPS:
        paths = find_files(word)
        counts = {"correct": 0, "wrong": 0, "unclear": 0, "total": 0}
        types = Counter()
        carriers = {}
        print(f"\nTesting {group} from: {', '.join(os.path.basename(p) for p in paths)}")

        for path in paths:
            for msg in mailbox.mbox(path):
                if args.limit and counts["total"] >= args.limit:
                    break
                subject = get_header(msg, "Subject")
                msg_id = get_header(msg, "Message-ID").strip() or f"{path}:{subject}"
                if msg_id in seen:
                    if seen[msg_id] != group:
                        conflicts.append(subject)
                    continue  # already tested this email
                seen[msg_id] = group

                sender = get_header(msg, "From")
                body = get_body(msg)
                carrier = get_carrier(msg)

                if args.sender_filter and not is_known_sender(sender):
                    guess, kind, raw_answer = False, None, "(filtered: unknown sender, model not asked)"
                else:
                    guess, kind, raw_answer = ask_model(sender, subject, body)
                counts["total"] += 1
                if guess is None:
                    counts["unclear"] += 1
                    status = "?"
                elif guess == expected:
                    counts["correct"] += 1
                    status = "ok"
                else:
                    counts["wrong"] += 1
                    status = "FILTERED" if raw_answer.startswith("(filtered") else "WRONG"
                if guess:
                    types[kind] += 1
                carriers.setdefault(carrier, Counter())
                carriers[carrier]["total"] += 1
                if status == "ok":
                    carriers[carrier]["correct"] += 1

                row = {
                    "file": os.path.basename(path),
                    "carrier": carrier,
                    "expected": "package" if expected else "not package",
                    "model_is_package": "" if guess is None else str(guess).lower(),
                    "model_type": kind or "",
                    "status": status,
                    "from": sender,
                    "subject": subject,
                    "raw_answer": raw_answer.strip(),
                }
                all_rows.append(row)
                if status != "ok":
                    mistakes.append(row)
                print(f"  [{counts['total']:>4}] {status:<5} {carrier:<7} {(kind or '-'):<9} {subject[:50]}")

        results[group] = counts
        type_counts[group] = types
        per_carrier[group] = carriers

    # ---------- summary ----------
    print("\n" + "=" * 60)
    print("Mode: model + sender filter" if args.sender_filter else "Mode: model only")
    total_correct = total = 0
    for group, word, expected in GROUPS:
        c = results[group]
        label = "recognized" if expected else "correctly rejected"
        print(f"{group}: {c['correct']}/{c['total']} {label}"
              f"  (wrong: {c['wrong']}, unclear answer: {c['unclear']})")
        if type_counts[group]:
            breakdown = ", ".join(f"{k}: {v}" for k, v in type_counts[group].most_common())
            print(f"    types chosen -> {breakdown}")
        if expected:
            for carrier, cc in sorted(per_carrier[group].items()):
                print(f"    {carrier:<16} {cc['correct']}/{cc['total']} correct")
        total_correct += c["correct"]
        total += c["total"]
    if total:
        print(f"\nOverall: {total_correct}/{total} correct ({100 * total_correct / total:.1f}%)")
    print(f"Time: {time.time() - start:.0f} seconds")
    if conflicts:
        print(f"\nWarning: {len(conflicts)} email(s) have BOTH a package and a not-package label "
              "(tested only once). Fix these in Gmail:")
        for s in conflicts[:10]:
            print(f"  - {s[:70]}")

    fieldnames = ["file", "carrier", "expected", "model_is_package", "model_type", "status",
                  "from", "subject", "raw_answer"]
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print("\nAll answers saved to results.csv (check the model_type column).")

    if mistakes:
        with open("mistakes.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mistakes)
        print(f"{len(mistakes)} mistakes saved to mistakes.csv.")


if __name__ == "__main__":
    main()
