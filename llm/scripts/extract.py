#!/usr/bin/env python3
"""
Step 2: can Mistral pull the carrier and the tracking number out of a package email?

Uses only your package emails (every .mbox with "Jijikat" in its name).
Put this file next to evaluate.py; it reuses its email-reading code.

Usage (in Terminal, same folder as before):
  python3 extract.py --limit 10   # quick test: first 10 emails
  python3 extract.py              # all package emails
  python3 extract.py --only dpd   # only files with "dpd" in the name (e.g. Jijikat-dpd.mbox)

How it is scored (you have no "correct tracking number" labels yet, so we use checks):
  carrier   -> compared with your Gmail carrier label (a real score)
  exact     -> DPD only: the "Pakketnummer" in the email is the right answer; without
               that label, a standalone 14-digit number; neither = "none". A real score.
               A second line shows the score after a simple format check on the
               model's answer (only 14-digit numbers kept): "LLM proposes, code checks".
  tracking  -> found_in_email : the number the model gave really appears in the email
               made_up        : the model gave a number that is NOT in the email (bad!)
               masked         : the model gave a partly hidden number like 3S****828
               too_short      : the model gave something too short to be real, like "96"
               none           : the model found no tracking number
               + a simple pattern check (regex) per carrier, to compare against

Output: extraction.csv - every email with the model's answer, for spot-checking.
Add a column "correct_tracking" there later, and you have real labels.

Everything runs on your Mac. No email leaves your computer.
"""
import argparse
import csv
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

import evaluate as ev  # reuse: get_header, get_carrier, find_files, MODEL, OLLAMA_URL

MAX_CHARS = 2000
CARRIER_NAMES = ["PostNL", "DHL", "UPS", "DPD"]

# Rough tracking-number shapes, used only as a comparison ("baseline"), not as the truth.
PATTERNS = {
    "PostNL": r"\b(3S[A-Z0-9]{8,15}|[A-Z]{2}\d{9}[A-Z]{2})\b",
    "DHL":    r"\b(JVGL\d{10,24}|JJD\d{10,24}|3S[A-Z0-9]{8,15}|[A-Z]{2}\d{9}[A-Z]{2}|\d{10})\b",
    "UPS":    r"\b(1Z[A-Z0-9]{16})\b",
    "DPD":    r"\b(\d{14})\b",
}

# Labelled numbers: when the email literally says "Pakketnummer: <number>", that number is
# the correct answer, and if the label is missing the correct answer is "none".
# Used to fill in correct_tracking automatically, so these carriers get a real score.
LABELLED_PATTERNS = {
    # "Pakketnummer: 11706850165688" or "pakketnummer(s): 04423114573603  04423114573606"
    "DPD": r"pakketnummer(?:\(s\)|s)?\s*:?\s*((?:\d{12,16}[\s,]*)+)",
}
# Fallback when there is no label: older DPD emails show the number on its own.
# Numbers right after these words are references, not tracking numbers, so they are skipped.
NOT_TRACKING_LABELS = r"(pakketreferentie|referentie|bestelnummer|klantnummer)\W{0,15}\S+"
UNLABELLED_PATTERNS = {
    "DPD": r"(?<![\d/.-])\b(\d{14})\b(?![\d/.-])",
}
# Validation step ("the LLM proposes, the code checks"): answers that don't have
# this exact shape are thrown away. Only used for the extra "after format check" score.
VALID_FORMATS = {
    "DPD": r"\d{14}",
}

PROMPT = """You extract information from Dutch or English parcel emails.
Answer with JSON only, in exactly this format:
{{"carrier": "PostNL" or "DHL" or "UPS" or "DPD" or "other" or null,
  "tracking_numbers": ["tracking number exactly as written in the email", ...]}}

Rules:
- "carrier" is the company delivering the parcel.
- "tracking_numbers" is a list. Usually it has one number; if the email is about several
  parcels, list each one separately. If there is no tracking number, use an empty list [].
- Copy each number exactly from the email (subject, text or links), without spaces.
  Never invent or guess one.
- A tracking number is a code of at least 8 letters and/or digits, usually labelled
  "track & trace", "zendingnummer", "pakketnummer", "parcel number" or "tracking number".
- These are NOT tracking numbers: order numbers (bestelnummer), customer or reference
  numbers (klantnummer, referentie, pakketreferentie), shop references, names, codes
  from inside web links, pick-up or receive codes
  (ontvangstcode), and numbers that are only partly visible (for example 3S****828).

From: {sender}
Subject: {subject}

{text}
"""


# ---------- reading emails ----------

def decoded_parts(msg):
    """Returns (plain_text, html_text) of the email, decoded."""
    plain, html_part = "", ""
    for part in msg.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment":
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
        if ctype == "text/plain" and not plain:
            plain = text
        elif ctype == "text/html" and not html_part:
            html_part = text
    return plain, html_part


def tracking_links(html_text, limit=5):
    """Links that look like tracking links: tracking numbers often only live in these."""
    links = []
    for url in re.findall(r'href=["\']([^"\']+)["\']', html_text, flags=re.I):
        url = html.unescape(url)
        if re.search(r"track|volg|trace|zending|shipment|parcel|pakket", url, re.I):
            if url not in links:
                links.append(url[:250])
        if len(links) >= limit:
            break
    return links


def text_for_model(msg):
    plain, html_part = decoded_parts(msg)
    text = plain if plain.strip() else ev.strip_html(html_part)
    text = re.sub(r"\s+", " ", text).strip()[:MAX_CHARS]
    links = tracking_links(html_part)
    if links:
        text += "\n\nLinks in the email:\n" + "\n".join(links)
    return text


def full_text(msg, subject):
    """Everything in the email (for checking that an answer really appears in it)."""
    plain, html_part = decoded_parts(msg)
    return html.unescape(" ".join([subject, plain, html_part]))


def readable_text(msg, subject):
    """The email as a person reads it: HTML tags removed, whitespace collapsed."""
    plain, html_part = decoded_parts(msg)
    text = " ".join([subject, plain, ev.strip_html(html_part)])
    return re.sub(r"\s+", " ", text)


def labelled_answer(carrier, msg, subject):
    """Returns the expected tracking number(s), "none", or "" if no rule for this carrier."""
    pattern = LABELLED_PATTERNS.get(carrier)
    if not pattern:
        return ""
    text = readable_text(msg, subject)
    hits = []
    for block in re.findall(pattern, text, flags=re.I):
        hits += re.findall(r"\d{12,16}", block)  # one label can list several numbers
    if not hits and carrier in UNLABELLED_PATTERNS:
        cleaned = re.sub(NOT_TRACKING_LABELS, " ", text, flags=re.I)
        hits = re.findall(UNLABELLED_PATTERNS[carrier], cleaned)
    numbers = []
    for h in hits:
        n = normalize(h)
        if n not in numbers:
            numbers.append(n)
    return " | ".join(numbers) if numbers else "none"


def normalize(s):
    return re.sub(r"[\s\-.]", "", s or "").upper()


# ---------- asking the model ----------

def ask_model(prompt):
    request_data = {
        "model": ev.MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        ev.OLLAMA_URL,
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
        if not isinstance(parsed, dict):
            raise ValueError
    except ValueError:
        return None, None, answer
    carrier = parsed.get("carrier")
    tracking = parsed.get("tracking_numbers", parsed.get("tracking_number"))
    carrier = carrier.strip() if isinstance(carrier, str) and carrier.strip() else None
    if tracking in (None, "", "null"):
        tracking = []
    elif isinstance(tracking, str):
        tracking = re.split(r"[\s,;|]+", tracking.strip())
    elif not isinstance(tracking, list):
        tracking = [str(tracking)]
    tracking = [str(t).strip() for t in tracking if str(t).strip() not in ("", "null", "None")]
    # match carrier name case-insensitively
    for name in CARRIER_NAMES:
        if carrier and carrier.lower() == name.lower():
            carrier = name
    return carrier, tracking, answer


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only test the first N emails")
    parser.add_argument("--only", default=None,
                        help='only use files whose name contains this word, e.g. --only dpd')
    args = parser.parse_args()

    paths = ev.find_files("jijikat")
    if args.only:
        paths = [p for p in paths if args.only.lower() in os.path.basename(p).lower()]
        if not paths:
            sys.exit(f"No package file with '{args.only}' in its name.")
    print("Reading package emails from:", ", ".join(os.path.basename(p) for p in paths))

    seen = set()
    rows = []
    stats = {}  # per labeled carrier: Counter
    start = time.time()

    for path in paths:
        for msg in mailbox.mbox(path):
            if args.limit and len(rows) >= args.limit:
                break
            subject = ev.get_header(msg, "Subject")
            msg_id = ev.get_header(msg, "Message-ID").strip() or f"{path}:{subject}"
            if msg_id in seen:
                continue
            seen.add(msg_id)

            sender = ev.get_header(msg, "From")
            label_carrier = ev.get_carrier(msg)
            everything = full_text(msg, subject)

            carrier, tracking, raw = ask_model(
                PROMPT.format(sender=sender, subject=subject, text=text_for_model(msg)))

            # tracking checks (every number the model gave must pass)
            everything_n = normalize(everything)
            if not tracking:
                tracking_result = "none"
            elif any("*" in t for t in tracking):
                tracking_result = "masked"      # partly hidden number: should have been []
            elif any(len(normalize(t)) < 8 for t in tracking):
                tracking_result = "too_short"   # e.g. "96": not a real tracking number
            elif all(normalize(t) in everything_n for t in tracking):
                tracking_result = "found_in_email"
            else:
                tracking_result = "made_up"

            pattern = PATTERNS.get(label_carrier)
            regex_hits = sorted(set(re.findall(pattern, everything))) if pattern else []
            regex_hits = [h if isinstance(h, str) else h[0] for h in regex_hits]
            model_set = {normalize(t) for t in tracking}
            regex_set = {normalize(h) for h in regex_hits}
            if model_set and regex_set:
                agrees = "yes" if model_set <= regex_set else "no"
            elif not model_set and not regex_set:
                agrees = "both found nothing"
            elif model_set:
                agrees = "only model found one"
            else:
                agrees = "only pattern found one"

            # real score where we have an automatic label
            expected = labelled_answer(label_carrier, msg, subject)
            validated_set = model_set
            if label_carrier in VALID_FORMATS:
                validated_set = {t for t in model_set if re.fullmatch(VALID_FORMATS[label_carrier], t)}
            if expected:
                expected_set = set() if expected == "none" else set(expected.split(" | "))
                exact = "yes" if model_set == expected_set else "NO"
                exact_validated = "yes" if validated_set == expected_set else "NO"
            else:
                exact = exact_validated = ""

            carrier_ok = carrier == label_carrier
            s = stats.setdefault(label_carrier, Counter())
            s["total"] += 1
            s["carrier_ok"] += carrier_ok
            s[tracking_result] += 1
            s["agree_" + agrees] += 1
            if exact:
                s["labelled"] += 1
                s["exact_ok"] += exact == "yes"
                s["exact_validated_ok"] += exact_validated == "yes"
                if expected == "none":
                    s["expected_none"] += 1
                    s["none_ok"] += exact == "yes"

            rows.append({
                "file": os.path.basename(path),
                "label_carrier": label_carrier,
                "model_carrier": carrier or "",
                "carrier_ok": "yes" if carrier_ok else "NO",
                "model_tracking": " | ".join(tracking),
                "tracking_check": tracking_result,
                "pattern_found": " | ".join(regex_hits[:3]),
                "model_vs_pattern": agrees,
                "correct_tracking": expected,  # automatic for DPD; fill in others yourself
                "exact_match": exact,
                "exact_after_format_check": exact_validated,
                "from": sender,
                "subject": subject,
                "raw_answer": raw.strip(),
            })
            flags = {"made_up": "MADE UP", "masked": "MASKED", "too_short": "TOO SHORT"}
            flag = flags.get(tracking_result, "" if carrier_ok else "carrier?")
            shown = " | ".join(tracking) or "-"
            print(f"  [{len(rows):>4}] {label_carrier:<7} -> {(carrier or '-'):<7} "
                  f"{shown[:30]:<30} {flag:<9} {subject[:40]}")

    # ---------- summary ----------
    print("\n" + "=" * 70)
    total = Counter()
    for carrier, s in sorted(stats.items()):
        total.update(s)
        print(f"{carrier}: {s['total']} emails")
        print(f"    carrier right:            {s['carrier_ok']}/{s['total']}")
        print(f"    tracking found in email:  {s['found_in_email']}   made up: {s['made_up']}   "
              f"masked: {s['masked']}   too short: {s['too_short']}   none: {s['none']}")
        if s["labelled"]:
            print(f"    EXACT MATCH (Pakketnummer rule): {s['exact_ok']}/{s['labelled']}   "
                  f"of which 'no number in email' answered correctly: "
                  f"{s['none_ok']}/{s['expected_none']}")
            print(f"    EXACT MATCH after format check (model + code): "
                  f"{s['exact_validated_ok']}/{s['labelled']}")
        print(f"    agrees with pattern:      {s['agree_yes']}   disagrees: {s['agree_no']}   "
              f"only pattern found one: {s['agree_only pattern found one']}   "
              f"only model found one: {s['agree_only model found one']}")
    if total["total"]:
        print(f"\nOverall carrier right: {total['carrier_ok']}/{total['total']} "
              f"({100 * total['carrier_ok'] / total['total']:.1f}%)")
        print(f"Overall tracking: found in email {total['found_in_email']}, made up {total['made_up']}, "
              f"masked {total['masked']}, too short {total['too_short']}, none {total['none']}")
    print(f"Time: {time.time() - start:.0f} seconds")

    with open("extraction.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)
    print("\nAll answers saved to extraction.csv")


if __name__ == "__main__":
    main()
