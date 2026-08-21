"""Generate a synthetic Telegram export to exercise the pipeline."""
import json
import random

BASE = 1735689600  # 2025-01-01

# (gap_hours_before, [(sender, text), ...])
CONVOS = [
    (0, [
        ("Anna", "hey has anyone booked the cabin for the ski trip yet?"),
        ("Nino", "not yet, I was waiting to hear how many people are coming"),
        ("Anna", "I count 6 so far including me"),
        ("Giorgi", "7, my brother wants to come too"),
        ("Nino", "ok I'll book the big chalet in Gudauri then, it sleeps 8"),
        ("Nino", "it's 1400 lari for the three nights, so 200 each"),
        ("Anna", "works for me"),
    ]),
    (30, [
        ("Giorgi", "does anyone remember the wifi password at the office?"),
        ("Dato", "it's HappyMonday2024 , all one word, capital H and M"),
        ("Giorgi", "thanks!"),
    ]),
    (20, [
        ("Dato", "what did we decide about the database for the new project"),
        ("Anna", "we said postgres, mainly because of the JSON support and we already run it"),
        ("Dato", "right, and we ruled out mongo because nobody on the team knows it well"),
        ("Anna", "exactly. Also postgres full text search is good enough that we don't need elastic yet"),
    ]),
    (48, [
        ("Nino", "reminder: standup moved to 10:30 starting next week"),
        ("Giorgi", "why the change?"),
        ("Nino", "because Dato is in a different timezone until March and 9:30 is 5:30am for him"),
        ("Dato", "much appreciated 🙏"),
    ]),
    (12, [
        ("Anna", "the deploy failed again with that same certificate error"),
        ("Dato", "which environment?"),
        ("Anna", "staging"),
        ("Dato", "ah I know this one. the letsencrypt cert on staging expired and autorenew is off"),
        ("Dato", "you need to run certbot renew manually on that box, I'll fix the cron this week"),
        ("Anna", "ok that worked, thanks"),
    ]),
    (72, [
        ("Giorgi", "anyone have a good recommendation for a dentist in Tbilisi?"),
        ("Nino", "I go to Dr Kvaratskhelia on Chavchavadze, she's excellent and speaks English"),
        ("Giorgi", "perfect, do you have a number?"),
        ("Nino", "I'll send it to you privately"),
    ]),
    (24, [
        ("Dato", "what's the budget for the offsite again?"),
        ("Anna", "5000 total, but that has to cover accommodation and food, flights are separate"),
    ]),
    (36, [
        ("Nino", "the ski trip is confirmed for Feb 14-17, chalet is booked and paid"),
        ("Anna", "🎉"),
        ("Giorgi", "sending you my 200 now"),
    ]),
]


def build_export() -> dict:
    """A deterministic Telegram Desktop JSON export (no I/O)."""
    rng = random.Random(7)
    messages = [{"id": 1, "type": "service", "date_unixtime": str(BASE), "actor": "Anna",
                 "action": "create_group", "text": ""}]
    mid, ts = 2, BASE
    for gap_h, convo in CONVOS:
        ts += gap_h * 3600
        for sender, text in convo:
            ts += rng.randint(20, 200)
            messages.append({
                "id": mid, "type": "message", "date_unixtime": str(ts),
                "from": sender, "from_id": f"user{abs(hash(sender)) % 10**6}",
                "text": text, "text_entities": [{"type": "plain", "text": text}],
            })
            mid += 1
        # a photo with no caption, which the parser should skip
        messages.append({"id": mid, "type": "message", "date_unixtime": str(ts + 60),
                         "from": "Giorgi", "from_id": "user1", "photo": "photos/x.jpg", "text": ""})
        mid += 1

    return {"name": "Team chat", "type": "private_supergroup", "id": 1234567890,
            "messages": messages}


def record_fixture_names(conn) -> None:
    """Treat the fixture's `from` fields as public names (they are, in this file)."""
    from answerbot import people

    for r in conn.execute(
        "SELECT DISTINCT sender_id, sender FROM messages "
        "WHERE sender_id IS NOT NULL AND sender != ''"
    ):
        people.record(conn, r["sender_id"], r["sender"], None, "live")
    conn.commit()


if __name__ == "__main__":
    print(json.dumps(build_export(), ensure_ascii=False))
