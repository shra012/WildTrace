"""Chat intent parsing: an OpenRouter LLM when a key is set, with a rule-based fallback.

The parser only decides what to do; jobs.py and runs.py do it. Intents:
  pick      - draw an existing trajectory of a category ("draw a cat")
  generate  - make a new trajectory from an unused photo ("make a new sitting dog")
  draw      - start drawing the latest ready job ("draw it", "start")
  chat      - anything else (answered with a short reply and suggestions)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import settings

ANIMALS = {
    "bird": "Bird", "birds": "Bird", "parrot": "Bird", "owl": "Bird", "eagle": "Bird",
    "cat": "Cat", "cats": "Cat", "kitten": "Cat", "kitty": "Cat",
    "dog": "Dog", "dogs": "Dog", "puppy": "Dog", "pup": "Dog",
    "fish": "Fish", "goldfish": "Fish",
    "frog": "Frog", "frogs": "Frog", "toad": "Frog",
    "horse": "Horse", "horses": "Horse", "pony": "Horse",
}
SUGGESTIONS = ["Draw a cat", "Make a new sitting dog", "Draw my face (upload a photo)"]


@dataclass
class Intent:
    action: str                      # pick | generate | draw | chat
    category: str | None = None
    hint: str | None = None          # breed / pose / view words, e.g. "sitting", "left profile"
    controller: str | None = None    # act | ik
    reply: str | None = None         # LLM-written reply for "chat"
    backend: str = "rules"
    extra: dict[str, Any] = field(default_factory=dict)


def _category(text: str) -> str | None:
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in ANIMALS:
            return ANIMALS[word]
    return None


def _hint(text: str, category: str | None) -> str | None:
    stop = {"a", "an", "the", "new", "me", "please", "draw", "make", "generate", "create", "show", "of", "one",
            "another", "fresh", "some", "can", "you", "i", "want", "would", "like", "to", "let's", "lets", "and",
            "with", "using", "ik", "act"}
    words = [w for w in re.findall(r"[a-z0-9/']+", text.lower()) if w not in stop and ANIMALS.get(w) != category]
    return " ".join(words) or None


def parse_rules(message: str, has_ready_job: bool) -> Intent:
    text = message.strip().lower()
    controller = "ik" if re.search(r"\bik\b|inverse kinematics", text) else ("act" if re.search(r"\bact\b", text) else None)
    category = _category(text)
    starts = re.search(r"\b(draw|start|go|run)\s+(it|this|that|now)\b", text) or re.fullmatch(r"(start|go|run|draw)[.!]*", text)
    if starts and not category:
        return Intent("draw", controller=controller)
    if category and re.search(r"\b(new|generate|create|make|fresh|another photo)\b", text):
        return Intent("generate", category=category, hint=_hint(text, category))
    if category:
        return Intent("pick", category=category, hint=_hint(text, category), controller=controller)
    if re.search(r"\b(face|portrait|selfie|me)\b", text):
        return Intent("chat", reply="Attach a face photo (paperclip or drag and drop) and I will turn it into line art.")
    if has_ready_job and re.search(r"\b(yes|yeah|sure|ok|okay|yep)\b", text):
        return Intent("draw", controller=controller)
    return Intent("chat")


_SYSTEM = """You route requests for WildTrace Studio, where a simulated xArm 7 robot draws line art with a pen.
Reply with one JSON object only: {"action": "pick"|"generate"|"draw"|"chat", "category": one of
["Bird","Cat","Dog","Fish","Frog","Horse"] or null, "hint": short breed/pose/view words or null,
"controller": "act"|"ik"|null, "reply": one or two short friendly sentences for the user}.
pick = draw an existing drawing of a named animal ("draw a cat", "show me a horse"). generate = make a NEW drawing from a fresh photo (words like new,
generate, create, make). draw = start drawing the current drawing, only when no animal is named (draw it, start, go, yes). chat = anything else;
for faces tell the user to attach a photo. Only these six animals exist; for others pick chat and say which exist."""


def parse_llm(message: str, history: list[dict[str, str]], has_ready_job: bool) -> Intent | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    context = f"A drawing is ready to draw: {'yes' if has_ready_job else 'no'}."
    messages = [{"role": "system", "content": _SYSTEM + "\n" + context}, *history[-6:], {"role": "user", "content": message}]
    try:
        response = httpx.post(
            f"{settings.OPENROUTER_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "X-Title": "WildTrace Studio"},
            json={"model": settings.CHAT_MODEL, "messages": messages, "temperature": 0.2,
                  "response_format": {"type": "json_object"}, "max_tokens": 200},
            timeout=20.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        data = json.loads(content[content.find("{"): content.rfind("}") + 1])
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return None
    action = data.get("action") if data.get("action") in ("pick", "generate", "draw", "chat") else "chat"
    category = data.get("category") if data.get("category") in set(ANIMALS.values()) else None
    if action == "draw" and category:
        action = "pick"  # "draw a cat" names a new drawing rather than starting the current one
        data["reply"] = None  # written for the wrong action
    if action in ("pick", "generate") and not category:
        action = "chat"
    controller = data.get("controller") if data.get("controller") in ("act", "ik") else None
    return Intent(action, category=category, hint=data.get("hint") or None, controller=controller,
                  reply=data.get("reply") or None, backend=f"openrouter: {settings.CHAT_MODEL}")


# Ordinals before counting words, and no "one": "the second one" means 2.
_ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
             "fifth": 5, "5th": 5, "sixth": 6, "6th": 6, "seventh": 7, "7th": 7, "eighth": 8, "8th": 8,
             "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def parse_face_choice(message: str, count: int) -> int | None:
    """Which face (1-based, left to right) a message picks: "2", "the second one", "left", "rightmost"."""
    text = message.strip().lower()
    number = re.search(r"\b([1-9])\b", text)
    if number:
        index = int(number.group(1))
    elif re.search(r"\b(left|leftmost)\b", text):
        index = 1
    elif re.search(r"\b(right|rightmost|last)\b", text):
        index = count
    elif re.search(r"\b(middle|center|centre)\b", text) and count % 2 == 1:
        index = (count + 1) // 2
    else:
        index = next((n for word, n in _ORDINALS.items() if re.search(rf"\b{word}\b", text)), None)
    return index if index is not None and 1 <= index <= count else None


def parse(message: str, history: list[dict[str, str]], has_ready_job: bool) -> Intent:
    rules = parse_rules(message, has_ready_job)
    if rules.action == "draw":
        return rules  # "draw it" / "start" is unambiguous; the LLM tends to re-pick from history
    return parse_llm(message, history, has_ready_job) or rules


def backend_name() -> str:
    return f"openrouter: {settings.CHAT_MODEL}" if os.environ.get("OPENROUTER_API_KEY") else "rules"
