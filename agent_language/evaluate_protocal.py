import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

load_dotenv()

TASK_COMPLETE_KEYWORD = "TASK_COMPLETE"
MAX_TURNS = 30
RESULTS_FILE = "results.json"


@dataclass
class TimeSlot:
    day: str
    location: str
    start: float  # hours in 24h (e.g. 10.5 = 10:30)
    end: float

    def contains(self, day: str, location: str, time: float, duration: float) -> bool:
        return (
            self.day == day
            and self.location.lower() == location.lower()
            and self.start <= time
            and time + duration <= self.end
        )


@dataclass
class Schedule:
    name: str
    slots: list[TimeSlot]

    def is_available(
        self, day: str, location: str, time: float, duration: float
    ) -> bool:
        return any(slot.contains(day, location, time, duration) for slot in self.slots)

    def to_natural(self) -> str:
        day_names = {
            "Mo": "Monday",
            "Tu": "Tuesday",
            "We": "Wednesday",
            "Th": "Thursday",
            "Fr": "Friday",
        }
        parts = []
        for slot in self.slots:
            start_str = _format_time(slot.start)
            end_str = _format_time(slot.end)
            parts.append(
                f"{day_names[slot.day]} in {slot.location}, {start_str}-{end_str}"
            )
        return "; ".join(parts)


def _format_time(t: float) -> str:
    hours = int(t)
    minutes = int((t - hours) * 60)
    if minutes == 0:
        return str(hours)
    return f"{hours}:{minutes:02d}"


def _parse_time(s: str) -> float:
    if ":" in s:
        h, m = s.split(":")
        return int(h) + int(m) / 60
    return float(s)


def verify_meeting(
    schedules: list[Schedule], day: str, location: str, time: float, duration: float
) -> tuple[bool, list[str]]:
    errors = []
    for schedule in schedules:
        if not schedule.is_available(day, location, time, duration):
            time_str = _format_time(time)
            errors.append(
                f"{schedule.name} is NOT available on {day} at {time_str} ({location})"
            )
    return len(errors) == 0, errors


DAY_ALIASES: dict[str, str] = {
    "monday": "Mo",
    "tuesday": "Tu",
    "wednesday": "We",
    "thursday": "Th",
    "friday": "Fr",
    "mon": "Mo",
    "tue": "Tu",
    "wed": "We",
    "thu": "Th",
    "fri": "Fr",
    "mo": "Mo",
    "tu": "Tu",
    "we": "We",
    "th": "Th",
    "fr": "Fr",
}


def parse_compact_result(text: str) -> tuple[str, str, float] | None:
    pattern = r"=>\s*([A-Za-z]{2,9})\[([A-Za-z]+)\](\d{1,2}(?::\d{2})?)\s*-\s*\d{1,2}(?::\d{2})?"
    match = re.search(pattern, text)
    if not match:
        return None
    raw_day = match.group(1).lower()
    day = DAY_ALIASES.get(raw_day, match.group(1))
    location = match.group(2)
    time = _parse_time(match.group(3))
    return day, location, time


@dataclass
class Session:
    client: OpenAI
    model: str
    name: str = ""
    system_prompt: str = ""
    messages: list[ChatCompletionMessageParam] = field(default_factory=list)
    total_completion_tokens: int = 0
    turns: int = 0

    def __post_init__(self) -> None:
        if self.system_prompt:
            sys_msg: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": self.system_prompt,
            }
            self.messages.append(sys_msg)

    def send(self, content: str) -> str:
        user_msg: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": content,
        }
        self.messages.append(user_msg)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=500,
        )
        assistant_content = response.choices[0].message.content or ""
        assistant_msg: ChatCompletionAssistantMessageParam = {
            "role": "assistant",
            "content": assistant_content,
        }
        self.messages.append(assistant_msg)

        if response.usage:
            self.total_completion_tokens += response.usage.completion_tokens

        self.turns += 1
        return assistant_content

    def is_complete(self) -> bool:
        if not self.messages:
            return False
        last = self.messages[-1]
        content = last.get("content")
        return (
            last["role"] == "assistant"
            and isinstance(content, str)
            and (TASK_COMPLETE_KEYWORD in content or "=>" in content)
        )


def negotiate(
    agent_a: Session, agent_b: Session, max_turns: int = MAX_TURNS
) -> list[dict[str, str]]:
    conversation: list[dict[str, str]] = []

    response = agent_a.send("Propose a meeting time.")
    conversation.append({"agent": agent_a.name, "content": response})

    for _ in range(max_turns):
        if agent_a.is_complete():
            break

        response = agent_b.send(response)
        conversation.append({"agent": agent_b.name, "content": response})
        if agent_b.is_complete():
            break

        response = agent_a.send(response)
        conversation.append({"agent": agent_a.name, "content": response})

    return conversation


MEETING_DURATION = 30  # minutes
DAYS = ["Mo", "Tu", "We", "Th", "Fr"]
CITIES = ["SF", "NYC"]
MIN_HOUR = 8
MAX_HOUR = 18


def generate_schedules(
    num_overlaps: int, rng: random.Random
) -> tuple[Schedule, Schedule]:
    days = DAYS[:]
    rng.shuffle(days)

    overlap_days = days[:num_overlaps]
    filler_days = days[num_overlaps:]

    a_slots: list[TimeSlot] = []
    b_slots: list[TimeSlot] = []

    for day in overlap_days:
        city = rng.choice(CITIES)
        overlap_start = rng.randint(MIN_HOUR + 1, MAX_HOUR - 2)
        overlap_end = rng.randint(
            overlap_start + 1, min(overlap_start + 3, MAX_HOUR - 1)
        )
        a_start = rng.randint(MIN_HOUR, overlap_start)
        a_end = rng.randint(overlap_end, MAX_HOUR)
        b_start = rng.randint(MIN_HOUR, overlap_start)
        b_end = rng.randint(overlap_end, MAX_HOUR)
        a_slots.append(TimeSlot(day, city, float(a_start), float(a_end)))
        b_slots.append(TimeSlot(day, city, float(b_start), float(b_end)))

    for day in filler_days:
        strategy = rng.choice(["a_only", "b_only", "diff_cities"])
        if strategy == "a_only":
            city = rng.choice(CITIES)
            start = rng.randint(MIN_HOUR, MAX_HOUR - 2)
            end = rng.randint(start + 2, MAX_HOUR)
            a_slots.append(TimeSlot(day, city, float(start), float(end)))
        elif strategy == "b_only":
            city = rng.choice(CITIES)
            start = rng.randint(MIN_HOUR, MAX_HOUR - 2)
            end = rng.randint(start + 2, MAX_HOUR)
            b_slots.append(TimeSlot(day, city, float(start), float(end)))
        else:
            city_a, city_b = rng.sample(CITIES, 2)
            start_a = rng.randint(MIN_HOUR, MAX_HOUR - 2)
            end_a = rng.randint(start_a + 2, MAX_HOUR)
            start_b = rng.randint(MIN_HOUR, MAX_HOUR - 2)
            end_b = rng.randint(start_b + 2, MAX_HOUR)
            a_slots.append(TimeSlot(day, city_a, float(start_a), float(end_a)))
            b_slots.append(TimeSlot(day, city_b, float(start_b), float(end_b)))

    day_order = {d: i for i, d in enumerate(DAYS)}
    a_slots.sort(key=lambda s: day_order[s.day])
    b_slots.sort(key=lambda s: day_order[s.day])

    return Schedule("T", a_slots), Schedule("J", b_slots)


def compute_valid_meetings(
    sched_a: Schedule, sched_b: Schedule, duration: float
) -> list[dict[str, str | float]]:
    valid: list[dict[str, str | float]] = []
    for slot_a in sched_a.slots:
        for slot_b in sched_b.slots:
            if (
                slot_a.day != slot_b.day
                or slot_a.location.lower() != slot_b.location.lower()
            ):
                continue
            overlap_start = max(slot_a.start, slot_b.start)
            overlap_end = min(slot_a.end, slot_b.end)
            if overlap_end - overlap_start >= duration:
                valid.append(
                    {
                        "day": slot_a.day,
                        "location": slot_a.location,
                        "start": overlap_start,
                        "end": overlap_end,
                    }
                )
    return valid


def run_trial(
    client: OpenAI,
    model: str,
    lang_spec: str,
    rng: random.Random,
) -> dict:
    num_overlaps = rng.choice([0, 1, 2])
    t_schedule, j_schedule = generate_schedules(num_overlaps, rng)
    duration = MEETING_DURATION / 60
    valid_meetings = compute_valid_meetings(t_schedule, j_schedule, duration)

    agent_t = Session(
        client=client,
        model=model,
        name="T",
        system_prompt=(
            f"You are T. Your availability: {t_schedule.to_natural()}\n"
            f"Meeting duration: {MEETING_DURATION} minutes.\n" + RULES + lang_spec
        ),
    )

    agent_j = Session(
        client=client,
        model=model,
        name="J",
        system_prompt=(
            f"You are J. Your availability: {j_schedule.to_natural()}\n"
            f"Meeting duration: {MEETING_DURATION} minutes.\n" + RULES + lang_spec
        ),
    )

    conversation = negotiate(agent_t, agent_j)

    combined_completion_tokens = (
        agent_t.total_completion_tokens + agent_j.total_completion_tokens
    )

    # Check if agents said NO_VALID_TIME
    said_no_valid = any("NO_VALID_TIME" in msg["content"] for msg in conversation)

    # Check if agents proposed a meeting
    meeting_result = None
    for msg in reversed(conversation):
        parsed = parse_compact_result(msg["content"])
        if parsed:
            meeting_result = parsed
            break

    correct = False
    errors: list[str] = []

    if said_no_valid and not meeting_result:
        if not valid_meetings:
            correct = True
        else:
            errors.append("Agent said NO_VALID_TIME but valid meetings exist")
    elif meeting_result:
        day, location, time = meeting_result
        correct, errors = verify_meeting(
            [t_schedule, j_schedule], day, location, time, duration
        )
    else:
        errors.append("No meeting proposed and no NO_VALID_TIME signal")

    combined_chars = sum(len(msg["content"]) for msg in conversation)

    return {
        "correct": correct,
        "errors": errors,
        "num_overlaps": num_overlaps,
        "valid_meetings": valid_meetings,
        "schedules": {
            "T": t_schedule.to_natural(),
            "J": j_schedule.to_natural(),
        },
        "combined_completion_tokens": combined_completion_tokens,
        "combined_chars": combined_chars,
        "total_turns": agent_t.turns + agent_j.turns,
        "agents": {
            agent_t.name: {
                "turns": agent_t.turns,
                "completion_tokens": agent_t.total_completion_tokens,
            },
            agent_j.name: {
                "turns": agent_j.turns,
                "completion_tokens": agent_j.total_completion_tokens,
            },
        },
        "meeting": (
            {
                "day": meeting_result[0],
                "location": meeting_result[1],
                "time": meeting_result[2],
            }
            if meeting_result
            else None
        ),
        "conversation": conversation,
    }


def run_experiment(
    client: OpenAI,
    model: str,
    lang_spec: str,
    n: int,
    experiment_id: str | None = None,
    max_workers: int = 8,
) -> dict:
    exp_id = experiment_id or "unnamed"
    trials = [None] * n

    def _run(i: int) -> tuple[int, dict]:
        rng = random.Random()
        return i, run_trial(client, model, lang_spec, rng)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run, i): i for i in range(n)}
        for future in as_completed(futures):
            i, trial = future.result()
            trials[i] = trial
            completed += 1
            status = "CORRECT" if trial["correct"] else "INCORRECT"
            print(
                f"[{completed}/{n}] {status} | "
                f"chars={trial['combined_chars']} | "
                f"tokens={trial['combined_completion_tokens']} | "
                f"turns={trial['total_turns']}"
            )

    experiment = {
        "experiment_id": exp_id,
        "model": model,
        "lang_spec": lang_spec,
        "num_trials": n,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trials": trials,
    }

    path = Path(RESULTS_FILE)
    results: list[dict] = []
    if path.exists():
        results = json.loads(path.read_text())
    results.append(experiment)
    path.write_text(json.dumps(results, indent=2) + "\n")

    correct_count = sum(1 for t in trials if t["correct"])
    chars = [t["combined_chars"] for t in trials]
    tokens = [t["combined_completion_tokens"] for t in trials]
    print(
        f"\nExperiment {exp_id}: "
        f"{correct_count}/{n} correct | "
        f"mean_chars={sum(chars) / len(chars):.0f} | "
        f"mean_tokens={sum(tokens) / len(tokens):.0f}"
    )

    return experiment


RULES = """\
Rules:
- You can ONLY be in the city listed for each day. You CANNOT travel or change cities.
- You can ONLY meet if BOTH people are in the SAME city on the SAME day.
- Reject any proposal where you are in a different city than the other person.
- When agreed, respond with => <day>[<city>]<start>-<end> and TASK_COMPLETE (e.g. => Fr[NYC]9-9:30)
- If no valid meeting time exists, respond with NO_VALID_TIME and TASK_COMPLETE
"""

LANG_SPECS: dict[str, str] = {
    "compact": """\
You communicate using a compact scheduling protocol. Here is the format:

M? d=<minutes> z=<timezone> w=<day range> p=<preference>
<name>: <day>[<city>]<start>-<end>,<start>-<end>;<day>[<city>]<start>-<end>
=> <day>[<city>]<start>-<end>

Example:
M? d=30 z=ET w=Mo-Fr p=earliest
T: Mo[SF]9-12;Tu[NYC]13-17;Th[SF]10-15;Fr[NYC]9-11
J: Mo[NYC]10-14;Tu[SF]9-12;We[SF]13-16;Th[NYC]11-15;Fr[NYC]9-11
=> Fr[NYC]9-9:30

- Times are in 24h format
- Days: Mo,Tu,We,Th,Fr
- Locations in brackets: [SF], [NYC]
- You MUST use this compact format for ALL messages, no natural language
- To propose: send your available slots in compact format
- To accept: respond with => <day>[<city>]<start>-<end>
- To reject/counter: send your slots that conflict and suggest alternatives
""",
    "natural": """\
Negotiate with the other person to find a 30-minute in-person meeting time.
Keep responses short (1-2 sentences).
""",
}


def main() -> None:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    model = "google/gemini-3-flash-preview"

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    for spec_name, lang_spec in LANG_SPECS.items():
        run_experiment(client, model, lang_spec, n, spec_name)


def evaluate_lang_spec(lang_spec: str, n: int = 5) -> float:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    model = "google/gemini-3-flash-preview"

    def _run(_: int) -> dict:
        return run_trial(client, model, lang_spec, random.Random())

    with ThreadPoolExecutor(max_workers=n) as executor:
        trials = list(executor.map(_run, range(n)))

    Path("sample_conversation.json").write_text(json.dumps(trials[0], indent=2) + "\n")

    return sum(t["combined_completion_tokens"] for t in trials) / len(trials)


if __name__ == "__main__":
    main()