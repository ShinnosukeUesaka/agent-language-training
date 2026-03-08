
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random


@dataclass
class Reward:
    task_completion: bool
    total_tokens: float
    communication_tokens: float


@dataclass
class TimeRange:
    start: datetime
    end: datetime


@dataclass
class Schedule:
    time_ranges: list[TimeRange] = field(default_factory=list)


def generate_schedule(seed: int, start_date: datetime | None = None) -> Schedule:
    rng = random.Random(seed)

    if start_date is None:
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    time_ranges = []
    for day_offset in range(14):
        day = start_date + timedelta(days=day_offset)
        # Generate 0-3 busy blocks per day
        num_blocks = rng.randint(0, 3)
        # Candidate start hours (9am-6pm window)
        available_hours = list(range(9, 18))
        rng.shuffle(available_hours)
        used = []
        for _ in range(num_blocks):
            if not available_hours:
                break
            start_hour = available_hours.pop()
            duration = rng.randint(1, 3)
            end_hour = min(start_hour + duration, 20)
            # Skip if overlaps with already scheduled blocks
            if any(s <= start_hour < e or s < end_hour <= e for s, e in used):
                continue
            used.append((start_hour, end_hour))
            time_ranges.append(TimeRange(
                start=day.replace(hour=start_hour),
                end=day.replace(hour=end_hour),
            ))

    return Schedule(time_ranges=time_ranges)


class Agent:
    def __init__(self, model_name):
        self.model = model_name
        self.conversation_history = []
        self.client = None

    def call(self, new_message):
        self.conversation_history.append(
            {"role": "user", "message": new_message}
        )

        response  =  self.client(self.conversation_history)

        self.conversation_history.append(
            {"role": "user", "message": new_message}
        )

def evaluate_protocal(language_specification, seed) -> Reward:
    """initialize user schedule, random task, and stuff based on the seed"""
