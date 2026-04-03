# An RL enviorment to train the ability to design communication protocols for terse agent to agent communication

Agent-to-agent communication will surpass human-to-human communication in the future. But right now, agents aren't great at using English — and we think we can do a lot better.

---

[![Watch the overview](https://img.youtube.com/vi/D6Q5o14w3ks/0.jpg)](https://youtu.be/D6Q5o14w3ks)

---

## The Problem

Here's a meeting scheduling task in plain English:

> *"I'm available Monday morning in San Francisco, from 9 to noon. Are you free then?"*
> *"Unfortunately I'm in New York on Monday. How about Tuesday?"*

They waste time on pleasantries. "Unfortunately." Filler. Human language carries centuries of social overhead that agents don't need.

## The Idea

We believe there will be a different compact language for every different task. Rather than forcing agents to communicate in English, give them the ability to come up with an **alien language** — one invented for the task at hand.

Crucially, this isn't just a prompt engineering trick. We built an **environment specifically to train this capability into models** — so that inventing efficient communication protocols becomes a learned skill, not a one-off hack.

The agent generates a language specification. The environment evaluates it. Two agents then communicate using that spec to complete the task, and the reward is simply: how many tokens did it take?

Train on that signal, and the language gets better.

## What Happens

With the initial language specification, agents use a lot of tokens. After training with GRPO, they use much less.

The specification the model converges on loves abbreviations. The final language looks something like English — but compressed, task-shaped, and surprisingly effective.

It's not quite alien. But it's not quite English either. It works really, really well.

## The Environment

- A language specification (a prompt) defines how two agents are allowed to communicate
- The agents negotiate to complete a task — here, scheduling a meeting
- Reward = total token usage to reach a correct answer (lower is better)
- A small model (Qwen3-1.7B) is trained with GRPO to write better and better specs

The model never watches the negotiation. It only sees the spec it wrote and the outcome. It learns, through trial and error, what kinds of language lead to efficient coordination.
