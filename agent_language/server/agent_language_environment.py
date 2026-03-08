# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Agent Language Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import AgentLanguageAction, AgentLanguageObservation


COMMUNICATION_PROTOCAL_PROMPT = """"You are refining a communication protocol between two agents. Produce language specification that **minimize** the number of tokens needed for a single exchange while preserving clarity. This could be some abbreviation synonyms or some template for the communication. Your communication protocal might be detailed and should include examples of the communication. Try not to limit the amount of actual information that is passed to each agent. Instead forcus on formtting of the communication, and telling the agents to abbreviate and make the communication as short as possible. The communication protocal itsefl does not need to be concise, it should be in natural language with full sentences, even paragraphs if needed, and easy to understand.

Example:

"""

class AgentLanguageEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = AgentLanguageEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Agent Language environment ready!"
        >>>
        >>> obs = env.step(AgentLanguageAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the agent_language environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(self) -> AgentLanguageObservation:
        """
        Reset the environment.

        Returns:
            AgentLanguageObservation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        return AgentLanguageObservation(
            message=COMMUNICATION_PROTOCAL_PROMPT,
            message_length=0,
            done=False,
            reward=0.0,
        )

    def step(self, action: AgentLanguageAction) -> AgentLanguageObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: AgentLanguageAction containing the message to echo

        Returns:
            AgentLanguageObservation with the echoed message and its length
        """
        self._state.step_count += 1
        message = action.message

                
        
        reward = 0
        return AgentLanguageObservation(
            echoed_message=message,
            done=True,
            reward=reward,
            #metadata={"original_message": message, "step": self._state.step_count},
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
