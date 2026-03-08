# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Agent Language Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import AgentLanguageAction, AgentLanguageObservation


class AgentLanguageEnv(
    EnvClient[AgentLanguageAction, AgentLanguageObservation]
):
    """
    Client for the Agent Language Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with AgentLanguageEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(AgentLanguageAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = AgentLanguageEnv.from_docker_image("agent_language-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(AgentLanguageAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: AgentLanguageAction) -> Dict:
        """
        Convert AgentLanguageAction to JSON payload for step message.

        Args:
            action: AgentLanguageAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "language_specification": action.language_specification,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AgentLanguageObservation]:
        """
        Parse server response into StepResult[AgentLanguageObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with AgentLanguageObservation
        """
        obs_data = payload.get("observation", {})
        observation = AgentLanguageObservation(
            message=obs_data.get("message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
