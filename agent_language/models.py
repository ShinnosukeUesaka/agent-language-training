# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Agent Language Environment.

The agent_language environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class AgentLanguageAction(Action):
    """Action for the Agent Language environment - just a message to echo."""

    language_specification: str = Field(..., description="Language Specification")


class AgentLanguageObservation(Observation):
    """Observation from the Agent Language environment - the echoed message."""

    message: str = Field(default="", description="Scenario")

class AgentLanguageState(State):
    """Custom state fields."""