
from agent_language.models import AgentLanguageAction
from agent_language.server.agent_language_environment import AgentLanguageEnvironment


class HFEnv:
    def __init__(self):
        self.env = AgentLanguageEnvironment()

    def reset(self):
        return self.env.reset()

    def submit_language_specification(self, language_specification: str):
        """
        Submit a language specification to the environment.

        Args:
            language_specification: Language specification to submit.

        Returns:
            Observation
        """
        return self.env.step(AgentLanguageAction(language_specification=language_specification))