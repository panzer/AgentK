from abc import ABC, abstractmethod


class AbstractBaseApplicationIO(ABC):

    @abstractmethod
    def spawn_agent_io() -> "AbstractBaseAgentIO":
        pass


class AbstractBaseAgentIO(ABC):

    @abstractmethod
    def __init__(display_name: str) -> None:
        pass

    @abstractmethod
    def agent_says(self, message: str) -> None:
        pass

    @abstractmethod
    def agent_thinks(self, thought: str) -> None:
        pass

    @abstractmethod
    def agent_errored(self, error: str) -> None:
        pass

    @abstractmethod
    def agent_uses_tool(self, tool_name: str, payload: object) -> None:
        pass

    @abstractmethod
    async def agent_requires_text_input(self, prompt: str) -> str:
        pass

    @abstractmethod
    def spawn_agent_io(self, display_name: str) -> "AbstractBaseAgentIO":
        pass

