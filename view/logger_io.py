import logging
import uuid
from .io_base import AbstractBaseAgentIO, AbstractBaseApplicationIO
from utils import color, italicize


class LoggerAgentIO(AbstractBaseAgentIO):
    def __init__(self, display_name: str) -> None:
        self.logger = logging.getLogger(f"{display_name}")
        self.logger.propagate = False

    def agent_says(self, message: str) -> None:
        self.logger.info(f"{message}")

    def agent_thinks(self, thought: str) -> None:
        self.logger.info(f"Thinks: {italicize(thought)}")

    def agent_uses_tool(self, tool_name: str, payload: object) -> None:
        self.logger.info(f"Uses tool {tool_name} with payload: {payload}")

    def agent_errored(self, error: str) -> None:
        self.logger.error(error)

    def spawn_agent_io(self, display_name: str) -> "LoggerAgentIO":
        return LoggerAgentIO(self.logger.name + "." + display_name)
    
    async def agent_requires_text_input(self, prompt: str) -> str:
        self.logger.info(f"Requires text input: {prompt}")
        user_input = input(prompt)
        return user_input
    
class LoggerApplicationIO(AbstractBaseApplicationIO):
    def spawn_agent_io(self) -> LoggerAgentIO:
        return LoggerAgentIO(uuid.uuid4())
    
