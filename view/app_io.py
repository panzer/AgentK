
from io_base import AbstractBaseAgentIO, AbstractBaseApplicationIO
from logger_io import LoggerAgentIO, LoggerApplicationIO

class AppIOManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppIOManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.agent_io = LoggerAgentIO()
        self.application_io = LoggerApplicationIO()

    def get_agent_io(self) -> AbstractBaseAgentIO:
        return self.agent_io

    def get_application_io(self) -> AbstractBaseApplicationIO:
        return self.application_io
