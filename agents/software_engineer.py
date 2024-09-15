from agents import BaseAgent
import utils

class SoftwareEngineer(BaseAgent):
    def __init__(self):
        system_prompt = """You are software_engineer, a ReAct agent that can create, modify, and delete code.

You have tools to manage files, run shell commands, and collaborate with other agents by assigning them tasks.

Importantly: boilderpate or imcomplete code solutions are not acceptable.
"""
        super().__init__("software_engineer", tools=utils.all_tool_functions(), system_prompt=system_prompt)

software_engineer = SoftwareEngineer().invoke