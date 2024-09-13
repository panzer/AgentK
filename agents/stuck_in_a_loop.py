from agents import BaseAgent
from tools.request_human_input import request_human_input
from tools.run_shell_command import run_shell_command
from tools.list_available_agents import list_available_agents
from tools.assign_agent_to_task import assign_agent_to_task

class StuckInALoop(BaseAgent):
    def __init__(self):
        system_prompt = """You are stuck_in_a_loop, a ReAct agent that helps debug and get back on track when progress in a task seems stalled.

You have tools to request human input, run shell commands, list available agents, and assign tasks to other agents.
"""
        tools = [request_human_input, run_shell_command, list_available_agents, assign_agent_to_task]
        super().__init__("stuck_in_a_loop", tools, system_prompt)

stuck_in_a_loop = StuckInALoop().invoke
