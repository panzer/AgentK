from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
import config
from agents import BaseAgent
import utils

class SoftwareEngineer(BaseAgent):
    def __init__(self):
        system_prompt = """You are software_engineer, a ReAct agent that can create, modify, and delete code.

You have tools to manage files, run shell commands, and collaborate with other agents by assigning them tasks.
"""
        super().__init__("software_engineer", tools=utils.all_tool_functions(), system_prompt=system_prompt)

    def reasoning(self, state: MessagesState):
        self.print_with_color("is thinking...")
        messages = state['messages']
        tooled_up_model = config.default_langchain_model.bind_tools(self.tools)
        response = tooled_up_model.invoke(messages)
        return {"messages": [response]}

    def check_for_tool_calls(self, state: MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]
        
        if last_message.tool_calls:
            if not last_message.content.strip() == "":
                self.print_with_color("thought this:")
                self.print_with_color(last_message.content)
            self.print_with_color("is acting by invoking these tools:")
            self.print_with_color([tool_call["name"] for tool_call in last_message.tool_calls])
            return "tools"
        
        return END

software_engineer = SoftwareEngineer().invoke