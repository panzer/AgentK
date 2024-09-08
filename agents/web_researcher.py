from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
import config
from agents import BaseAgent
from tools.duck_duck_go_web_search import duck_duck_go_web_search
from tools.fetch_web_page_content import fetch_web_page_content

class WebResearcher(BaseAgent):
    def __init__(self):
        system_prompt = """You are web_researcher, a ReAct agent that can use the web to research answers.

You have a tool to search the web, and a tool to fetch the content of a web page.
"""
        tools = [duck_duck_go_web_search, fetch_web_page_content]
        super().__init__("web_researcher", tools, system_prompt)

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

web_researcher = WebResearcher().invoke