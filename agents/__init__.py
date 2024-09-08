from abc import ABC, abstractmethod
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

class BaseAgent(ABC):
    def __init__(self, name: str, tools: list, system_prompt: str):
        self.name = name
        self.tools = tools
        self.system_prompt = system_prompt
        self.workflow = StateGraph(MessagesState)
        self.workflow.add_node("reasoning", self.reasoning)
        self.workflow.add_node("tools", ToolNode(self.tools))
        self.workflow.set_entry_point("reasoning")
        self.workflow.add_conditional_edges(
            "reasoning",
            self.check_for_tool_calls,
        )
        self.workflow.add_edge("tools", "reasoning")
        self.customize_workflow()
        self.graph = self.compile()

    def customize_workflow(self):
        pass

    def compile(self):
        return self.workflow.compile()

    @abstractmethod
    def reasoning(self, state: MessagesState):
        pass

    @abstractmethod
    def check_for_tool_calls(self, state: MessagesState) -> Literal["tools", END]:
        pass

    def invoke(self, task: str) -> str:
        return self.graph.invoke(
            {"messages": [SystemMessage(self.system_prompt), HumanMessage(task)]}
        )

    def print_with_color(self, message: str):
        print(f"\033[94m{self.name}\033[0m: {message}")
