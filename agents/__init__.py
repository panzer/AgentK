from abc import ABC
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage, ToolCall
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
import config

class BaseAgent(ABC):
    def __init__(self, name: str, tools: list, system_prompt: str):
        self.name = name
        self.tools = tools
        self.system_prompt = system_prompt
        self.init_workflow()
        self.workflow.set_entry_point("reasoning")
        self.workflow.add_conditional_edges(
            "reasoning",
            self.check_for_tool_calls,
        )
        self.workflow.add_edge("tools", "reasoning")
        self.customize_workflow()
        self.graph = self.compile()

    def init_workflow(self):
        self.workflow = StateGraph(MessagesState)
        self.workflow.add_node("reasoning", self.reasoning)
        self.workflow.add_node("tools", ToolNode(self.tools))

    def customize_workflow(self):
        pass

    def compile(self):
        return self.workflow.compile()

    def reasoning(self, state: MessagesState):
        messages = state['messages']
        tooled_up_model = config.default_langchain_model.bind_tools(self.tools)
        response = tooled_up_model.invoke(messages)
        return {"messages": [response]}

    def check_for_tool_calls(self, state: MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]
        
        if last_message.tool_calls:
            if not last_message.content.strip() == "":
                self.think(last_message.content)
            for tool_call in last_message.tool_calls:
                self.announce_tool_call(tool_call)
            return "tools"
        
        return END

    def invoke(self, task: str) -> str:
        return self.graph.invoke(
            {"messages": [SystemMessage(self.system_prompt), HumanMessage(task)]}
        )

    def say(self, message: str):
        print(f"\033[94m{self.name}\033[0m: {message}")

    def think(self, thought: str):
        print(f"\033[94m{self.name}\033[0m: \033[90m\033[3m{thought}\033[0m")

    def announce_tool_call(self, tool_call: ToolCall):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        self.say(f"\033[94mUsing tool \033[92m{tool_name}\033[0m {tool_args}\033[0m")
