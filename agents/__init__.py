from abc import ABC
from typing import Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolCall
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
import config
from utils import color, italicize

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
    
    @property
    def gpt_model(self) -> BaseChatModel:
        return config.default_langchain_model

    def reasoning(self, state: MessagesState):
        messages = state['messages']
        tooled_up_model = self.gpt_model.bind_tools(self.tools)
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
        print(f"{color(self.name, 'blue')}: {message}")

    def think(self, thought: str):
        self.say(italicize(color(thought, 'grey')))

    def announce_tool_call(self, tool_call: ToolCall):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        MAX_LEN = 20
        pretty_args = {k: (v[:MAX_LEN] + '...' if len(v) > MAX_LEN else v) for k, v in tool_args.items()}
        self.say(f"{italicize(color('Using tool', 'grey'))} {color(tool_name, 'green')} {italicize(color(pretty_args, 'grey'))}")
