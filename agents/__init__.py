from abc import ABC
import traceback
from typing import Literal
import sys

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolCall
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
import config
import utils
from view.io_base import AbstractBaseAgentIO

class BaseAgent(ABC):
    def __init__(self, name: str, tools: list, system_prompt: str, io: AbstractBaseAgentIO):
        self.name = name
        self.tools = tools
        self.system_prompt = system_prompt
        self.io = io
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
        self.io.agent_says(message)
        # print(f"{color(self.name, 'blue')}: {message}")

    def think(self, thought: str):
        self.io.agent_thinks(thought)
        # self.say(italicize(color(thought, 'grey')))

    def announce_tool_call(self, tool_call: ToolCall):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        MAX_LEN = 20
        def abbr_value(v) -> str:
            full_str = str(v)
            if len(full_str) > MAX_LEN:
                return full_str[:MAX_LEN] + '...'
            else:
                return v
        pretty_args = {k: abbr_value(v) for k, v in tool_args.items()}
        self.io.agent_uses_tool(tool_name, pretty_args)
        # self.say(f"{italicize(color('Using tool', 'grey'))} {color(tool_name, 'green')} {italicize(color(pretty_args, 'grey'))}")

    def assign_agent_to_task(self, agent_name: str, task: str) -> str:
        try:
            agent_module = utils.load_module(f"agents/{agent_name}.py")
            agent_function = getattr(agent_module, agent_name)
            result = agent_function(task=task)
            del sys.modules[agent_module.__name__]  # unload module so updates can be reflected for next run
            response = result["messages"][-1].content
            self.io.reasoning(f"{agent_name} responded {response}")
            return response
        except Exception as e:
            exception_trace = traceback.format_exc()
            error = f"An error occurred while assigning {agent_name} to task {task}:\n {e}\n{exception_trace}"
            self.io.agent_errored(error)
            return error
