from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, MessagesState
import config
from agents import BaseAgent

import base64
import pathlib
from langchain_core.tools import tool

### Private tools not shared with other agents
@tool
def load_image_from_filepath(filepath: str):
    """Given a path to a local file, load the image so that it can be used for analysis."""
    path = pathlib.Path(filepath)
    base64_image = base64.b64encode(path.read_bytes()).decode('utf-8')
    extension = path.suffix.lstrip('.')
    print("used the tool")
    return {
        "messages": [
            HumanMessage([{"type": "image_url", "image_url": {"url": f"data:image/{extension};base64,{base64_image}", "detail": "low"}}]),
        ]
    }

@tool
def load_image_from_url(url: str):
    """Given a publicly accessable http or https URL, load the image so that it can be used for analysis."""
    return {
        "messages": [
            HumanMessage([{"type": "image_url", "image_url": {"url": url, "detail": "low"}}]),
        ]
    }

class ImageAnalystState(MessagesState):
    image_url: dict

class ImageAnalyst(BaseAgent):
    def __init__(self):
        system_prompt = """You are image_analyst, a ReAct agent that can view, describe, and draw conclusions from images.
        Be sure to answer the question given to you about the image. You have tools to load images so that you can see them.
        If you cannot load the image for some reason, either a technical error, or not enough information to locate the image, respond looking for more information (or give up).
        """
        tools = [load_image_from_filepath, load_image_from_url]
        self.tools_by_name = {tool.name: tool for tool in tools}
        super().__init__("image_analyst", tools, system_prompt)

    def init_workflow(self):
        self.workflow = StateGraph(ImageAnalystState)
        self.workflow.add_node("reasoning", self.reasoning)
        self.workflow.add_node("tools", self.tool_node)
        
    def tool_node(self, state: ImageAnalystState):
        print("tool node")
        state_update = {}
        for tool_call in state["messages"][-1].tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            state_update.update(observation)
        # print("state update")
        # print(state_update)
        return state_update

    def reasoning(self, state: ImageAnalystState):
        self.print_with_color("is thinking...")
        messages = state['messages']
        print(messages[-1])
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

image_analyst = ImageAnalyst().invoke
