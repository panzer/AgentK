from typing import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, merge_content
from langgraph.graph import StateGraph
import config
from agents import BaseAgent

import base64
import httpx
import pathlib
from langchain_core.tools import tool

### Private tools not shared with other agents
@tool
def load_image_from_filepath(filepath: str):
    """Given a path to a local file, load the image so that it can be used for analysis."""
    path = pathlib.Path(filepath)
    base64_image = base64.b64encode(path.read_bytes()).decode('utf-8')
    extension = path.suffix.lstrip('.')
    return [
        # {"type": "text", "text": "Here is the image you requested. Now, complete the initial task."},
        {"type": "image_url", "image_url": {"url": f"data:image/{extension};base64,{base64_image}", "detail": "low"}}
    ]

@tool
def load_image_from_url(url: str):
    """Given a publicly accessable http or https URL, load the image so that it can be used for analysis."""
    response = httpx.get(url)
    if response.is_success:
        return [
            # {"type": "text", "text": "Here is the image you requested. Now, complete the initial task."},
            {"type": "image_url", "image_url": {"url": url, "detail": "low"}}
        ]
    else:
        return [
            {"type": "text", "text": f"Sorry, the GET request for that URL returned Status {response.status_code}. Might there be another URL or local file where we can get the same image?"},
        ]

class ImageAnalystState(TypedDict):
    messages: list[AnyMessage]
    used_tools: bool

class ImageAnalyst(BaseAgent):
    def __init__(self):
        system_prompt = """You are image_analyst, a ReAct agent that can view, describe, and draw conclusions from images.
        Be sure to answer the question given to you about the image. If images are already given by the user, assume those are what they want you to analyze.
        If you cannot load the image for some reason, either a technical error, or not enough information to locate the image, respond looking for more information (or give up)."""
        tools = [load_image_from_filepath, load_image_from_url]
        self.tools_by_name = {tool.name: tool for tool in tools}
        super().__init__("image_analyst", tools, system_prompt)

    def init_workflow(self):
        self.workflow = StateGraph(ImageAnalystState)
        self.workflow.add_node("reasoning", self.reasoning)
        self.workflow.add_node("tools", self.tool_node)
        
    def tool_node(self, state: ImageAnalystState):
        image_content = []
        last_message = state["messages"][-1]
        if not last_message.content.strip() == "":
            self.think(last_message.content)
        for tool_call in last_message.tool_calls:
            self.announce_tool_call(tool_call)
            tool = self.tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            # state_update.update(observation)
            image_content.extend(observation)
        def first_text(content: str | list[str | dict]) -> str:
            if type(content) is str:
                return content
            else:
                if type(content[0]) is str:
                    return content[0]
                else:
                    for c in content:
                        if type(c) is dict and c.get("type") == "text":
                            return c.get("text")
            print("WARN: NO VIABLE FIRST MESSAGE FOUND")
            return "Describe this image"  # fallback
        
        return {
            "messages": [
                state["messages"][0],  # keep system message
                HumanMessage(merge_content(
                    [{"type": "text", "text": first_text(state["messages"][1].content)}],  # assume second message is "human" task originally given
                    image_content,
                ))
            ],
            "used_tools": True,
        }

    def reasoning(self, state: ImageAnalystState):
        self.say("is thinking...")
        messages = state['messages']
        print(f"{state.get('used_tools')=}")
        print([f"{m.__class__.__name__}: {m.content}" for m in messages])
        if not state.get("used_tools"):
            tooled_up_model = config.default_langchain_model.bind_tools(self.tools)
        else:
            tooled_up_model = config.default_langchain_model

        response = tooled_up_model.invoke(messages)
        return {"messages": messages + [response]}  # because we are not using MessagesState, manually append messages when desired

image_analyst = ImageAnalyst().invoke
