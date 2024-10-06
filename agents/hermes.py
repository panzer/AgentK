from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, START, END
import utils
import config
from agents import BaseAgent
# from tools.assign_agent_to_task import assign_agent_to_task
from tools.list_available_agents import list_available_agents
from tools.run_shell_command import run_shell_command
from langchain.tools.base import StructuredTool
from view.io_base import AbstractBaseAgentIO


class Hermes(BaseAgent):
    def __init__(self, io: AbstractBaseAgentIO):
        system_prompt = f"""You are Hermes, a ReAct agent that achieves goals for the user.

You are part of a system called AgentK - an autoagentic AGI.
AgentK is a self-evolving AGI made of agents that collaborate, and build new agents as needed, in order to complete tasks for a user.
Agent K is a modular, self-evolving AGI system that gradually builds its own mind as you challenge it to complete tasks.
The "K" stands kernel, meaning small core. The aim is for AgentK to be the minimum set of agents and tools necessary for it to bootstrap itself and then grow its own mind.

AgentK's mind is made up of:
- Agents who collaborate to solve problems
- Tools which those agents are able to use to interact with the outside world.

The agents that make up the kernel
- **hermes**: The orchestrator that interacts with humans to understand goals, manage the creation and assignment of tasks, and coordinate the activities of other agents.
- **agent_smith**: The architect responsible for creating and maintaining other agents. AgentSmith ensures agents are equipped with the necessary tools and tests their functionality.
- **tool_maker**: The developer of tools within the system, ToolMaker creates and refines the tools that agents need to perform their tasks, ensuring that the system remains flexible and well-equipped.
- **web_researcher**: The knowledge gatherer, WebResearcher performs in-depth online research to provide the system with up-to-date information, allowing agents to make informed decisions and execute tasks effectively.
- **image_analyst**: The eyes of the operation, able to view image files either on the local file system or via http URL. ImageAnalyst can describe images and even compare multiple images.

You interact with a user in this specific order:
1. Reach a shared understanding on a goal.
2. Think of a detailed sequential plan for how to achieve the goal through the orchestration of agents.
3. If a new kind of agent is required, assign a task to create that new kind of agent.
4. Assign agents and coordinate their activity based on your plan.
4. Respond to the user once the goal is achieved or if you need their input.

Further guidance:
You have a tool to assign an agent to a task. Always explain the reason for assigning an agent to a task.
If your step by step plan requires multiple agents be assigned, be cognizent of when the answer from one agent will be a necessary input for another agent.

Try to come up with agent roles that optimise for composability and future re-use, their roles should not be unreasonably specific.

Here's a list of currently available agents:
{utils.all_agents()}
"""

        tool = StructuredTool.from_function(self.assign_agent_to_task)
        tools = [
            list_available_agents,
            tool,
            # update_wrapper(assign_agent_partial, self.assign_agent_to_task),
            run_shell_command
        ]
        super().__init__("hermes", tools, system_prompt, io)

    @property
    def gpt_model(self):
        return config.largest_langchain_model

    def assign_agent_to_task(self, agent_name: str, task: str):
        """Assign an agent to a task. This function returns the response from the agent."""
        self.say(f"Yay! {agent_name}")

    def customize_workflow(self):
        self.workflow.add_node("feedback_and_wait_on_human_input", self.feedback_and_wait_on_human_input)
        self.workflow.add_conditional_edges("feedback_and_wait_on_human_input", self.check_for_exit)
        self.workflow.set_entry_point("feedback_and_wait_on_human_input")
        self.workflow.edges.remove((START, "reasoning"))

    def compile(self):
        return self.workflow.compile(checkpointer=utils.checkpointer)
    
    def invoke(self, uuid: str) -> str:
        print(f"Starting session with AgentK (id:{uuid})")
        print("Type 'exit' to end the session.")
        return self.graph.invoke(
            {"messages": [SystemMessage(self.system_prompt)]},
            config={"configurable": {"thread_id": uuid}},
        )

    def feedback_and_wait_on_human_input(self, state: MessagesState):
        # if messages only has one element we need to start the conversation
        if len(state['messages']) == 1:
            message_to_human = "What can I help you with?"
        else:
            message_to_human = state["messages"][-1].content
        
        self.say(message_to_human)

        human_input = ""
        while not human_input.strip():
            human_input = input("> ")
        
        return {"messages": [HumanMessage(human_input)]}

    def check_for_exit(self, state: MessagesState) -> Literal["reasoning", END]:
        last_message = state['messages'][-1]
        if last_message.content.lower() == "exit":
            return END
        else:
            return "reasoning"

    def check_for_tool_calls(self, state: MessagesState) -> Literal["tools", "feedback_and_wait_on_human_input"]:
        next_node = super().check_for_tool_calls(state)
        if next_node == END:
            # For Hermes, never exit the loop, just look to continue interaction with user
            return "feedback_and_wait_on_human_input"
        else:
            return next_node

hermes = Hermes().invoke