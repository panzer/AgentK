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

web_researcher = WebResearcher().invoke