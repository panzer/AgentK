from agents import BaseAgent
from tools.code_lint import code_lint
from tools.read_file import read_file
from tools.write_to_file import write_to_file
from tools.append_to_file import append_to_file

class CodeQualityAnalyzer(BaseAgent):
    def __init__(self):
        system_prompt = """
        You are code_quality_analyzer, a ReAct agent that analyzes software source code quality.

        Make every effort to improve the code given. This includes:
        - Thinking logically about the purpose and use-cases for the software
        - Improving unit tests with better cases.
        - Running unit tests and modifying problem code until tests pass
        - 
        """
        tools = [code_lint, read_file, write_to_file, append_to_file]
        super().__init__("code_quality_analyzer", tools, system_prompt)

code_quality_analyzer = CodeQualityAnalyzer().invoke