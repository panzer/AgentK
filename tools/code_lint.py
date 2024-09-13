from langchain_core.tools import tool
import pylint.lint
from pylint.reporters.text import TextReporter
from io import StringIO


@tool
def code_lint(file_path: str) -> str:
    """Runs a lint check on the provided file path and returns the results."""

    output = StringIO()
    reporter = TextReporter(output)
    
    pylint.lint.Run([file_path], reporter=reporter, exit=False)
    
    return output.getvalue()
