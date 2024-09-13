import os
from langchain_core.tools import tool

@tool
def append_to_file(file: str, file_contents: str) -> str:
    """Write the contents to an existing file, will not create a new file."""
    if not os.path.exists(file):
        raise FileExistsError(f"File {file} does not exist and will need to be created using write_to_file tool.")

    print(f"Appending to file: {file}")
    with open(file, 'a') as f:
        f.write(file_contents)

    return f"File {file} editted successfully."