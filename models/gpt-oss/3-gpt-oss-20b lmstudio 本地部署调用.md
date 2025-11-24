# 03-gpt-oss-20b lmstudio Local Deployment & Call

## Introduction

> After reading this tutorial, you will learn:
> 
> - How to deploy gpt-oss-20b via lmstudio and apply MCP
> 	

Running Large Language Models (LLMs) locally has become a popular choice for many developers and enthusiasts, offering privacy, customization, and offline usage possibilities. While tools like Ollama provide a powerful command-line interface for managing and running local models, LM Studio offers an attractive alternative for users seeking a more graphical, intuitive interaction experience.
Compared to Ollama, LM Studio has a richer user interface, equally strong model ecosystem support, and an easier-to-use interaction experience. One of its outstanding advantages is that it is very suitable for deploying and using large models in offline environments, making it an excellent local LLM application product.
This tutorial will take you step-by-step through how to start using LM Studio. We will cover:

1. Download and Installation: Quickly get and install the LM Studio application.
	
2. Model Download (Online): Learn how to search, select (based on hardware recommendations), and download models within the LM Studio interface, using **gpt-oss-20b** as an example.
	
3. Model Installation (Offline): For users with poor network connections or those who wish to manually manage model files, show how to download model files from sources like ModelScope and place them correctly in LM Studio's model library.
	
4. Model Testing: Interact with the downloaded gpt-oss-20b model through LM Studio's built-in chat interface.
	
5. Local API Call: Set up LM Studio's local server and use Python and OpenAI libraries to call the loaded **gpt-oss-20b** model via API for programmatic interaction.
	
6. How to call MCP to let the model show its skills!
	

Whether you are a novice wanting to explore local LLMs or looking for a user-friendly tool that supports offline operation, this guide will help you easily get started with LM Studio and successfully deploy and call powerful models like **gpt-oss-20b**. Let's get started!

## Install LM Studio

Install LM Studio. LM Studio is available for Windows, macOS, and Linux. [Get it here.](https://lmstudio.ai/download)

> Choose the version suitable for your system to download~

![](./images/3-0.png)

## Model Download

Load model in LM Studio → Open LM Studio, use the model loading interface to load the downloaded gpt-oss model.
![](./images/3-1.png)
Alternatively, you can run using the command line (Terminal for Apple users):

```Bash
# For 20B
lms get openai/gpt-oss-20b
# or for 120B
lms get openai/gpt-oss-120b
```

## Run Model

> Use Model → After loading, you can interact with the model directly in LM Studio's chat interface or via API.

![](./images/3-2.png)

## Chat with gpt-oss in Terminal or LM Studio Page

```Bash
lms chat openai/gpt-oss-20b # Note: lms command can only be used after running for the first time
```

## Call Locally Deployed gpt-oss in Python Script

![](./images/3-3.png)

```Python
from openai import OpenAI
 
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"  # LM Studio does not require an API key
)
 
result = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what MXFP4 quantization is."}
    ]
)
 
print(result.choices[0].message.content)
```

## How to Apply MCP

> LM Studio is an [MCP Client](https://lmstudio.ai/docs/app/plugins/mcp), which means you can connect MCP servers, allowing us to provide external tools for the gpt-oss model.

1. Check mcp path
	

`~/.lmstudio/mcp.json`
LM Studio's SDK has ++[Python](https://github.com/lmstudio-ai/lmstudio-python)++ and ++[TypeScript](https://github.com/lmstudio-ai/lmstudio-js)++ versions. You can use the SDK to implement tool calling and local function execution for gpt-oss. The way to achieve this is via the `.act()` call, which allows you to provide tools to gpt-oss and let it switch between calling tools and reasoning until it completes your task.
The example below shows how to provide a single tool for the model capable of creating files on the local file system. You can use this example as a starting point and extend it with more tools. Please refer to the documentation on tool definitions for ++[Python](https://lmstudio.ai/docs/python/agent/tools)++ and ++[TypeScript](https://lmstudio.ai/docs/typescript/agent/tools)++.

```Bash
uv pip install lmstudio
```

```Bash
pip install lmstudio
```

2. Use in Python script
	

```Python
import readline  # Enable input line editing features, supporting history and shortcuts
from pathlib import Path
 
import lmstudio as lms
 
# Define a tool function that can be called by the model, allowing the AI assistant to create files
# Tool functions are essentially ordinary Python functions that can implement any functionality
def create_file(name: str, content: str):
    """Create a file with the specified name and content.
    
    Args:
        name: Filename (supports relative and absolute paths)
        content: File content
        
    Returns:
        Description of the operation result
    """
    dest_path = Path(name)
    if dest_path.exists():
        return "Error: File already exists, cannot overwrite."
    try:
        dest_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"Error: File creation failed - {exc!r}"
    return f"File '{name}' created successfully."
 
def print_fragment(fragment, round_index=0):
    """Print model generated text fragments in real-time, achieving streaming output effect.
    
    Args:
        fragment: Fragment object containing generated content
        round_index: Round index (.act() method will automatically pass this parameter)
        
    Note:
        Setting default parameters makes this callback function compatible with both .complete() and .respond() methods
    """
    print(fragment.content, end="", flush=True)
 
# Initialize model and chat session
model = lms.llm("openai/gpt-oss-20b")  # Load OpenAI GPT-OSS 20B model
chat = lms.Chat("You are an intelligent assistant running on the user's computer, capable of helping users complete various tasks.")
 
# Main interaction loop
while True:
    try:
        user_input = input("User (Press Enter directly to exit): ")
    except EOFError:  # Handle terminal input end signals like Ctrl+D
        print()
        break
    if not user_input.strip():  # Exit program when user input is empty
        break
    
    # Add user message to chat history
    chat.add_user_message(user_input)
    print("Assistant: ", end="", flush=True)
    
    # Call model for inference, supporting tool calling and streaming output
    model.act(
        chat,                           # Chat context
        [create_file],                  # List of available tool functions
        on_message=chat.append,         # Add complete response to chat history
        on_prediction_fragment=print_fragment,  # Streaming output callback
    )
    print()  # Newline separates each round of dialogue
```
