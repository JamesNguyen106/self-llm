# 01-GLM-4.1V-Thinking vLLM Deployment and Usage

[GLM-4.1V-Thinking](https://github.com/THUDM/GLM-4.1V-Thinking) is a new version of the VLM open-source model launched by Zhipu AI based on the [GLM-4-9B-0414](https://github.com/THUDM/GLM-4) base model. It introduces a thinking paradigm and comprehensively improves model capabilities through Reinforcement Learning with Curriculum Sampling (RLCS), achieving the strongest performance among visual language models at the 10B parameter level.

## Environment Preparation

The basic experimental environment for this article is as follows:

```
----------------
PyTorch  2.5.1
Python  3.12(ubuntu22.04)
CUDA  12.4
GPU  A800-80GB(80GB) * 1
----------------
```

1. Clone the GLM4.1V project repository

```bash
git clone https://github.com/THUDM/GLM-4.1V-Thinking.git
cd GLM-4.1V-Thinking
```

2. Change `pip` source for acceleration, download and install dependency packages

```bash
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope
pip install -r GLM-4.1V-Thinking/requirements.txt
```

Key dependencies include:

- torch>=2.7.1
- gradio>=5.35.0
- PyMuPDF>=1.26.1
- av>=14.4.0 (for video processing)
- accelerate>=1.6.0
- Latest transformers and vLLM from GitHub

## Model Download

Models can be obtained from Hugging Face and ModelScope repositories. You can download them explicitly or execute repository code to download automatically upon first use.

### ModelScope

Create a new .py/.ipynb file, copy the following code and execute it: Use the `snapshot_download` function in modelscope to download the model. The first parameter is the model name, and the parameter `cache_dir` is the custom model download path.

```python
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-8B', cache_dir='/root/autodl-tmp', revision='master')
```

### Hugging Face CLI

```bash
huggingface-cli download THUDM/GLM-4.1V-9B-Thinking --local-dir /root/autodl-tmp/GLM-4.1V-9B-Thinking
```

ps: Remember to modify the corresponding `cache_dir` / `local_dir` to your model download path~

## Transformers Usage Code

Use the `transformers` library as a rewriting interaction script for inference testing.

```python
import argparse
import re
import torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration

def build_content(image_paths, video_path, text):
    content = []
    if image_paths:
        for img_path in image_paths:
            content.append({"type": "image", "url": img_path})
    if video_path:
        content.append({"type": "video", "url": video_path})
    content.append({"type": "text", "text": text})
    return content

processor = AutoProcessor.from_pretrained("/root/autodl-tmp/ZhipuAI/GLM-4.1V-9B-Thinking", use_fast=True)
model = Glm4vForConditionalGeneration.from_pretrained(
    "/root/autodl-tmp/ZhipuAI/GLM-4.1V-9B-Thinking", torch_dtype=torch.bfloat16, device_map="cuda:0"
)
messages = []
messages.append(
    {
        "role": "user",
        "content": [
            {"type": "video", "url": "/root/autodl-tmp/media/demo3.mp4"},
            {"type": "text", "text": "What is the content of this video?"},
        ],
    }
)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    padding=True,
).to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=25000,
    repetition_penalty=1.0,
    do_sample=1.0 > 0,
    top_k=2,
    temperature=1.0,
)
raw = processor.decode(
    output[0][inputs["input_ids"].shape[1] : -1], skip_special_tokens=False
)
match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
answer = match.group(1).strip() if match else ""
messages.append(
    {"role": "assistant", "content": [{"type": "text", "text": answer}]}
)
print(f"Assistant: {raw}")
```

Example video file used:

[vdieo-1.mp4](images/vdieo-1.mp4)

Code execution result is as follows:

![image-2.png](images/image-2.png)

```
Assistant: <think>The user now needs to describe the video content. First look at the screen: mainly a Logi brand keyboard, with hands operating the keys. Break down every detail: the keyboard is black with a silver upper edge, has number keys, function keys (such as ins, delete, home, etc.), hand movements are clicking different keys (such as number keys, function keys), background has a desktop, headphones (white headphones and black headphone cord), and other items. Need to describe in chronological order or scene, explaining that the hand is operating on the keyboard, the interaction of the keys, and the environmental elements.

First, the video shows a person operating a Logi brand keyboard. In the picture, you can see a black keyboard with a silver upper edge, and the keyboard has number keys (0-9, etc.) and function keys (such as ins, delete, home, end, page up/down, etc.). A hand clicks different keys on the keyboard, such as first clicking the number key "2", then "8", etc., showing key operations. In the background, you can see a desktop with white headphones, black headphone cords, and other items, creating an office or desktop scene atmosphere. The overall is a close-up display of keyboard operation, focusing on key interaction and keyboard appearance.

Now organize the language: The video shows a hand operating on a Logi brand black keyboard, which has a silver upper edge and contains number keys and function keys (such as ins, delete, home, end, page up/down, etc.). The hand clicks different keys in turn, and the background shows the desktop, white headphones, and black headphone cords, presenting a scene of operating the keyboard on the desktop, focusing on the interaction of keyboard keys and keyboard appearance details.</think><answer>The video shows a hand operating on a Logi brand black keyboard in a desktop environment. The keyboard has a silver upper edge and contains number keys (0 - 9) and function keys (such as "ins", "delete", "home", "end", "page up/down", etc.). The hand clicks different keys (such as number keys, function keys) in turn. In the background, items such as a desktop, white headphones, and black headphone cords are visible, presenting a close-up scene of operating the keyboard, focusing on the interaction of keyboard keys and keyboard appearance details.
```

## vLLM Usage Code Preparation

First, import the `LLM` and `SamplingParams` classes from the `vLLM` library. The `LLM` class is the main class for running offline inference using the `vLLM` engine. The `SamplingParams` class specifies the parameters of the sampling process, used to control and adjust the randomness and diversity of the generated text.

`vLLM` provides a very convenient wrapper. We can directly pass in the model name or model path without manually initializing the model and tokenizer.

Create a new `vllm_model.py` file in the `/root/autodl-tmp` path and enter the following content:

```python
# vllm_model.py
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json

def prepare_model(model_path, max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=2048):
    # Initialize vLLM inference engine
    llm = LLM(model=model_path, tokenizer=model_path, max_model_len=max_model_len,trust_remote_code=True)
    return llm

def get_completion(prompts, llm, max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=2048):
    stop_token_ids = [151329, 151336, 151338]
    # Create sampling parameters. temperature controls the diversity of generated text, top_p controls the probability of nucleus sampling
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    # Initialize vLLM inference engine
    outputs = llm.generate(prompts, sampling_params)
    return outputs

# Initialize vLLM inference engine
model_path = '/root/autodl-tmp/ZhipuAI/GLM-4.1V-9B-Thinking'
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = prepare_model(model_path)

prompt = "What is the factorial of 5?"
messages = [
    {"role": "user", "content": prompt}
]

# Apply chat template from template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

outputs = get_completion(text, llm, max_tokens=1024, temperature=1, top_p=1, max_model_len=2048)
print(outputs[0].outputs[0].text)
```

Code execution result is as follows:

![image-3.png](images/image-3.png)

GLM-4.1V-9B-Thinking supports multiple types of multimodal inputs, but with specific limitations:

| **Input Type** | **Maximum Allowed Quantity** | **Format Support** |
| --- | --- | --- |
| Image | 10 images (Gradio), 300 (API) | JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP |
| Video | 1 video | MP4, AVI, MKV, MOV, WMV, FLV, WEBM, MPEG, M4V |
| Document | 1 PDF or 1 PPT | PDF, PPT, PPTX (internally converted to images) |

## Create OpenAI API Compatible Server

`GLM-4.1V-Thinking` is compatible with the `OpenAI API` protocol, so we can directly use `vLLM` to create an `OpenAI API` server. `vLLM` makes it very convenient to deploy a server that implements the `OpenAI API` protocol. By default, it will start the server at [http://localhost:8000](http://localhost:8000/). The server currently hosts one model at a time and implements list models, `completions`, and `chat completions` endpoints.

- `completions`: Basic text generation task, the model generates a piece of text after a given prompt. This type of task is usually used to generate articles, stories, emails, etc.
- `chat completions`: Dialogue-oriented task, the model needs to understand and generate dialogue. This type of task is usually used to build chatbots or dialogue systems.

When creating a server, we can specify parameters such as model name, model path, chat template, etc.

- `-host` and `-port` parameters specify the address.
- `-model` parameter specifies the model name.
- `-chat-template` parameter specifies the chat template.
- `-served-model-name` specifies the name of the served model.
- `-max-model-len` specifies the maximum length of the model.

```bash
vllm serve /root/autodl-tmp/ZhipuAI/GLM-4.1V-9B-Thinking
    --served-model-name GLM-4.1V-9B-Thinking
    --max_model_len 25000
    --limit-mm-per-prompt '{"image": 32, "video": 1}'
    --allowed-local-media-path /
```

![image-4.png](images/image-4.png)

- View the current model list via `curl` command

```bash
curl http://localhost:8000/v1/models
```

The returned value is as follows

```json
{
	"object":"list",
	"data":[
		{
			"id":"GLM-4.1V-9B-Thinking",
			"object":"model",
			"created":1753411607,
			"owned_by":"vllm",
			"root":"/root/autodl-tmp/ZhipuAI/GLM-4.1V-9B-Thinking",
			"parent":null,
			"max_model_len":8192,
			"permission":[
				{
						"id":"modelperm-7a0d15b7334b4d19b024c2829c96b44a",
						"object":"model_permission",
						"created":1753411607,
						"allow_create_engine":false,
						"allow_sampling":true,
						"allow_logprobs":true,
						"allow_search_indices":false,
						"allow_view":true,
						"allow_fine_tuning":false,
						"organization":"*",
						"group":null,
						"is_blocking":false
						}
					]
			}
		]
}
```

- Test `OpenAI Completions API` using `curl` command

```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "GLM-4.1V-9B-Thinking",
        "prompt": "What is 456*123? <think>\n",
        "max_tokens": 1024,
        "temperature": 0
    }'
```

The returned value is as follows

![image-5.png](images/image-5.png)

• Single image example using `Python` script to request `OpenAI Chat Completions API`

```bash
# vllm_openai_completions.py
import argparse
import os
from openai import OpenAI

def get_media_type(file_path):
    video_extensions = {".mp4", ".avi", ".mov"}
    image_extensions = {".jpg", ".jpeg", ".png"}
    _, ext = os.path.splitext(file_path.lower())
    return (
        "video_url"
        if ext in video_extensions
        else "image_url"
        if ext in image_extensions
        else None
    )

def create_content_item(media_path, media_type):
    if media_path.startswith(("http://", "https://")):
        url = media_path
    else:
        url = "file://" + media_path

    if media_type == "video_url":
        return {"type": "video_url", "video_url": {"url": url}}
    else:
        return {"type": "image_url", "image_url": {"url": url}}

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-xxx", # Fill in arbitrarily, just to pass interface parameter validation
)

media_path = "/root/autodl-tmp/media/demo1.png"
media_type = get_media_type(media_path)
messages = [
    {
        "role": "user",
        "content": [
            create_content_item(media_path, media_type),
            {"type": "text", "text": "What is this picture?"},
        ],
    }
]
print("=========Messages=========")
print(messages)
response = client.chat.completions.create(
    model="GLM-4.1V-9B-Thinking",
    messages = messages,
    temperature = 1.0,
    top_p = None,
    extra_body = {
        "skip_special_tokens": False,
        "repetition_penalty": 1.0,
    },
)
print("=========Answer=========")
print(response.choices[0].message.content.strip())
```

The image used in the example is as follows:

![image-6.png](images/image-6.png)

The result obtained is as follows:

![image-7.png](images/image-7.png)

• Multi-image analysis using `Python` script to request `OpenAI Chat Completions API`

```python
# vllm_openai_completions.py
# The above code remains unchanged
media_path_1 = "/root/autodl-tmp/media/demo2-1.JPG"
media_path_2 = "/root/autodl-tmp/media/demo2-2.JPG"

messages = [
    {
        "role": "user",
        "content": [
            create_content_item(media_path_1, media_type),
            create_content_item(media_path_2, media_type),
            {"type": "text", "text": "What are the possible connections between these two images?"},
        ],
    }
]
```

Pass in the paths of multiple images separately for analysis. GLM-4.1V-Thinking can analyze up to 10 images simultaneously.

The images used in the example are as follows:
<div style="display: flex; justify-content: center; gap: 20px; margin: 20px 0;">
    <img src="images/image-8.jpg" alt="image-8" style="width: 300px; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <img src="images/image-9.jpg" alt="image-9" style="width: 300px; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
</div>

Code execution result is as follows:

Assistant: Based on the provided reference information, there may be the following connections between these two images:

1. Healthy Lifestyle Connection
The first image shows sugar-free Coca-Cola, emphasizing "0 sugar 0 fat 0 calories", indicating that the product is positioned as a healthy drink choice.
The second image shows a weighing scale and weight record (42.7 kg), which is directly related to weight management.
Both reflect modern people's concern for healthy diet and weight management.
2. Time and Behavior Connection
The date "10/23" in the second image may represent a specific date, related to the Coca-Cola consumption in the first image.
It may be that on October 23, this person chose sugar-free Coke as a healthy drink and recorded their weight.
3. Psychological Motivation Connection
The sugar-free Coke in the first image may represent a healthy alternative choice to reduce sugar and calorie intake.
"YOU'RE GONNA LOVE IT!" in the second image may be encouraging oneself to stick to a healthy lifestyle, including choosing sugar-free drinks.
4. Product and Consumer Connection
The first image is product information of Coca-Cola.
The second image may be a record or feedback from a consumer after using sugar-free Coke, showing the health benefits of the product.
These connections indicate that these two images may tell a story about a lifestyle of enhanced health awareness and focus on weight management, where sugar-free Coke is part of a healthy diet choice.
