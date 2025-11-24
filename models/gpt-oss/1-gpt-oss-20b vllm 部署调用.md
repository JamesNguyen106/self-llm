# **01-GPT-OSS-20b** **vLLM** **Deployment & Inference**

## Introduction

> After reading this tutorial, you will learn:
> 
> - How to deploy and invoke gpt-oss capabilities locally using transformers!
> 	
> - How to deploy and invoke gpt-oss capabilities using vLLM!
> 	

GPT-OSS is an open-source large language model series released by OpenAI, containing two versions: gpt-oss-120b and gpt-oss-20b. Both models adopt the MoE (Mixture-of-Experts) Transformer architecture, support a context length of 128K, and use the Apache 2.0 license, allowing free use and commercial applications. The GPT-OSS-120B model is almost on par with the OpenAI o4-mini model in core reasoning benchmarks, while being able to run efficiently on a single 80GB GPU. The GPT-OSS-20B model achieves similar results to the OpenAI o3-mini model in common benchmarks and can run on edge devices equipped with only 16GB of memory, making it ideal for on-device applications, local inference, or rapid iteration without expensive infrastructure. Both models perform strongly in tool use, few-shot function calling, CoT reasoning (as shown in results in the Tau-Bench agent evaluation suite), and HealthBench tests (even surpassing proprietary models such as OpenAI o1 and GPT-4o).

vLLM supports the following two sizes of gpt-oss:

- [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
	- Smaller model
		
	- Requires only about 16GB VRAM
		
	- Can run on consumer-grade graphics cards such as A100 or H20
		
- [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
	- Larger full-size model
		
	- Best performance with VRAM ≥ 60GB
		
	- Can run on single H100 or multi-GPU setups
		

## **Environment Preparation**

Configuring the environment can be a headache, so we have prepared a mirror for students:
- ⚠️**Note**: Since vLLM version 0.10.1, which supports the gpt-oss model, has not been officially released yet (2025.8.6), you need to install vLLM and gpt-oss dependencies from source. It is recommended to start a new python 3.12 environment to avoid affecting the existing environment.
- ⚠️**Note:** Many dependencies of vllm-gpt-oss are the latest. If the environment is not configured with the latest versions, problems will occur!!! If you need to upgrade conda, use the upgrade cuda command:

```Bash
conda search cuda-toolkit --channel nvidia
conda install cuda-toolkit=<new_version_number> --channel nvidia
nvcc --version # Check if upgrade is successful

export CUDA_HOME=$CONDA_PREFIX 
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH # Set path after conda upgrade
```

The experimental basic environment for this article is as follows:

> PyTorch 2.9.0
> Python 3.12(ubuntu22.04)
> CUDA 12.8
> GPU NVIDIA H20-96GB \* 1

1. Clone the code repository
	

```Bash
git clone https://github.com/huggingface/gpt-oss-recipes.git
cd gpt-oss-recipes
```

2. Change `pip` source for acceleration, download and install dependency packages
	

```Bash
conda create -n gpt_oss_vllm python=3.12
conda activate gpt_oss_vllm

python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

export HF_ENDPOINT="https://hf-mirror.com"
pip install -U huggingface_hub
```

3. `gpt-oss` dependencies
	

```Python
pip install -U transformers kernels torch accelerate
```

## **Model Download**

Models can be obtained from Hugging Face and ModelScope repositories. You can download explicitly, or execute repository code to download automatically upon first use.

### **Hugging Face** **CLI**

```Bash
huggingface-cli download openai/gpt-oss-20b --local-dir gpt-oss-20b
```

ps: Remember to modify the corresponding `cache_dir` / `local_dir` to your model download path~

## **transformers** **Usage**

### Call via Script

Use the `transformers` library as a rewritten interaction script for inference to perform call testing

```Python
from transformers import pipeline
import torch

model_id = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
```

![](./images/1-0.png)

### Call via Service

```Bash
transformers serve
transformers chat localhost:8000 --model-name-or-path openai/gpt-oss-20b
```

ps: Remember to modify to your model download path~
![](./images/1-1.png)

### Adjust Generation Parameters

You can adjust the detail level of inference through system prompts. For example, set high reasoning level:

```JSON
messages = [
    {"role": "system", "content": "Reasoning: high"},
    {"role": "user", "content": "Explain the basic principles of quantum computing"}
]
```

## **vLLM** **Call Code Preparation**

### vLLM Environment Configuration

Note: Compatibility between vLLM versions is poor, please identify carefully

```Bash
pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128
 
# !!!Install FlashInfer!!!
pip install flashinfer-python==0.2.10
```

### Start Server and Download Model

vLLM provides a command that will automatically download the model from HuggingFace and start an OpenAI-compatible server on it.
Run the following command based on the required model size in the terminal session on the server.

```Bash
# For 20B, can be replaced with locally downloaded directory
vllm serve openai/gpt-oss-20b
 
# For 120B, can be replaced with locally downloaded directory
vllm serve openai/gpt-oss-120b
```

![](./images/1-2.png)

### Test

```Python
VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 vllm serve openai/gpt-oss-20b --served-model-name gpt-oss-20b --trust_remote_code --port 8801
```

![](./images/1-3.png)

### Reference Links

- Online experience URL: [https://gpt-oss.com/](https://gpt-oss.com/?utm_source=aihub.cn)
	
- Huggingface: [https://huggingface.co/openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/?utm_source=aihub.cn)
	
- vLLM official gpt-oss documentation: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#gpt-oss-vllm-usage-guide
