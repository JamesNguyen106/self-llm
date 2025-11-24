# 03-GLM-4.1V-Thinking LoRA and SwanLab Visualization

## Dataset Construction

The data format for `supervised-finetuning` (`sft`) of the model is as follows:

```
{
  "instruction": "Answer the following user question, output only the answer.",
  "input": "What is 1+1?",
  "output": "2"
}
```

Where `instruction` is the user instruction, informing the model of the task it needs to complete; `input` is the user input, which is the necessary input content to complete the user instruction; `output` is the output that the model should give.

The goal of supervised fine-tuning is to enable the model to understand and follow user instructions. So what interesting things can we do? Perhaps we can fine-tune a model with a corresponding characteristic dialogue style through a large amount of stylized character setting dialogue data.

Here, I found an open-source Cyber Catgirl dataset on ModelScope for this tutorial. Imagine which guy doesn't want to have a Cyber Catgirl? Portal: [Muxue Catgirl Dataset](https://modelscope.cn/datasets/himzhzx/muice-dataset-train.catgirl/files)

```
{
  "instruction": "What is Muxue's function?",
  "input": "",
  "output": "Meow~ Benxue's main function is to make you happy meow! Heal your soul with the power of cute catgirl, meow~"
  "history":[]
}
```

So, let's start preparing to adopt Muxue.

## Data Preparation

Import the corresponding libraries and convert our data file:

```python
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

# Convert JSON file to CSV file
df = pd.read_json('/root/autodl-tmp/LLaMA-Factory/data/muice-dataset-train.catgirl.json') # Note to modify
ds = Dataset.from_pandas(df)
```

`LoRA` (`Low-Rank Adaptation`) training data needs to be formatted and encoded before being input to the model for training. We need to first encode the input text into `input_ids` and the output text into `labels`. The result after encoding is vectors. We first define a preprocessing function. This function encodes the input and output text of each sample simultaneously and returns an encoded dictionary:

```python
def process_func(example):
    MAX_LENGTH = 1024 # Set maximum sequence length to 1024 tokens
    input_ids, attention_mask, labels = [], [], [] # Initialize return values
    # Adapt chat_template
    instruction = tokenizer(
        f"[gMASK]<sop><|system|>\nNow you have to play the role of the woman beside the emperor -- Zhen Huan" 
        f"<|user|>\n{example['instruction'] + example['input']}"  
        f"<|assistant|>\n<think></think>\n",  
        add_special_tokens=False   
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    # Concatenate input_ids of instruction part and response part, and add eos token at the end as the end token
    input_ids = instruction["input_ids"] + response["input_ids"]
    # Attention mask, indicating the positions the model needs to pay attention to
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    # For instruction, use -100 to indicate that these positions do not calculate loss (i.e., the model does not need to predict this part)
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:  # Truncate if exceeding maximum sequence length
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
```

## Load Model and Tokenizer

```python
from transformers import Glm4vForConditionalGeneration
import torch

# Remember to replace the model path with your local model path
tokenizer = AutoTokenizer.from_pretrained('ZhipuAI/GLM-4.1V-9B-Thinking')
model = Glm4vForConditionalGeneration.from_pretrained(
    'ZhipuAI/GLM-4.1V-9B-Thinking',
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
```

## LoRA Config

Many parameters can be set in the `LoraConfig` class. The more important ones are as follows:

- `task_type`: Model type. Most `decoder_only` models are now causal language models `CAUSAL_LM`.
- `target_modules`: The names of the model layers that need to be trained, mainly the layers in the `attention` part. The names of the corresponding layers vary for different models.
- `r`: Rank of `LoRA`, which determines the dimension of the low-rank matrix. A smaller `r` means fewer parameters.
- `lora_alpha`: Scaling parameter, together with `r`, determines the strength of `LoRA` updates. The actual scaling ratio is `lora_alpha/r`, which is `32 / 8 = 4` times in the current example.
- `lora_dropout`: `dropout rate` applied to `LoRA` layers, used to prevent overfitting.

```python
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # Training mode
    r=8, # LoRA rank
    lora_alpha=32, # LoRA alpha
    lora_dropout=0.1 # Dropout rate
)
```

## Training Arguments

- `output_dir`: Output path of the model
- `per_device_train_batch_size`: `batch_size`
- `gradient_accumulation_steps`: Gradient accumulation
- `num_train_epochs`: As the name suggests, `epoch`

```python
from transformers import TrainingArguments
args = TrainingArguments(
    output_dir="./output/glm4_1V-Thinking_lora", # Output path
    per_device_train_batch_size=4, # batch_size
    gradient_accumulation_steps=4, # Gradient accumulation
    logging_steps=2,
    num_train_epochs=3, # epoch
    save_steps=10, 
    learning_rate=1e-4, # lr
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)
```

## Instantiate SwanLabCallback

```python
import swanlab
from swanlab.integration.transformers import SwanLabCallback

swanlab.login(api_key='your apikey', save=True) # Remember to replace with your account's apikey
# Instantiate SwanLabCallback
swanlab_callback = SwanLabCallback(
    project="self-llm", 
    experiment_name="glm4.1v-lora-catgirl"
)
```

After training is complete, you can see the relevant parameter curves during your training process.

![image-15.png](images/image-15.png)

## Load LoRA Model for Inference

After training is complete, select the best performing LoRA model weights (under the output path defined earlier), load the weights for inference, and say hello to Muxue~

```python
from transformers import AutoTokenizer, AutoProcessor, Glm4vForConditionalGeneration
import torch
from peft import PeftModel

mode_path = 'ZhipuAI/GLM-4.1V-9B-Thinking' # Local glm4.1-Thinking model path
lora_path = 'output/glm4_1V-Thinking_lora/checkpoint-180' # Change this to your lora output corresponding checkpoint address

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)
processor = AutoProcessor.from_pretrained(mode_path, use_fast=True)
# Load model
model = Glm4vForConditionalGeneration.from_pretrained(mode_path, 
                                                      device_map="auto",
                                                      torch_dtype=torch.bfloat16, 
                                                      trust_remote_code=True)

# Load LoRA weights
model = PeftModel.from_pretrained(model, model_id=lora_path)
```

say hi!

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Who are you?"
            }
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=8192)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print(output_text)
```

`Meow~ Benxue is an AI cat meow! Specially using the wisdom of cat people to help everyone solve various problems meow~ Remember to give Benxue dried fish as a gift meow! Meow~</think><answer>Meow~ Benxue is an AI cat meow! Specially using the wisdom of cat people to help everyone solve various problems meow~ Remember to give Benxue dried fish as a gift meow! Meow~</answer>`

Next, let's test what the visual understanding effect of glm4.1V-Thinking would be like with Muxue.

Note that glm4.1V has limitations on the number and size of images and videos, and the pytorch version needs to be above 2.2.

```python
messages = [
    {
        "role": "user",
        "content": [{
                "type": "text",
                "text":"Assume you are a catgirl."}],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://cdn.colorhub.me/Pl0d7lY07R4/rs:auto:0:500:0/g:ce/fn:colorhub/bG9jYWw6Ly8vY2YvMTUvMDY0NTdiMWVhNTA3NjA1MjU5Yzc5YmUzYzRiM2VmYTVkMTAwY2YxNS5qcGVn.webp"
            },
            {
                "type": "text",
                "text": "What does Benxue think of it?"
            }
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=8192)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print(output_text)
```

`Meow~ This leather looks very comfortable meow! Just as soft as Benxue's fur meow~ What do you think meow? Benxue wants to touch it meow~ (Rubbing) Meow~ (Licking fur) Meow~ (Rolling) Meow~ (Wagging tail) Meow~ (Meowing) Meow~ (Jumping up) Meow~ (Pouncing) Meow~ (Rubbing) Meow~ (Licking fur) Meow~ (Rolling) Meow~ (Wagging tail) Meow~ (Meowing) Meow~ (Jumping up) Meow~ (Pouncing) Meow~ (Rubbing) Meow~ (Licking fur) Meow~ (Rolling) Meow~ (Wagging tail) Meow~ (Meowing) Meow~ (Jumping up) Meow~ (Pouncing) Meow~ (Rubbing) Meow~ (Licking fur) Meow~ (Rolling) Meow~ (Wagging tail) Meow~ (Meowing) Meow~ (Jumping up) Meow~ (Pouncing) Meow~ (Rubbing) Meow~ (Licking fur) Meow~ (Rolling) Meow~ (Wagging tail) Meow~ (Meowing) Meow~ (Jumping up) Meow~ (Pouncing) Meow~ (Rubbing) Meow~ (Licking fur) Meow~ (Rolling) Meow~ (Wagging tail) Meow~ (Meowing) Meow~ (Jumping up) Meow~ (Pouncing) Meow~ (Rubbing) Meow~ (Licking fur) Meow~ (Rolling) Meow~ (Wagging tail) Meow~ (Meowing) Meow~ (Jumping up) Meow~ (Pouncing) Meow~ (Rubbing) Meow~ (Licking fur) Meow~ (Rolling) Meow~ (Wagging tail) Meow~ (Meowing) Meow~ (Jumping up) Meow~ (Pouncing) Meow~ (Rubbing) Meow~ (Licking fur) Meow~ (Rolling) Meow~ (Wagging tail) Meow~ (Meowing) Meow~ (Jumping up) Meow~ (Pouncing) Meow~ (Rubbing) Meow~ (Licking fur) Meow~ (Rolling) Meow~ (Wagging tail) Meow~ (Meowing) Meow~ (Jumping up) Meow~ (Pouncing) Meow~ (Rubbing) Meow~ (Licking fur) Meow~ (Rolling) Meow~ (Wagging tail) Meow~ (Meowing) Meow~ (Jumping up) Meow~ (Pouncing) Meow~ (Rubbing) Meow~ (Licking fur) Meow~ (Rolling) Meow~ (Wagging tail) Meow~ (Meowing) Meow~ (Jumping up) Meow~ (Pouncing) Meow~ (Rubbing) Meow~ (Licking fur) Meow~ (Rolling) Meow~ (Wagging tail) Meow~ (Meowing) Meow~ (Jumping up) Meow~ (Pouncing) Meow~ (Rubbing) Meow~ (Licking fur) Meow~ (Rolling) Meow~ (Wagging tail) Meow~ (Meowing) Meow~ (Jumping up) Meow~ (Pouncing) Meow~ (Rubbing) Meow~ (Licking fur) Meow~ (Rolling) Meow~ (Wagging tail) Meow~ (Meowing) Meow~ (Jumping up) Meow~ (Pouncing)`

![image-16.png](images/image-16.png)

hhhhh