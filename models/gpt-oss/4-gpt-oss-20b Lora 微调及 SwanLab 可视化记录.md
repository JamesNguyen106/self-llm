# 04-gpt-oss-20b LoRA Fine-tuning and SwanLab Visualization

## Introduction

> After reading this tutorial, you will learn:
> 	
> - LoRA fine-tuning of gpt-oss-20b!
>
> - Use ms-swift for LoRA fine-tuning of gpt-oss-20b!	

### Data Preparation

> Download multilingual reasoning dataset

```Python
from datasets import load_dataset
dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
```

### Data Visualization

```Python
User:
    ¿Cuál es el capital de Australia?
Assistant reasoning:
    Okay, der Benutzer fragt nach der Hauptstadt Australiens. Ich erinnere mich, dass Canberra die Hauptstadt ist. Ich
    sollte das bestätigen. Lass mich sehen, ob es irgendwelche potenziellen Verwirrungen gibt. Der Benutzer könnte auch
    an der größten Stadt interessiert sein. Die größte Stadt ist Sydney, aber die Hauptstadt ist Canberra. Ich sollte
    das klarstellen. Vielleicht auch erwähnen, dass Canberra eine geplante Stadt ist und nicht die größte. Der Benutzer
    könnte auch nach der Geografie fragen. Vielleicht erwähne ich, dass Canberra im südwestlichen Teil der Australian
    Capital Territory liegt. Ich sollte die Antwort präzise und freundlich halten. Vielleicht auch erwähnen, dass
    Canberra oft mit Sydney verwechselt wird. Ich sollte sicherstellen, dass die Antwort klar und korrekt ist.
Assistant response:
    La capital de Australia es **Canberra**. Aunque es la ciudad más pequeña de las principales capitales del país, fue
    elegida en 1908 como la sede del gobierno federal para equilibrar la influencia entre las ciudades de Sydney y
    Melbourne. Canberra está ubicada en el Territorio de la Capital Australiana (ACT), en el este de Australia.
```

### Environment Configuration

```Python
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "peft>=0.17.0" "transformers>=4.55.0" trackio
```

### Fine-tuning

> Load tokenizer

```Python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b") # Replace with your downloaded directory~
```

> Then we can use the tokenizer's `apply_chat_template()` method to format the message:

```Python
messages = dataset[0]["messages"]
conversation = tokenizer.apply_chat_template(messages, tokenize=False)
print(conversation)
```

> To prepare the model for training, let's first download the weights from [Hugging Face Hub](https://huggingface.co/).
> 
> We will use the 🤗 `AutoModelForCausalLM` class in Transformers to load the model

```Python
import torch
from transformers import AutoModelForCausalLM, Mxfp4Config

quantization_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
)

model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs)
```

This will load the configuration required for training the model. `attn_implementation` is set to eager for better performance, and `use_cache` is set to `False` because we will use gradient checkpointing to fine-tune the model.

If you are familiar with Transformers, you might notice that we are using `Mxfp4Config` for quantization. This is a specific configuration for OpenAI models that allows us to use mixed precision training, which includes a special 4-bit floating point format called ++[MXFP4](https://en.wikipedia.org/wiki/Block_floating_point)++ optimized for AI workloads.

> Test a message

```Python
messages = [
    {"role": "user", "content": "¿Cuál es el capital de Australia?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

output_ids = model.generate(input_ids, max_new_tokens=512)
response = tokenizer.batch_decode(output_ids)[0]
print(response)
```

> Configure LoRA parameters

```Python
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
    target_parameters=[
        # MoE expert layer projections, add or remove as needed
        "7.mlp.experts.gate_up_proj",
        "7.mlp.experts.down_proj",
        "15.mlp.experts.gate_up_proj",
        "15.mlp.experts.down_proj",
        "23.mlp.experts.gate_up_proj",
        "23.mlp.experts.down_proj",
    ],
)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
```

Note: The `openai/gpt-oss-20b` model is a ++[Mixture of Experts (MoE)](https://huggingface.co/blog/moe)++ architecture. In addition to targeting attention layers (`target_modules="all-linear"`), it is important to include projection layers in expert modules. PEFT facilitates this via the `target_parameters` argument, which allows you to specify expert-specific layers such as `mlp.experts.down_proj` and `mlp.experts.gate_up_proj`.

```Python
from datasets import DatasetDict

max_length = 4096

def format_and_tokenize(example):
    # Expect "messages" field to exist (consistent with your example)
    messages = example["messages"]
    # Do not add generation_prompt; let the model learn the complete conversation unfolding
    text = tokenizer.apply_chat_template(
        messages, tokenize=False
    )
    # Tokenize directly as a whole, labels=inputs (handled by collator)
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    return tokens

tokenized = ds.map(format_and_tokenize, remove_columns=ds.column_names)
# Simply split a validation set (optional)
splits = tokenized.train_test_split(test_size=0.01, seed=42)
train_ds, eval_ds = splits["train"], splits["test"]

```

> Fine-tuning parameter settings

```Python
from dataclasses import dataclass
import torch
from transformers import Trainer, TrainingArguments, default_data_collator

@dataclass
class CausalDataCollator:
    tokenizer: AutoTokenizer
    mlm: bool = False
    def __call__(self, features):
        # default_data_collator will convert input_ids/attention_mask to tensors
        batch = default_data_collator(features)
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"].clone()
        return batch

collator = CausalDataCollator(tokenizer)

training_args = TrainingArguments(
    output_dir="gpt-oss-20b-multilingual-reasoner",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1.0,
    learning_rate=2e-4,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    warmup_ratio=0.03,
    logging_steps=1,
    save_steps=200,
    save_total_limit=2,
    bf16=True,
    gradient_checkpointing=True,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=collator,
)

trainer.train()

```
> Upload training process to SwanLab
> Visualize training process via SwanLab

![](./images/4-0.png)

### SwanLab

![](./images/4-1.png)

> [SwanLab](https://github.com/swanhubx/swanlab) is an open-source model training recording tool for AI researchers, providing training visualization, automatic logging, hyperparameter recording, experiment comparison, multi-person collaboration, and other functions. On `SwanLab`, researchers can discover training problems based on intuitive visualization charts, compare multiple experiments to find research inspiration, and break down team communication barriers through online link sharing and organization-based multi-person collaborative training.

#### Why record training?

Compared to software development, model training is more like an experimental science. Behind a high-quality model are often thousands of experiments. Researchers need to constantly try, record, compare, and accumulate experience to find the best model structure, hyperparameters, and data ratio. In this process, how to efficiently record and compare is crucial for improving research efficiency.

#### Where to use?

It is recommended to register an account on the [SwanLab official website](https://swanlab.cn/) first, and then select it during the SFT initialization phase.


> Set to your own api_key~
```Python
from transformers import TrainerCallback

try:
    import swanlab
    class SwanLabCallback(TrainerCallback):
        def __init__(self, project="swift-robot"):
            swanlab.init(project=project)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                for k,v in logs.items():
                    if isinstance(v, (int,float)):
                        swanlab.log({k: v})
    trainer.add_callback(SwanLabCallback(project="swift-robot"))
except Exception as e:
    print("SwanLab disabled:", e)
```

> Merge weights
```Python
from peft import PeftModel

# Load base model first
infer_kwargs = dict(attn_implementation="eager", torch_dtype="auto", use_cache=True, device_map="auto")
base_model = AutoModelForCausalLM.from_pretrained(model_id, **infer_kwargs).cuda()

# Load LoRA adapter back (use training output directory)
peft_model = PeftModel.from_pretrained(base_model, "gpt-oss-20b-multilingual-reasoner")
# Merge and unload LoRA
merged = peft_model.merge_and_unload()
merged.eval()

# Generate
messages = [
    {"role": "system", "content": "reasoning language: German"},
    {"role": "user", "content": "¿Cuál es el capital de Australia?"},
]
inp = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(merged.device)
gen = merged.generate(inp, max_new_tokens=512, do_sample=True, temperature=0.6)
print(tokenizer.batch_decode(gen)[0])


```

![](./images/4-5.png)


## ms-swift Fine-tuning

> Here is a framework fine-tuning tutorial based on `ms-swift`
> There are many fine-tuning frameworks, and whichever one you choose leads to the same goal. Why choose ms-swift, see:

- 🍎 Model Type: Supports 450+ pure text large models, 150+ multimodal large models, and All-to-All full modal models, sequence classification models, Embedding model training to deployment full process.
Dataset Type: Built-in 150+ pre-training, fine-tuning, human alignment, multimodal and other types of datasets, and supports custom datasets.
Hardware Support: CPU, RTX series, T4/V100, A10/A100/H100, Ascend NPU, MPS, etc.
- 🍊 Lightweight Training: Supports LoRA, QLoRA, DoRA, LoRA+, ReFT, RS-LoRA, LLaMAPro, Adapter, GaLore, Q-Galore, LISA, UnSloth, Liger-Kernel and other lightweight fine-tuning methods.
Distributed Training: Supports distributed data parallel (DDP), device\_map simple model parallel, DeepSpeed ZeRO2 ZeRO3, FSDP and other distributed training technologies.
Quantization Training: Supports training of BNB, AWQ, GPTQ, AQLM, HQQ, EETQ quantized models.
RLHF Training: Supports human alignment training methods such as DPO, GRPO, RM, PPO, KTO, CPO, SimPO, ORPO for pure text large models and multimodal large models.
- 🍓 Multimodal Training: Supports training of different modal models of image, video and voice, supports training of VQA, Caption, OCR, Grounding tasks.
Interface Training: Provides training, inference, evaluation, and quantization capabilities in an interface way, completing the full link of large models.
Pluginization and Extension: Supports custom model and dataset extension, supports customization of components such as loss, metric, trainer, loss-scale, callback, optimizer, etc.
- 🍉 Toolbox Capabilities: Not only provides training support for large models and multimodal large models, but also covers their inference, evaluation, quantization and deployment full processes.
Inference Acceleration: Supports PyTorch, vLLM, LmDeploy inference acceleration engines, and provides OpenAI interface to provide acceleration for inference, deployment and evaluation modules.
Model Evaluation: Uses EvalScope as the evaluation backend, supports 100+ evaluation datasets to evaluate pure text and multimodal models.
Model Quantization: Supports quantization export of AWQ, GPTQ and BNB, exported models support using vLLM/LmDeploy inference acceleration, and support continued training.

### Environment Configuration

1. Basic Environment Configuration
	

> PyTorch 2.6.0
> 
> Python 3.12(ubuntu22.04)
> 
> CUDA 12.4
> 
> GPU NVIDIA H20-96GB \* 4

2. LoRA Environment Configuration
	

```Bash
pip install ms-swift==3.7.0
pip install deepspeed
pip install swanlab
pip install -U transformers kernels torch
```

### Data Preparation

> Build Dataset
> 
> Refer to [Custom Dataset — swift 3.8.0.dev0 Documentation](https://swift.readthedocs.io/zh-cn/latest/Customization/%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86.html) for more definition methods
> 
> Here uses the simplest way consistent with the official structure
> 
> Mine is a translation task, so my example data is as follows:

```Bash
  {
    "messages": [
      {
        "role": "user",
        "content": "Source Text：There, they found the body of Saroja Balasubramanian, 53, covered with blood-stained blankets. Style Guide：The translation style must be style: wikinews/Crime and Law"
      },
      {
        "role": "assistant",
        "content": "<think>\nxxx</think>\n\n在那里，他们发现了 53 岁的萨罗贾·巴拉苏布拉曼尼亚的尸体，盖着血迹斑斑的毯子。"
      }
    ]
  },
```

> Or you can use any open source dataset
> 
> Here, other students found an open source Cyber Catgirl dataset on ModelScope for this tutorial. Imagine which guy doesn't want to have a Cyber Catgirl?

Dataset Portal: [Muxue Catgirl Dataset](https://modelscope.cn/datasets/himzhzx/muice-dataset-train.catgirl/files)

```JSON
{
  "instruction": "What is Muxue's function?",
  "input": "",
  "output": "Meow~ Benxue's main function is to make you happy meow! Heal your soul with the power of cute catgirl, meow~"
  "history":[]
}
```

### LoRA Fine-tuning

> Write bash script

```Bash
MASTER_PORT=$PORT \                             # Communication port for distributed training main process, use environment variable $PORT
NPROC_PER_NODE=4 \                              # Number of processes per node (usually equal to number of GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 \                  # Specify GPU numbers to use
swift sft --deepspeed zero3\                    # Use swift's sft training command and enable DeepSpeed ZeRO-3 optimization
    --model /root/autodl-tmp/gpt-oss-20b \      # Model path (replace with your own model directory)
    --dataset /root/autodl-tmp/train.json \     # Dataset path (replace with your own training data)
    --train_type lora \                         # Training type is LoRA (Low-Rank Adaptation)
    --torch_dtype bfloat16 \                    # Calculation precision set to bfloat16
    --num_train_epochs 35 \                     # Total training epochs
    --per_device_train_batch_size 1 \           # Training batch size per device
    --per_device_eval_batch_size 1 \            # Validation batch size per device
    --learning_rate 1e-4 \                      # Learning rate
    --lora_rank 8 \                             # LoRA rank (low-rank decomposition dimension)
    --lora_alpha 32 \                           # LoRA scaling factor
    --target_modules all-linear \               # Target module types to apply LoRA
    --gradient_accumulation_steps 16 \          # Gradient accumulation steps
    --eval_steps 50 \                           # Evaluate every 50 steps
    --save_steps 50 \                           # Save model every 50 steps
    --save_total_limit 2 \                      # Keep at most 2 latest checkpoints
    --logging_steps 5 \                         # Log every 5 steps
    --max_length 8192 \                         # Maximum sequence length
    --output_dir output \                       # Model output directory
    --warmup_ratio 0.05 \                       # Learning rate warmup ratio
    --dataloader_num_workers 4 \                # DataLoader worker threads
    --use_liger_kernel true \                   # Enable liger kernel optimization
    --load_from_cache_file false \              # Whether to load data from cache file
    --loss_scale ignore_empty_think \           # Ignore loss of empty think tags
    --save_strategy epoch\                      # Save strategy: save once per epoch
    --model_author gxb \                        # Model author name
    --model_name gxb-gpt-oss-20b-agent-distill \# Model name
    --report_to swanlab \                       # Report training logs to SwanLab
    --swanlab_project swift-robot               # SwanLab project name
```

### Test Effect

```Bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters v0-20250811-150539/checkpoint-551 \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

![](./images/4-2.png)

### Merge Weights

```Bash
swift export \
    --adapters v0-20250811-150539/checkpoint-551 \
    --merge_lora true
```

![](./images/4-3.png)

### Inference

> Write inference script

```Bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model v0-20250811-150539/checkpoint-551 \
    --val_dataset SFT/dataForSFT/val.json \
    --max_new_tokens 2048 \
    --result_path SFT/infer_output/_sft_1epoch.jsonl
```

![](./images/4-4.png)
