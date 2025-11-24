# 05-gpt-oss-20b DPO Fine-tuning and SwanLab Visualization

## Introduction

> After reading this tutorial, you will learn how to construct DPO data and use ms-swift to fine-tune gpt-oss-20b!

## Why do DPO?

In large model fine-tuning, SFT (Supervised Fine-Tuning) is usually done first, followed by DPO (Direct Preference Optimization). This is because SFT uses high-quality labeled data to let the model learn the basic capabilities and output format of the task first, while DPO uses paired good and bad answers to guide the model to better align with human preferences. If DPO is done directly, the model may not be able to generate qualified answers yet, the preference signal noise is large, and it is difficult to converge; SFT followed by DPO can optimize to "do well" on the basis of "knowing how to do".

## Environment Configuration

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

## Data Preparation

> Build dataset, example data is as follows
> 
> Refer to [Custom Dataset — swift 3.8.0.dev0 Documentation](https://swift.readthedocs.io/zh-cn/latest/Customization/%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86.html) for more definition methods
> 
> Here uses the simplest way consistent with the official structure
> 
> Example data is as follows [Structure should contain one positive sample and one negative sample]:

```JSON
{"messages": [{"role": "system", "content": "You are a helpful and harmless assistant"}, {"role": "user", "content": "Tell me tomorrow's weather"}, {"role": "assistant", "content": "Tomorrow will be sunny"}], "rejected_response": "I don't know"}
{"messages": [{"role": "system", "content": "You are a helpful and harmless math calculator"}, {"role": "user", "content": "What is 1+1"}, {"role": "assistant", "content": "Equals 2"}, {"role": "user", "content": "Add 1 more"}, {"role": "assistant", "content": "Equals 3"}], "rejected_response": "I don't know"}
```

Data on translation task:

```JSON
    {
        "messages": [
            {"role": "system", "content": "You are an expert machine translation specialist in the field of Technical Writing, highly proficient in translating from Chinese to English."}, 
            {"role": "user", "content": "Translate the following Chinese text to English: 步骤一：添加域名"}, 
            {"role": "assistant", "content": "Step 1: Add a domain name"}], 
        "rejected_response": "Step 1: Adding domain names"
    }
```

## Fine-tuning

> Construct DPO script

```Bash
MASTER_PORT=$PORT \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf --rlhf_type dpo \
    --deepspeed zero3 \
    --model models/gpt-oss-20b \ # Replace with your own model path
    --dataset  dpo_data.json \ # Replace with your own data path
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 4e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --save_strategy epoch \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir /opt/tiger/Agent-distill/finetune/gpt_output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --use_liger_kernel true \
    --load_from_cache_file false \
    --loss_scale ignore_empty_think \
    --save_strategy epoch\
    --model_author gxb \
    --model_name gxb-gpt-oss-20b-dpo \
    --report_to swanlab \ # Use swanlab to record experiments
    --swanlab_project swift-robot
```

> Training process visualization

![](./images/5-0.png)

> Video memory usage

![](./images/5-1.png)

### SwanLab

![](./images/5-2.png)

> [SwanLab](https://github.com/swanhubx/swanlab) is an open-source model training recording tool for AI researchers, providing training visualization, automatic logging, hyperparameter recording, experiment comparison, multi-person collaboration, and other functions. On `SwanLab`, researchers can discover training problems based on intuitive visualization charts, compare multiple experiments to find research inspiration, and break down team communication barriers through online link sharing and organization-based multi-person collaborative training.

#### Why record training?

Compared to software development, model training is more like an experimental science. Behind a high-quality model are often thousands of experiments. Researchers need to constantly try, record, compare, and accumulate experience to find the best model structure, hyperparameters, and data ratio. In this process, how to efficiently record and compare is crucial for improving research efficiency.

#### Where to use?

It is recommended to register an account on the [SwanLab official website](https://swanlab.cn/) first, and then select during the initialization phase of SFT and GRPO training
`--report_to swanlab \ # Report training logs to SwanLab`

### Merge Weights

> Subsequent steps are consistent with SFT

```Bash
swift export \
    --adapters v0-20250812-150539/checkpoint-20 \
    --merge_lora true
```

### Inference

> Write inference script

```Bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model v0-20250812-150539/checkpoint-20-merged \
    --val_dataset val.json \
    --max_new_tokens 2048 \
    --result_path dpo_1epoch.jsonl
```
