# 02-gpt-oss-20b EvalScope Evaluation

## What is Large Model Evaluation

Large Language Model (LLM) evaluation refers to the process of comprehensively assessing the performance of LLMs in various tasks and scenarios. The purpose of evaluation is to measure the model's general capabilities, specific domain performance, efficiency, robustness, safety, and other aspects, in order to optimize model design, guide technology selection, and promote model deployment in practical applications.
The main content of evaluation includes the following aspects:

- General Capabilities: Assess the model's basic capabilities in language understanding, generation, reasoning, etc.
	
- Specific Domain Performance: Performance evaluation for specific tasks (such as mathematical reasoning, code generation, sentiment analysis, etc.).
	
- Efficiency and Resource Consumption: Including model training and inference time, computational resource requirements, etc.
	
- Robustness and Reliability: Assess the model's stability when facing noise, adversarial attacks, or input perturbations.
	
- Ethics and Safety: Detect whether the model generates harmful content, or has bias or discrimination.
	

EvalScope is the official model evaluation and performance benchmarking framework launched by the ModelScope community. It has built-in multiple common test benchmarks and evaluation metrics, such as MMLU, CMMLU, C-Eval, GSM8K, ARC, HellaSwag, TruthfulQA, MATH, and HumanEval, etc.; supports multiple types of model evaluation, including LLM, multimodal LLM, embedding model, and reranker model. EvalScope is also suitable for various evaluation scenarios, such as end-to-end RAG evaluation, arena mode, and model inference performance stress testing. In addition, through seamless integration with the ms-swift training framework, evaluation can be initiated with one click, realizing full-link support from model training to evaluation. Official website address: [https://evalscope.readthedocs.io/zh-cn/latest/get\_started](https://evalscope.readthedocs.io/zh-cn/latest/get_started)

## EvalScope Evaluation Usage Method

> In order to use the model more conveniently and improve inference speed, we use vLLM to start a Web service compatible with OpenAI format.

1. Create and activate a new conda environment:
	

```Bash
conda create -n gpt_oss_vllm python=3.12
conda activate gpt_oss_vllm
```

2. Install related dependencies:
	

```Bash
# Install PyTorch-nightly and vLLM
pip install --pre vllm==0.10.1+gptoss \    
            --extra-index-url https://wheels.vllm.ai/gpt-oss/ \    
            --extra-index-url https://download.pytorch.org/whl/nightly/cu128
# Install FlashInfer
pip install flashinfer-python==0.2.10
# Install evalscope
pip install evalscope[perf] -U
```

3. Start model service
	

> We successfully started the gpt-oss-20b model service on H20 GPU

```Bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 vllm serve openai/gpt-oss-20b --served-model-name gpt-oss-20b --trust_remote_code --port 8801
```

## Inference Speed Test

> We use EvalScope's inference speed test function to evaluate the model's inference speed.

Test Environment:

- Graphics Card: H20-96GB \* 1
	
- vLLM Version: 0.10.1 + gptoss
	
- Prompt Length: 1024 tokens
	
- Output Length: 1024 tokens
	

```Bash
evalscope perf \  
    --parallel 1 10 50 100 \  
    --number 5 20 100 200 \  
    --model gpt-oss-20b \  
    --url http://127.0.0.1:8801/v1/completions \  
    --api openai \  
    --dataset random \  
    --max-tokens 1024 \  
    --min-tokens 1024 \  
    --prefix-length 0 \  
    --min-prompt-length 1024 \  
    --max-prompt-length 1024 \  
    --log-every-n-query 20 \  
    --tokenizer-path openai-mirror/gpt-oss-20b \  
    --extra-args '{"ignore_eos": true}'
```

```Plain
╭──────────────────────────────────────────────────────────╮
│ Performance Test Summary Report                          │
╰──────────────────────────────────────────────────────────╯

Basic Information:
┌───────────────────────┬──────────────────────────────────┐
│ Model                 │ gpt-oss-20b                      │
│ Total Generated       │ 332,800.0 tokens                 │
│ Total Test Time       │ 154.57 seconds                   │
│ Avg Output Rate       │ 2153.10 tokens/sec               │
└───────────────────────┴──────────────────────────────────┘


                                    Detailed Performance Metrics                                    
┏━━━━━━┳━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃      ┃      ┃      Avg ┃      P99 ┃    Gen. ┃      Avg ┃     P99 ┃      Avg ┃     P99 ┃   Success┃
┃Conc. ┃  RPS ┃  Lat.(s) ┃  Lat.(s) ┃  toks/s ┃  TTFT(s) ┃ TTFT(s) ┃  TPOT(s) ┃ TPOT(s) ┃      Rate┃
┡━━━━━━╇━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│    1 │ 0.15 │    6.811 │    6.854 │  150.34 │    0.094 │   0.096 │    0.007 │   0.007 │    100.0%│
│   10 │ 0.96 │   10.374 │   10.708 │  986.63 │    0.865 │   1.278 │    0.009 │   0.010 │    100.0%│
│   50 │ 2.47 │   20.222 │   22.612 │ 2529.14 │    2.051 │   5.446 │    0.018 │   0.020 │    100.0%│
│  100 │ 3.37 │   29.570 │   35.594 │ 3455.61 │    2.354 │   6.936 │    0.027 │   0.028 │    100.0%│
└──────┴──────┴──────────┴──────────┴─────────┴──────────┴─────────┴──────────┴─────────┴──────────┘


               Best Performance Configuration               
 Highest RPS         Concurrency 100 (3.37 req/sec)         
 Lowest Latency      Concurrency 1 (6.811 seconds)          

Performance Recommendations:
• The system seems not to have reached its performance bottleneck, try higher concurrency
```

## Benchmark Test

> We use EvalScope's benchmark test function to evaluate the model's capabilities.
> Here we take the AIME2025 mathematical reasoning benchmark as an example to test the model's capabilities.

Run test script:

```Python
from evalscope.constants import EvalType
from evalscope import TaskConfig, run_task
task_cfg = TaskConfig(    
    model='gpt-oss-20b',  # Model name    
    api_url='http://127.0.0.1:8801/v1',  # Model service address    
    eval_type=EvalType.SERVICE, # Evaluation type, here use service evaluation    
    datasets=['aime25'],  # Dataset to test    
    generation_config={        
    'extra_body': {"reasoning_effort": "high"}  # Model generation parameters, set to high reasoning level here    
    },    eval_batch_size=10, # Batch size for concurrent testing    
    timeout=60000, # Timeout in seconds
    )
run_task(task_cfg=task_cfg)
```

Output as follows: The test result here is 0.8. You can try different model generation parameters, test multiple times, and check the results.

```Plain
+-------------+-----------+---------------+-------------+-------+---------+---------+
| Model       | Dataset   | Metric        | Subset      |   Num |   Score | Cat.0   |
+=============+===========+===============+=============+=======+=========+=========+
| gpt-oss-20b | aime25    | AveragePass@1 | AIME2025-I  |    15 |     0.8 | default |
+-------------+-----------+---------------+-------------+-------+---------+---------+
| gpt-oss-20b | aime25    | AveragePass@1 | AIME2025-II |    15 |     0.8 | default |
+-------------+-----------+---------------+-------------+-------+---------+---------+
| gpt-oss-20b | aime25    | AveragePass@1 | OVERALL     |    30 |     0.8 | -       |
+-------------+-----------+---------------+-------------+-------+---------+---------+ 
```

## Concurrency Test

```Bash
MODEL="gpt-oss-20b"
NUMBER=100
PARALLEL=20

evalscope perf \
    --url "http://localhost:8801/v1/chat/completions" \
    --parallel ${PARALLEL} \
    --model ${MODEL} \
    --number ${NUMBER} \
    --api openai \
    --dataset openqa \
    --stream \
    --swanlab-api-key 'your-swanlab-api-key' \
    --name "${MODEL}-number${NUMBER}-parallel${PARALLEL}"
```

- `--url`: Specify the API interface address of the model service, here is the locally deployed vLLM service address.
	
- `--parallel`: Specify the number of threads for concurrent requests, here set to 2 threads.
	
- `--model`: Specify the name of the model to be evaluated, here is **gpt-oss-20b**.
	
- `--number`: Specify the number of requests to be sent by each thread, here set to 100 requests.
	
- `--api`: Specify the API type used for evaluation, here is openai.
	
- `--dataset`: Specify the dataset used for evaluation, here is openqa.
	
- `--stream`: Specify whether to use streaming output, here set to true.
	
- `--swanlab-api-key`: Specify the API key of swanlab, here needs to be replaced with the actual API key.
	
- `--name`: Specify the name of the evaluation task, here is gpt-oss-20b-number100-parallel5.
	

The test results can be viewed on my experimental results [perf\_benchmark](https://swanlab.cn/@twosugar/perf_benchmark/overview), as shown in the figure below:

```SQL
Benchmarking summary:
+-----------------------------------+-----------+
| Key                               |     Value |
+===================================+===========+
| Time taken for tests (s)          |  289      |
+-----------------------------------+-----------+
| Number of concurrency             |    5      |
+-----------------------------------+-----------+
| Total requests                    |  100      |
+-----------------------------------+-----------+
| Succeed requests                  |  100      |
+-----------------------------------+-----------+
| Failed requests                   |    0      |
+-----------------------------------+-----------+
| Output token throughput (tok/s)   |  514.391  |
+-----------------------------------+-----------+
| Total token throughput (tok/s)    |  547.177  |
+-----------------------------------+-----------+
| Request throughput (req/s)        |    0.346  |
+-----------------------------------+-----------+
| Average latency (s)               |   14.1385 |
+-----------------------------------+-----------+
| Average time to first token (s)   |    0.0413 |
+-----------------------------------+-----------+
| Average time per output token (s) |    0.0095 |
+-----------------------------------+-----------+
| Average inter-token latency (s)   |    0.0095 |
+-----------------------------------+-----------+
| Average input tokens per request  |   94.75   |
+-----------------------------------+-----------+
| Average output tokens per request | 1486.59   |
+-----------------------------------+-----------+
2025-08-12 01:12:44,012 - evalscope - INFO - 
Percentile results:
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
| Percentiles | TTFT (s) | ITL (s) | TPOT (s) | Latency (s) | Input tokens | Output tokens | Output (tok/s) | Total (tok/s) |
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
|     10%     |  0.0171  | 0.0093  |  0.0094  |   2.0817    |      84      |      221      |    104.1461    |   109.3644    |
|     25%     |  0.0178  | 0.0093  |  0.0094  |   9.2178    |      87      |      960      |    105.2331    |   110.6657    |
|     50%     |  0.0191  | 0.0094  |  0.0094  |   19.256    |      94      |     2048      |    105.7487    |   111.2425    |
|     66%     |  0.0245  | 0.0095  |  0.0095  |   19.3416   |      98      |     2048      |    105.8855    |   113.9922    |
|     75%     |  0.0252  | 0.0095  |  0.0095  |   19.3696   |     100      |     2048      |    106.062     |    116.157    |
|     80%     |  0.0262  | 0.0095  |  0.0095  |   19.3809   |     104      |     2048      |    106.1607    |   121.9178    |
|     90%     |  0.0277  | 0.0097  |  0.0096  |   19.4948   |     107      |     2048      |    106.2971    |   152.2758    |
|     95%     |  0.0323  | 0.0099  |  0.0099  |   19.8081   |     109      |     2048      |    106.3563    |   178.6802    |
|     98%     |  0.5273  | 0.0102  |   0.01   |   20.7537   |     111      |     2048      |    106.4233    |   210.0931    |
|     99%     |  0.5283  | 0.0108  |  0.0101  |   20.7543   |     114      |     2048      |    107.4007    |   211.7394    |
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
```
