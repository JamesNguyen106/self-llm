# 4-MiniMax-M2 EvalScope

## The Significance and Value of Model Evaluation
Model evaluation is a crucial step that moves a model from "usable" to "usable and trustworthy." On one hand, systematic benchmark and stress testing quantify a model's real performance across knowledge, reasoning, alignment, and safety dimensions, helping to identify capability gaps, data biases, and robustness issues, thereby preventing the deployment of models with unknown risks. On the other hand, unified metrics and reproducible experiments provide comparable references for model selection, version iteration, and resource allocation, guiding engineering optimizations (e.g., context length, concurrency, inference parameters) to achieve better cost‑performance and user experience. For businesses, high‑quality evaluation reduces decision‑making and operational costs, continuously monitors regression and drift, and forms a "train‑evaluate‑deploy‑monitor" loop that accelerates model value delivery.

## Introduction to EvalScope
EvalScope is a model evaluation and performance benchmarking framework released by ModelScope. It includes many common benchmarks and evaluation metrics such as MMLU, CMMLU, C‑Eval, GSM8K, ARC, HellaSwag, TruthfulQA, MATH, and HumanEval. It supports evaluation of various model types, including LLMs, VLMs, embedding models, and reranker models. EvalScope can be used in multiple scenarios, such as end‑to‑end RAG evaluation, arena mode, and model inference performance testing. Moreover, through seamless integration with the ms‑swift training framework, it enables one‑click evaluation, providing a full‑link from model training to evaluation.

## How to Use EvalScope for Evaluation

> To simplify model usage and improve inference speed, we use SGLang to launch an OpenAI‑compatible service.

1. Install the required dependencies:

```bash
pip install sglang==0.5.5
pip install modelscope==1.31.0
pip install evalscope==1.1.1
pip install bfcl-eval==2025.10.27.1 
```

> For users who may encounter environment setup issues, we have prepared a MiniMax‑M2 environment image on the AutoDL platform. Click the link below to create an AutoDL example directly.

> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/mimimax-m2***

2. Start the model service

```bash
python -m sglang.launch_server \
  --model-path MiniMaxAI/MiniMax-M2 \
  --tp-size 8 \
  --ep-size 8 \
  --tool-call-parser minimax-m2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --reasoning-parser minimax-append-think \
  --port 8000 \
  --mem-fraction-static 0.85
```

## MiniMax‑M2 Performance

![minimax-benchmark-results](./images/fig-4-2-minimax-benchmark-results.png)

As shown, the model's tool usage and deep‑search capabilities are comparable to the best overseas models. While its programming performance lags slightly behind the top foreign models, it already ranks among the best domestically. Let's test the model on other tasks ourselves.

## IQuiz Test

Below we evaluate the model's **IQ and EQ** capabilities.

We will use the EvalScope framework on the IQuiz dataset, which contains 40 IQ questions and 80 EQ multiple‑choice questions, including classic problems from the LLM era:

- Which is larger: 9.8 or 9.11?
- How many "r" characters appear in the words "strawberry" and "blueberry"?
- Liu Yu is on vacation when he is asked to drive a leader to the airport. He is upset about his vacation being ruined, so he brakes hard. While in the car, the leader says: "Xiao Liu, this feels like the ancient capital Xi'an, as if I'm riding a horse carriage." What does the leader mean?

You can try the questions yourself [here](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/dataPeview).

Run the following command in the terminal:

```bash
evalscope eval \
  --model MiniMaxAI/MiniMax-M2 \
  --api-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --eval-type server \
  --eval-batch-size 16 \
  --datasets iquiz \
  --work-dir outputs/iquiz/MiniMax-M2
```

**Results:**

```
+------------+-----------+----------+----------+-------+---------+---------+
| Model      | Dataset   | Metric   | Subset   |   Num |   Score | Cat.0   |
+============+===========+==========+==========+=======+=========+=========+
| MiniMax-M2 | iquiz     | mean_acc | IQ       |    40 |  0.825  | default |
+------------+-----------+----------+----------+-------+---------+---------+
| MiniMax-M2 | iquiz     | mean_acc | EQ       |    80 |  0.6375 | default |
+------------+-----------+----------+----------+-------+---------+---------+
| MiniMax-M2 | iquiz     | mean_acc | OVERALL  |   120 |  0.7    | -       |
+------------+-----------+----------+----------+-------+---------+---------+
```

## Mathematics Ability Test

> Here we use **AIME2025** as an example to test the model's mathematical capability.

AIME (American Invitational Mathematics Examination) is a competition for high‑school to advanced students, focusing on algebra, geometry, number theory, and combinatorics. It emphasizes multi‑step reasoning and precise calculation, often requiring long chains of thought. This benchmark is suitable for assessing a model's analysis, decomposition, and derivation abilities on complex math problems, serving as an important reference for "reasoning depth" and "computational accuracy."

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='MiniMaxAI/MiniMax-M2',
    api_url='http://localhost:8000/v1',
    api_key='EMPTY',
    eval_type='server',
    datasets=['aime25'],
    eval_batch_size=16,
    dataset_args={
        'aime25': {
            'subset_list': [
                'AIME2025-I',      # you can select only one subset
                # 'AIME2025-II',   # uncomment to evaluate the second subset
            ],
        }
    },
    generation_config={
        'temperature': 0,
        'max_tokens': 65536,  # AIME questions require long outputs for reasoning
    },
    use_cache='outputs/aime25/MiniMax-M2',  # cache directory
    limit=5,  # evaluate only the first 5 questions for quick testing
)

run_task(task_cfg=task_cfg)
```

**Results:**

```
+------------+-----------+----------+------------+-------+---------+---------+
| Model      | Dataset   | Metric   | Subset     |   Num |   Score | Cat.0   |
+============+===========+==========+============+=======+=========+=========+
| MiniMax-M2 | aime25    | mean_acc | AIME2025-I |     5 |       1 | default |
+------------+-----------+----------+------------+-------+---------+---------+
```

## Code Ability Test

**LiveCodeBench** evaluates code generation with executable verification. It uses realistic programming problems and unit tests to assess end‑to‑end functional correctness and robustness. This benchmark reflects a model's ability to understand requirements, synthesize runnable code, pass test cases, and handle edge conditions.

```bash
evalscope eval \
  --model MiniMaxAI/MiniMax-M2 \
  --api-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --eval-type server \
  --eval-batch-size 16 \
  --datasets live_code_bench \
  --work-dir outputs/live_code_bench/MiniMax-M2\
  --limit 10
```

**Results:**

```
+------------+-----------------+----------+----------------+-------+---------+---------+
| Model      | Dataset         | Metric   | Subset         |   Num |   Score | Cat.0   |
+============+=================+==========+================+=======+=========+=========+
| MiniMax-M2 | live_code_bench | pass@1   | release_latest |    10 |       0 | default |
+------------+-----------------+----------+----------------+-------+---------+---------+
```

## Agent Ability Test

**BFCL V4** (Berkeley Function Calling Leaderboard V4) is a benchmark that evaluates large language models' ability to call functions/tools, especially in complex programming and tool‑calling tasks. It focuses on accuracy when invoking specific functions across languages (Python, Java, JavaScript) and REST APIs.

BFCL V4 has two main tracks:
1. **Basic Function‑Calling** – assesses accuracy of calling specific functions, handling programming languages and REST APIs.
2. **Agentic Version** – extends the basic track to evaluate how models operate as agents in noisy, error‑prone environments (e.g., server errors, rate limits, permission issues). It also includes multi‑hop web search to test complex question answering.

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='MiniMaxAI/MiniMax-M2',
    api_url='http://localhost:8000/v1',
    api_key='EMPTY',
    eval_type='server',
    datasets=['bfcl_v4'],
    eval_batch_size=10,
    dataset_args={
        'bfcl_v4': {
            'subset_list': [
                'simple_python',
                'simple_java',
                'simple_javascript',
                'multiple',
                'parallel',
                'parallel_multiple'
            ],
            'extra_params':{
                'underscore_to_dot': True,  # convert underscores to dots in function names
                'is_fc_model': True,        # enable function‑calling specific config
            }
        }
    },
    generation_config={'temperature': 0},
    use_cache='outputs/bfcl_v4',
    limit=3,  # quick test; remove for full evaluation
)

run_task(task_cfg=task_cfg)
```

**Results:**

```
+------------+-----------+----------+-------------------+-------+---------+---------+
| Model      | Dataset   | Metric   | Subset            |   Num |   Score | Cat.0   |
+============+===========+==========+===================+=======+=========+=========+
| MiniMax-M2 | bfcl_v4   | acc      | multiple          |     3 |       1 | default |
+------------+-----------+----------+-------------------+-------+---------+---------+
| MiniMax-M2 | bfcl_v4   | acc      | parallel          |     3 |       1 | default |
+------------+-----------+----------+-------------------+-------+---------+---------+
| MiniMax-M2 | bfcl_v4   | acc      | parallel_multiple |     3 |  0.6667 | default |
+------------+-----------+----------+-------------------+-------+---------+---------+
| MiniMax-M2 | bfcl_v4   | acc      | simple_java       |     3 |  0.3333 | default |
+------------+-----------+----------+-------------------+-------+---------+---------+
| MiniMax-M2 | bfcl_v4   | acc      | simple_javascript |     3 |  0.3333 | default |
+------------+-----------+----------+-------------------+-------+---------+---------+
| MiniMax-M2 | bfcl_v4   | acc      | simple_python     |     3 |       1 | default |
+------------+-----------+----------+-------------------+-------+---------+---------+
| MiniMax-M2 | bfcl_v4   | acc      | NON_LIVE          |    18 |  0.8055 | -       |
+------------+-----------+----------+-------------------+-------+---------+---------+
| MiniMax-M2 | bfcl_v4   | acc      | OVERALL           |    18 |  0.0806 | -       |
+------------+-----------+----------+-------------------+-------+---------+---------+
```

## Visualizing Evaluation Results

First, install the optional visualization package:

```bash
pip install 'evalscope[app]'
```

Then start the service and open `http://127.0.0.1:7861` to view the visualized results.

```bash
evalscope app
```
