# Practical Guide to Fine-tuning BGE-M3 Embedding Model for Code Retrieval Scenarios based on PyTorch

## Overview

In this article, we will explain the principles of embedding model training code and the practice of embedding model training. This includes model fine-tuning data preparation, model fine-tuning, and model evaluation. Evaluation will be conducted on 20% of the data separated from the training data and the T2Retrieval dataset on the C-MTEB leaderboard.

## Explanation of Embedding Model Training Principles

### 1. Basic Concepts

The core goal of the Embedding model is to map text, images, or audio content into a high-dimensional vector space. For example, we use query and document embeddings in this case. Through the embedded vectors, text has semantic similarity. Similar texts are closer in the vector space, and we can calculate this using cosine similarity or dot product.

In this training, we used Contrastive Learning to fine-tune the BGE-M3 model. You can understand it as treating each query and its corresponding positive sample as a pair. Within the same batch, all other query-doc pairs are treated as negative samples. The model learns to increase the similarity of positive sample pairs and decrease the similarity of negative sample pairs.

### 2. Training Method: In-batch Negatives

This training uses the **In-batch Negatives** strategy, which is a popular strategy in contrastive learning and is widely used. So before we start, we need to introduce batch negative samples. First, let's introduce its core idea:

- In a batch, each query forms a positive sample pair with its corresponding positive document.
- Other query-document pairs in the same batch serve as negative samples.
- This provides `batch_size - 1` negative samples for each query without increasing additional computational costs.

From the core idea, you can see that if our batch size is larger, does it mean that each query can obtain more negative samples (`batch_size - 1`), the contrast signal in training is stronger, and the model is easier to pull positive samples closer and push negative samples farther, thereby learning finer-grained semantic representations?

In this experiment, I used batch sizes of 8, 16, and 80 to train the model. The actual training effect is also obvious. When the batch size is larger, the loss is smoother, and when the batch size is smaller, the fluctuation of loss is more obvious. Although there is a big gap with the batch size used in the original model training or fine-tuning, it is already close to the batch size used in BGE-M3 model fine-tuning. Because BGE-M3 uses a batch size of 192 when fine-tuning long sequence data. However, the graphics card I am currently using can only support a batch size of 80, which requires about 45GB of video memory.

I also checked the batch size settings in two related papers of BGE. In the "Training Recipe" section 3.4 of the C-Pack (BGE) paper, I saw that when introducing the "General purpose fine-tuning" stage, they pointed out relying purely on in-batch negative samples and used a large batch size of up to 19,200 to improve the discrimination of embeddings. It feels like a very large batch, and the video memory occupied requires a very large cluster to support.

In Section 4.2 "Detailed Analysis", the paper compared the impact of different batch sizes (256, 2,048, and 19,200) through experiments, and observed that as the batch size expands, the embedding quality improves consistently, with the most significant improvement in retrieval performance. So in general, batch size is relatively important in contrastive learning, and having large video memory plays a considerable factor in training such models.

For the completeness of the tutorial, of course, we also have to look at the paper corresponding to BGE-M3.

Appendix B.1 "Experimental Hyperparameters" and Table 9 of the M3-Embedding paper also have some descriptions about batch size design.

They are divided into two stages. One is the unsupervised data pre-training stage. In this stage, the batch size changes dynamically, depending on the sequence length of the training data.

For shorter sequences (0-500 tokens), the total batch size can reach 67,200; while for very long sequences (7000-8192 tokens), the batch size is 9,984.

I guess it is because the video memory is not large enough. Although they didn't say it, I know that the longer the sequence, the larger the video memory required. Even when we use BGE-M3 for embedding ourselves, if the doc length is close to 8k, not only the inference time is prolonged, but the video memory usage will also increase a lot. Interested friends can try it yourself and tell me.

The other stage is the fine-tuning stage. In this stage, the batch size also changes with the sequence length, but overall it is smaller than the previous stage. For short sequences, the batch size is 1,152, and for long sequences, it is 192. I don't know the reason clearly. Maybe their machines were allocated to other teams, or fine-tuning doesn't need so many machines, emm...

The table content is as follows:

<table><tr><td rowspan="2">Length Range</td><td colspan="2">Batch Size</td></tr><tr><td>Unsupervised</td><td>Fine-tuning</td></tr><tr><td>0-500</td><td>67,200</td><td>1,152</td></tr><tr><td>500-1000</td><td>54,720</td><td>768</td></tr><tr><td>1000-2000</td><td>37,248</td><td>480</td></tr><tr><td>2000-3000</td><td>27,648</td><td>432</td></tr><tr><td>3000-4000</td><td>21,504</td><td>336</td></tr><tr><td>4000-5000</td><td>17,280</td><td>336</td></tr><tr><td>5000-6000</td><td>15,072</td><td>288</td></tr><tr><td>6000-7000</td><td>12,288</td><td>240</td></tr><tr><td>7000-8192</td><td>9,984</td><td>192</td></tr></table>
Table 9: Detailed total batch size used in training for data with different sequence length ranges.

Interested friends can read the paper "C-Pack: Packed Resources For General Chinese Embeddings", which mentions many tricks on how the BGE model is trained.
The paper for BGE-M3 is: "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation"

Our core formulas are mainly the following two. For the query vector $Q$ and document vector $P$ in the batch:

Similarity Matrix:

$$
S = \frac{Q\,P^T}{\tau}
$$

Loss Function:

$$
L = \text{CrossEntropy}(S, y)
$$

There are two symbols to note:

- $\tau$ is the temperature parameter
- $y = [0, 1, 2, \dots, N-1]$ is the label vector, where $N$ is the batch size

### 3. Key Technical Components

#### 3.1 Vector Normalization

```python
q_emb = F.normalize(q_emb, p=2, dim=-1)
p_emb = F.normalize(p_emb, p=2, dim=-1)
```

Use L2 normalization to ensure all vectors are on the unit sphere:

$$
\|q\|_2 = 1, \quad \|p\|_2 = 1
$$

Where p=2: specifies using the L-p norm, here p=2 is the L2 (Euclidean) norm. You can also use p=1 (L1), which has a Chinese name called Manhattan norm, but L1 is not used for normalization here.

#### 3.2 Temperature Parameter

```python
temperature = 0.02
sim = q_emb @ p_emb.t() / self.temperature
```

The temperature parameter $\tau$ controls the sharpness of the similarity distribution. A smaller temperature value makes it easier for the model to distinguish between similar and dissimilar samples. From the formula, the temperature value is a scaling factor for the similarity score. Placing it in the denominator allows it to scale all similarity scores simultaneously.

When we set the temperature relatively low (close to 0), such as 0.02, the scaled value will become larger. If the temperature is equal to 1 or greater than 1, the scaled value will be smaller than the original value.

The temperature we use here is similar to the temperature control logic of LLM. If you have some understanding of the token probability mechanism of large language models, do you recall that you can use temperature to control the output when making a request? A lower temperature will make the model output more deterministic, while a higher temperature will make the model output more diverse.

LLM's token sampling strategy generally chooses random sampling methods, which leads to different text outputs each time. If a deterministic method is chosen, such as greedy search (selecting the highest probability each time), it will have the effect that the output result is the same each time. In principle, even with random sampling, the token with higher probability is more likely to be sampled.

Okay, that's enough about LLM. Let's use an example to illustrate. Use a higher temperature $\tau = 1$.

When the temperature is 1, we apply the Softmax function directly to the original similarity scores.

The calculation process is as follows:
$P(\text{Positive}) = \frac{e^{0.9/1}}{e^{0.9/1} + e^{0.8/1} + e^{0.5/1} + e^{0.2/1}}$

$P(\text{Positive}) = \frac{2.46}{2.46 + 2.23 + 1.65 + 1.22} = \frac{2.46}{7.56} \approx 0.325$

We can also calculate the probabilities of other samples:

- Probability of Negative Sample 1: $2.23 / 7.56 \approx 0.295$
- Probability of Negative Sample 2: $1.65 / 7.56 \approx 0.218$
- Probability of Negative Sample 3: $1.22 / 7.56 \approx 0.162$

In this case, the probability of the positive sample is only 32.5%. The most difficult negative sample to distinguish (Negative Sample 1) obtained a probability of 29.5%. Thinking from the perspective of calling a large model, have you noticed that we rarely use temperature=1 for model calls? Because when temperature=1, the probability difference of each token is actually not that large, which easily leads to the model outputting garbled characters or nonsense.

So for model training, the probability distribution given by the model is relatively smooth, and it does not have strong confidence that the positive sample is the only correct answer. In this way, the penalty signal calculated by the loss function is not strong enough, and the model training effect will be worse.

Another case, use a lower temperature $\tau = 0.1$.

Now, we lower the temperature to 0.1. Use temperature to scale the similarity score:

- **Positive Sample**: $0.9 / 0.1 = 9.0$
- **Negative Sample 1**: $0.8 / 0.1 = 8.0$
- **Negative Sample 2**: $0.5 / 0.1 = 5.0$
- **Negative Sample 3**: $0.2 / 0.1 = 2.0$

Notice that the gap in original scores (e.g., only 0.1 difference between positive sample and negative sample 1) is magnified by 10 times (now a difference of 1.0).

Next, we apply the Softmax function to these magnified scores:

$P(\text{Positive}) = \frac{e^{9.0}}{e^{9.0} + e^{8.0} + e^{5.0} + e^{2.0}}$

$P(\text{Positive}) = \frac{8103.1}{8103.1 + 2981.0 + 148.4 + 7.4} = \frac{8103.1}{11239.9} \approx 0.721$

Let's look at the probabilities of other samples:

- Probability of Negative Sample 1: $2981.0 / 11239.9 \approx 0.265$
- Probability of Negative Sample 2: $148.4 / 11239.9 \approx 0.013$
- Probability of Negative Sample 3: $7.4 / 11239.9 \approx 0.001$
  
The above content is the result calculated by our code, which is placed a little below. By lowering the temperature, the probability of the positive sample has changed from 32.5% to 72.1%. Hasn't the confidence of our positive sample significantly improved!

And the probability of the most difficult negative sample has changed from 29.5% to 26.5%. The probabilities of other less relevant negative samples have basically decreased, that is, approaching the positive sample: the probability ratio of negative samples is 1:0.

To make it easier for everyone to see the gap between the two, I visualized their probability changes. You can see the changes from the picture below.

![Probability distribution change under different temperatures](./images/不同温度下概率分布的变化.png)

The above is a line chart, which may not be easy to see the degree of change between the two. Let's change to a bar chart.

![Probability distribution change under different temperatures](images/不同温度下概率分布的变化-柱形图.png)

From the probability distribution in the picture, can we see that the difference between before and after is that low temperature will make the probability distribution sharper? I mean it will make the probability of the first one higher and higher, and others will decrease.

Or highlight the probability of correct samples.

When calculating the loss, since the predicted probability (72.1%) is still about 28% away from the expected close to 100%, and our expectation is that the output result of the model is that the ratio between positive samples and negative samples approaches 1:0, or 0.9:0.1 is also acceptable.

Therefore, further amplifying the difference between positive and negative samples, that is, making the distinction between the two more obvious, is our training direction.

Thinking from another angle, the amplified value will actually be placed in the position of the exponent. The growth of the exponent is explosive. Just amplifying a little bit will cause the numerical gap to be opened quickly. You can see the following picture:

![Exponential function of e](images/e的指数函数.png)

The figure shows the function image of $e^x$ where $x$ is from 0-10. The further back, the value will approach infinity, almost impossible to display. Just adding a slight power will cause a very obvious gap between sample probabilities.

If you want to try it yourself, you can use the following code to test what effect changing the temperature will have:

```python
import numpy as np

scores = np.array([0.9, 0.8, 0.5, 0.2])

temperatures = [1, 0.1]

probabilities = {}
for tau in temperatures:
    # Scale similarity scores
    scaled_scores = scores / tau
    # Apply Softmax function
    exp_scores = np.exp(scaled_scores)
    probabilities[tau] = exp_scores / exp_scores.sum()

probabilities
```

Output result after running the code:

```text
{1: array([0.32554809, 0.2945681 , 0.21822141, 0.1616624 ]),
0.1: array([7.20924938e-01, 2.65213463e-01, 1.32042008e-02, 6.57398449e-04])}
```

From the output results, we keep 3 decimal places. The probability distribution when the temperature is 1 is [0.326, 0.295, 0.218, 0.162], and the probability distribution when the temperature is 0.1 is [0.721, 0.265, 0.013, 0.001].

#### 3.3 Loss Function

Use Cross Entropy Loss function to train the model:

```python
labels = torch.arange(sim.size(0), device=sim.device, dtype=torch.long)
loss = F.cross_entropy(sim, labels)
```

We can understand cross-entropy loss as converting embedding learning into a classification problem, that is, within a batch, matching each query with its corresponding correct document is regarded as a classification task.

Where each row of the similarity matrix represents the classification score of the query for all candidate documents, and the diagonal elements should be the highest score of the correct category.

By creating labels [0, 1, 2, ..., batch_size-1], let the model learn to classify the i-th query to the i-th document. Let's explain through the loss function below.

For the $i$-th query, the loss function is:

$$
L_i = -\log\left(\frac{\exp(s_{i,i}/\tau)}{\sum_{j=1}^{N}\exp(s_{i,j}/\tau)}\right)
$$

Where $s_{i,j}$ is the similarity score between the $i$-th query and the $j$-th document.

##### Introduction to Loss Function Formula

Let's look at some contents of the loss function.

**1. Similarity Matrix Calculation**

$S = \frac{QP^T}{\tau}$

- $Q$: Query vector matrix, shape is $N \times D$, where $N$ is batch size, D is vector dimension
- $P$: Document vector matrix, shape is $N \times D$
- $P^T$: Transpose of document vector matrix, shape is $D \times N$
- $QP^T$: Matrix multiplication, result is $N \times N$ similarity matrix
- $\tau$: Temperature parameter (0.02), used to adjust similarity distribution

Here, the cosine similarity of each query and all documents is calculated through matrix calculation. You need a little basis in linear algebra.

**2. Loss Function for Single Query**

$$
L_i = -\log\left(\frac{\exp(s_{i,i}/\tau)}{\sum_{j=1}^{N}\exp(s_{i,j}/\tau)}\right)
$$

- $s_{i,i}$: Similarity between the $i$-th query and its corresponding positive sample (diagonal element)
- $s_{i,j}$: Similarity between the $i$-th query and the $j$-th document
- $\exp(s_{i,j}/\tau)$: Exponential scaling of similarity
- Numerator: Similarity of positive sample pair
- Denominator: Sum of similarities of all documents (denominator of softmax)

Mainly used to maximize the probability of positive sample pairs and minimize the probability of negative sample pairs, that is, as we said before, the probability of positive samples should approach 1, and the probability of negative samples should approach 0.

**Softmax Function**

> For details, please refer to pytorch documentation: <https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html>
> Pytorch statement: Applies the Softmax function to an n-dimensional input Tensor using rescaling them so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.

The score part in the loss function is actually the application of the softmax function (before taking log):

$$
\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{n}\exp(x_j)}
$$

![softmax function curve](images/softmax函数曲线图-数学函数图像.png)

From the above figure, it can be seen that an important property of the softmax function is:

Any value passed through the softmax function will become a number between 0 and 1.

This is because through exponential operation, both numerator and denominator use the exp() function, ensuring all values are positive. From the initial exponential function graph of e, it can be seen that the e exponential function is always positive. Also, due to the normalization effect, each value is divided by the sum of all values, ensuring the final result is between 0 and 1.

In our code case:

- $x_i = s_{i,i}/\tau$ (Similarity of positive sample pair)
- $x_j = s_{i,j}/\tau$ (Similarity of all documents)
- $\frac{\exp(s_{i,i}/\tau)}{\sum_{j=1}^{N}\exp(s_{i,j}/\tau)}$ is the specific application of the softmax function

**Why use softmax?**

The main advantage of using the Softmax function is that it can convert the model's output scores (Logits) into a probability distribution. From its calculation results, it can be seen that all values will be mapped to the range (0, 1), and their sum is 1. This is convenient for subsequent logarithmic (log) operations to calculate cross-entropy loss.

**Why use logarithmic function?**

In the loss function, we use $-\log$ to calculate the loss. There are several important reasons for this:

![Logarithmic function y=log(x) image](images/01-2.png)

From the logarithmic function image, it can be seen that when the value is closer to 1, log(x) is closer to 0, and the loss is smaller. When the value is closer to 0, log(x) is closer to a very large negative number, and -log(x) is closer to a very large positive number.

Considering from a mathematical perspective, the closer probability $p$ is to 1, the closer loss -log(p) is to 0. The closer probability $p$ is to 0, the closer loss -log(p) is to infinity. This characteristic makes the model penalize wrong predictions more severely.

Before taking the log, we also just performed the softmax operation on the logarithmic value, so that when the model predicts correctly (probability close to 1), the loss is very small, and when the model predicts incorrectly (probability close to 0), the loss is very large. This prompts the model to strive to learn correct features.

**3. Overall Loss Function**

$$
L = \text{CrossEntropy}(S, y)
$$

Meaning of symbols:

- $S$: Similarity matrix
- $y = [0, 1, 2, ..., N-1]$: Label vector, representing the correct match position for each query
- Cross-entropy loss converts the similarity matrix into a probability distribution and calculates the difference with the true labels

**4. Specific Form of Similarity Matrix S**

Assume batch size = 3, then the shape of similarity matrix $S$ is $3 \times 3$:

$$
S = \begin{bmatrix}
s_{0,0} & s_{0,1} & s_{0,2} \\
s_{1,0} & s_{1,1} & s_{1,2} \\
s_{2,0} & s_{2,1} & s_{2,2}
\end{bmatrix}
$$

**Specific Numerical Example**:

Assume the similarity matrix after temperature scaling might be:

$$
S =
\begin{bmatrix}
8.5 & 2.1 & 1.8 \\
1.2 & 9.3 & 0.9 \\
0.5 & 1.7 & 8.1
\end{bmatrix}
$$

**Meaning of Matrix Elements**:

- **Diagonal Elements** (Positive Sample Pairs):
  - $s_{0,0} = 8.5$: Similarity between query 0 and document 0 (should be positive sample)
  - $s_{1,1} = 9.3$: Similarity between query 1 and document 1 (should be positive sample)
  - $s_{2,2} = 8.1$: Similarity between query 2 and document 2 (should be positive sample)

- **Off-diagonal Elements** (Negative Sample Pairs):
  - $s_{0,1} = 2.1$: Similarity between query 0 and document 1 (negative sample)
  - $s_{0,2} = 1.8$: Similarity between query 0 and document 2 (negative sample)
  - $s_{1,0} = 1.2$: Similarity between query 1 and document 0 (negative sample)
  - etc...

You can use the following code to calculate:

```python
import math

s_00 = 8.5
s_01 = 2.1
s_02 = 1.8
tau = 0.02

# Calculate scaled similarity
scaled_s_00 = s_00 / tau
scaled_s_01 = s_01 / tau
scaled_s_02 = s_02 / tau

# Calculate exponential value
exp_s_00 = math.exp(scaled_s_00)
exp_s_01 = math.exp(scaled_s_01)
exp_s_02 = math.exp(scaled_s_02)

scaled_s_00, scaled_s_01, scaled_s_02, exp_s_00, exp_s_01, exp_s_02
```

Calculation output result:

```python
(425.0,
105.0,
90.0,
3.759713994046786e+184,
3.989519570547216e+45,
1.2204032943178408e+39)
```

Continue to calculate the loss value:

```python
import math

# Calculate numerator and denominator
numerator = math.exp(425)
denominator = math.exp(425) + math.exp(105) + math.exp(90)

# Calculate loss
loss = -math.log(numerator / denominator)
loss
```

Output:

```text
-0.0
```

The output result is very close to the expected result we obtained from the e exponential function graph above, because $\exp(425)$ is already much larger than $\exp(105)$.

During the model training process, the model parameters are updated through gradient descent, increasing the diagonal elements (making the similarity of positive sample pairs higher) and decreasing the off-diagonal elements (making the similarity of negative sample pairs lower), ultimately making the diagonal elements much larger than the off-diagonal elements.

This is why in the process of training the model, the diagonal values of the similarity matrix will be significantly larger than other positions.

**4. Actual Code Implementation**

```python
# Calculate similarity matrix
sim = q_emb @ p_emb.t() / self.temperature

# Generate labels (diagonal indices)
labels = torch.arange(sim.size(0), device=sim.device, dtype=torch.long)

# Calculate cross entropy loss
loss = F.cross_entropy(sim, labels)
```

The actual code here mentions cross-entropy loss, but actually cross_entropy is based on CrossEntropyLoss. We can check the source code of CrossEntropyLoss:

**CrossEntropyLoss Chinese Translation Version:**

```python
class CrossEntropyLoss(_WeightedLoss):
    r"""This criterion computes the cross entropy loss between input logits and target.

    It is useful when training a classification problem with C classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The input is expected to contain raw, unnormalized scores for each class.
    input has to be a Tensor of size :math:`(C)` for unbatched input,
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1` for the
    K-dimensional case.
    The latter is useful for higher dimension inputs, such as computing cross entropy loss per-pixel for 2D images.

    The target that this criterion expects should contain either:

    - Class indices in the range :math:`[0, C)` where :math:`C` is the number of classes; if
      `ignore_index` is specified, this loss also accepts this class index (this index
      may not necessarily be in the class range). The unreduced (i.e. with :attr:`reduction`
      set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
      :math:`d_1, ..., d_k` for the K-dimensional case. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}} l_n, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

      Note that this case is equivalent to the combination of :class:`~torch.nn.LogSoftmax` and
      :class:`~torch.nn.NLLLoss`.

    - Probabilities for each class; useful when labels beyond a single class per minibatch item
      are required, such as for blended labels, label smoothing, etc. The unreduced (i.e. with
      :attr:`reduction` set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
      :math:`d_1, ..., d_k` for the K-dimensional case. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \frac{\sum_{n=1}^N l_n}{N}, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

    .. note::
        The performance of this criterion is generally better when `target` contains class
        indices, as this allows for optimized computation. Consider providing `target` as
        class probabilities only when a single class label per minibatch item is too restrictive.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
            of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

    Shape:
        - Input: :math:`(C)`, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of K-dimensional loss.
        - Target: If containing class indices, shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with
          :math:`K \geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`.
          If containing class probabilities, same shape as the input and each value should be between :math:`[0, 1]`.
        - Output: If reduction is 'none', shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of K-dimensional loss, depending on the shape of the input. Otherwise, scalar.


        where:

        .. math::
            \begin{aligned}
                C ={} & \text{number of classes} \\
                N ={} & \text{batch size} \\
            \end{aligned}

    Examples:

        >>> # Example of target with class indices
        >>> loss = nn.CrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
        >>>
        >>> # Example of target with class probabilities
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5).softmax(dim=1)
        >>> output = loss(input, target)
        >>> output.backward()
    """
```

**Mathematical Formula of CrossEntropy:**

The above content may not be suitable for direct viewing. Let's change the expression to see what the formula looks like. For the case of discrete labels (category indices), the formula of the cross-entropy loss function is:

$$L = -\frac{1}{N}\sum_{i=1}^{N}\log\left(\frac{\exp(x_{i,y_i})}{\sum_{j=1}^{C}\exp(x_{i,j})}\right)$$

Where:

- $N$ is batch size
- $C$ is the number of categories (equal to batch size in our case)
- $x_{i,j}$ is the prediction score of the $i$-th sample on the $j$-th category
- $y_i$ is the true category label of the $i$-th sample

**Specific Form in our embedding code:**

$$
L = -\frac{1}{N}\sum_{i=1}^{N}\log\left(\frac{\exp(s_{i,i}/\tau)}{\sum_{j=1}^{N}\exp(s_{i,j}/\tau)}\right)
$$

Note in the formula:

$s_{i,j}$ is the element in the similarity matrix $S$
$\tau$ is the temperature parameter

**Formula Derivation Process:**

1. **Softmax Function**: Convert prediction scores to probability distribution

$$
p_{i,j} = \frac{\exp(x_{i,j})}{\sum_{k=1}^{C}\exp(x_{i,k})}
$$

2. **Cross Entropy**: Measure the difference between predicted probability and true label

$$
L_i = -\sum_{j=1}^{C}y_{i,j}\log(p_{i,j})
$$

3. **Simplified Form**: For one-hot encoded true labels, $y_{i,j}=1$ only when $j=y_i$

$$
L_i = -\log(p_{i,y_i}) = -\log\left(\frac{\exp(x_{i,y_i})}{\sum_{j=1}^{C}\exp(x_{i,j})}\right)
$$

4. **Batch Average**: Average the loss of the entire batch

$$
L = -\frac{1}{N}\sum_{i=1}^{N}\log\left(\frac{\exp(x_{i,y_i})}{\sum_{j=1}^{C}\exp(x_{i,j})}\right)
$$

The essence of this loss function is **InfoNCE Loss** (Info Noise Contrastive Estimation), which allows the model to learn to distinguish between positive and negative sample pairs through contrastive learning. The contrastive learning loss function used by MoCo is InfoNCE loss, which is used to train the model, and it is similar to the formula we introduced this time.

Friends interested in InfoNCE Loss can read the following article:
> Connection between Contrastive Learning Loss (InfoNCE loss) and Cross Entropy Loss, and the role of temperature coefficient - Youngshell's article - Zhihu
> <https://zhuanlan.zhihu.com/p/506544456>
> For more hardcore content, you can read this:
> <https://lilianweng.github.io/posts/2021-05-31-contrastive/>

### 4. Training Optimization Strategy

#### 4.1 Mixed Precision Training (FP16)

```python
with torch.cuda.amp.autocast(enabled=cfg['fp16']):
    out = self.model(**batch)
```

Mixed precision training is a technique that uses floating-point numbers of different precisions simultaneously for deep learning model training. In this training, we use FP16 (half-precision floating-point numbers) for most calculations, while keeping key operations in FP32 (single-precision floating-point numbers) to ensure numerical stability. This method can significantly reduce video memory usage and accelerate training while maintaining the model's training effect. Through PyTorch's Automatic Mixed Precision (AMP) function, the system can automatically select the appropriate data type and prevent gradient underflow through gradient scaling.

Friends interested in mixed precision training can go to the link below to view more specific content:
<https://docs.pytorch.org/docs/stable/amp.html>

#### 4.2 Gradient Accumulation

```python
accumulation = self.cfg['accumulation_steps']
loss = out['loss'] / accumulation
```

Simulate a larger batch size by accumulating gradients:
$$\nabla_{\text{effective}} = \frac{1}{K}\sum_{k=1}^{K}\nabla_k$$

Where $K$ is the accumulation steps.
Gradient accumulation is mainly to solve the problem of insufficient video memory. When training deep learning models, we usually tend to use larger batch sizes. However, the problem is that increasing the batch size will directly lead to an increase in video memory usage.

So, we are more likely to adjust the batch size as much as possible within the allowable range of our video memory. Of course, the batch size is not the bigger the better, but more of a moderate value.

Let me give an example to help everyone understand.

Suppose our video memory size only allows us to set batch_size = 8, but we found through experiments that using batch_size = 256 works better.

At this time, gradient accumulation can be used.

Take a batch of data with size 8, perform one forward propagation and backward propagation, and calculate the gradient. But do not update the model parameters immediately, but accumulate this gradient.

Repeat the first step, take another batch of data to calculate the gradient, and continue to accumulate it to the previous gradient. Repeat this process until we reach our expected number of times.

After accumulating gradients of 8 small batches, we get a gradient equivalent to 256 samples, calculate the total gradient through these 256 samples, and update the model parameters once.

Through the process here, everyone should be able to understand what gradient accumulation is, but our code actually does not combine well with batch negative samples. Everyone should be able to find this problem by looking at the source code.

Suppose our goal is to simulate a large batch of size 256 to obtain enough negative samples, but our video memory is only enough to hold a batch of size 32. So, we set the gradient accumulation steps to 8 (because 32×8=256).

Step 1: The model reads the first small batch of size 32. When calculating the loss of a sample in this batch, it can only see the other 31 samples in this small batch as negative samples. Then calculate the gradient of this small batch and store it.

Step 2: The model reads the second small batch of size 32. Similarly, the samples in this batch can only see 31 other samples in their own batch as negative samples. The calculated gradient is accumulated to the gradient of the first step.

...This process repeats 8 times.

After step 8: The model accumulates gradients of 8 small batches, which is equivalent to a batch of size 256 at the gradient level. Then, the model uses this accumulated gradient to perform a parameter update.

Here comes the key problem: for the calculation of the loss function, the negative sample pool of each sample has not expanded. The negative samples it can see are always limited to the 31 inside its small batch (micro-batch). Although we simulated a large batch of 256 at the parameter update level, at the loss calculation level, the source range of batch negative samples is still 32.

So can we think about whether we can only store their most critical information, that is, the "representation vectors" (embeddings) needed when calculating contrastive loss.

The answer is yes, this is the core idea of algorithms like MoCo (Momentum Contrast).

Unfortunately, our code does not use his algorithm logic.

However, interested students can see how this paper does it.

This is a paper from 2020, the address is at: <https://arxiv.org/pdf/1911.05722>
The name of the paper is: "Momentum Contrast for Unsupervised Visual Representation Learning"

#### 4.3 Learning Rate Scheduling

Use cosine annealing learning rate scheduler:

```python
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
```

![Learning Rate Scheduling](images/01-3.png)

Simply put, this function creates a learning rate scheduling strategy:

1. Warm-up stage: The learning rate starts at 0 and increases linearly to the set initial learning rate.
2. Cosine decay stage: After the warm-up is completed, the learning rate gradually decreases from the initial value to 0 according to the shape of the cosine function.

This strategy is very common in deep learning and can help the model converge stably in the early stage of training and fine-tune parameters in the later stage.
Friends interested in learning rate scheduling can refer to the link below to see detailed explanations: <https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html>

### 5. Evaluation Metrics

#### 5.1 Recall@1

$$\text{Recall@1} = \frac{1}{N}\sum_{i=1}^{N}1[\text{rank}_i = 0]$$

Measures the proportion of the model ranking the correct answer first.

#### 5.2 MRR (Mean Reciprocal Rank)

$$\text{MRR} = \frac{1}{N}\sum_{i=1}^{N}\frac{1}{\text{rank}_i + 1}$$

Considers the ranking position of the correct answer, giving higher reward scores to correct answers ranked higher.

### 6. Data Processing Flow

#### 6.1 Data Format

Training data uses JSONL format, each line contains:

```json
{
    "query": "Write function to find the sum of all items in the given dictionary.",
    "pos_doc": "def return_sum(dict):\n  sum = 0\n  for i in dict.values():\n    sum = sum + i\n  return sum",
    "meta": {"task_id": 796}
}
```

**Field Description**:

- `query`: Query text, used to test the model's understanding of the query
- `pos_doc`: Positive sample document, text content related to the query
- `meta`: Metadata

#### 6.2 Tokenization

```python
tokenizer(queries, padding=True, truncation=True, max_length=query_max_len, return_tensors='pt')
```

**Parameter Description**:

- `queries`: List of input query texts
- `padding=True`: Automatic padding, padding all sequences to the same length, ensuring consistent input length in the batch
- `truncation=True`: Automatic truncation, truncating when text exceeds maximum length
- `max_length=query_max_len`: Set maximum length limit, text exceeding this length will be truncated
- `return_tensors='pt'`: Return PyTorch tensor format, convenient for direct input into the model for calculation

Perform tokenization on queries and documents separately, supporting padding and truncation to handle texts of different lengths.

### 7. BGE-M3 Model Introduction

For more specific model introduction, you can read the bge-m3 model card on hf:
Address: <https://huggingface.co/BAAI/bge-m3>

#### 7.1 Model Overview

BGE-M3 is a multi-functional embedding model with the following characteristics:

- **Multi-Functionality**: Supports three common retrieval functions simultaneously: dense retrieval, multi-vector retrieval, and sparse retrieval
- **Multi-Linguality**: Supports over 100 working languages
- **Multi-Granularity**: Able to handle inputs of different granularities, from short sentences to long documents up to 8192 tokens

Among the multiple common retrieval functions supported by bge-m3, if you need to know more details, you can go to the link below to read the corresponding paper or code.

- **Dense retrieval**: Map text to a single embedding vector, e.g., [DPR](https://arxiv.org/abs/2004.04906), [BGE-v1.5](https://github.com/FlagOpen/FlagEmbedding)
- **Sparse retrieval**: Vocabulary matching, vector size equal to vocabulary size, most positions set to zero, only calculating weights for tokens appearing in the text. e.g., BM25, [unicoil](https://arxiv.org/pdf/2106.14807.pdf), and [splade](https://arxiv.org/abs/2107.05720)
- **Multi-vector retrieval**: Use multiple vectors to represent text, e.g., [ColBERT](https://arxiv.org/abs/2004.12832)

#### 7.2 Model Specifications

| Model Name | Dimension | Sequence Length | Introduction |
|:----:|:---:|:---:|:---|
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 1024 | 8192 | Multilingual; Unified fine-tuning from bge-m3-unsupervised (dense, sparse, and colbert) |
| [BAAI/bge-m3-unsupervised](https://huggingface.co/BAAI/bge-m3-unsupervised) | 1024 | 8192 | Multilingual; Contrastive learning from bge-m3-retromae |
| [BAAI/bge-m3-retromae](https://huggingface.co/BAAI/bge-m3-retromae) | -- | 8192 | Multilingual; Extend max_length of [xlm-roberta](https://huggingface.co/FacebookAI/xlm-roberta-large) to 8192 and further pre-train via [retromae](https://github.com/staoxiao/RetroMAE) |

## Explanation of Embedding Model Training Practice

### Project File Structure Description

This project is a complete embedding model fine-tuning practice project, mainly containing the following files and directories:

**Core Files:**

- `01-train_embedding_with_validation.ipynb` - Main training and validation Jupyter notebook
- `02-C-MTEB_retrieval_evaluation.ipynb` - Notebook for C-MTEB retrieval evaluation
- `requirements.txt` - List of project dependency packages

**Model Related Directories:**

- `bge-m3/` - Original BGE-M3 model files, containing pre-trained weights and configuration
- `models/bge-m3-finetuned/` - Model checkpoints saved during fine-tuning
  - `best_checkpoint/` - Model checkpoint with best performance on validation set
  - `checkpoint-epoch-*/` - Checkpoints for each training epoch
  - `final_model/` - Final model after training completion

**Data Files:**

- `mbpp.jsonl` - Full version of MBPP dataset
- `mbpp_train.jsonl` - Training set data
- `mbpp_val.jsonl` - Validation set data
- `sanitized-mbpp.json` - Cleaned MBPP data

**Evaluation Results:**

- `c_mteb_results/` - C-MTEB evaluation results
  - `BGE-M3/` - Evaluation results of original BGE-M3 model
  - `BGE-M3-Finetuned/` - Evaluation results of fine-tuned model

**Documentation:**

- `README.md` - Project main documentation

### Practice Session

#### Environment Configuration

First enter the `train_embedding_with_validation.ipynb` file in the current directory.
Before entering, please prepare the python environment and install the basic environment configuration required by the notebook.

**Method 1: Install using requirements.txt**

```bash
pip install -r requirements.txt
```

**Method 2: Manually install core dependencies**

```bash
# Install notebook and ipykernel
pip install notebook ipykernel

# Install deep learning related dependencies
pip install torch transformers sentence-transformers

# Install bge ecosystem embedding model dedicated library
pip install FlagEmbedding mteb

# Install data processing and scientific computing libraries
pip install numpy matplotlib tqdm scipy scikit-learn pandas seaborn
```

#### Check Library Versions

**Method 1: Use pip list to check**

```bash
# Check versions of all installed libraries
pip list

# Check versions of specific libraries
pip list | grep -E "(torch|transformers|sentence-transformers|FlagEmbedding|mteb)"
```

**Method 2: Use Python code to check**

```python
import torch
import transformers
import sentence_transformers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import scipy
import tqdm
import seaborn as sns

print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"sentence-transformers: {sentence_transformers.__version__}")
print(f"numpy: {np.__version__}")
print(f"matplotlib: {plt.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"tqdm: {tqdm.__version__}")
print(f"seaborn: {sns.__version__}")

try:
    import FlagEmbedding
    print(f"FlagEmbedding: {FlagEmbedding.__version__}")
except:
    print("FlagEmbedding: Not installed")

try:
    import mteb
    print(f"mteb: {mteb.__version__}")
except:
    print("mteb: Not installed")
```

**Current Environment Version Information**

The following are the main library versions used in the development environment of this project:

| Library Name | Version Number |
|--------|--------|
| torch | 2.7.1 |
| transformers | 4.53.0 |
| sentence-transformers | 5.0.0 |
| numpy | 2.2.5 |
| pandas | 2.2.3 |
| matplotlib | 0.1.7 (matplotlib-inline) |
| tqdm | 4.67.1 |
| scipy | 1.16.0 |
| scikit-learn | 1.7.0 |

**CUDA Environment Configuration**

The CUDA hardware and software configuration of the development environment of this project is as follows:

| Configuration Item | Details |
|--------|----------|
| PyTorch Version | 2.7.1+cu126 |
| CUDA Version | 12.6 |
| CUDA Device Count | 1 |
| GPU Device Name | NVIDIA RTX A6000 |

NVIDIA RTX A6000, with 48GB video memory. This experiment requires at least 24GB video memory. If your graphics card video memory is insufficient, you can appropriately reduce the batch size. Prioritize graphics cards with more than 24GB, because 48GB is also a bit insufficient in the C-MTEB evaluation task, which requires a long evaluation time, about 6 hours. Of course, I did not turn on the maximum batch size. The current evaluation video memory is about at least 17GB. You can adjust the batch size.

After installation is complete, you can open `train_embedding_with_validation.ipynb` to start the fine-tuning practice of the embedding model.

#### Model Download

Before starting, let's download the bge-m3 model.
Here we will use modelscope for model download, so install modelscope first using the following command:

```bash

pip install modelscope
```

Next, let's use modelscope to download the bge-m3 model. Use the following command to download:

```bash
modelscope download --model BAAI/bge-m3  --local_dir ./bge-m3
```

This command must be executed in the `models/BGE-M3-finetune-embedding-with-valid` directory. That is to say, your model needs to be downloaded into `models/BGE-M3-finetune-embedding-with-valid`. If you are still unclear, you can see an expected directory structure diagram below (at the end of the dataset preparation section).

#### Dataset Preparation

You can go to the following address to download the dataset and put it in the current directory:

<https://modelscope.cn/datasets/opencompass/mbpp>

After opening the link, you can see the page looks like this:

![mbpp dataset](images/01-4.png)

You can switch to the data preview page to check what our data looks like:

![Data preview page](images/01-5.png)

If you have finished reading, or want to continue with data preparation work, you can switch to the data file page and then download the dataset:

![Data file download page](images/01-6.png)

If you complete the model training and evaluation, you can finally get such a directory structure:

![Directory structure description](images/01-7.png)

Where `c_mteb_results` is the evaluation result after completing C-MTEB. If you have not completed it, it will not be output. `c_mteb_results` is also the cache of our evaluation. If you complete a complete model cache, it will be loaded directly from here next time.

You only need to download the two files pointed to by the arrows from modelscope:

![Two files pointed to by arrows](images/01-8.png)

After downloading, ensure that your file name is the same as the file name in my example directory. If it is different, please change it to the same name. `sanitized-mbpp.json` is the dataset we use this time.

Because it is a cleaned file, `mbpp.jsonl` is an uncleaned data file, the data volume will be larger, about 800 items. The cleaned `sanitized-mbpp.json` file has nearly 500 items of data.

`mbpp_val.jsonl` and `mbpp_train.jsonl` are both separated from `sanitized-mbpp.json`, with a ratio of 8:2, one for training and one for testing.

These two files are generated by code and do not need to be downloaded, but are generated after running the code, so don't be surprised if you don't see these two files at the beginning.

#### Run Code

After opening the notebook file, we run the code sequentially from top to bottom to complete this experiment reproduction.

Now please switch to `train_embedding_with_validation.ipynb` to start this experiment.

The fine-tuning task requires nearly 48GB of video memory to complete. If your graphics card does not have such large video memory, you can adjust the batch size to 8 or 16 and try again. By running the code, check the video memory usage to ensure that the video memory does not overflow.

In the ubuntu environment, you can use the following command to view video memory usage:

```bash
watch -n 2 nvidia-smi
```

If you are on a Windows system, this command is not available. It is recommended to use the wsl environment for operation.

When the batch size is set to 1, it is the minimum video memory requirement. After my test, it requires 12.5GB video memory. So if you want to try running this experiment, please prepare at least a 13GB graphics card. If you use a batch size of 80 for training, it takes about 5 minutes to complete this fine-tuning training. The final fine-tuning effect evaluation also takes about 1 minute, so the total time is about 6 minutes.

If you have reproduced the model fine-tuning task in the `train_embedding_with_validation.ipynb` file.

Then your next step can be to start evaluating the model we fine-tuned. We chose the T2Retrieval dataset on the C-MTEB leaderboard. You can use the notebook file `C-MTEB_retrieval_evaluation.ipynb` to reproduce this evaluation experiment.

This experiment compares the performance of the model on T2Retrieval before and after fine-tuning. The experiment takes a long time. It takes 6 hours to complete the evaluation in my environment with an A6000 48GB graphics card, which is quite long. It is recommended to use a graphics card with at least 24GB for this C-MTEB evaluation.

#### Evaluation Experiment Results

In the final visualization part, we can see that the first figure is the training configuration of the model, as shown below.

![Model training configuration](images/01-9.png)

This model training used the following configuration parameters:

- **Model Architecture**: Based on BGE-M3 multilingual embedding model, supporting 1024-dimensional vector output and 8192-length sequence processing
- **Training Epochs**: 10 epochs, evaluation and checkpoint saving are performed after each epoch
- **Batch Size**: 80, combined with gradient accumulation steps of 2, achieving equivalent 160 batch size training
- **Learning Rate**: 2e-4, using cosine annealing scheduling strategy, first 10% steps used for warm-up
- **Sequence Length**: Query and document are both set to 512 tokens, balancing computational efficiency and semantic understanding capability
- **Optimizer**: AdamW optimizer, weight decay 0.01, using gradient clipping to prevent gradient explosion
- **Mixed Precision Training**: Enable FP16 mixed precision to reduce video memory usage and accelerate training
- **Temperature Parameter**: 0.02, used to adjust the sharpness of similarity distribution
- **Data Processing**: 2 worker processes loading data in parallel to improve data preprocessing efficiency

The MBPP dataset was used during the training process, divided into training set (341 samples) and validation set (86 samples) at a ratio of 8:2. The In-batch Negatives strategy was used for contrastive learning. Recall@1 and MRR metrics were evaluated on the validation set after each epoch, and the best performing model checkpoint was saved.

The second figure shows the loss function of model training and the comparison of model test effects before and after fine-tuning, as shown below.

![Loss function of model training and comparison of model test effects before and after fine-tuning](images/01-10.png)

The figure consists of three parts: The step losses on the left show that the loss per step during training dropped rapidly from about 0.8 to close to 0, or around 0.1, accompanied by some fluctuations and occasional sudden increases (indicating per-step noise/mini-batch differences or gradient accumulation effects).

The epoch metrics in the middle show that the training converged very stably at the epoch level: epoch_loss dropped from about 0.77 to close to 0.03, epoch_acc rose from about 0.80 to close to 0.99, meaning that on our training set, the accuracy of the model is close to 100%, indicating that the model's performance on the training set has reached a very good level. However, it is normal for the model's performance on the training set to reach close to 100%, so we also have a test set, which is the 20% data separated from the dataset at the beginning. This part of the data was not added to the training. The test performance has also improved, proving that our model training has improved at least in code retrieval capabilities.

The Evaluation comparison on the right is the test on the validation set (in the notebook, the fine-tuned model will be used as the baseline, and the un-fine-tuned bge_m3 as a comparison).

The values in the figure show that after fine-tuning (baseline), Recall@1=0.837, MRR=0.893, while the un-fine-tuned bge_m3 has Recall@1=0.779, MRR=0.838, indicating that fine-tuning brought about +0.058 Recall@1 and about +0.055 MRR improvement on this validation set. If you want to achieve better results, I think you can improve from the data part. For example, we only used nearly 500 items of cleaned data, but there are more than 800 items of uncleaned data pairs available for us to use.

From the visualization results, the training process converged well, and after fine-tuning, the model improved the retrieval quality in code retrieval scenarios. Interested friends can try to use uncleaned data for model training, or add strategies like dropout to retrain, and compare their training effects.

#### Evaluation Results on C-MTEB Before and After Fine-tuning

![Evaluation Results on C-MTEB Before and After Fine-tuning](images/01-11.png)

The overall metrics of the fine-tuned model on C-MTEB (such as ndcg, map, recall, precision, mrr at @1/@3/@5/@10, etc.) generally decreased compared to the original BGE-M3.

This is an expected transfer effect, because when we trained, first, the amount of data was insufficient, second, the parameter adjustment was not the optimal parameter, and third, there are still many parts of our code that can be improved, such as adding the queue mechanism adjustment of Moco mentioned earlier, etc.

We intentionally shifted the model capability from general semantic retrieval to code retrieval, or we wanted the model to perform capability transfer in specific scenarios, so it is normal for general capabilities to decline. It would be outrageous if they didn't decline.

During fine-tuning, contrastive learning objectives with code-based positive/negative samples and In-batch Negatives were used. We can also add strategies similar to SimCLR, etc.

Correspondingly, in code retrieval related validation sets and MBPP experiments, fine-tuning brought improvements on the code retrieval test set. For example, on our 20% test set in 8:2, verification Recall@1 increased from 0.779 to 0.837, and MRR increased from 0.838 to 0.893, indicating that the retrieval quality of the model in code retrieval scenarios has been well improved, and the accuracy rate on the training set can reach 99%. It can be confirmed that the recall rate after training will be greatly improved.

This embedding model fine-tuning experiment ends here~

If there is a chance later, I will bring you fine-tuning experiments closer to actual scenarios!
