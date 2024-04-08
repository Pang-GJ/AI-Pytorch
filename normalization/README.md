# 归一化

本文将探讨 BatchNorm、LayerNorm、RMSNorm 等归一化方法的原理与实现，借助 kimi chat 翻译自[Batch Normalization, Layer Normalization and Root Mean Square Layer Normalization: A Comprehensive Guide with Python Implementations](https://afterhoursresearch.hashnode.dev/batch-normalization-layer-normalization-and-root-mean-square-layer-normalization-a-comprehensive-guide-with-python-implementations)。

稳定和加速神经网络的训练常常依赖于所采用的归一化技术。虽然归一化背后的理论看起来简单直接，但其实际应用却有多种不同的表现形态，每种都有其独特的优点和缺点。

本文将探讨三种流行的归一化方法：

- 批量归一化（BatchNorm）
- 层归一化（LayerNorm）
- 平方根均值层归一化（RMSNorm）

我们将涵盖：

- 每种技术背后的数学原理
- 计算复杂度的讨论
- 每种方法的优缺点

## 批量归一化（BatchNorm）

### 概述

批量归一化由 Sergey Ioffe 和 Christian Szegedy[1]提出，旨在训练期间对给定小批量中每一层的输出在每个特征维度上进行归一化。简单来说，它使用在小批量中所有实例计算得出的统计数据（均值和方差）。

### 公式

输出 $\hat{x}$ 的计算公式为：

$$\hat{x} = \frac{x - \mathbb{E}_{\text{mini-batch}}(x)}{\sqrt{Var_{\text{mini-batch}}(x) + \epsilon}} \cdot \gamma + \beta$$

这里，$\mathbb{E}_{\text{mini-batch}}(x)$ 和 $Var_{\text{mini-batch}}(x)$ 是在小批量上按特征计算的均值和方差，$\epsilon$ 是一个小的常数，用于数值稳定性。$\gamma$ 和 $\beta$ 分别是可学习的缩放和偏移参数。

### 运行统计

批量归一化还需要计算和存储均值和方差的运行统计。在训练期间，这些统计数据以指数移动平均（EMA）的形式计算，使用一个标量动量项 $\alpha$ 更新，即 $y*{EMA_i} = \alpha y*{EMA\_{i-1}} + (1 - \alpha)y_i$，其中 $i$ 是当前训练步骤。在推理期间，使用存储的运行统计数据来归一化单个样本。

### 属性

与不进行归一化相比，批量归一化：

- 减少了内部协变量偏移（即减少了层输入分布的变化）
- 加速了收敛
- 使得学习率可以更高
- 对初始化不那么敏感

### Python 实现

```python
class BatchNorm(nn.Module):
    def __init__(
        self,
        size: int,
        eps: float = 1e-5,
    ):
        """
        批量归一化。
        假设输入 x 的形状为 (batch, seq_len, d_model)。

        参数:
            size: 特征维度的形状（即 d_model）
            eps: 为了数值稳定性。默认值为 1e-5。
        """
        super(BatchNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(size), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(size), requires_grad=True)

    def forward(self, x):
        x_var, x_mean = torch.var_mean(x, dim=[0,1], keepdim=True, correction=0)
        x_std = torch.sqrt(x_var + self.eps)

        x_norm = (x - x_mean) / x_std

        return self.gamma.unsqueeze(0).unsqueeze(1) * x_norm + self.beta.unsqueeze(0).unsqueeze(1)
```

假设我们的输入 $x$ 的形状为 (batch, seq_len, d_model)，在批量归一化中，我们在批处理和序列长度维度（分别为 0 和 1）上进行归一化，但保持特征维度（d_model）不变。这是因为 BatchNorm 旨在稳定小批量中每个特征的分布。

## 层归一化（LayerNorm）

### 概述

与批量归一化不同，层归一化对批量中每个单独的数据点的特征进行归一化，使其对批量大小的变化不太敏感。

### 公式

输出 $\hat{x}$ 的计算方式与批量归一化类似，但在计算 $\mathbb{E}(x)$ 和 $Var(x)$ 的轴上有所不同。

$$\hat{x} = \frac{x - \mathbb{E}_{\text{features}}(x)}{\sqrt{Var_{\text{feature}}(x) + \epsilon}} \cdot \gamma + \beta$$

这里，$\mathbb{E}_{\text{features}}(x)$ 和 $Var_{\text{features}}(x)$ 是在特征维度上计算的均值和方差。

### 属性

比批量归一化对批量大小的敏感度低

在序列模型中表现良好

稳定训练

加速收敛

### Python 实现

```python
class LayerNorm(nn.Module):
    def __init__(
        self,
        size: int,
        eps: float = 1e-5,
    ):
        """
        层归一化。
        假设输入 x 的形状为 (batch, seq_len, d_model)。

        参数:
            size: 特征维度的形状（即 d_model）
            eps: 为了数值稳定性。默认值为 1e-5。
        """
        super(Layernorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(size), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(size), requires_grad=True)

    def forward(self, x):
        x_var, x_mean = torch.var_mean(x, dim=-1, keepdim=True, correction=0)
        x_std = torch.sqrt(x_var + self.eps)

        x_norm = (x - x_mean) / x_std

        return self.gamma.unsqueeze(0).unsqueeze(1) * x_norm + self.beta.unsqueeze(0).unsqueeze(1)
```

假设我们的输入 $x$ 的形状为 (batch, seq_len, d_model)，层归一化在批处理中每个序列的特征维度（d_model）上进行归一化。这样做的目的是使单个数据点的所有特征归一化到零均值和单位方差，从而使模型对输入特征的规模不那么敏感。

## 平方根均值层归一化（RMSNorm）

### 概述

RMSNorm[3] 是 LayerNorm 的一个变体，它使用均方根 $\mathbb{E}(x^2)$ 而不是标准差进行重新缩放，并且不使用重新居中操作。作者假设 LayerNorm 中的重新居中不变性质是可有可无的，只保留了重新缩放不变性质。

### 公式

输出 $\hat{x}$ 的计算公式为：

$$\hat{x} = \frac{x}{ \sqrt{\mathbb{E}\_{\text{feature}}(x^2) + \epsilon}} \cdot \gamma$$

### 属性

计算上比 LayerNorm 更简单，因此更高效

### Python 实现

```python
class RMSNorm(nn.Module):
    def __init__(
        self,
        size: int,
        eps: float = 1e-5,
    ):
        """
        平方根均值层归一化。
        假设输入 x 的形状为 (batch, seq_len, d_model)。

        参数:
            size: 特征维度的形状（即 d_model）
            eps: 为了数值稳定性。默认值为 1e-5。
        """
        super(RMSnorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(size), requires_grad=True)

    def forward(self, x):
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps) # 也可以使用弗罗贝尼乌斯范数来计算 rms
        x_norm = x / rms

        return self.gamma.unsqueeze(0).unsqueeze(1) * x_norm
```

假设我们的输入 $x$ 的形状为 (batch, seq_len, d_model)，对于 RMS 层归一化，与 LN 一样，我们在特征维度（d_model）上进行归一化。我们使用序列中每个数据点的特征值的均方根进行归一化。这种方法计算效率高，并且对异常值更加稳健。

## 计算复杂度和内存要求

批量归一化：需要存储运行统计数据，这使得并行化更加困难。

层归一化：计算上不那么密集，因为不需要运行统计数据。

RMSNorm：由于没有重新居中操作，计算上甚至比 LayerNorm 更不密集。

## 使用场景和建议

批量归一化：这种技术在卷积架构中特别强大，因为批量大小通常足够大，以便对均值和方差的估计可靠。然而，对于像 RNN 和 Transformer 这样的模型，批量归一化并不理想，因为序列长度可能会变化。它依赖于运行统计数据，这也给在线学习场景带来挑战，并在尝试跨多个设备并行化模型时增加了复杂性。

层归一化：对于像 RNN 和 Transformer 这样的序列模型非常有效。它也是小批量大小情况下的理想选择，因为它独立计算每个数据点的统计信息，消除了估计总体统计信息所需的大批量。

RMSNorm：如果计算效率是你的首要考虑因素，RMSNorm 提供了一个比 LayerNorm 更简单的方程，计算强度更低。在几个 NLP 任务上的实验表明，RMSNorm 在质量上与 LayerNorm 相当，但提高了运行速度。

## 结论

选择合适的归一化技术对你的深度学习模型至关重要。本文旨在提供批量归一化、层归一化和平方根均值层归一化的理论及实践概览。提供的 Python 实现应该能帮助你更深入地理解这些技术的细节。

通过这些归一化技术的应用，我们可以更好地训练和优化深度学习模型，从而在各种任务中取得更好的性能。理解每种归一化方法的原理和实现细节，可以帮助我们根据具体的应用场景和需求，选择最合适的归一化策略。

## References

[1] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, Sergey Ioffe and Christian Szegedy, 2015

[2] Layer normalization, Jimmy Lei Ba et al., 2016

[3] Root Mean Square Layer Normalization, Biao Zhang and Rico Sennrich, 2019
