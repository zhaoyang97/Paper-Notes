# Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics

**会议**: CVPR2025  
**arXiv**: [2603.13085](https://arxiv.org/abs/2603.13085)  
**代码**: 待确认  
**领域**: 深度学习理论  
**关键词**: Neural Tangent Kernel, 注意力机制, 核方法, influence function, 对抗鲁棒性

## 一句话总结

通过 NTK 框架揭示线性化注意力机制不会收敛到无穷宽 NTK 极限（谱放大效应使 Gram 矩阵条件数立方化，需宽度 $m = \Omega(\kappa^6)$），并引入「影响可塑性」概念量化这一非收敛的双面后果：注意力比 ReLU 网络高 6-9 倍的可塑性既增强了任务适配能力，也加剧了对抗脆弱性。

## 研究背景与动机

1. **NTK 理论的基本预测**：Neural Tangent Kernel 理论预测足够宽的网络在训练中核保持近似不变（lazy training），可用核方法精确分析学习动态
2. **注意力机制缺乏理论刻画**：注意力的非线性动态使其大部分游离于 NTK 理论框架之外，学习过程中的灵活性缺乏严格理论表征
3. **惊人的经验发现**：标准 ReLU 网络随宽度增加 NTK 距离单调递减（符合预期），但带注意力的网络 NTK 距离不降反升或非单调，表明注意力从未进入核 regime
4. **核心问题**：注意力为什么不收敛到 NTK 极限？这种非收敛对训练数据依赖性意味着什么？
5. **切入角度**：设计参数无关的线性化注意力 $f^{\text{att}}(\mathbf{X}) = \mathbf{X}\mathbf{X}^T\mathbf{X}$，建立与数据依赖 Gram 诱导核的精确对应，从而实现严格的理论分析

## 方法详解

### 核心理论框架

1. **线性化注意力定义**：$f^{\text{att}}(\mathbf{X}) = \mathbf{X}\mathbf{X}^T\mathbf{X}$，对应 identity QKV 投影 + 线性化 softmax（$\exp(A_{ij}) \approx 1 + A_{ij}$）的标准注意力，保留了注意力核心的二次交互结构，相当于未归一化的 Nadaraya-Watson 估计器
2. **MLP-Attn 架构**：线性化注意力预处理（输出做 $\ell_2$ 归一化）→ 两层 ReLU MLP（$f = \frac{1}{\sqrt{m}} \sum_r a_r \sigma(\mathbf{w}_r^T \tilde{\mathbf{x}})$）。注意力层参数无关，仅 MLP 权重 $\mathbf{w}_r$ 参与训练，$a_r \in \{-1, +1\}$ 固定

### 关键定理

1. **Theorem 4.1（数据依赖 Gram 诱导核）**：线性化注意力诱导核为 $K_{\text{LinAttn}}(\mathbf{x}_i, \mathbf{x}_j) = \sum_{k,\ell} (\mathbf{x}_i^T \mathbf{x}_k)(\mathbf{x}_k^T \mathbf{x}_\ell)(\mathbf{x}_\ell^T \mathbf{x}_j)$，矩阵形式 $\mathbf{K} = \mathbf{G}^3$（$\mathbf{G} = \mathbf{X}\mathbf{X}^T$），呈现传递相似性链 $i \to k \to \ell \to j$：影响从 $\mathbf{x}_i$ 经中间点传播到 $\mathbf{x}_j$
2. **Theorem 4.2（序列架构 NTK）**：MLP-Attn 无穷宽极限 NTK 为 $K_{\text{seq}}(\mathbf{x}, \mathbf{x}') = \mathbb{E}_{\mathbf{w}}[\sigma'(\mathbf{w}^T \tilde{\mathbf{x}}) \sigma'(\mathbf{w}^T \tilde{\mathbf{x}}')] \cdot \langle \tilde{\mathbf{x}}, \tilde{\mathbf{x}}' \rangle$，由于 $f^{\text{att}}$ 参数无关，仅 $\mathbf{w}_r$ 梯度贡献
3. **Theorem 4.7（谱放大与 NTK 非收敛）**：注意力变换使条件数立方化 $\kappa(\tilde{\mathbf{G}}) = \kappa(\mathbf{G})^3$，NTK 收敛所需宽度为 $m = \Omega(\kappa(\mathbf{G})^6/\epsilon^2)$。MNIST 上 $\kappa(\mathbf{G}) \approx 1.2 \times 10^3$ → 需 $m \gg 10^{18}$；CIFAR-10 上 $\kappa(\mathbf{G}) \approx 8.7 \times 10^3$ → 需 $m \gg 10^{24}$，远超实际宽度
4. **Proposition 4.5（数据依赖核灵敏度）**：注意力核灵敏度依赖于整个数据集的相关结构 $|K_{\text{LinAttn}}(\mathbf{x}_i + \delta, \mathbf{x}_j) - K_{\text{LinAttn}}(\mathbf{x}_i, \mathbf{x}_j)| \leq \|\mathbf{G}\mathbf{x}_j\|_1 \cdot \epsilon$，$\|\mathbf{G}\mathbf{x}_j\|_1$ 随数据集规模和相关密度增长；对比多项式核的 $O(\epsilon)$ 灵敏度与数据无关

### 影响可塑性（Influence Malleability）

- **影响函数**：基于 NTK 的 leave-one-out 公式 $I(\mathbf{x}_i, \mathbf{x}_{\text{test}})$，通过经验有限宽核矩阵 $(\mathbf{K}_m + \lambda \mathbf{I})^{-1}$ 高效计算，无需重训练
- **翻转率定义**：选取 top-$\tau$（$\tau=0.1$）高影响力训练样本，施加 PGD 对抗扰动（$\epsilon=0.3$），计算影响力符号翻转的比例
- **互补度量**：原始与扰动后影响力排名的 Spearman 秩相关系数 $\rho$，越低表示可塑性越高
- **三种干预策略**：Curated（移除高影响样本）、Transformed（替换为对抗版本）、Adversarial（全数据 PGD 扰动）

## 实验关键数据

### NTK 非收敛验证

| 模型 | 数据集 | $m=16$ | $m=1024$ | $m=4096$ | 趋势 |
|------|--------|--------|----------|----------|------|
| 2L-ReLU | MNIST | 45.1 | 39.9 | 39.2 | ↓ 收敛 |
| MLP-Attn | MNIST | 10.3 | 33.3 | 43.4 | ↑ 非单调后发散 |
| 2L-ReLU | CIFAR-10 | 246.2 | 101.7 | 56.9 | ↓ 收敛 |
| MLP-Attn | CIFAR-10 | 3.7 | 10.4 | 12.6 | ↑ 单调发散 |

MNIST 非单调 vs CIFAR-10 单调递增反映了 Gram 矩阵结构差异：MNIST 较低的 $\kappa(\mathbf{G})$ 允许小宽度下出现暂态近 lazy regime，而 CIFAR-10 从初始化起就处于特征学习 regime。

### 影响可塑性（10 类分类，$\epsilon=0.3$）

| 数据集 | 模型 | FGSM | PGD | MIM |
|--------|------|------|-----|-----|
| MNIST | 2L-ReLU | 4.1% | 3.3% | 3.4% |
| MNIST | MLP-Attn | **34.6%** | **28.9%** | **21.9%** |
| CIFAR-10 | 2L-ReLU | 3.3% | 3.1% | 3.2% |
| CIFAR-10 | MLP-Attn | **26.4%** | **19.1%** | **20.5%** |

MLP-Attn 翻转率比 ReLU 高 6-9 倍。FGSM 产生最高翻转率，PGD 在 MNIST 上产生最大比值（8.8×）。

### 二分类场景（$\epsilon=0.3$）

| 数据集 | 模型 | FGSM | PGD | MIM |
|--------|------|------|-----|-----|
| MNIST (3 vs 8) | 2L-ReLU | 8.4% | 8.4% | 8.6% |
| MNIST (3 vs 8) | MLP-Attn | **25.9%** | **41.0%** | **40.5%** |
| CIFAR-10 (cars vs planes) | 2L-ReLU | 15.2% | 15.5% | 15.3% |
| CIFAR-10 (cars vs planes) | MLP-Attn | 14.3% | 14.0% | 14.8% |

MNIST 二分类中注意力优势 3-5×；CIFAR-10 二分类中优势消失（≈1×），与 Theorem 4.7 一致——二分类 CIFAR-10 的 $\kappa(\mathbf{G})$ 较低，立方条件数放大效应减弱。

### 对抗训练分析

| 数据集 | 模型 | 标准训练 | 对抗训练 |
|--------|------|----------|----------|
| MNIST | 2L-ReLU | 3.3% | 43.4% |
| MNIST | MLP-Attn | 28.9% | 42.2% |
| CIFAR-10 | 2L-ReLU | 3.1% | 36.5% |
| CIFAR-10 | MLP-Attn | 19.1% | 38.6% |

对抗训练大幅提升 ReLU 可塑性（3.3% → 43.4%），但 MLP-Attn 在标准训练下就天然具有高可塑性（28.9%）。两种不同机制产生可塑性：(1) 架构性——注意力的 Gram 诱导核天然创造灵敏度；(2) 训练诱导性——对抗增强迫使特征重学习。注意力的灵敏度是内禀的而非外部施加的。

## 亮点

1. **理论优雅**：从线性化注意力出发建立精确核对应（$\mathbf{K} = \mathbf{G}^3$），再通过谱放大解释非收敛，因果链完整
2. **"影响可塑性"概念新颖**：将 NTK 非收敛的理论发现转化为可度量的实际指标——训练数据依赖性的动态变化
3. **双面性洞察深刻**：同一个数据依赖核机制既是注意力的力量之源（数据依赖核在目标与数据分布对齐时降低逼近误差）也是脆弱性之源（高可塑性使对抗操纵更容易）
4. **实验与理论精确一致**：经验条件数 $\kappa(\mathbf{G}) \approx 10^3$ 精确预测了观测到的非收敛行为；二分类场景中 $\kappa(\mathbf{G})$ 降低时优势消失也验证了理论
5. **可推广性分析**：堆叠 $k$ 层线性化自注意力产生 $\mathbf{G}^{2k+1}$，条件数 $\kappa(\mathbf{G})^{2k+1}$，进一步加剧非收敛；截断注意力（保留 top-$r$ 奇异分量）可恢复收敛

## 局限性

- 仅分析线性化注意力（identity QKV + 线性化 softmax），与实际 softmax 注意力存在差距；softmax 的逐行归一化可能进一步放大非收敛
- 实验限于 MNIST/CIFAR-10 和两层网络（$m \leq 4096$），未扩展到更大规模架构或更复杂数据
- Theorem 4.7 仅给出初始化时 NTK 偏差的下界，未直接预测训练后轨迹——宽度增大导致 NTK 距离上升是互补效应（更大网络有更多特征学习容量）
- 影响可塑性度量依赖扰动预算 $\epsilon$ 和选择阈值 $\tau$ 的具体取值，虽然补充实验确认了不同 $\epsilon \in \{0.1, 0.2, 0.3, 0.5\}$ 下排序一致

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次揭示注意力 NTK 非收敛及其影响可塑性后果
- 实验充分度: ⭐⭐⭐ 理论验证充分但实验规模受限
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰严谨，实验与理论紧密呼应
- 价值: ⭐⭐⭐⭐ 为理解注意力机制的根本特性提供新视角
