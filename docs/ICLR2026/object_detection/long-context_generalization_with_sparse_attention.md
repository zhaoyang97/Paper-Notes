# Long-Context Generalization with Sparse Attention

**会议**: ICLR 2026
**arXiv**: [2506.16640](https://arxiv.org/abs/2506.16640)
**代码**: [deep-spin/asentmax](https://github.com/deep-spin/asentmax)
**领域**: LLM效率 / 注意力机制
**关键词**: 稀疏注意力, 长上下文泛化, α-entmax, 长度外推, Transformer

## 一句话总结

提出 ASEntmax（Adaptive-Scalable Entmax），用可学习温度的 α-entmax 替代 softmax 注意力，从理论和实验两方面证明稀疏注意力能实现 1000× 长度外推，解决 softmax 在长上下文下的注意力弥散（dispersion）问题。

## 研究背景与动机

1. **Softmax 的注意力弥散问题**：随着上下文长度 $n$ 增大，softmax 将概率质量分散到所有 token 上，导致相关 token 的注意力权重趋近于零。理论上，当 $n \to \infty$ 时，softmax 的归一化熵趋近 1（完全均匀分布），即 **complete dispersion**。
2. **长度外推失败的根本原因**：模型在短序列上训练时学到的注意力模式无法迁移到长序列——softmax 在长序列中的权重分布与短序列截然不同，导致检索和推理能力崩溃。
3. **已有长上下文方案的局限**：RoPE 外推、ALiBi 等位置编码方法只处理位置信息，不解决注意力分布本身的弥散问题；Scalable Softmax (SSMax) 通过缩放因子缓解但缺乏理论保证。
4. **稀疏注意力的理论优势**：α-entmax 等稀疏变换能将不相关 token 的注意力精确置零，天然避免弥散。但此前缺乏严格的理论分析来解释为何稀疏注意力有助于长度外推。
5. **三大理论性质的缺失**：需要形式化证明稀疏注意力在以下方面优于 softmax：(1) 非消失注意力；(2) 集中度韧性（concentration resilience）；(3) 表征保持（representational preservation）。
6. **自适应稀疏度的需求**：不同注意力头在不同层可能需要不同程度的稀疏性，固定 α 过于僵硬，需要可学习的自适应机制。

## 方法详解

### 整体框架

ASEntmax 在标准 Transformer 的注意力机制中，将 softmax 替换为带有可学习温度 $\theta$ 的 α-entmax。具体而言，注意力权重计算从 $\text{softmax}(QK^T/\sqrt{d})$ 变为 $\alpha\text{-entmax}(QK^T/(\sqrt{d} \cdot \theta))$，其中 $\alpha > 1$ 控制稀疏程度，$\theta$ 为每个注意力头独立学习的温度参数。

### 关键设计

**1. α-entmax 稀疏变换**

- 是 softmax 的推广：当 $\alpha = 1$ 退化为 softmax，$\alpha = 2$ 为 sparsemax
- 核心特性：输出中包含精确的零值，自动将不相关 token 的注意力置零
- 可微分，支持端到端训练

**2. 三大理论性质**

- **Non-vanishing Attention（非消失注意力）**：对于 $\alpha > 1$，向序列中添加不相关 token 不会减少相关 token 的注意力权重。形式化地，若新增 token 的 score 低于阈值，则现有 token 的注意力权重完全不变。Softmax 则无论新增 token 的相关性如何，都会减少所有现有 token 的权重。
- **Concentration Resilience（集中度韧性）**：α-entmax 的注意力熵上界为 $O(\log s)$（$s$ 为支撑集大小），而非 softmax 的 $O(\log n)$（$n$ 为序列长度）。这意味着即使序列长度增大 1000×，只要相关 token 数量 $s$ 不变，注意力集中度就保持不变。
- **Representational Preservation（表征保持）**：在 $L$ 层 Transformer 中，softmax 的梯度路径数为 $O(n^L)$，导致深层网络中表征坍缩；α-entmax 将其降为 $O(s^L)$，有效保持不同输入的可区分性。

**3. 可学习温度 θ（ASEntmax）**

- 每个注意力头学习独立的温度参数 $\theta$
- $\theta$ 大 → 更稀疏（高温加剧稀疏化）；$\theta$ 小 → 更接近 dense（低温缓解稀疏化）
- 允许模型自适应地在稀疏和稠密注意力之间插值，不同头可选择不同策略

**4. Non-dispersion 性质**

- Softmax 完全弥散：归一化熵 $H(\text{softmax}(z))/\log n \to 1$（当 $n \to \infty$）
- α-entmax 保持集中：归一化熵有界，不随 $n$ 增长而趋近 1
- 这是长度外推能力的理论基石

### 损失函数/训练策略

- 使用标准语言模型训练目标（next-token prediction，交叉熵损失）
- 温度 $\theta$ 通过反向传播与模型参数联合优化
- $\alpha$ 通常固定为 1.5（实验中验证的最优值），也可设为可学习
- 在短序列（如长度 64）上训练，直接在长序列（如 65K）上测试

## 实验关键数据

### 主实验

Associative Recall 任务（训练长度 64）的长度外推准确率：

| 方法 | 64 | 256 | 1K | 4K | 16K | 65K |
|------|-----|------|------|------|------|------|
| Softmax | 99.8% | 52.1% | 12.3% | 3.1% | 0.8% | 0.2% |
| SSMax | 99.7% | 89.4% | 71.2% | 45.6% | 28.3% | 15.1% |
| Adaptive Temp | 99.6% | 91.2% | 78.5% | 52.3% | 34.7% | 21.4% |
| **ASEntmax** | **99.9%** | **99.5%** | **99.1%** | **98.2%** | **96.8%** | **95.3%** |

### 消融实验

α 和温度可学习性的影响（Associative Recall, 测试长度 16K）：

| 配置 | 准确率 | 说明 |
|------|--------|------|
| ASEntmax (α=1.5, θ 可学习) | **96.8%** | 最优配置 |
| α-entmax (α=1.5, 固定温度) | 88.4% | 缺乏自适应能力 |
| α-entmax (α=2.0, 固定温度) | 82.1% | 过度稀疏导致信息损失 |
| ASEntmax (α 可学习, θ 可学习) | 95.2% | α 学习不稳定，略有下降 |
| Softmax + Adaptive Temp | 34.7% | 温度无法解决 softmax 的根本弥散问题 |

### 关键发现

1. **1000× 外推**：训练长度 64 → 测试长度 65K，ASEntmax 保持 95.3% 准确率，softmax 降至 0.2%
2. **语言建模优势**：在长上下文 LM 评估中，ASEntmax 在 8× 训练长度时的困惑度趋势显著优于 softmax 和 SSMax
3. **检索能力保持**：在远超训练长度的 needle-in-a-haystack 测试中，ASEntmax 保持高检索成功率
4. **稀疏度自适应**：不同层和头学到了不同的温度值，验证了自适应机制的必要性——底层倾向更 dense，高层倾向更 sparse

## 亮点与洞察

- **理论深度扎实**：三大性质（non-vanishing, concentration resilience, representational preservation）的形式化证明是论文最大贡献，为稀疏注意力的长度外推优势提供了严格的数学基础
- **Dispersion 概念的提出**：将 softmax 的长上下文失败统一归因为"弥散"，并用归一化熵定量刻画，概念清晰且有说服力
- **$O(s^L)$ vs $O(n^L)$ 的洞察**：揭示了稀疏注意力在深层网络中的本质优势——梯度路径的组合爆炸被稀疏性有效抑制
- **简洁的实现**：仅替换 softmax 为 α-entmax + 可学习温度，无需额外架构修改，工程实现友好

## 局限性 / 可改进方向

1. **计算效率**：α-entmax 的前向/反向传播涉及排序操作，复杂度为 $O(n \log n)$，比 softmax 的 $O(n)$ 更高；尽管稀疏输出可加速后续计算，但注意力计算本身更慢
2. **预训练成本**：需要从头预训练或全量微调，不能简单作为 drop-in replacement 应用于已有预训练模型
3. **大规模验证不足**：实验主要在中等规模模型上进行，尚未在 7B+ 参数的大模型上验证
4. **与 FlashAttention 的兼容性**：稀疏注意力的不规则访存模式可能与 FlashAttention 等硬件优化方法冲突
5. **α 值的选择**：虽然实验表明 1.5 较优，但缺乏理论指导来确定最优 α

## 相关工作与启发

- **Scalable Softmax (SSMax)**：通过 $\log n$ 偏置项缩放 softmax logits，缓解弥散但不根治——本文的理论分析解释了为何 SSMax 效果有限
- **RoPE / ALiBi / YaRN**：位置编码层面的长度外推方法，与 ASEntmax 是正交的改进方向，可组合使用
- **Entmax (Peters et al., 2019)**：α-entmax 的原始工作，主要用于 NLP 分类和翻译任务，本文首次将其与长上下文外推联系起来
- **Sparse Transformer (Child et al., 2019)**：结构化稀疏注意力，与 α-entmax 的数据驱动稀疏不同
- **Gated Attention / Linear Attention**：替代 softmax 的其他方案，但缺乏 α-entmax 的理论保证

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 三大理论性质的形式化证明具有开创性，将稀疏注意力与长度外推建立了严格的数学联系
- **实验充分度**: ⭐⭐⭐⭐ — 合成任务和语言建模均有覆盖，1000× 外推结果令人印象深刻，但缺乏大规模模型验证
- **写作质量**: ⭐⭐⭐⭐⭐ — 理论推导清晰，概念层次分明，dispersion 的定义和可视化非常直观
- **价值**: ⭐⭐⭐⭐ — 为长上下文 LLM 提供了一个理论上有保证的新方向，但工程落地仍需解决效率和兼容性问题
