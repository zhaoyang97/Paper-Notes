# Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders

**会议**: ACL 2025  
**arXiv**: [2505.05111](https://arxiv.org/abs/2505.05111)  
**代码**: [https://github.com/Aatrox103/multilingual-llm-features](https://github.com/Aatrox103/multilingual-llm-features)  
**领域**: LLM / NLP  
**关键词**: Sparse Autoencoders, Multilingual LLM, Language-Specific Features, Mechanistic Interpretability, Steering Vectors  

## 一句话总结

利用 Sparse Autoencoders (SAEs) 分析多语言 LLM 的内部表示，发现存在强烈的语言特定 SAE features，这些 features 不仅与语言特有 token 相关还与语言上下文相关，消融它们只影响对应语言能力，且多个语言 features 之间存在协同效应；进一步利用这些 features 增强 steering vectors 实现对生成语言的精确控制。

## 研究背景与动机

### 多语言 LLM 的机制理解

随着 Gemini 1.5、Qwen2、LLaMA 3 等模型强调多语言能力，理解 LLM 如何在内部处理不同语言的机制变得至关重要。

### 现有分析方法的局限性

1. **Neuron-based 方法**（识别语言特定神经元）：受 "superposition" 问题影响——单个神经元可能编码多个不相关概念，导致分析不可靠
2. **Internal-activation-based 方法**（使用最终层 unembedding matrix 获取中间层 token 分布）：除最后几层外误差很大，因为不同层的激活分布差异较大

### SAE 的优势

Sparse Autoencoders 将 LLM 激活分解为稀疏线性组合的 SAE features，具有三个优势：
- 可应用于单个 token，比 neuron-based 方法更单义
- 每层独立训练，比 activation-based 方法在跨层分析时更可靠
- 多语言平行数据天然适合识别单语 features

## 方法详解

### 整体框架

研究分为四个层次递进的部分：
1. **发现语言特定 features**：提出单语性度量指标
2. **分析 code-switching**：证明 features 与语言上下文（而非仅 token）相关
3. **消融实验**：验证 features 对语言能力的因果影响
4. **Steering vectors 增强**：用 features 作为 gating 信号实现语言控制

### 关键设计

#### 1. 单语性度量指标 ν

给定 K 种语言的残差流集合 $\mathcal{D} = \{\mathcal{D}_1, ..., \mathcal{D}_K\}$，feature s 对语言 L 的单语性：

$$\nu_s^L = \mu_s^L - \gamma_s^L$$

其中 $\mu_s^L$ 是 feature s 在语言 L 上的平均激活，$\gamma_s^L$ 是其在其他语言上的平均激活。$\nu$ 越大表示该 feature 越强烈地与特定语言相关。

**发现**：
- top-4 features 的 ν 值显著高于随机 feature（随机 feature 接近零）
- 大多数语言中，rank #1 feature 的 ν 值明显高于其他 features
- 某些语言的 rank #2 feature 也有较大的 ν 值

#### 2. Code-Switching 实验设计

使用 GPT-4o 生成各语言句子（以名词结尾），然后将名词替换为其他语言的等价词。例如：

- 原始（西班牙语前缀 + 西班牙语名词）
- Code-switch（西班牙语前缀 + 法语名词）
- 独立名词（无前缀）

计算在有/无前缀情况下结尾名词的语言特定 feature 激活值。

**实验结果**：
- 西班牙语前缀增强了非西班牙语名词的西班牙语 feature 激活
- 深层比浅层增强更显著
- 法语名词（同语系）比韩语名词（不同语系）增强更多
- 西班牙语前缀降低了非西班牙语名词原始语言的 feature 激活

#### 3. Directional Ablation

通过投影消除 feature 方向来"零化"语言特定 features：

$$x' \leftarrow x - \hat{d}\hat{d}^\intercal x$$

- 消融后用修改的残差流继续前向传播
- 测量不同语言文本的 CE loss 变化

#### 4. 协同效应分析

对比单独消融 top-1 和 top-2 法语 features 与同时消融的效果：
- 法语文本：同时消融的 CE loss 增加 > 单独效果之和（协同效应）
- 西班牙语/日语文本：同时消融 ≈ 单独效果之和（无协同效应）

#### 5. 增强 Steering Vectors

传统 steering vector 通过正/负 prompt 集的均值激活差计算：

$$v = \frac{1}{|\mathcal{X}_+|}\sum_{x \in \mathcal{X}_+} a_L(x) - \frac{1}{|\mathcal{X}_-|}\sum_{x \in \mathcal{X}_-} a_L(x)$$

**改进**：用语言特定 features 作为 gating 信号控制 steering vectors，实现更精确的语言切换控制。

### 损失函数 / 训练策略

本文属于分析性工作，不涉及模型训练。SAE 使用现有的预训练版本：
- Gemma Scope 用于 Gemma 2 2B/9B
- Llama Scope 用于 Llama-3.1-8B
- 分析数据：Flores-10（从 Flores-200 中提取的 10 种语言子集）

## 实验关键数据

### 主实验

**Adversarial Language Identification 任务**（Gemma 2 2B）：

| 方法 | Es 成功率/其他CE | Fr | Ja | Ko | Zh |
|------|---------|---|---|---|---|
| SV L1 | 92.1/4.7 | 92.6/4.5 | 86.1/5.4 | 95.2/5.3 | 84.7/5.2 |
| **SAE L3** | **95.8/4.2** | **96.7/4.2** | **89.2/4.0** | **95.4/4.4** | 71.9/4.3 |

**Gemma 2 9B 的 Cross-Lingual Continuation 任务**：

| 方法 | Es 成功率/其他CE | Fr | Vi | Ko |
|------|---------|---|---|---|
| SV L1 | 82.2/4.1 | 85.3/4.0 | 83.6/4.1 | 93.0/4.6 |
| **SAE L3** | **96.2/3.4** | **94.6/2.9** | **95.3/2.8** | 93.6/4.3 |

### 关键发现

1. **语言特定 features 确实存在**：在 Gemma 2 2B/9B 和 Llama 3.1 8B 上一致观察到
2. **features 不仅是 token 级别的**：code-switching 实验证明它们编码了语言上下文信息
3. **消融影响是语言特定的**：消融法语 features 主要增加法语文本的 CE loss，对其他语言几乎无影响
4. **协同效应仅在目标语言内存在**：多个法语 features 同时消融对法语的影响 > 单独效果之和
5. **语言相似性有影响**：法语 rank #2 feature 在某些层也是西班牙语 top-2 features，解释了消融法语 features 对西班牙语有一定影响
6. **SAE 增强的 steering vectors 更优**：在成功率/对其他语言影响的平衡上优于普通 steering vectors
7. **中文控制较难**：SAE 方法在中文上的 steering 效果不如其他语言，可能因为中文的特征更分散

## 亮点与洞察

1. **机制可解释性的新工具**：首次系统地用 SAE 分析 LLM 的多语言机制，比 neuron-based 和 activation-based 方法更可靠
2. **对"superposition"问题的有效回应**：SAE 将多语义神经元分解为单义 features，绕过了 superposition 问题
3. **从分析到应用的完整链条**：不仅分析了 features 的存在和性质，还展示了实际应用（增强 steering vectors）
4. **Code-switching 实验设计精巧**：通过控制变量（前缀语言 × 名词语言）清晰展示了上下文依赖性
5. **协同效应的发现**：揭示了同一语言的多个 features 之间的非线性互作，对理解 LLM 内部表示结构有重要意义

## 局限性 / 可改进方向

1. **主要关注非英语语言**：英语作为主要训练语言有不同特性，但本文未深入分析英语的 features
2. **SAE 自身的局限性**：SAE 的稀疏性假设可能不完全成立，重建误差可能忽略重要信息
3. **语言集合有限**：仅分析 10 种语言，缺少非洲、大洋洲等资源极低语言
4. **因果关系需更多验证**：消融实验显示相关性，但模型可能有冗余编码路径
5. **Steering vectors 的实际应用场景有限**：语言控制的需求相比内容安全控制更小众
6. **未分析训练数据的影响**：语言特定 features 的强度是否与训练数据中该语言的占比正相关？

## 相关工作与启发

- **Mechanistic Interpretability 方向**：继 Bricken et al. (2023) 和 Cunningham et al. (2023) 的 SAE 用于 LLM 可解释性后，将其推广到多语言维度
- **Steering Vectors (Turner et al., 2024)**：本文展示了 SAE features 可作为 gating 信号改善传统 steering vectors
- **语言中立性假说的挑战**：此前一些研究认为 LLM 在中间层使用"语言中立"表示（Wendler et al., 2024），本文发现语言特定 features 在所有层都存在，暗示该假说需修正
- **Directional Ablation (Arditi et al., 2024)**：消融特定方向来去除 LLM 能力的方法在安全领域已有应用（去除 refusal）
- **启发**：SAE features 可能为其他维度的可解释性（如领域特定、风格特定、情感特定）提供类似的分析框架

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-------------|------|
| 创新性 | 8 | SAE 分析多语言机制的视角新颖，gating 改进 steering vectors 有巧思 |
| 实验充分性 | 9 | 多模型、多语言、code-switching、消融、协同效应、应用全覆盖 |
| 写作质量 | 8 | 层次递进清晰，图表直观 |
| 实用价值 | 7 | 分析性贡献为主，steering vectors 应用场景偏窄 |
| 总分 | 8 | 高质量的机制分析工作，对理解多语言 LLM 有重要意义 |
