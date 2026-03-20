# A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders

**会议**: NeurIPS 2025  
**arXiv**: [2409.14507](https://arxiv.org/abs/2409.14507)  
**代码**: https://github.com/lasr-spelling/sae-spelling  
**领域**: AI Safety / 可解释性 / 机械可解释性  
**关键词**: Sparse Autoencoder, Feature Absorption, Feature Splitting, 机械可解释性, LLM内部表征

## 一句话总结
发现并系统研究了 SAE 中的"特征吸收"现象：看似单义的 SAE latent 会在特定 token 上不激活，其特征方向被更具体的子 latent "吸收"，这是层级特征+稀疏性损失的必然结果，对 SAE 用于可靠解释 LLM 构成根本挑战。

## 研究背景与动机
1. **领域现状**：Sparse Autoencoder (SAE) 是当前机械可解释性的核心工具，旨在将 LLM 的稠密激活分解为可解释的稀疏特征方向
2. **现有痛点**：
   - SAE latent 的精度/召回率不稳定，有些看似追踪某个概念的 latent 在应该激活时不激活
   - 仅看 max activating examples 容易产生虚假的可解释性幻觉
   - 之前的 feature splitting 研究只关注了"好的"分裂（如大小写分裂），忽略了问题分裂
3. **核心矛盾**：SAE 的稀疏性损失鼓励减少同时激活的 latent 数量，但当特征形成层级关系时（如"以S开头"是"short"的父特征），SAE 会通过将父特征方向吸收到子 latent 中来增加稀疏性，导致父 latent 不再在该 token 上激活
4. **本文要解决什么？** 识别、定义、量化并解释 SAE 中的特征吸收现象
5. **切入角度**：从拼写任务（首字母预测）入手，用 linear probe 作为 ground truth，系统对比 SAE latent 的精度/召回率
6. **核心idea一句话**：SAE 在层级特征上的稀疏优化必然导致特征吸收，使得 SAE latent 不是可靠的分类器

## 方法详解

### 整体框架
以"预测 token 首字母"为测试任务。输入：LLM 残差流激活。方法：(1) 训练 linear probe 作为 ground truth 分类器；(2) 找到与 probe 方向余弦相似度最高的 SAE latent 作为"首字母 latent"；(3) 对比两者的精度/召回率；(4) 通过消融实验验证吸收的因果效应。输出：特征吸收率的量化指标。

### 关键设计

1. **Toy Model 证明吸收的必然性**:
   - 做什么：在 4 个特征的简单设置中，当特征 1 只在特征 0 激活时才激活（层级关系），SAE 学到的 latent 0 会在特征 1 激活时不激活，同时 latent 1 的解码器吸收特征 0 方向
   - 核心思路：独立特征时 SAE 完美恢复；加入层级共现后，SAE 编码器学习 "¬feat1 ∧ feat0" 而非 "feat0"，因为这样只需激活 1 个 latent 而非 2 个，稀疏性更好
   - 设计动机：提供解析证明，在层级设置中吸收确实降低 SAE 损失

2. **首字母拼写实验设计**:
   - 做什么：用 ICL prompt 让模型预测 token 首字母，提取残差流激活
   - 核心思路：训练 26 个首字母的 linear probe，与 SAE latent 逐一对比。k-sparse probing 检测 feature splitting，消融实验检测 absorption
   - 设计动机：首字母是明确的二元特征（以/不以某字母开头），有清晰的 ground truth

3. **特征吸收率指标 (Feature Absorption Rate)**:
   - 做什么：量化吸收发生的频率
   - 核心思路：找到 k 个 feature split latent 都不激活但 probe 正确分类的 false negative token，对这些 token 做 integrated-gradients 消融。如果某个非首字母 latent 的消融效应最大且与 probe 余弦相似度 > 0.025，判定为吸收。absorption_rate = num_absorptions / probe_true_positives
   - 设计动机：保守估计吸收率，只计算因果上确认的吸收案例

### 损失函数 / 训练策略
- SAE 使用标准 L1 loss 或 JumpReLU/TopK 架构
- Linear probe 使用带 L1 正则的逻辑回归
- k-sparse probing 先用 L1 权重选 top-k latent，再训练标准 probe

## 实验关键数据

### 主实验
在 Gemma-2-2B 上使用 Gemma Scope SAE (16k和65k宽度)，以及自训练的 Qwen2 0.5B 和 Llama 3.2 1B SAE

| 配置 | 精度 | 召回率 | F1 | 说明 |
|------|------|--------|-----|------|
| Linear Probe | ~0.98 | ~0.98 | ~0.98 | Ground truth baseline |
| SAE latent (best) | ~0.95 | ~0.70 | ~0.81 | 最佳 SAE latent |
| k=5 sparse probe | ~0.95 | ~0.90 | ~0.92 | 多 latent 组合改善 |

所有测试的 SAE 都存在特征吸收现象，没有 SAE 能匹配 linear probe 性能

### 消融实验
| L0 范围 | 吸收率 | 说明 |
|---------|--------|------|
| 低 L0 (~25) | ~15-20% | 高精度低召回，吸收严重 |
| 中 L0 (~50-100) | ~10-15% | 最佳 F1 区间 |
| 高 L0 (~200) | ~5-8% | 低精度高召回 |

### 关键发现
- **吸收在所有测试的 SAE 中都存在**：包括 Gemma Scope、自训练 L1 SAE、TopK SAE，跨 Gemma、Qwen、Llama 三种模型
- **更宽更稀疏的 SAE 吸收更严重**：65k SAE 比 16k 吸收率高，低 L0 吸收更多
- **调整超参数无法根本解决**：改变 SAE 宽度或稀疏度只能部分缓解
- **案例研究**："starts with S" latent 6510 在 token "short" 上不激活，取而代之的是 token-aligned latent 1085（"short"词义），后者的解码器包含一个小的"starts with S"方向分量

## 亮点与洞察
- **可解释性幻觉的系统揭示**：看似单义的 latent 实际上有隐藏的 false negative，仅看 max activating examples 会被误导。这对整个 SAE-based interpretability 范式是重要警示
- **从 toy model 到大规模验证的完整链条**：先在 toy model 中解析证明吸收必然发生，再在数百个真实 SAE 上验证，方法论非常扎实
- **因果验证**：不仅发现吸收 latent 存在，还通过消融实验证明吸收的 probe 方向分量确实因果介导了模型行为
- **吸收率指标**：提供了可复用的定量指标，可用于评估未来的 SAE 改进方法

## 局限性 / 可改进方向
- 吸收率指标依赖消融实验，只能应用到注意力尚未移动信息的早期层（<layer 17），晚期层无法检测
- 指标是保守下界估计，不捕捉多个 latent 共同吸收或主 latent 弱激活的情况
- 仅在首字母拼写任务上验证，其他类型的层级特征（如语义层级）需要进一步研究
- 未提出解决方案，只是诊断了问题。可能的方向：Meta-SAE、group lasso、层级稀疏编码

## 相关工作与启发
- **vs Anthropic SAE 工作 [Bricken et al.]**: 他们首次描述 feature splitting 并以正面现象看待，本文发现 splitting 的病态形式 absorption
- **vs Meta-SAE [Bussman et al.]**: Meta-SAE 在 SAE 解码器上再训练 SAE，可以分解 latent 为子成分，可能是解决吸收的方向
- **vs k-sparse probing [Gurnee et al.]**: 本文复用其方法检测 feature splitting，但首次用于检测 absorption

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统定义并研究 feature absorption，对 SAE 范式提出根本性质疑
- 实验充分度: ⭐⭐⭐⭐⭐ 数百个 SAE，三种模型，toy model + 真实验证，因果分析完整
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，案例分析精彩，图表设计优秀
- 价值: ⭐⭐⭐⭐⭐ 对 AI Safety 和机械可解释性领域有重大影响
