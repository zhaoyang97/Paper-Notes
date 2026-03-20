# Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation

**会议**: AAAI 2026 (Oral)  
**arXiv**: [2511.11255v2](https://arxiv.org/abs/2511.11255v2)  
**代码**: 无  
**领域**: 推荐系统 / 信息检索  
**关键词**: 生成式推荐, LLM对齐, 协同过滤, DPO, 语义-协同ID  

## 一句话总结

提出统一三层对齐框架 Align³GR，在 token 级（双端 SCID）、行为建模级（多任务 SFT）和偏好级（渐进式 DPO）系统性弥合 LLM 与推荐系统之间的语义-行为鸿沟。

## 背景与动机

LLM 作为生成式推荐器（Generative Recommender）直接端到端输出推荐物品是近期趋势，但核心困难在于 LLM 的语言建模目标与推荐系统的用户隐式偏好建模之间存在本质差距：语言侧关注语义信息与 next-token prediction，推荐侧关注交互行为信息。现有工作通常只在 tokenization、SFT、偏好 RL 三个环节中的某一个做对齐，缺乏系统性的多层级联合优化。此外，已有方法在 tokenization 阶段往往只编码 item 而忽略 user，偏好对齐阶段也多依赖静态离线数据，无法应对真实场景中用户偏好的动态变化。

## 核心问题

1. 如何在 token 层面同时建模用户和物品的语义与协同信号，而非孤立编码？
2. 如何在 SFT 阶段让 LLM 不仅学会推荐行为模式，还能理解 user token 的语义含义？
3. 如何通过渐进式偏好优化（由易到难）持续提升模型，突破静态 DPO 的性能天花板？

## 方法详解

### 整体框架

Align³GR 是一个统一的三层对齐流水线：Token 级对齐 → 行为建模级对齐 → 偏好级对齐。以 Llama2-7B 为骨干，使用 LoRA 做参数高效微调。

### 关键设计

1. **Token 级对齐：双端 SCID Tokenization**
   - 对用户和物品分别提取语义特征（冻结的 T5 编码器）和协同特征（冻结的 DIN 编码器），拼接后通过 SC Encoder（MLP）融合为统一的 SC embedding
   - 使用 3 层 RQ-VAE（每层 256 个 codebook embedding，维度 32）将 SC embedding 量化为离散的 SCID token
   - 训练目标包含两部分：用户-物品行为对齐损失 $\mathcal{L}_{\text{U2I}}$（sampled-softmax）和 RQ-VAE 重构/量化损失，通过两阶段切换超参 $\alpha, \gamma$ 进行训练——先稳定行为对齐（$\alpha=1, \gamma=0$），再聚焦量化学习（$\alpha=0.1, \gamma=1$）
   - 推理时用户和物品模块独立部署，各自生成 SCID

2. **行为建模级对齐：增强多任务 SFT**
   - 基于 LC-Rec 的多任务 SFT（序列预测、非对称预测、意图推断、偏好推理），但做了两个关键增强：
   - **注入 User SCID**：在所有任务 prompt 中加入用户 SCID token，提供更丰富的上下文
   - **双向对齐任务**（$B_2$）：text→SCID（从用户画像预测 SCID）和 SCID→text（从 SCID 重构用户画像），显式建立 SCID token 与真实语义的对应关系

3. **偏好级对齐：渐进式 DPO**
   - 基于 Softmax-DPO（每样本 1 正例 + 20 负例），分两阶段渐进学习：
   - **SP-DPO（Self-Play DPO）**：模型与自身博弈生成多样化训练数据，利用 SCID 的分层特性按 prefix-ngram 匹配度分三阶段（Easy/Medium/Hard），从完全不同的正负例逐步过渡到 prefix 高度重叠但仍不同的正负例
   - **RF-DPO（Real-world Feedback DPO）**：利用真实用户反馈构建偏好数据，反馈分三级（disliked/neutral/liked），同样渐进训练——Easy 阶段用强烈不喜欢作负例，Hard 阶段用中性（曝光未点击）作更难负例
   - 每个阶段微调后的模型成为下一阶段的参考模型 $\pi_\theta^i \to \pi_{\text{ref}}^{i+1}$

### 损失函数 / 训练策略

- Token 级：$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{U2I}} + \gamma \cdot (\mathcal{L}_{\text{User RQ}} + \mathcal{L}_{\text{Item RQ}})$，两阶段训练
- 行为建模级：多任务 SFT loss + 双向对齐辅助 loss
- 偏好级：Softmax-DPO loss，渐进式 SP-DPO → RF-DPO，每阶段模型迭代更新参考策略
- 骨干 Llama2-7B + LoRA，AdamW 优化器，batch size 1024，训练 20,000 步，beam width 20

## 实验关键数据

| 数据集 | 指标 | Align³GR | EAGER-LLM (之前SOTA) | 提升 |
|--------|------|----------|----------------------|------|
| Instruments | R@5 | 0.1103 | 0.0991 | +11.3% |
| Instruments | R@10 | 0.1442 | 0.1224 | +17.8% |
| Instruments | N@5 | 0.0947 | 0.0851 | +11.3% |
| Instruments | N@10 | 0.1113 | 0.0926 | +20.2% |
| Beauty | R@10 | 0.0994 | 0.0830 | +19.8% |
| Beauty | N@10 | 0.0529 | 0.0459 | +15.3% |
| Yelp | R@10 | 0.0679 | 0.0569 | +19.3% |
| Yelp | N@10 | 0.0403 | 0.0315 | +27.9% |

**工业 A/B 测试**（约 4000 万用户，多周测试）：
| 模型 | Recall@100 | Revenue 提升 |
|------|------------|-------------|
| TIGER | 0.229 | +0.555% |
| Align³GR | 0.242 | +1.432% |

### 消融实验要点

- **Token 级**：单端→双端 tokenization 有显著提升；加入协同特征（CF）进一步提升；U-I 对齐损失与双端+CF 配合效果最佳
- **行为建模级**：注入 User SCID 到 prompt 带来一致提升；双向对齐任务 $B_2$ 贡献最大，说明 LLM 需要显式的语义-结构化映射监督
- **偏好级**：Self-Play 将 R@10 从 0.1295 提升到 0.1356；渐进式学习再提升到 0.1396；加入真实反馈 RF-DPO + 渐进策略达到最佳 0.1442

## 亮点

- **系统性对齐设计**：三层对齐（token/behavior/preference）形成完整流水线，每层都有明确的对齐目标，且实验证明各层贡献互补
- **双端 SCID**：不同于仅编码 item 的已有方法，同时建模用户与物品的语义-协同联合表示，并通过 U2I 行为损失做联合优化
- **渐进式 DPO**：从 SP-DPO 到 RF-DPO、从 Easy 到 Hard 的课程学习策略，解决了静态 DPO 在动态推荐场景中的局限
- **工业验证充分**：不仅有 3 个公开数据集的离线实验，还有 4000 万用户规模的在线 A/B 测试，Revenue 提升 1.432%

## 局限性 / 可改进方向

- 骨干模型仅使用 Llama2-7B，未探索更大或更新的 LLM（如 Llama3）对效果的影响
- RQ-VAE 的 codebook 大小固定为 256，对于超大规模物品库可能存在 codebook 碰撞问题
- RF-DPO 的反馈分级（disliked/neutral/liked）较粗粒度，更细粒度的反馈信号可能进一步提升效果
- 公开数据集上用 LLM 情感分析代替真实用户反馈做 RF-DPO，可能引入噪声
- 用户历史限制为最近 20 个交互，长序列建模能力未被充分探索

## 与相关工作的对比

- **vs LC-Rec**：LC-Rec 只做 item tokenization + 多任务 SFT，Align³GR 增加了双端 SCID 和渐进式 DPO，全面超越
- **vs EAGER-LLM**：EAGER-LLM 在 token 级引入协同信号，但仍是单端（item 侧），且无偏好对齐；Align³GR 的双端 tokenization + 行为建模增强 + 渐进 DPO 实现了全方位提升
- **vs LETTER**：LETTER 提出可学习 tokenizer 但无用户建模和偏好优化，Align³GR 在所有指标上显著优于 LETTER
- **vs 标准 DPO**：传统 DPO 依赖静态离线数据，Align³GR 的渐进式 SP-DPO + RF-DPO 实现持续自我改进和真实反馈适应

## 启发与关联

- 三层对齐的设计思路（token/behavior/preference）可以迁移到其他 LLM 适配下游任务的场景（如 LLM+搜索、LLM+广告）
- 渐进式 DPO 的 Easy-to-Hard 课程学习策略在偏好标签噪声大的场景（如推荐、广告）特别有价值
- 双端 tokenization 思路提示我们：在 LLM 做推荐时，user 和 item 应当在同一框架内联合建模，而非独立处理

## 评分

- 新颖性: ⭐⭐⭐⭐ （三层对齐的系统设计有创新，但各模块如 RQ-VAE、DPO 本身并非新技术）
- 实验充分度: ⭐⭐⭐⭐⭐ （3 个公开数据集 + 工业 A/B 测试 + 详细消融）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，但部分公式符号较密集）
- 价值: ⭐⭐⭐⭐ （工业落地价值高，但学术新颖性中等偏上）
