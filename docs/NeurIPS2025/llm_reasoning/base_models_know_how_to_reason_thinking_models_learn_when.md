# Base Models Know How to Reason, Thinking Models Learn When

**会议**: NeurIPS 2025  
**arXiv**: [2510.07364](https://arxiv.org/abs/2510.07364)  
**代码**: https://github.com/cvenhoff/thinking-llms-interp  
**领域**: LLM 推理 / 机械可解释性 / 表征工程  
**关键词**: Thinking Models, Steering Vectors, SAE, 推理机制, 基座模型

## 一句话总结
通过无监督 SAE 聚类发现 thinking model 的推理机制分类，然后用 steering vector 在基座模型上激活这些潜在推理能力，混合模型恢复高达 91% 的 thinking-base 性能差距（无需权重更新），证明基座模型已具备推理能力，thinking model 只是学会了"何时"部署它们。

## 研究背景与动机
1. **领域现状**：DeepSeek R1、QwQ 等 thinking model 通过长链推理显著超越基座模型，但其优势来源不清楚
2. **现有痛点**：
   - 对 thinking model 为何有效存在多种竞争假说：学到新能力 vs 更好的结构化 vs 复用已有能力 vs 只是更多计算
   - 现有分析依赖人工检查推理轨迹，主观且可能遗漏模式
3. **核心矛盾**：thinking model 是"学会了新的推理方法"还是"学会了在正确时机使用已有方法"？
4. **本文要解决什么？** 提供因果证据证明基座模型已具备推理能力
5. **切入角度**：用 SAE 无监督聚类发现推理类别，再用 steering vector 在基座模型上激活对应能力
6. **核心idea一句话**：预训练教会模型"如何推理"，后训练（RLVR）教会模型"何时推理"

## 方法详解

### 整体框架
三阶段：(1) 无监督推理机制分类——对 thinking model 推理轨迹的句子级激活做 SAE 聚类，发现 ~15 种推理类别；(2) 提取 steering vector——对每种推理类别计算基座和thinking模型间的激活差；(3) 混合模型——基座模型生成时，用分类器检测当前应用哪种推理机制，注入对应 steering vector。

### 关键设计

1. **无监督推理分类（Top-K SAE 聚类）**:
   - 做什么：从 thinking model 的推理轨迹中自动发现可解释的推理机制类别
   - 核心思路：在 43 万句推理轨迹上训练 restricted-decoder SAE（潜维度 10-25，远小于输入维度 1536+），每个潜在特征对应一种推理机制。用完整性+独立性+一致性三维评分选择最佳配置
   - 设计动机：避免人工标注偏见，确保分类完整且可解释

2. **Steering Vector 提取与应用**:
   - 做什么：将每种推理机制编码为一个方向向量，注入基座模型即可激活对应能力
   - 核心思路：对SAE每个聚类，收集基座和thinking模型在该类别句子上的平均激活差作为 steering vector。生成时对每个 token 计算 SAE 激活，确定最活跃推理类别，注入对应 vector
   - 设计动机：如果基座模型已有潜在推理能力，那么一个简单的方向偏移就应能激活它

3. **混合模型推理（Hybrid Model）**:
   - 做什么：基座模型+选择性 steering = 恢复 thinking 性能
   - 核心思路：基座模型负责主 token 生成，每个位置用 thinking 模型的 SAE 激活分类器检测当前推理类别，施加对应 steering vector。通过困惑度选择最佳系数和窗口大小。仅 12% 的 token 被 steer
   - 设计动机：仅用 15 个 steering vector 就能显著提升，排除了"steering 只是偏向特定 token"的替代解释

### 损失函数 / 训练策略
- SAE：Top-K 稀疏自编码器，K 约束同时活跃的推理类别数
- 混合模型：无需训练，零权重更新，仅在推理时注入 steering vector
- 选择标准：按 thinking model 困惑度最低来选 steering 强度

## 实验关键数据

### 主实验
GSM8K 和 MATH500 上的混合模型性能

| 基座模型 | Thinking模型 | 基座 Acc | 混合 Acc | Thinking Acc | Gap Recovery |
|---------|-------------|---------|---------|-------------|-------------|
| Llama-3.1-8B | R1-Distill-8B | 37.8% | 63.4% | 83.4% | **56.1%** |
| Qwen2.5-14B | R1-Distill-14B | 90.8% | 93.0% | 94.2% | **64.7%** |
| Qwen2.5-32B | R1-Distill-32B | 92.6% | 94.4% | 94.8% | **81.8%** |
| Qwen2.5-32B | QwQ-32B | 92.6% | 94.8% | 96.4% | 57.9% |

### 消融实验
| 配置 | 说明 |
|------|------|
| 随机 steering vector | 性能下降，排除偶然性 |
| 错误类别 steering | 性能下降，说明类别匹配重要 |
| 仅 12% token 被 steer | 大部分token由基座模型自主生成 |

### 关键发现
- 仅 15 个 steering vector 即可恢复高达 91% 的性能差距（MATH500 上 Qwen2.5-Math-1.5B）
- Llama-8B 获得最大绝对提升（+25.6%），说明基座模型确实有潜在推理能力
- 蒸馏训练的模型（R1-Distill）和 RLVR 训练的模型（QwQ）都适用
- 推理机制分类在 15-25 类时最优，与认知科学中的"基本推理操作"数量一致

## 亮点与洞察
- **重新定义后训练的角色**：RLVR 不是教模型"如何推理"，而是教"何时推理"——这改变了我们对 thinking model 训练的理解
- **无监督推理分类的方法论贡献**：用 restricted SAE 做聚类是发现 AI 认知结构的新工具
- **极简因果验证**：仅用 15 个 vector steer 12% 的 token 就有显著效果，是极强的因果证据
- **跨架构跨训练方法泛化**：在 Qwen、Llama、蒸馏型、RLVR 型上都有效

## 局限性 / 可改进方向
- 混合模型依赖 thinking model 的 SAE 分类器——实际部署时仍需 thinking model
- 仅在数学推理任务（GSM8K/MATH500）上验证，代码/逻辑推理未测试
- Qwen2.5-Math-1.5B 的 gap recovery 为 0%（基座已很强），说明基座足够强时 steering 无额外收益
- 句子级聚类粒度可能不够精细

## 相关工作与启发
- **vs Gandhi et al.**: 他们认为 thinking model 学到新能力，本文证据指向复用已有能力
- **vs Ward et al.**: 最接近的工作，也发现 RL 复用预训练表征，但本文提供了更完整的因果验证
- **vs Steering vector 工作 [Turner et al.]**: 之前用于控制风格/情感，本文首次用于激活推理机制

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "基座已会推理，thinking 只学何时用"的洞察极具影响力
- 实验充分度: ⭐⭐⭐⭐ 3个基座+4个thinking模型交叉验证完整
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑严密，图表设计精美
- 价值: ⭐⭐⭐⭐⭐ 改变对 thinking model 训练范式的理解
