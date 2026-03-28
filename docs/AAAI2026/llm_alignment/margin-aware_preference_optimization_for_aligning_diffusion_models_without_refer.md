# Margin-aware Preference Optimization for Aligning Diffusion Models without Reference

**会议**: AAAI 2026  
**arXiv**: [2406.06424](https://arxiv.org/abs/2406.06424)  
**代码**: https://github.com/mapo-t2i/mapo  
**领域**: 扩散模型 / 对齐RLHF  
**关键词**: 偏好对齐, 扩散模型, 文本图像生成, reference-free, DPO

## 一句话总结
提出 MaPO（Margin-aware Preference Optimization），一种无需参考模型的偏好对齐方法，通过直接优化 Bradley-Terry 模型下偏好/非偏好输出的似然 margin 来对齐 T2I 扩散模型，在风格适配、安全生成、通用偏好对齐等 5 个领域均超越 DPO 和专用方法。

## 研究背景与动机

1. **领域现状**：RLHF/DPO 等偏好对齐方法已广泛用于将 T2I 扩散模型（如 SDXL）与人类偏好对齐。这些方法通常依赖一个冻结的参考模型（reference model）做 KL 散度正则化，以确保训练稳定性。
2. **现有痛点**：作者发现 T2I 扩散模型存在严重的 **"reference mismatch"（参考模型不匹配）** 问题——当偏好数据的分布与参考模型差距较大时（如学习新的艺术风格或个性化特定对象），参考模型反而会**阻碍有效适配**。视觉模态的非结构化特性使得这个问题比 LLM 场景更严重。
3. **核心矛盾**：reference mismatch 越大，DPO 等方法的性能退化越严重。但实际应用中经常需要将模型适配到与预训练分布差异很大的偏好（如从写实到动漫风格），这正是 reference mismatch 最严重的场景。
4. **本文要解决什么？** 设计无需参考模型的偏好对齐方法，彻底消除 reference mismatch 对 T2I 扩散模型对齐的负面影响。
5. **切入角度**：直接在 Bradley-Terry 偏好模型下最大化偏好和非偏好输出之间的似然 margin，同时最大化偏好输出的似然，不锚定任何参考模型。
6. **核心 idea**：将 T2I 对齐统一为无参考的成对偏好优化，同时学习通用风格特征和特定偏好。

## 方法详解

### 整体框架
MaPO 基于 Bradley-Terry 模型，直接优化偏好图像和非偏好图像的似然 margin。与 DPO 不同，MaPO 不需要维护冻结的参考模型，目标函数包含两部分：(1) 最大化偏好和非偏好输出的 likelihood margin；(2) 同时最大化偏好输出的似然（防止两边都下降）。

### 关键设计

1. **无参考偏好优化（Reference-free Preference Optimization）**:
   - **做什么**：在 Bradley-Terry 模型下直接优化 margin，不依赖参考模型。
   - **核心思路**：标准 DPO 的损失函数形如 $\mathcal{L}_\text{DPO} = -\log \sigma(\beta [\log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}])$，需要参考模型 $\pi_\text{ref}$。MaPO 去掉参考模型，直接优化 $\log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)$（似然 margin），同时加上 $\log \pi_\theta(y_w|x)$ 项防止两侧似然同时下降。
   - **设计动机**：参考模型在 reference mismatch 场景下成为"锚定障碍"——它限制模型向偏好分布移动的自由度。移除参考模型相当于释放了这个约束，让模型可以自由地向偏好分布适配。

2. **构造 Reference Mismatch 场景的数据集**:
   - **做什么**：创建 Pick-Style 和 Pick-Safety 两个数据集，分别模拟 reference-chosen mismatch 和 reference-rejected mismatch。
   - **核心思路**：Pick-Style 通过在 prompt 前加"Disney style animated image"或"Pixel art style image"作为偏好图像 prompt，加"Realistic 8k image"作为非偏好 prompt，模拟风格偏好偏移（参考模型远离偏好风格）。Pick-Safety 则用"Sexual, nudity"前缀生成非偏好图像，模拟安全偏好（参考模型远离非偏好内容）。
   - **设计动机**：现有偏好数据集不能直接控制 reference mismatch 程度，需要专门构造数据来验证 MaPO 在不同 mismatch 程度下的优势。

3. **统一多任务 T2I 对齐**:
   - **做什么**：将 5 个不同的 T2I 任务（安全生成、风格适配、文化表示、个性化、通用偏好）统一为成对偏好优化。
   - **设计动机**：传统方法需要为不同任务设计不同的对齐策略（如 DreamBooth 用于个性化），MaPO 的无参考框架足够灵活，可以适配所有场景。

### 损失函数 / 训练策略
- MaPO 损失：margin-aware loss + 偏好似然最大化项
- 基于 SDXL 模型，比 DPO 减少 14.5% 的训练时间（不需要维护参考模型的 forward pass）
- 也不需要额外 GPU 内存存储参考模型

## 实验关键数据

### 主实验

| 任务 | 数据集 | MaPO | DPO | 说明 |
|------|--------|------|-----|------|
| 风格适配 | Pick-Style (Cartoon) | **显著优于** | 受 ref mismatch 限制 | mismatch 越大优势越明显 |
| 安全生成 | Pick-Safety | **显著优于** | 受 ref mismatch 限制 | 安全场景优势明显 |
| 通用偏好 | Pick-a-Pic v2 | **优于** | 次优 | 即使 mild mismatch 也有提升 |
| Imgsys排名 | 公开排行榜 | **第7名** | 第20名 | 超过21/25 SOTA T2I模型 |
| 个性化 | - | **超过DreamBooth** | - | 替代专用方法 |

关键发现：MaPO 的优势随 reference mismatch 程度增大而**急剧增长**。

### 消融/分析

| 分析维度 | 关键发现 |
|---------|---------|
| Reference mismatch 程度 | mismatch 越大，DPO 退化越严重，MaPO 优势越明显 |
| 训练效率 | 比 DPO 减少 14.5% 训练时间（无需参考模型 forward） |
| 内存效率 | 无需存储参考模型，更 memory-friendly |

## 亮点与洞察
- **精准识别了 reference mismatch 问题**：在 T2I 场景中系统性地分析了参考模型不匹配的问题和影响，这个分析本身就是重要贡献。很多人直接套用 DPO 到扩散模型但忽视了这个关键差异。
- **方法极简但有效**：MaPO 的核心改动就是去掉参考模型并加 margin loss，实现简单但效果好。在 Imgsys 上排第7名（DPO 第20名）非常有说服力。
- **统一多任务框架**：一个方法覆盖安全、风格、个性化等多个任务，避免了为每个任务调不同方法的麻烦。

## 局限性 / 可改进方向
- 无参考模型可能在某些场景下缺乏正则化，导致模式坍塌风险
- 仅在 SDXL 上验证，对其他扩散架构（如 DiT-based）的适用性未知
- Pick-Style 和 Pick-Safety 是合成数据集，真实用户偏好可能更复杂
- 缺少对 margin 超参数敏感性的详细分析

## 相关工作与启发
- **vs Diffusion-DPO**: DPO 直接搬到扩散模型，保留参考模型，在 reference mismatch 时性能受限。MaPO 去掉参考模型，在 mismatch 严重时优势巨大。
- **vs DreamBooth**: DreamBooth 是专门的个性化方法，MaPO 作为通用对齐方法在个性化场景也能超越 DreamBooth。
- **vs SimPO (LLM 场景)**: SimPO 也探索了无参考模型的偏好优化，但是在 LLM 场景。MaPO 将这个思路引入 T2I 扩散模型并解决了视觉模态特有的 reference mismatch 问题。

## 评分
- 新颖性: ⭐⭐⭐⭐ 精准识别 reference mismatch 问题并提出简洁解决方案
- 实验充分度: ⭐⭐⭐⭐ 5个任务领域 + Imgsys 公开排行榜验证
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，动机充分
- 价值: ⭐⭐⭐⭐ 对 T2I 扩散模型对齐有直接实用价值

## 补充说明
- 该工作的方法论和实验设计对相关领域有参考价值
- 后续工作可在更多场景和更大规模上验证方法的泛化性和可扩展性
- 与近期相关工作的结合（如与 RL/MCTS/多模态方法的交叉）有潜在研究价值
- 建议结合实际应用需求评估该方法的部署可行性和计算效率
- 数据集和评估指标的选择可能影响结论的普适性，需在更多 benchmark 上交叉验证

## 补充说明
- 该工作的方法论和实验设计对相关领域有参考价值
- 后续工作可在更多场景和更大规模上验证方法的泛化性和可扩展性
- 与近期相关工作的结合（如与 RL/MCTS/多模态方法的交叉）有潜在研究价值
