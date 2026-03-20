# Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences

**会议**: ICCV 2025  
**arXiv**: [2506.02095](https://arxiv.org/abs/2506.02095)  
**代码**: [https://cyclereward.github.io/](https://cyclereward.github.io/)  
**领域**: 多模态VLM / 图文对齐 / 奖励建模  
**关键词**: cycle consistency, reward model, image-text alignment, preference learning, DPO, self-supervised  

## 一句话总结
提出CycleReward，利用cycle consistency作为自监督信号替代人工偏好标注——将caption用T2I模型重建为图像再比较相似度来排序，构建866K偏好对数据集CyclePrefDB，训练的奖励模型在detailed captioning上比HPSv2/PickScore/ImageReward高6%+，且DPO训练后提升VLM在多个VL任务上的性能，无需任何人工标注。

## 研究背景与动机
1. **领域现状**：图文对齐度量是多模态学习的核心挑战。现有奖励模型（ImageReward、HPSv2、PickScore）依赖大规模人工偏好标注，成本高且难以扩展。GPT-4V标注虽可用但昂贵、闭源且rate-limited。
2. **现有痛点**：(1) 人工偏好数据收集成本高、难以规模化；(2) 现有偏好数据主要针对短文本（~20 token），无法有效评估长描述性caption；(3) CLIP等嵌入式方法对长文本不敏感。
3. **核心矛盾**：长描述性caption越来越重要（ShareGPT4V、LLaVA等生成的详细描述），但缺乏能有效评估它们的alignment metric。直接比较图文很难，但把文本映射回图像空间后，图图比较就容易得多。
4. **本文要解决什么**：用cycle consistency构建无需人工标注的偏好数据集和奖励模型，特别针对长描述性caption的对齐评估。
5. **切入角度**：经典的cycle consistency思想——$x \xrightarrow{F} y \xrightarrow{G} x'$，如果caption $y$越准确，重建图像$x' = G(y)$就越接近原图$x$。把这个相似度作为偏好信号而非直接作为metric。
6. **核心idea一句话**：用cycle consistency score排序caption/image候选，构建偏好数据集训练奖励模型，实现无人工标注的图文对齐学习。

## 方法详解

### 整体框架
(1) Cycle consistency scoring: Image→Caption(多个模型生成)→T2I重建→DreamSim相似度；Text→Image(多个模型生成)→I2T重建→SBERT相似度。(2) 偏好排序：相似度更高的为preferred。(3) 构建866K偏好对数据集CyclePrefDB。(4) 训练奖励模型CycleReward（BLIP backbone + Bradley-Terry loss）。

### 关键设计

1. **Cycle Consistency作为偏好信号**:
   - **Image-to-Text方向**：给定图像$x$，11个I2T模型（BLIP2, LLaVA系列, InternVL2系列）生成不同质量的caption $\{y_i\}$，用Stable Diffusion 3将每个caption反向重建为图像$G(y_i)$，用DreamSim计算$d_{img}(x, G(y_i))$。更高相似度 = preferred。
   - **Text-to-Image方向**：给定caption $y$，4个T2I模型（SD1.5, SDXL, SD3, FLUX）×3 seeds生成候选图像$\{x_i\}$，用LLaVA-1.5-13B将每张图reverse-caption为$F(x_i)$，用SBERT计算$d_{text}(y, F(x_i))$。更高相似度 = preferred。
   - 设计动机：直接比较跨模态（图文）不如同模态（图图或文文）比较准确。Cycle consistency巧妙地将跨模态对齐问题转化为同模态相似度问题。

2. **CyclePrefDB数据集**:
   - 866K偏好对（398K I2T + 468K T2I），基于7.6K DCI数据集的高分辨率图像和dense caption。
   - 平均文本长度56 token，远高于HPDv2 (19 token)和Pick-A-Pic (24 token)—— 特别适合评估detailed captioning。
   - 数据过滤：移除重复、过滤奖励差距过小的对（$|r_i - r_j| < \tau_{sim}$）和preferred奖励过低的对。
   - 设计动机：使用多个不同能力的模型（从BLIP2到InternVL2-40B）生成质量梯度明显的候选，确保偏好对的信息量。

3. **CycleReward奖励模型**:
   - 架构：BLIP backbone（ViT-L/16 + BERT-base + 5层MLP + 标量输出头）。
   - 三种变体：I2T-only、T2I-only、Combo（联合训练$\mathcal{L} = \mathcal{L}_{text} + \lambda \mathcal{L}_{img}$）。
   - 训练证明了一个关键结论：**蒸馏cycle consistency训练的奖励模型优于直接使用raw cycle consistency score**——即使多seed平均的raw score也无法达到CycleReward的性能（+4%在DetailCaps上），因为奖励模型学到了比pixel重建更丰富的对齐概念。

### DPO应用
- I2T方向：用CyclePrefDB-I2T对Qwen-VL-Chat做DPO → 在detailed captioning提升同时，在perception/reasoning/hallucination等VL任务上全面提升，比VLFeedback（GPT-4V标注）效果相当或更好。
- T2I方向：用CyclePrefDB-T2I对SD1.5做Diffusion DPO → 在T2I-CompBench和PartiPrompts上超过或媲美Pick-A-Pic（人工标注851K对）。

## 实验关键数据

### 对齐度量（Pairwise Accuracy）

| 方法 | DetailCaps-4870 | GenAI-Bench | 标注类型 |
|------|----------------|-------------|---------|
| CLIPScore | 51.66 | 49.73 | 无 |
| VQAScore (11B) | 50.24 | **64.13** | 无 |
| HPSv2 | 54.34 | 56.13 | 人工 |
| PickScore | 51.01 | 57.05 | 人工 |
| ImageReward | 50.70 | 56.70 | 人工 |
| Raw Cycle Consistency | 56.46 | 52.52 | 自监督 |
| **CycleReward-Combo** | **60.50** | 55.52 | 自监督 |

CycleReward在detailed captioning上比所有人工标注方法高6%+，比VQAScore (11B, 24×更大)高10.26%。

### Best-of-N采样
- LLaVA-WD和DeCapBench上CycleReward的BoN增益最大（远超VQAScore、ImageReward）
- T2I-CompBench上与ImageReward（人工标注）相当，在complex prompts上甚至更好

### DPO结果

I2T DPO (Qwen-VL-Chat):

| 模型 | DeCapBench | LLaVA-W | MMMU | MME-P | MMHal |
|------|-----------|---------|------|-------|-------|
| Base | 26.47 | 61.67 | 73.10 | 1460.2 | 2.99 |
| DPO w/ VLFeedback (GPT-4V标注) | 28.03 | 69.17 | 76.39 | 1551.5 | 3.32 |
| **DPO w/ CyclePrefDB-I2T** | **30.63** | **70.00** | 74.13 | 1485.7 | 3.11 |

### 消融

| Similarity Metric | DetailCaps | GenAI |
|-------------------|-----------|-------|
| DreamSim (ours) | **58.02** | 53.49 |
| LPIPS | 53.16 | 52.97 |
| CLIP | 57.90 | 53.30 |

### 关键发现
- **蒸馏奖励模型 > 原始cycle consistency score**：CycleReward在所有benchmark上优于raw cycle consistency（+4% on DetailCaps），因为：(1) 奖励模型学到了超越pixel重建的高层对齐概念（如Figure 7的红鸟vs蓝鸟）；(2) 奖励模型更快（单次forward vs T2I重建）；(3) 可微分。
- **无需人工标注即可匹敌人工标注方法**：IRDB-Cycle（在ImageRewardDB数据上用cycle consistency重标注）与原始ImageReward效果相当——证明cycle consistency是人工偏好的有效proxy。
- **DPO泛化性惊人**：CyclePrefDB-I2T只包含captioning指令，但DPO后在perception、reasoning、hallucination等多个任务上全面提升，说明detailed captioning能力的提升对通用VL能力有正迁移。
- **I2T decoder越强越好**：InternVL2-26B作为text-to-image cycle的I2T decoder显著优于LLaVA-1.5-13B (+5.47% on DetailCaps)，因为更强的LLM可以更准确地重建caption。
- **DreamSim优于LPIPS和CLIP做图像相似度**：因为DreamSim专门建模人类视觉相似性判断。

## 亮点与洞察
- **"cycle一致性不需要实时计算"的关键insight**：之前Image2Text2Image等方法直接用cycle consistency作为metric（慢、不可微）。CycleReward将其作为偏好信号训练奖励模型，获得了更好的性能+更快的推理+可微性——这是"离线蒸馏优于在线计算"的又一例证。
- **理论连接到PMI**：cycle consistency score在理论上等价于log p(x,y) + PMI(x,y)，即同时度量了pair的似然和互信息。这提供了为什么cycle consistency作为对齐信号有效的理论基础。
- **"training-free data annotation at scale"范式**：不用人工、不用GPT-4V，只用开源模型的cycle consistency就能自动标注百万级偏好数据。这对降低RLHF/DPO的数据获取成本有重要意义。

## 局限性 / 可改进方向
- 依赖T2I/I2T模型的质量——重建质量差时会产生错误偏好。
- Stable Diffusion 3的77-token限制制约了更长文本的评估。
- 在text-to-image generation上VQAScore (11B)仍然更强——CycleReward更擅长captioning评估。
- 继承了底层模型的偏差（如DreamSim偏好前景相似、SD3生成偏差等）。
- 未探索视频/音频等其他模态的cycle consistency。

## 相关工作与启发
- **vs ImageReward**: ImageReward用137K人工偏好,CycleReward用866K自动偏好。在detailed captioning上CycleReward显著更强(+9.8%)，因为human preference数据偏向短文本。
- **vs VQAScore**: VQAScore用24×更大模型(11B vs 477M)在T2I上更好，但在detailed captioning上CycleReward+10.26%。两者互补——VQAScore擅长组合关系，CycleReward擅长详细描述。
- **vs VisVM (之前batch)**: VisVM也用CLIP作为PRM构建self-supervision for VLM improvement。CycleReward更进一步——使用cross-modal cycle consistency而非单模态相似度，且应用到DPO训练。两者都展示了"无需人工标注的VLM自改进"这一重要方向。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Cycle consistency作为大规模偏好信号是全新且优雅的想法，理论连接到PMI清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 5个alignment benchmark+BoN+DPO(I2T和T2I)+消融(metric/decoder/数据规模/filtering)+Winoground+与human preference的agreement rate，极其全面
- 写作质量: ⭐⭐⭐⭐⭐ 清晰优雅，Figure 1的overview、Figure 7的failure analysis都很好
- 价值: ⭐⭐⭐⭐⭐ 替代人工标注的自监督偏好学习范式，对降低VLM对齐成本有重大意义
