---
title: >-
  [论文解读] Reason2Drive: Towards Interpretable and Chain-Based Reasoning for Autonomous Driving
description: >-
  [ECCV 2024][自动驾驶][chain-based reasoning] 构建 Reason2Drive 基准数据集（600K+ 视频-文本对，覆盖感知-预测-推理链式任务），提出 ADRScore 评估链式推理正确性的新指标，并设计 Prior Tokenizer + Instructed Vision Decoder 框架增强 VLM 的目标级感知和推理能力，在自动驾驶推理任务上显著超越所有基线。
tags:
  - "ECCV 2024"
  - "自动驾驶"
  - "chain-based reasoning"
  - "VLM"
  - "benchmark dataset"
  - "interpretable decision-making"
---

# Reason2Drive: Towards Interpretable and Chain-Based Reasoning for Autonomous Driving

**会议**: ECCV 2024  
**arXiv**: [2312.03661](https://arxiv.org/abs/2312.03661)  
**代码**: [https://github.com/fudan-zvg/Reason2Drive](https://github.com/fudan-zvg/Reason2Drive)  
**领域**: 自动驾驶  
**关键词**: autonomous driving, chain-based reasoning, VLM, benchmark dataset, interpretable decision-making

## 一句话总结
构建 Reason2Drive 基准数据集（600K+ 视频-文本对，覆盖感知-预测-推理链式任务），提出 ADRScore 评估链式推理正确性的新指标，并设计 Prior Tokenizer + Instructed Vision Decoder 框架增强 VLM 的目标级感知和推理能力，在自动驾驶推理任务上显著超越所有基线。

## 研究背景与动机
1. **领域现状**：大型视觉-语言模型（VLMs）因其复杂推理能力在自动驾驶领域引起广泛关注。VLM 有望为自动驾驶提供更好的可解释性和泛化能力，相比端到端方法（将系统视为黑盒直接从传感器输入到控制信号）和基于规则的方法（依赖大量人工规则），VLM 能提供明确的决策依据。
2. **现有痛点**：(a) 缺乏数据集：现有驾驶语言数据集（Talk2Car、NuScenesQA、DriveLM 等）过度简化了复杂驾驶过程为简单 QA（布尔回答或有限多选），缺乏解释决策过程的推理链标注。(b) 评估缺陷：BLEU、CIDEr 等传统文本指标仅从整体文本质量衡量，不考虑推理步骤与最终结论的因果关系——无法判断模型的推理链是否真正支持了正确的决策。
3. **核心矛盾**：自动驾驶不是简单的 QA 过程，而是感知（Perception）→ 预测（Prediction）→ 推理（Reasoning）的多步链式决策。但现有数据集和评估体系都无法支撑这种链式推理的研究。
4. **本文要解决什么？**：(a) 构建大规模、多源、包含链式推理标注的自动驾驶基准；(b) 设计专门评估链式推理正确性的指标；(c) 增强 VLM 利用目标级感知先验的能力来提升推理准确性。
5. **切入角度**：将自动驾驶决策过程结构化为 "感知→预测→推理" 三步链，从数据集构建、评估指标、模型架构三个层面系统性解决问题。
6. **核心idea一句话**：通过大规模链式推理数据集 + 因果感知的评估指标 + 感知先验增强的 VLM 架构，系统性地推进可解释自动驾驶推理研究。

## 方法详解

### 整体框架
**数据集层**：从 nuScenes、Waymo、ONCE 三个公开数据集解析标注构建 object-centric 数据库 → 手工设计问题模板生成感知/预测/推理三类 QA 对 → GPT-4 验证和增强多样性 → 最终 632,955 个视频-文本对。

**评估层**：提出 ADRScore 指标体系，包含推理对齐（RA）、冗余（RD）、缺失步骤（MS）三个维度，聚合为 $ADRScore = \frac{1}{3}(RA + RD + MS)$，以及考虑视觉元素精度的 ADRScore-S 变体。

**模型层**：基于 InstructBLIP 架构，输入：视频帧序列 + 感知先验（目标位置和运动信息）→ Vision Encoder（EVA-CLIP ViT-G/14）提取图像特征 → Prior Tokenizer（2层 MLP + RoIAlign + 位置编码）提取感知先验特征 → Q-Former 对齐到文本空间 → LLM 生成文本回答（含 `<LOC>` 和 `<MOT>` 特殊 token）→ Instructed Vision Decoder 从 token embedding 解码出目标位置和轨迹预测。

### 关键设计

1. **Reason2Drive 数据集**:
    - 做什么：构建最大规模的语言驱动自动驾驶推理基准
    - 核心思路：将驾驶标注解析为 object-centric 数据库（每帧存储目标的类别、属性、位置、运动等），然后与手工模板结合生成三类任务的 QA 对。任务分为目标级和场景级两个粒度：
     - **感知任务**（39%，246K）：识别目标类别、属性（移动状态、距离）、位置等
     - **预测任务**（34%，216K）：预测目标未来轨迹、移入/移出车道、转弯方向等
     - **推理任务**（27%，171K）：分析当前感知和预测状态，推导驾驶策略和风险评估
    - 设计动机：现有数据集最多覆盖13K个样本且仅限简单QA，而驾驶决策需要理解完整的"看到什么→会发生什么→该怎么做"因果链

2. **ADRScore 评估指标**:
    - 做什么：评估生成推理链的因果正确性，而非仅仅文本相似度
    - 核心思路：将生成的推理步骤 $\vec{h}=\{h_1,...,h_N\}$ 与标准步骤 $\vec{r}=\{r_1,...,r_K\}$ 进行对齐。基于 BERT 句子嵌入的余弦相似度计算对齐值。三个子指标：
     - **推理对齐 RA** = 平均对齐度（生成步骤与标准步骤的匹配程度）
     - **冗余 RD** = 最小对齐度（惩罚多余的无关步骤）
     - **缺失步骤 MS** = 反向对齐的最小值（检测遗漏的关键步骤）
     - **ADRScore-S**：当步骤包含视觉元素时（位置坐标、轨迹），用几何 MSE 替代语义相似度，更严格地评估空间推理
    - 设计动机：传统 BLEU/CIDEr 指标对推理能力差异的区分度很低——不同推理水平的模型得分差距很小，无法有效 benchmark

3. **Prior Tokenizer（感知先验分词器）**:
    - 做什么：将目标级视觉先验（位置、运动信息）编码为 LLM 可理解的 token
    - 核心思路：用 RoIAlign 从图像特征中提取区域级特征 $f_r$，用位置编码函数 $E(\cdot)$ 将几何位置和运动信息映射到相同维度，两者相加后通过 2 层 MLP 投影：$f_p = F_p(f_r + E(P))$。最终感知先验 token 与视觉 token 一起通过 Q-Former 对齐到文本空间
    - 设计动机：直接将坐标/轨迹作为文本输入 LLM 会导致信息损失（文本描述难以完整捕获动态场景的空间细节），视觉特征级的编码更高效准确

4. **Instructed Vision Decoder（指导式视觉解码器）**:
    - 做什么：从 LLM 输出中解码出精确的目标位置和运动轨迹预测
    - 核心思路：扩展 LLM 词汇表加入 `<LOC>` 和 `<MOT>` 两个特殊 token。当 LLM 需要生成感知预测时输出这些 token，提取其最后一层特征通过 MLP 投影得到 hidden embedding $f_h$，再与视觉特征一起送入 Transformer decoder 解码：$\hat{P} = D(f_v, f_h)$。解码器包含特征对齐层和任务特定 head（目标检测 + 轨迹预测）
    - 设计动机：仅用语言模型当解码器无法输出准确的感知结果，而驾驶场景中感知准确性是可靠推理的前提。受 LISA 启发，将感知能力直接嵌入多模态 LLM

### 损失函数 / 训练策略
- **总损失**：$\mathcal{L} = \mathcal{L}_{txt} + \lambda_{per}\mathcal{L}_{per}$，$\lambda_{per}=1.0$
- **文本损失** $\mathcal{L}_{txt}$：自回归交叉熵损失
- **感知损失** $\mathcal{L}_{per}$：二元交叉熵（分类）+ MSE（回归），$\lambda_{reg}=0.25$
- **两阶段训练**：
    - 预训练阶段：从 InstructBLIP 初始化，冻结 LLM 和 Vision Encoder，训练 Prior Tokenizer、Q-Former 和 Instructed Vision Decoder
    - 微调阶段：用 LoRA 高效微调 LLM，冻结 Vision Encoder 和 Prior Tokenizer，全量微调 Instructed Vision Decoder
- 训练配置：AdamW 优化器，weight decay 0.01，cosine 学习率调度（max 3e-4），batch size 8 × 8 V100 GPU

## 实验关键数据

### 主实验

| 方法 | LLM | ADRScore | ADRScore-S | B@4 | METEOR | ROUGE | CIDEr |
|------|-----|----------|------------|------|--------|-------|-------|
| Blip-2 | OPT-2.7B | 0.296 | 0.162 | 0.361 | 0.249 | 0.443 | 0.174 |
| Blip-2 | FlanT5-XL | 0.310 | 0.171 | 0.368 | 0.256 | 0.451 | 0.187 |
| InstructBLIP | FlanT5-XL | 0.329 | 0.187 | 0.376 | 0.269 | 0.462 | 0.196 |
| InstructBLIP | Vicuna-7B | 0.351 | 0.214 | 0.408 | 0.294 | 0.484 | 0.211 |
| MiniGPT-4 | Vicuna-7B | 0.338 | 0.203 | 0.396 | 0.286 | 0.475 | 0.219 |
| Ours | FlanT5-XL | 0.457 | 0.420 | 0.451 | 0.349 | 0.520 | 0.292 |
| **Ours** | **Vicuna-7B** | **0.463** | **0.432** | **0.457** | **0.356** | **0.529** | **0.298** |

### 消融实验

**任务组合消融**：

| Perception | Prediction | Reasoning | ADRScore | ADRScore-S |
|------------|------------|-----------|----------|------------|
| ✓ | | | 0.282 | 0.253 |
| ✓ | ✓ | | 0.297 | 0.264 |
| | | ✓ | 0.351 | 0.323 |
| ✓ | | ✓ | 0.407 | 0.364 |
| ✓ | ✓ | ✓ | **0.463** | **0.432** |

**视觉输入和感知先验消融**：

| 图像级 | 视频级 | 区域级 | 位置编码 | ADRScore | ADRScore-S |
|--------|--------|--------|----------|----------|------------|
| ✓ | | | | 0.414 | 0.379 |
| | ✓ | | | 0.431 | 0.394 |
| | ✓ | ✓ | | 0.447 | 0.418 |
| | ✓ | ✓ | ✓ | **0.463** | **0.432** |

**感知预测质量**：

| 预测类型 | 指标 | MiniGPT-4 | Kosmos-2 | Ours |
|----------|------|-----------|----------|------|
| Bounding box | Accuracy | 0.723 | 0.745 | **0.806** |
| Trajectory | ADE | 2.334 | 2.563 | **1.875** |

### 关键发现
1. **推理数据最关键**：仅用推理任务训练（0.351 ADRScore）远优于仅用感知（0.282）或感知+预测（0.297），推理数据对指令微调至关重要，但感知和预测数据额外提升了 +4.1% 和 +6.8%
2. **ADRScore 区分度远高于传统指标**：不同模型在 BLEU/CIDEr 上的差距很小（0.361→0.457），但 ADRScore 差距显著（0.296→0.463），ADRScore-S 差距更大（0.162→0.432），有效解决了评估含糊问题
3. **视觉先验编码显著提升**：区域级特征 (+2.4% ADRScore-S) 和位置编码 (+1.4%) 的逐步加入持续提升性能，证明感知先验的重要性
4. **感知准确性驱动推理质量**：目标检测准确率 80.6%，轨迹 ADE 1.875，显著优于基线——准确的感知是可靠推理的基础
5. **良好的泛化能力**：在 nuScenes 上训练迁移到 Waymo+ONCE 上，ADRScore-S 仅从 0.443 降至 0.385，而基线模型（MiniGPT-4）从 0.263 暴跌至 0.130
6. **下游规划任务受益**：预训练 Reason2Drive 后再微调控制信号预测，速度 RMSE 从 3.743 降至 2.842，转向 RMSE 从 5.926 降至 4.866，说明链式推理训练有效提升下游规划能力

## 亮点与洞察
1. **系统性贡献**：从数据集（Reason2Drive）→ 评估指标（ADRScore）→ 模型架构（Prior Tokenizer + Instructed Vision Decoder）三位一体地推进自动驾驶推理研究
2. **ADRScore 设计精妙**：将推理链评估拆为对齐/冗余/缺失三个维度，ADRScore-S 进一步用几何误差替代语义相似度评估空间推理，非常适合驾驶场景
3. **感知-推理协同增强**：Prior Tokenizer 和 Instructed Vision Decoder 形成了感知→推理→感知的闭环——推理依赖感知，推理过程中又输出更精确的感知结果
4. **数据集构建方法可复用**：object-centric database + 模板化 QA 生成 + GPT-4 增强的流程可以应用到其他需要结构化标注的领域

## 局限性 / 可改进方向
1. **自动标注的质量上限**：数据集由模板 + GPT-4 自动生成，推理链的多样性和深度受限于模板设计；更复杂的因果推理可能需要人工标注
2. **远距离目标的感知困难**：论文承认 VLM 对远距离风险目标的识别能力不足，这在高速驾驶场景中是关键安全问题
3. **自车运动的影响**：自车位移导致的相对运动容易干扰目标运动状态判断（如将停止的目标判断为移动中），需要更好的运动补偿机制
4. **评估指标的 BERT 依赖**：ADRScore 的语义相似度基于 BERT 句子嵌入，可能不是驾驶领域最优的语义模型；可尝试驾驶领域微调的编码器
5. **缺乏多传感器融合**：仅使用前视摄像头图像，真实驾驶需要激光雷达、多摄像头、GPS等多源信息融合

## 相关工作与启发
- **DriveLM** [Contributors, 2023]：构建驾驶场景 VQA 数据集，但仅覆盖感知信息缺乏推理链。Reason2Drive 在此基础上加入了完整的决策推理过程
- **LISA** [Lai et al., 2023]：将分割能力嵌入多模态 LLM，启发了 Instructed Vision Decoder 的设计——将感知能力直接集成到 VLM
- **ROSCOE** [Golovneva et al., 2022]：提出面向推理链的评估指标体系，启发了 ADRScore 的设计
- **ADriver-I** [Jia et al., 2023]：通用世界模型用于自动驾驶，启发了控制信号预测的下游验证实验
- **启发**：Reason2Drive 的链式推理框架可以扩展到其他需要多步决策的场景（如机器人操作、医疗诊断），ADRScore 的设计范式也可以迁移到评估任何链式推理任务

## 评分
- ⭐⭐⭐⭐ 新颖性：数据集和评估指标的设计有新意，Prior Tokenizer 有巧思，但模型架构整体较为常规
- ⭐⭐⭐⭐⭐ 实验充分度：主实验、多维消融（任务组合/视觉输入/解码器/编码器/泛化性/下游任务）、GPT-4 评估验证，非常全面
- ⭐⭐⭐⭐ 写作质量：论文结构清晰，图表丰富，数据集构建和评估指标的阐述详细
- ⭐⭐⭐⭐⭐ 价值：系统性贡献（数据集+指标+方法）对自动驾驶VLM研究有重要推动作用，数据集和指标可被广泛复用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MindDriver: Introducing Progressive Multimodal Reasoning for Autonomous Driving](../../CVPR2026/autonomous_driving/minddriver_introducing_progressive_multimodal_reasoning_for_autonomous_driving.md)
- [\[ECCV 2024\] Neural Volumetric World Models for Autonomous Driving](neural_volumetric_world_models_for_autonomous_driving.md)
- [\[CVPR 2026\] DriveCombo: Benchmarking Compositional Traffic Rule Reasoning in Autonomous Driving](../../CVPR2026/autonomous_driving/drivecombo_benchmarking_compositional_traffic_rule_reasoning_in_autonomous_drivi.md)
- [\[ICCV 2025\] SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](../../ICCV2025/autonomous_driving/seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)
- [\[ECCV 2024\] SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving](seflow_a_self-supervised_scene_flow_method_in_autonomous_driving.md)

</div>

<!-- RELATED:END -->
