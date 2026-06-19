---
title: >-
  [论文解读] RoadSceneVQA: Benchmarking Visual Question Answering in Roadside Perception Systems for Intelligent Transportation System
description: >-
  [AAAI 2026][自动驾驶][视觉问答] 提出 RoadSceneVQA——首个面向路侧感知场景的大规模视觉问答数据集（34,736 QA 对），并设计了 RoadMind 模型，通过认知锚点融合（CAF）和辅助解耦思维链（AD-CoT）显著提升轻量级 MLLM 在交通场景推理中的表现，在 0.9B 参数下即可超越 8B 模型。
tags:
  - "AAAI 2026"
  - "自动驾驶"
  - "视觉问答"
  - "路侧感知"
  - "多模态大语言模型"
  - "思维链推理"
  - "视觉-语言融合"
---

# RoadSceneVQA: Benchmarking Visual Question Answering in Roadside Perception Systems for Intelligent Transportation System

**会议**: AAAI 2026  
**arXiv**: [2511.18286](https://arxiv.org/abs/2511.18286)  
**代码**: [github.com/GuanRunwei/RS-VQA](https://github.com/GuanRunwei/RS-VQA)  
**领域**: 自动驾驶 / 智能交通  
**关键词**: 视觉问答, 路侧感知, 多模态大语言模型, 思维链推理, 视觉-语言融合

## 一句话总结

提出 RoadSceneVQA——首个面向路侧感知场景的大规模视觉问答数据集（34,736 QA 对），并设计了 RoadMind 模型，通过认知锚点融合（CAF）和辅助解耦思维链（AD-CoT）显著提升轻量级 MLLM 在交通场景推理中的表现，在 0.9B 参数下即可超越 8B 模型。

## 研究背景与动机

### 路侧感知的优势与现状

路侧感知相较于车载感知具有独特优势：自上而下的视角能更清晰地观察交通参与者的状态和行为，更全面地理解整体场景。然而当前路侧感知系统主要聚焦于实例级别的自动化任务（检测、跟踪、轨迹预测、交通流预测），存在以下关键问题：

**缺乏人在环的感知**：仅强调实例级别的识别，在事件级或整体场景级理解方面不足

**可扩展性和可解释性有限**：在复杂环境中灵活识别未预见的物体和事件的能力不足

**缺乏语义推理能力**：现有基准仅测量感知准确性，无法评估模型是否理解隐含的交通规则

### 现有 VQA 数据集的不足

| 问题 | 说明 |
|------|------|
| 大多以车载视角为主 | Talk2Car、NuScenes-QA、DriveLM 等均为驾驶场景 |
| 缺乏推理问题 | 多数数据集仅关注显式属性识别，不涉及交通规则推理 |
| 路侧VQA刚起步 | TUM-VideoQA 虽为路侧但不含推理问题 |

### 核心挑战

RoadSceneVQA 首次要求模型回答如"行人是否违反交通规则？"这类需要综合信号状态、空间上下文和行为动态的推理问题，这需要三重能力的融合：
- 视觉-语义定位（关联行人位置与信号状态）
- 交通规则知识内化
- 反事实因果推理（推断"如果灯是绿色，骑车人是否还会违规？"）

## 方法详解

### 整体框架

RoadMind 模型包含三个核心组件：

1. **自适应视觉编码**：将输入图像分解为 patch 序列 + 全局下采样图像，通过 InternViT 提取高层特征，Pixel Shuffle 增强特征密度
2. **CogniAnchor Fusion (CAF)**：认知锚点融合模块，语言驱动的视觉注意力
3. **AD-CoT**：辅助解耦思维链，利用 GPT-4o 生成的推理过程增强轻量级模型

最终输出：$\mathbf{A} = \mathtt{Qwen2.5}(\mathtt{Concat}(\mathtt{CAF}(\mathbf{V}, \mathbf{T}^{OC}), \mathbf{T}^{OC}))$

### 关键设计

#### 1. 人机协作标注系统（CH-MA）

**功能**：构建高质量、可扩展的 VQA 标注框架。

**三阶段流程**：
- **Stage A**：QwenVL-Max 根据定制 prompt 和路侧图像生成 4 个候选 QA 对，标注员选择最高质量的
- **Stage B**：标注员修正和精炼所选 QA 对，确保事实准确性、上下文对齐和语言清晰度
- **Stage C**：7 名标注员的质量控制面板进行投票审查，仅多数通过的样本纳入数据集

**设计动机**：解决四大挑战——人工提问的主观偏差、孤立关注单个参与者而忽略上下文、问题多样性依赖标注员专业度、模板生成缺乏自然性。

#### 2. CogniAnchor Fusion (CAF)

**功能**：解决 MLLM 中简单 token 拼接导致的视觉-语言交互问题。

**核心思路**：受人类视觉-语言协同认知启发，以文本驱动的方式预锚定潜在的感兴趣区域，采用线性注意力机制实现高效的语言驱动图像注意力：

$$\mathtt{AW}(\mathbf{Q_i}^T, \mathbf{K}) = [\phi(\mathbf{Q_i}^T)^\top \phi(\mathbf{K_1}), \ldots, \phi(\mathbf{Q_i}^T)^\top \phi(\mathbf{K_N})]^\top - \frac{1}{N}\sum_{s=1}^N \phi(\mathbf{Q_i}^T)^\top \phi(\mathbf{K_s}) + \frac{1}{N}$$

通过调整计算顺序（先聚合 key-value 对再与 query 交互），复杂度从传统 Softmax 注意力的 $O(N^2)$ 降至 $O(N)$。

**设计动机**：标准的 token 拼接有两大缺陷——(1) 背景噪声等无关视觉 token 干扰关键文本 token 的目标定位；(2) 信息交互不平衡，主导性视觉特征压制文本信号。CAF 借鉴 InLine Attention 解决线性注意力的非单射问题。

#### 3. Assisted Decoupled Chain-of-Thought (AD-CoT)

**功能**：通过知识蒸馏与认知迁移增强轻量级路侧 MLLM 的推理能力。

**核心思路**：
1. 将输入图像和 CoT 提示送入 GPT-4o，生成辅助推理上下文（包括感知推理过程和结论性答案）
2. 推理过程与原始问题拼接，作为 RoadMind 的增强输入
3. GPT-4o 的结论性答案与人工标注的真值配对，构建多任务学习目标

**损失函数**：

$$L_{\text{MTL}} = \frac{1}{\sigma_{\text{hard}}^2} \sum_{l=1}^L \log p(\mathbf{y}_l^{\text{hard}} | \mathbf{y}_{<l}^{\text{hard}}, \mathbf{x}, \mathbf{q}) + \frac{1}{\sigma_{\text{soft}}^2} \sum_{l=1}^{\min(L,L')} D_{\text{KL}}(p_l^{\text{GPT}} || \hat{p}_l) + \log \sigma_{\text{hard}} + \log \sigma_{\text{soft}}$$

其中 $\sigma_{\text{hard}}$ 和 $\sigma_{\text{soft}}$ 是两个可学习的不确定性权重。

**设计动机**：轻量级路侧 MLLM 的推理能力有限，通过 GPT-4o 作为软监督先验实现知识迁移，使小模型获得大模型的推理能力。

### 损失函数 / 训练策略

- 预训练模型微调 1 epoch，初始学习率 1e-5
- 冻结视觉编码器，解冻 LLM 和 MLP 投影器
- 输入图像 resize 为 448×448，最大序列长度 16384
- AdamW 优化器，weight decay 0.05，cosine scheduler + warm-up ratio 0.03
- 4 × A100 GPU 训练，每 GPU batch size 为 1

## 实验关键数据

### 主实验

#### RoadSceneVQA 数据集总体性能

| 模型 | LLM | 参数量 | Exact Match | CIDEr | SPICE | GPT-Score |
|------|-----|--------|-------------|-------|-------|-----------|
| MiniCPM-o 2.6 (未微调) | LLaMA3 | 8B | 0.021 | 0.661 | 0.124 | 0.428 |
| InternVL3 | Qwen 2.5 | 0.9B | 0.142 | 1.656 | 0.170 | 0.403 |
| **RoadMind** | **Qwen 2.5** | **0.9B** | **0.144** | **1.867** | **0.188** | **0.440** |
| InternVL3 | Qwen 2.5 | 2B | 0.151 | 1.834 | 0.201 | 0.465 |
| **RoadMind** | **Qwen 2.5** | **2B** | 0.142 | 1.705 | **0.219** | **0.489** |
| Qwen2.5-VL | Qwen2.5 | 7B | 0.152 | 1.689 | 0.213 | 0.497 |
| InternVL3 | Vicuna | 8B | 0.161 | 1.735 | 0.208 | 0.532 |
| **RoadMind** | **Qwen 2.5** | **8B** | 0.157 | 1.836 | **0.221** | **0.554** |

RoadMind-0.9B 的 GPT-Score（0.440）已超越未微调的 MiniCPM-o 2.6（8B, 0.428）和微调的 MobileVLM v2（1.7B, 0.417）。

#### CODA-LM 泛化性能

| 模型 | GTS↑ | 全部↑ | 车辆↑ | VRU↑ | 标志↑ | STS↑ |
|------|------|------|------|------|------|------|
| InternVL1.5-20B | 38.38 | 61.53 | 63.77 | 53.14 | 50.57 | 41.18 |
| **RoadMind-8B** | **48.50** | **70.65** | **74.25** | **59.78** | 47.43 | **54.28** |

RoadMind-8B 在 CODA-LM 上超越了 InternVL1.5-20B，证明了强大的跨场景泛化能力。

### 消融实验

#### CAF 和 AD-CoT 迁移性能（MiniCPM-o 2.6, GPT-Score）

| 配置 | GPT-Score | 说明 |
|------|-----------|------|
| 无微调 | 0.428 | 基线 |
| LoRA | 0.452 | +5.6% |
| SFT | 0.527 | +23.1% |
| SFT + CAF | 0.533 | CAF 贡献 +1.1% |
| **SFT + CAF + AD-CoT** | **0.549** | AD-CoT 贡献 +3.0% |

#### 视觉-语言融合方式对比

| 融合方式 | 参数量 | FLOPs | ROUGH-L | METEOR | SPICE |
|----------|--------|-------|---------|--------|-------|
| 纯拼接 (Concat) | - | - | 0.366 | 0.397 | 0.187 |
| LCA+Concat | 1.049M | 371.86M | 0.397 | 0.386 | 0.201 |
| CA+Concat | 1.063M | 495.41M | 0.418 | 0.422 | 0.217 |
| **CAF+Concat** | **0.924K** | **61.08M** | **0.425** | **0.411** | **0.221** |

CAF 以最少的参数（仅 924 参数）和最低的计算量（61M FLOPs）实现最优性能。

#### AD-CoT 各组件贡献

| CoT 方式 | 感知 METEOR | 感知 GPT-Score | 推理 METEOR | 推理 GPT-Score |
|----------|-----------|---------------|-----------|---------------|
| **AD-CoT（完整）** | **0.420** | **0.568** | **0.339** | **0.445** |
| 仅原始问题输入 | 0.392 | 0.523 | 0.301 | 0.401 |
| 仅GT训练 | 0.428 | 0.549 | 0.325 | 0.439 |
| 仅GPT答案训练 | 0.387 | 0.491 | 0.331 | 0.395 |
| MCoT | 0.422 | 0.536 | 0.319 | 0.431 |

### 关键发现

1. **感知 vs 推理差距显著**：RoadMind-8B 在感知类问题上 GPT-Score 达 0.53-0.60，但推理类仅 0.34-0.45，高层推理仍是挑战
2. **CAF 以极少参数实现最优**：仅 924 个参数 + 61M FLOPs，远低于传统交叉注意力的 1M+ 参数 / 495M FLOPs
3. **AD-CoT 的多任务学习关键**：同时使用 GT 答案和 GPT-4o 答案训练比任一单独使用都好
4. **小模型通过设计超越大模型**：RoadMind-0.9B > MiniCPM-o 2.6 (8B)

## 亮点与洞察

- **首个路侧推理型 VQA 数据集**：从感知识别转向法规感知的认知推理评估
- **人机协作标注效率高**：CH-MA 系统在质量和效率之间取得平衡
- **CAF 设计精巧**：线性注意力 + InLine 技巧解决了非单射问题，实现 O(N) 复杂度
- **知识蒸馏的巧妙应用**：GPT-4o 的推理过程作为输入增强，结论答案作为软标签，双重知识迁移

## 局限与展望

- 数据集场景数仅 26 个（基于 Rope3D），场景多样性有限
- 推理类问题的表现仍有较大提升空间
- AD-CoT 依赖 GPT-4o 生成思维链，训练成本较高
- 当前仅基于单帧图像，未利用视频时序信息
- 路侧视角的交通规则推理需要更丰富的法规知识注入

## 相关工作与启发

- **DriveLM/CODA-LM**：车载 VQA 的代表性工作，RoadSceneVQA 将 VQA 扩展至路侧
- **InLine Attention**：高效线性注意力的关键技术，被 CAF 采用
- **CoT 蒸馏**：GPT-4o 作为教师模型的范式在自动驾驶领域越来越流行

## 评分

- 新颖性: ⭐⭐⭐⭐ （首个路侧推理 VQA + CAF 融合设计新颖）
- 实验充分度: ⭐⭐⭐⭐⭐ （多尺度模型、多数据集、全面消融）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，图表丰富）
- 价值: ⭐⭐⭐⭐ （数据集和方法对路侧智能交通有重要推动）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] STRIDE-QA: Visual Question Answering Dataset for Spatiotemporal Reasoning in Urban Driving Scenes](stride-qa_visual_question_answering_dataset_for_spatiotemporal_reasoning_in_urba.md)
- [\[AAAI 2026\] ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts](expertad_enhancing_autonomous_driving_systems_with_mixture_of_experts.md)
- [\[AAAI 2026\] Smart: A Surrogate Model for Predicting Application Runtime in Dragonfly Systems](smart_a_surrogate_model_for_predicting_application_runtime_in_dragonfly_systems.md)
- [\[CVPR 2026\] Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule](../../CVPR2026/autonomous_driving/perception_characteristics_distance_measuring_stability_and_robustness_of_percep.md)
- [\[AAAI 2026\] SparseCoop: Cooperative Perception with Kinematic-Grounded Queries](sparsecoop_cooperative_perception_with_kinematic-grounded_queries.md)

</div>

<!-- RELATED:END -->
