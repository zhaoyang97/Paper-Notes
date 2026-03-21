# GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration

**会议**: CVPR 2025  
**arXiv**: [2503.17709](https://arxiv.org/abs/2503.17709)  
**代码**: 待确认  
**领域**: LLM Agent  
**关键词**: GUI Agent, 跨应用泛化, 探索视频, 图引导推理, 多任务评估

## 一句话总结
提出 GUI-Xplore 数据集（312 个应用、32K+ QA 对、五层级任务）和 Xplore-Agent 框架（Action-aware GUI 建模 + GUI Transition Graph 推理），通过模拟"先探索再推理"的人类策略，在陌生应用上比 SOTA GUI Agent 提升约 10% StepSR。

## 研究背景与动机
1. **领域现状**：GUI Agent 旨在自动化人-设备交互，现有方法通过大规模预训练数据（如 AITW、Mind2Web）学习通用 GUI 操作知识，或在 WebArena 等交互环境中训练。
2. **现有痛点**：两个维度的泛化不足——(a) **跨应用**：不同开发者设计的应用具有结构性差异（交互逻辑、页面层级），现有数据集忽视这些差异导致在陌生应用上性能急剧下降；(b) **跨任务**：大多数数据集只关注导航/自动化这一种任务类型，无法评估 Agent 对应用的全方位理解。
3. **核心矛盾**：缺乏一种机制让 Agent 在推理阶段获取应用特定知识。现有预训练范式只在训练阶段学习通用知识，推理时面对新应用没有上下文。
4. **本文要解决什么？** (1) 如何让 GUI Agent 快速适应陌生应用（跨应用泛化）；(2) 如何评估 Agent 的多层次 GUI 能力（跨任务泛化）。
5. **切入角度**：模拟人类使用陌生软件的方式——先探索一遍应用了解布局和功能，再基于探索获得的知识完成具体任务。
6. **核心idea一句话**：给 GUI Agent 输入一段应用探索视频作为先验知识，将视频建模为 GUI Transition Graph 辅助推理。

## 方法详解

### 整体框架
Xplore-Agent 是一个两阶段框架：(1) **Action-aware GUI Modeling** 阶段：从探索视频中提取关键帧和操作序列，将非结构化视频转化为结构化的文本探索序列；(2) **Graph-Guided Environment Reasoning** 阶段：将探索序列聚类为 GUI Transition Graph，然后用图的节点和边信息作为 prompt 输入 LLM 完成五类下游 QA 任务。

### 关键设计

1. **Action-aware Frame Extraction**:
   - 做什么：从探索视频中智能提取标记操作起止的关键帧（而非均匀采样）
   - 核心思路：利用 YUV 色彩空间的亮度差（Y-Diff）检测相邻帧间的视觉变化。GUI 操作前后通常伴随明显的界面跳转，Y-Diff 峰值精准标注操作边界。从平均 20 分钟视频中提取约 115 个关键帧（对比 1FPS 的 629 帧）
   - 设计动机：均匀采样会产生大量冗余的静态帧，而 GUI 视频的操作模式（点击→跳转→静止）天然适合基于变化检测的关键帧提取

2. **Exploration Sequence Generation（VH + Action 生成）**:
   - 做什么：将关键帧转化为文本表征——用 LVLM（基于 Pix2Struct 目标微调的 QwenVL-7B）为每个关键帧生成简化的 View Hierarchy，再通过预训练的 Action Generation 模块推断相邻关键帧间的操作类型和目标元素
   - 核心思路：图像信息→文本压缩，保留语义同时大幅降低 token 消耗
   - 设计动机：约 200 个关键帧如果直接输入 VLM 会超出上下文窗口限制，转为文本后可被 LLM 高效处理

3. **GUI Clustering + Transition Graph**:
   - 做什么：将线性的探索序列聚合为图结构。LVLM 逐帧判断新关键帧是否属于已有的页面节点（功能类别匹配），若匹配则归入该节点，否则创建新节点。最终节点代表页面聚类中心（附功能描述），边代表操作关系
   - 核心思路：利用 GPT 的语义理解能力做 online clustering，无需 VH ground truth
   - 设计动机：线性操作流无法捕捉应用的非线性页面跳转关系，图结构天然适合建模多对多的页面转换

4. **GUI-Xplore 数据集**:
   - 312 个应用（207 自动探索 + 105 人工探索），6 大类 33 子类
   - 115 小时探索视频，平均每应用 23.73 分钟
   - 32,569 个 QA 对覆盖五层级任务：Application Overview（全局功能总结）、Page Analysis（单页功能分析）、Application Usage（操作序列推理）、Action Recall（操作时序定位）、Action Sequence Verification（操作拓扑验证）
   - 所有任务统一为五选一多项选择题

### 训练策略
- Action-aware GUI Modeling 阶段微调 QwenVL-7B
- Graph-Guided Reasoning 阶段使用 GPT 做页面聚类和下游 QA（无需训练）

## 实验关键数据

### 主实验：跨应用自动化
| 方法 | Ele. Acc. | Op. Acc. | StepSR |
|------|-----------|----------|--------|
| GPT | 5.06% | 66.12% | 4.02% |
| AUTO-UI | 7.40% | 24.87% | 2.17% |
| SeeClick | 6.64% | – | 6.64% |
| CogAgent | 17.18% | 73.54% | 15.80% |
| **Xplore-Agent** | **30.73%** | **84.63%** | **30.39%** |

### 跨任务性能对比
| 方法 | Overview | Page | Usage | Recall | SeqVerify | Avg. |
|------|----------|------|-------|--------|-----------|------|
| GPT (8frame) | 96.88% | 82.12% | 66.48% | 22.6% | 28.85% | 59.39% |
| VideoTree | 89.75% | 91.05% | 65.73% | 21.70% | 21.61% | 57.97% |
| **Xplore-Agent** | **99.25%** | **92.86%** | **68.21%** | **24.36%** | **36.54%** | **64.24%** |

### 消融实验
| 配置 | Avg. Acc. | Token 数 | 说明 |
|------|-----------|----------|------|
| w/o Clustering | - | 5,144,107 | token 爆炸无法运行 |
| Rule-based Clustering | 61.87% | 63,771 | 基于 VH 和截图相似度 |
| GPT Clustering (Full) | **64.24%** | 45,199 | 更少 token 且更高精度 |

### 关键发现
- Xplore-Agent 在跨应用自动化上 StepSR 比 CogAgent 提升近 15 个百分点（15.80% → 30.39%），主要归功于探索视频提供的应用先验
- 所有模型在 Environment Understanding（Overview, Page）上表均远优于 Operational Behavior（Recall, SeqVerify），说明现有 VLM 缺乏对页面间关系和交互行为的建模能力
- 增加输入帧数反而降低端到端模型性能，表明更多输入并不总是更好，信息压缩和结构化是关键
- Action-aware 关键帧提取将帧数从 629 降至 115，在不损失性能的前提下减少 82% 计算量

## 亮点与洞察
- **"探索即适应"范式创新**：打破传统 GUI Agent 的"预训练→推理"模式，引入推理时的应用特定先验，类似人类使用新软件前先熟悉一遍的行为模式。这一范式可能比无限堆积预训练数据更高效
- **图结构建模应用环境**：GUI Transition Graph 巧妙地将线性视频流转化为结构化知识表示，且 GPT 聚类比 rule-based 聚类同时减少 token 用量和提升精度，说明 LLM 的语义理解在结构化任务中有独特优势
- **五层级任务设计**：从全局理解到局部操作、从静态页面到时序行为，构建了完整的 GUI Agent 能力评估体系

## 局限性 / 可改进方向
- 探索视频需要预先录制，真实场景中 Agent 应具备自主探索能力（explore-on-the-fly），这是更本质的挑战
- 当前只输出文本答案而非具体 GUI 操作（如坐标点击），距离端到端 GUI 自动化还有距离
- 数据收集面临隐私问题（应用截图可能包含敏感信息）
- 图聚类对 GPT 有较强依赖，开源替代方案的效果未知
- 自动探索（DroidBot）的覆盖范围有限，人工探索成本高，可考虑基于 RL 的自主探索策略

## 相关工作与启发
- **vs CogAgent**: CogAgent 通过大规模预训练获取通用 GUI 操作能力，但面对陌生应用性能差（StepSR 仅 15.80%），Xplore-Agent 通过探索先验将其提升到 30.39%
- **vs VideoTree**: 同为两阶段方法，但 VideoTree 用均匀采样+通用视频理解，而 Xplore-Agent 用 GUI 特定的关键帧提取+图推理，在 SeqVerify 上优势尤为显著（36.54% vs 21.61%）
- **vs Mind2Web/AITW**: 这些数据集提供操作轨迹但忽略应用结构差异，GUI-Xplore 通过探索视频补充应用级知识

## 评分
- 新颖性: ⭐⭐⭐⭐ 探索-推理范式和 GUI Transition Graph 概念新颖，但 two-stage pipeline 中各组件（VH 生成、GPT 聚类）较为标准
- 实验充分度: ⭐⭐⭐⭐ 跨应用和跨任务两个维度的评估全面，消融完整，但部分实验细节（如 GPT 聚类的 prompt）不够透明
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，框架图直观，但 Section 3/4 的组织略显冗余
- 价值: ⭐⭐⭐⭐ 提出了 GUI Agent 泛化的新范式和配套 benchmark，对后续研究有明确指引
