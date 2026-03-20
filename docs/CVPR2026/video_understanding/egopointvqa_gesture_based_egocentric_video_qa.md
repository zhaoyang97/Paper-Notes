# EgoPointVQA: Gesture-Based Egocentric Video Question Answering

**会议**: CVPR 2026  
**arXiv**: [2603.12533](https://arxiv.org/abs/2603.12533)  
**代码**: 待公开（作者承诺 release code/model/dataset）  
**领域**: 第一人称视频理解 / 多模态问答 / 手势理解  
**关键词**: egocentric VQA, pointing gesture, deictic reasoning, hand intent tokens, MLLM  

## 一句话总结
提出 EgoPointVQA 数据集（4000 合成 + 400 真实第一人称视频）和 HINT 方法，通过 3D 手部关键点编码为手势意图 token 并与视觉 token 交织输入 MLLM，使模型能理解用户指向手势并回答指示性问题，HINT-14B 达到 68.1% 准确率，超越 InternVL3-14B 6.6 个百分点。

## 背景与动机
随着 AR/VR 设备和智能眼镜（Apple Vision Pro、Meta Ray-Ban）的普及，第一人称 AI 助手需要理解用户通过指向手势和指示代词（"this"、"that"）表达的空间引用。但现有 MLLM 在此任务上严重不足：(1) 训练数据中缺少手势丰富的第一人称视频；(2) 架构上没有显式编码手势信息的机制，全局整合视觉和文本输入，无法将指示表达与手指指向的物体关联。即使 GPT-4o 在此任务上仅 46.8% 平均准确率，GPT-5 也只有 62.6%。

## 核心问题
如何让 MLLM 从第一人称视频中理解用户的指向手势，并正确回答包含指示代词的问题？

## 方法详解

### 整体框架
EgoPointVQA 系统包含两部分：(1) 数据集与评估基准——定义 6 类指示推理任务，构建合成+真实视频及多选 QA 对；(2) HINT 方法——在标准 MLLM 的视觉流之外增加手势意图流，将 3D 手部关键点编码为 token 与视觉 token 交织输入。

### 关键设计

1. **EgoPointVQA 数据集**：
   - **合成视频**：4000 个，用 AI2-THOR 模拟器生成，184 个室内场景，12000 个视点，MIXAMO 动画 + 逆运动学对齐指尖与目标物体，448×448 30FPS
   - **真实视频**：400 个，20 名参与者（12 国籍）使用 Meta Ray-Ban 眼镜拍摄，1536×2048 30FPS，3-8 秒
   - **6 类任务**：Reference（识别物体）、Counting（计数同类）、Spatial（相对位置/深度）、Temporal（多手势时序）、Attribute（颜色/形状/材质）、Feedback（功能/适用性）
   - **QA 生成**：三阶段管线——Stage 1 用 InternVL3-78B 提取密集场景信息 → Stage 2 生成结构化多选 QA → Stage 3 用 GPT-4o 改写为指示代词形式
   - 训练集：18073 QA（合成全部 + 100 真实视频的 640 QA），测试集：672 QA（300 真实视频）

2. **HINT (Hand Intent Tokens)**：
   - **3D 手部姿态提取**：每帧用 WiLoR（鲁棒的 in-the-wild 手部重建模型）提取 21 个 3D 关键点 K_t ∈ ℝ^{21×3}
   - **Keypoint Adapter**：将 63 维特征（flatten 后）经 LayerNorm → W₁(63→d_h) → GeLU → W₂(d_h→d) 映射为与 LLM 同维度的单个 Hand Intent Token H_t。检测置信度 < τ=0.5 时不插入 token
   - **Frame-Keypoint 交织**：在输入序列中，每帧的视觉 token 后紧跟该帧的 H_t，使 LLM 在自回归生成中能同时 attend 视觉和手势信息

3. **训练策略**：仅训练 Keypoint Adapter + LoRA 微调视觉编码器和 LLM，backbone 参数冻结。AdamW + cosine schedule + warmup 0.03，batch size 32，1 epoch，混合合成+真实数据

### 损失函数 / 训练策略
标准自回归语言模型损失：p(X_a | V, X_q, H) = Π p(x_i | V, X_{q,<i}, X_{a,<i}, H_{<i})，其中 H 显式提供手势条件信号。

## 实验关键数据

| 方法 | 规模 | Refer. | Temporal | Spatial | Count | Attr. | Feed. | Avg |
|------|------|--------|----------|---------|-------|-------|-------|-----|
| GPT-5 | - | 75.6 | 53.6 | 62.3 | 50.0 | 56.1 | 77.8 | 62.6 |
| GPT-4o | - | 56.1 | 29.5 | 43.1 | 44.8 | 41.5 | 65.7 | 46.8 |
| InternVL3 | 14B | 63.1 | 66.1 | 61.4 | 50.0 | 58.5 | 77.2 | 62.7 |
| **HINT** | **14B** | **73.8** | **69.6** | **64.9** | **54.2** | **63.4** | **82.5** | **68.1** |
| InternVL3 | 8B | 66.1 | 57.5 | 63.2 | 33.3 | 51.3 | 76.8 | 58.0 |
| **HINT** | **8B** | **75.0** | **66.1** | **64.9** | **35.4** | **61.0** | **79.8** | **63.7** |

- HINT-14B 平均 68.1%，超过 InternVL3-14B 5.4 个百分点
- 人类表现 95.9%，仍有 ~28% 的差距
- HINT token 仅占总 token <1%，推理时间从 2.58s 增加到 2.84s（+10%）
- 在标准视频理解 benchmark（Video-MME/MVBench/EgoSchema）上性能与 baseline 持平，无灾难性遗忘

### 消融实验要点
- 仅 SFT（无 HINT）：Reference 从 66.1→68.5，加 HINT 后→75.0，说明数据+架构缺一不可
- 合成+真实数据最优（75.0%），仅用合成 69.0%，仅用真实 67.3%
- 手势建模方式对比：可视化关键点 57.1%，可视化箭头 70.2%，文本关键点 68.5%，**HINT 75.0%**——说明让模型自学手部几何信息优于人工编码
- 置信度阈值 τ=0.5 最优；τ=0.7 过严（64.9%），τ=0.1 太松（66.7%）
- 移除手势后性能暴跌至 41.7%（Reference），证明手势是核心线索

## 亮点
- 填补了重要研究空白：指向手势驱动的第一人称 VQA 此前几乎没有研究
- HINT 设计简洁有效：仅一个 2 层 MLP adapter + 交织策略，几乎不增加 token 开销
- 数据集构建管线完整：AI2-THOR+IK+InternVL3-78B+GPT-4o，可复现性好
- 消融实验全面：对比了 5 种手势建模方式、阈值、帧采样、数据组成等
- 偏差分析严谨：text-only 和 choices-only baseline 均接近随机，证明无捷径

## 局限性 / 可改进方向
- 手势仅限指向，未扩展到抓取、挥手等其他交互手势
- WiLoR 在运动模糊和遮挡下关键点不准，是主要失败模式
- 数据集规模有限（672 测试 QA），统计置信度可能不足
- 合成数据与真实场景仍有域差距
- 仅 multiple-choice 评估，未测试开放式回答

## 与相关工作的对比
- vs EgoGPT/Ego-R1：后者关注长期记忆和习惯分析的第一人称 VQA，不处理手势引用
- vs Ferret/Osprey/DAM：这些区域级 VQA 方法需要显式给定 bbox/mask，HINT 从自然手势推断
- vs Set-of-Mark/ViP-LLaVA：依赖人工视觉标注（标签、涂鸦），HINT 使用自然手势信号
- vs VGLLM-QA：利用 3D 几何先验但不处理手势，在本任务上仅 48.9%

## 启发与关联
- 将 off-the-shelf 手部重建模型的输出轻量编码为 token 的思路可推广到其他 body language 理解
- 手势 token 与 visual token 的交织策略类似于多模态 token mixing 范式
- 对 AR/VR 交互和辅助技术（视障辅助）有直接应用价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个手势驱动的第一人称 VQA 数据集和方法，问题定义新颖
- 实验充分度: ⭐⭐⭐⭐ 15 个 baseline 对比、多种消融、偏差分析、人类表现对比
- 写作质量: ⭐⭐⭐⭐ 任务定义清晰，图表丰富，数据集构建细节充分
- 价值: ⭐⭐⭐⭐ 对 AR/VR 交互和具身 AI 有重要推动作用
