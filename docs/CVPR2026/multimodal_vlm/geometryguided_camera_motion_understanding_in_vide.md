# Geometry-Guided Camera Motion Understanding in VideoLLMs

**会议**: CVPR 2026  
**arXiv**: [2603.13119](https://arxiv.org/abs/2603.13119)  
**代码**: 待发布  
**领域**: 视频理解 / 视觉语言模型 / 3D视觉  
**关键词**: 相机运动理解, VideoLLM, 几何引导, 3D基础模型, 运动原语, 结构化提示  

## 一句话总结
通过 benchmarking-diagnosis-injection 框架系统揭示 VideoLLM 的相机运动盲区，并利用冻结 3DFM (VGGT) 提取几何线索 + 轻量时序分类器 + 结构化提示注入，无需微调即可显著提升 VideoLLM 的细粒度相机运动理解。

## 背景与动机
相机运动 (pan/tilt/dolly 等) 是电影语法的核心几何信号，直接影响叙事、注意力引导和空间布局表达。然而现有 VideoLLM 主要针对高层语义 (物体识别、动作理解) 优化，缺少显式的相机运动监督。实验发现，多数 VideoLLM 在相机运动 VQA 上的准确率接近随机猜测 (25%)，说明这一关键信号被严重忽视。更值得注意的是，对 CameraBench 进行专门微调的模型甚至比原始 Qwen2.5-VL 表现更差，揭示了常规微调路线的问题。

## 核心问题
1. VideoLLM 在细粒度相机运动原语识别上系统性失败，原因是什么？
2. 如何在不修改 VideoLLM 权重的前提下注入可靠的相机运动信息？

## 方法详解

### 整体框架
三阶段流程: (1) 从冻结的 3DFM (VGGT) 提取逐帧 camera token; (2) 轻量 Transformer 时序分类器预测约束感知的运动标签; (3) 将逐秒运动标签序列作为结构化提示注入 VideoLLM 推理。整个管线即插即用、与 VideoLLM 无关。

### 关键设计
1. **CameraMotionDataset**: 基于 ReCamMaster 的 MultiCamVideo 构建 12,274 个 1 秒片段，每段从精确外参矩阵确定性标注 15 种原子运动原语 (pan-left/right, tilt-up/down, dolly-in/out 等)，人工验证一致率 93%。
2. **约束感知多标签分类**: 定义 15 类原语间的互斥矩阵 $\mathbf{M} \in \{0,1\}^{K \times K}$，训练时引入不兼容正则 $\mathcal{L}_{inc} = \sum M_{ij} p_i p_j$ 和基数正则 $\mathcal{L}_{card}$，确保预测组合物理上合理。
3. **Probing 诊断**: 对 Qwen2.5-VL 冻结 ViT 各层用 Q-Former 探针读取相机运动信号，发现第 7 层 (第一个全注意力层) 性能最高，之后随深度递减——说明 token 压缩和语义对齐训练抹掉了运动线索。
4. **VGGT-Q-Former 蒸馏**: 将 1.2B 参数的 VGGT camera token 蒸馏到仅 8.72M 参数的 Q-Former，吞吐量提升 5.3×，峰值显存降至 39%，instance accuracy 仅下降 8.13%。

### 损失函数 / 训练策略
- 主损失: BCE $\mathcal{L}_{bce}$
- 约束正则: $\mathcal{L}_{inc} = \sum M_{ij} p_i p_j$ 惩罚互斥原语共现
- 基数正则: $\mathcal{L}_{card}$ 限制每段预测 1~3 个标签
- 蒸馏: MSE 回归 loss $\mathcal{L}_{reg} = \sum \|\tilde{c}_t - c'_t\|^2$，三阶段渐进训练

## 实验关键数据

| 方法 | Instance Acc | Macro-F1 | Weighted-F1 |
|------|-------------|----------|-------------|
| VGGT + 约束 | **0.738** | **0.87** | **0.92** |
| VGGT 无约束 | 0.572 | 0.79 | 0.84 |
| VGGT-Q-Former 蒸馏 | 0.638 | 0.83 | 0.87 |
| Q-Former probing | 0.450 | 0.69 | 0.74 |

- 多数现成 VideoLLM 在 CameraMotionVQA 上接近随机准确率 25%
- 注入运动标签后，VideoLLM 描述从模糊运动语句转变为含方向、时序结构的影视叙事风格

### 消融实验要点
- 去掉约束正则，instance accuracy 从 73.8% 降至 57.2%，说明互斥约束至关重要
- Probing 实验: ViT 第 7 层 (浅层全注意力) 运动信号最强，到第 31 层几乎消失
- 蒸馏 vs 完整 VGGT: 吞吐量 23.36 vs 4.39 samples/s，精度损失可控
- 时序卷积 vs 平均池化: 去掉时序建模后精度明显下降

## 亮点 / 我学到了什么
- 用 probing 定量诊断 "信息在哪丢失" 是理解大模型瓶颈的优秀方法论
- 约束感知标签体系设计巧妙: 互斥矩阵 + 基数正则从损失函数层面保证预测的物理合理性
- 结构化提示注入无需训练权重即可改变模型的推理行为模式

## 局限性 / 可改进方向
- 数据集为合成数据 (UE5 渲染)，真实视频上的泛化性有待验证
- 仅覆盖外参运动 (pan/tilt/dolly)，zoom 等内参变化未处理
- 仅探索了 VGGT 一个 3DFM backbone，未对比其他几何模型
- static 类预测不可靠，VGGT 的重建先验假设相机运动，静态段可能 OOD

## 与相关工作的对比
- **CameraBench**: 提供原语级运动标注和 VQA 评估，但标签来自人工标注且无精确相机参数；本文用外参确定性标注更可靠
- **SpatialVID**: 提供逐帧深度和 pose 驱动的运动指令，但用于视频生成而非理解；本文反向利用几何信号增强理解
- **Shot-by-Shot**: 用 shot-level 电影语法线索引导描述生成，但不涉及原语级运动识别

## 评分
- 新颖性: ⭐⭐⭐⭐ (诊断-注入框架设计完整，约束多标签分类新颖)
- 实验充分度: ⭐⭐⭐⭐ (benchmarking + probing + 蒸馏 + 定性分析)
- 写作质量: ⭐⭐⭐⭐ (逻辑清晰，图表专业)
- 价值: ⭐⭐⭐ (面向影视视频理解场景)
