---
title: >-
  [论文解读] STRIDE-QA: Visual Question Answering Dataset for Spatiotemporal Reasoning in Urban Driving Scenes
description: >-
  [AAAI 2026 Oral][自动驾驶][VQA] 构建了自动驾驶领域最大规模时空推理VQA数据集STRIDE-QA（270K帧、16M QA对），定义了三类时空推理任务（物体间空间/自车空间/自车时空），通过微调VLM使空间定位成功率从近零提升至55%、时序一致性从0提升至28.4%。 领域现状：VLM（如GPT-4o…
tags:
  - "AAAI 2026 Oral"
  - "自动驾驶"
  - "VQA"
  - "时空推理"
  - "自车视角"
  - "3D标注"
  - "VLM微调"
---

# STRIDE-QA: Visual Question Answering Dataset for Spatiotemporal Reasoning in Urban Driving Scenes

**会议**: AAAI 2026 Oral  
**arXiv**: [2508.10427](https://arxiv.org/abs/2508.10427)  
**代码**: [turingmotors/stride-qa](https://turingmotors.github.io/stride-qa/)  
**领域**: 自动驾驶 / 视觉问答 / 时空推理  
**关键词**: VQA, 时空推理, 自车视角, 3D标注, VLM微调

## 一句话总结
构建了自动驾驶领域最大规模时空推理VQA数据集STRIDE-QA（270K帧、16M QA对），定义了三类时空推理任务（物体间空间/自车空间/自车时空），通过微调VLM使空间定位成功率从近零提升至55%、时序一致性从0提升至28.4%。

## 研究背景与动机

**领域现状**：VLM（如GPT-4o、Qwen2.5-VL）已被应用于自动驾驶场景理解和决策支持。现有驾驶VQA数据集包括nuScenes-QA（460K QA对）、nuPlanQA（1M QA对）等。

**核心痛点**：现有VLM训练于静态网页图文对，缺乏精确的时空推理能力。已有驾驶VQA数据集要么只支持物体中心视角，要么缺乏时间维度的3D对齐标注，且规模有限。

**关键gap**：没有一个数据集能同时支持物体间空间推理、自车空间推理和自车时空推理（包含未来预测）这三类任务。现有方法在时空预测一致性上得分接近零。

**本文切入**：利用东京100小时驾驶数据，通过全自动标注pipeline（BEVFusion 3D检测 + PubTracker跟踪 + SAM 2.1分割 + 可见性过滤），构建16M QA对的大规模时空推理VQA数据集。

**核心idea**：为VLM提供大规模、物理grounded的时空推理监督信号，使其具备从"看图说话"到"预测未来运动"的能力跃迁。

**传感器配置**：64通道LiDAR + 6个相机（前后2880×1860、侧面1920×1240、60° FOV、360°覆盖）+ IMU + RTK-GNSS，20秒clip分割。

## 方法详解

### 整体框架
多传感器驾驶数据（6相机+64线LiDAR，2Hz同步） → 全自动标注Pipeline → QA对生成 → VLM微调。标注Pipeline包含7个模块：关键帧采样、3D目标检测、多目标跟踪、属性提取、语义分割、可见性过滤、问题生成。

数据规模统计：训练集含5.30M定性QA + 10.17M定量QA，验证集含0.28M定性 + 0.69M定量。评估集由409个场景组构成，包含5317个QA对，覆盖6种动态交互场景。

### 关键设计一：三类VQA任务定义
- **物体间空间QA (Object-centric Spatial)**：基于单帧，判断两个非自车智能体之间的空间关系（定性问题如相对位置、定量问题如"距离1.53m"）
- **自车空间QA (Ego-centric Spatial)**：基于单帧，描述智能体相对于自车的距离、朝向、大小
- **自车时空QA (Ego-centric Spatiotemporal)**：输入2Hz采样的4帧上下文（t∈{-1.5,-1.0,-0.5,0}s），预测未来t∈{1,2,3}s时刻目标智能体的距离、航向角和速度

### 关键设计二：全自动标注Pipeline
7个模块化组件：(1) 1Hz关键帧采样（每clip 2-17s窗口）；(2) BEVFusion激光雷达-相机融合3D检测（45类物体，mAP=0.701，位置误差仅13.6cm）；(3) PubTracker基于点的3D跟踪（无需外观特征，AMOTA=0.676）；(4) 属性提取器（自车坐标系下的欧氏距离、航向角、速度）；(5) SAM2.1 Large分割（逐实例2D掩码）；(6) 可见性过滤（IoU>0.3 + 覆盖率>0.8 + SAM置信度>0.8 三重过滤）；(7) 模板化QA生成（Region [X]引用标识）。

### 关键设计三：评估指标体系
- **LSR (Localization Success Rate)**：距离误差<±25% GT值 且 航向误差<±10° 时为成功
- **MLSR (Mean LSR)**：序列级LSR均值，衡量时序稳定性
- **TLC (Temporal Localization Consistency)**：所有4个时间步都成功的严格一致性度量

### 损失函数
微调采用LoRA（rank=16, alpha=32, dropout=0.05），在16块H100上训练2个epoch。学习率5e-5（LLM）/2e-6（视觉编码器）/1e-5（投影头），全局batch size=64。优化器为AdamW（β1=0.9, β2=0.999, weight decay=0.1），cosine scheduler + 5% warmup。推理温度设为0以确保确定性输出。
输入为4帧前视图（532×336分辨率）+ 由Set-of-Mark方法生成的Region ID标注掩码。

## 实验

### 主实验：STRIDE-QA Bench时空推理

| 模型 | LSR@0s | LSR@1s | LSR@2s | LSR@3s | MLSR | TLC |
|------|--------|--------|--------|--------|------|-----|
| GPT-4o | 18.1 | 6.6 | 6.1 | 7.6 | 9.6 | 0.7 |
| GPT-4o mini | 4.6 | 2.0 | 0.7 | 0.7 | 2.0 | 0.0 |
| Qwen2.5-VL-7B | 1.0 | 3.4 | 4.4 | 1.0 | 2.4 | 0.0 |
| SpatialRGPT-8B | 0.5 | 0.2 | 0.2 | 0.0 | 0.2 | 0.0 |
| Cosmos-Reason1-7B | 1.5 | 3.2 | 2.0 | 1.5 | 2.0 | 0.0 |
| **STRIDE-Qwen2.5-VL-7B** | **96.3** | **46.2** | **38.4** | **38.9** | **55.0** | **28.4** |
| STRIDE-Cosmos-Reason1-7B | 96.8 | 43.5 | 37.4 | 36.2 | 53.5 | 25.4 |

### 消融：SpatialRGPT-Bench空间推理

| 模型 | 原始定性↑ | 原始定量↑ | Obj空间定量↑ | Ego空间定性↑ | Ego空间定量↑ |
|------|-----------|-----------|-------------|-------------|-------------|
| GPT-4o | 80.5 | 32.5 | 39.4 | 55.7 | 27.7 |
| Qwen2.5-VL-7B | 67.2 | 24.4 | 12.8 | 47.1 | 29.3 |
| STRIDE-Qwen2.5-VL-7B | 69.5 | 37.5 | **61.5** | **77.9** | **70.3** |
| STRIDE-Cosmos-Reason1-7B | 71.1 | 30.0 | 58.7 | 79.9 | 68.9 |

### 关键发现
1. **通用VLM在时空推理上几乎完全失败**：所有基线模型TLC≈0，说明缺乏时序一致性推理能力
2. **微调效果巨大**：STRIDE-Qwen2.5-VL-7B的LSR@0s从1.0%跃升至96.3%（96倍提升），MLSR达55.0
3. **出视野预测是主要瓶颈**：Maintain State场景（目标在视野内）LSR衰减平缓，Oncoming Pass等OOV场景LSR急剧下降，是TLC仅28.4%的主因
4. **跨域迁移有效**：在外部SpatialRGPT-Bench上，Ego空间定量从29.3%跃升至70.3%（+41pt）
5. **基线模型行为分析**：基线VLM预测稀疏且系统性偏差，重复相似的错误猜测，说明其默认了一种简单的记忆行为而非基于视觉上下文推理
6. **六类动态场景差异显著**：Oncoming Pass（OOV率100%）和Maintain State（OOV率5%）的LSR衰减模式截然不同，验证了OOV是性能下降的主因
7. **数据集统计丰富**：vehicle类占主体（440K实例），pedestrian（139K）、large vehicle（119K）也有充足覆盖

## 亮点
- 16M QA对 × 270K帧：目前自动驾驶时空推理VQA领域最大规模数据集
- 首个同时支持物体间、自车空间、自车时空三类推理的统一框架
- 完全自动的标注Pipeline，3D检测误差仅13.6cm，可扩展到更大规模
- LSR/MLSR/TLC三层指标递进评估，精准量化VLM时空推理能力的各维度
- 数据采集覆盖东京多种场景：交通拥堵、施工区域、行人密集十字路口
- 隐私保护完善：Dashcam Anonymizer自动模糊人脸和车牌

## 局限性
1. **单前视野评估**：仅使用前视图（60° FOV），目标离开视野后预测能力急剧下降，多相机融合的效果未探索
2. **仅LoRA微调**：受限于计算资源，未探索全参数微调的性能上界
3. **缺少下游任务评估**：未验证时空推理能力对运动规划、行为预测等安全关键任务的收益
4. **无跨数据集泛化**：作为首个时空推理benchmark，暂无可比的公开数据集进行交叉验证
5. **模板化QA局限性**：QA对由模板生成，缺乏自然语言多样性，可能限制VLM的语言理解泛化
6. **速度估计误差较大**：检测pipeline的mAVE=1.28 m/s，在速度维度的QA质量存在一定噪声

## 相关工作
- **空间感知VLM**：SpatialVLM（CVPR 2024）、Spatial-RGPT（NeurIPS 2024）——静态场景空间推理，缺乏时间维度
- **驾驶VQA数据集**：nuScenes-QA（AAAI 2024, 460K对）、nuPlanQA（2025, 1M对）、TUMTraffic-VideoQA（ICML 2025, 87.3K）——规模和任务覆盖均不如STRIDE-QA
- **驾驶VLM**：DriveLM（ECCV 2024）、Senna-VLM、Cosmos-Reason1-7B（NVIDIA 2025）——用于场景理解和高级决策
- **ToD3Cap/NuPrompt**：聚焦自车空间描述和物体referring，但缺乏时空预测任务
- **Refer-KITTI**：仅6小时、818条referring表达式，规模远不足

## 评分
⭐⭐⭐⭐ — 数据集规模和标注pipeline质量扎实，三类任务定义清晰有实际意义。核心发现（通用VLM时空推理≈0、微调后巨大提升）具有重要参考价值。局限在于仅前视野评估，且未验证对下游规划的实际收益。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] RoadSceneVQA: Benchmarking Visual Question Answering in Roadside Perception Systems for Intelligent Transportation System](roadscenevqa_benchmarking_visual_question_answering_in_roadside_perception_syste.md)
- [\[AAAI 2026\] Understanding Dynamic Scenes in Egocentric 4D Point Clouds](understanding_dynamic_scenes_in_ego_centric_4d_point_clouds.md)
- [\[CVPR 2026\] HybridDriveVLA: Vision-Language-Action Model with Visual CoT reasoning and ToT Evaluation for Autonomous Driving](../../CVPR2026/autonomous_driving/hybriddrivevla_vision-language-action_model_with_visual_cot_reasoning.md)
- [\[CVPR 2026\] TopoHR: Hierarchical Centerline Representation for Cyclic Topology Reasoning in Driving Scenes with Point-to-Instance Relations](../../CVPR2026/autonomous_driving/topohr_hierarchical_centerline_representation_for_cyclic_topology_reasoning_in_d.md)
- [\[AAAI 2026\] Fine-Grained Representation for Lane Topology Reasoning](fine-grained_representation_for_lane_topology_reasoning.md)

</div>

<!-- RELATED:END -->
