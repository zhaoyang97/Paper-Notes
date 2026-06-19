---
title: >-
  [论文解读] FSboard: Over 3 Million Characters of ASL Fingerspelling Collected via Smartphones
description: >-
  [CVPR 2025][人体理解][手语指拼] 发布 FSboard——迄今最大的 ASL 指拼（fingerspelling）识别数据集（320万字符、266小时视频、147位聋人签名者用智能手机自拍录制），聚焦手机文字输入场景，基线模型用 MediaPipe + ByT5 达到 11.1% CER，为指拼作为手机输入方式提供了坚实的数据基础。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "手语指拼"
  - "ASL识别"
  - "大规模数据集"
  - "移动端手势输入"
  - "聋人社区"
---

# FSboard: Over 3 Million Characters of ASL Fingerspelling Collected via Smartphones

**会议**: CVPR 2025  
**arXiv**: [2407.15806](https://arxiv.org/abs/2407.15806)  
**代码**: 数据集公开发布（CC BY 4.0）  
**领域**: 其他  
**关键词**: 手语指拼, ASL识别, 大规模数据集, 移动端手势输入, 聋人社区

## 一句话总结

发布 FSboard——迄今最大的 ASL 指拼（fingerspelling）识别数据集（320万字符、266小时视频、147位聋人签名者用智能手机自拍录制），聚焦手机文字输入场景，基线模型用 MediaPipe + ByT5 达到 11.1% CER，为指拼作为手机输入方式提供了坚实的数据基础。

## 研究背景与动机

**领域现状**：手语翻译（尤其是 ASL 到英语）质量在稳步提升，但距实际可用仍有很大差距。参与式 ML 方法论建议将这一宏大目标分解为能立即为聋人/听障社区带来具体收益的中间里程碑。

**现有痛点**：现有指拼数据集规模太小——之前最大的 ChicagoFSWild+ 只有 30 万字符、14 小时视频，且是从网络爬取的低分辨率裁剪片段。数据不足严重制约了模型性能。此外，大量所谓"指拼识别"研究实际只是静态手形分类，忽略了实际高速指拼中的协同发音效应和词间空格问题。

**核心矛盾**：数据规模与收集成本之间的矛盾——高质量指拼数据需要招募聋人签名者、统一硬件和场景，成本高；但模型性能严重依赖于数据规模和多样性。

**本文目标** (1) 构建一个规模比现有最大数据集大 10 倍以上的 ASL 指拼数据集；(2) 将数据集聚焦在手机文字输入这一实际应用场景上；(3) 提供高质量基线模型证明数据集的有效性。

**切入角度**：受 Hassan et al. 用户研究启发——聋人用指拼输入手机文本的速度（42.5 wpm）比触屏键盘（31.9 wpm）更快且错误更少。这说明指拼键盘有真实的应用价值，但需要大规模数据来训练可靠的识别模型。

**核心 idea**：通过招募 147 位聋人用 Pixel 4A 手机自拍录制指拼构建迄今最大 ASL 指拼数据集，覆盖 MacKenzie 短语、URL、地址、电话号码、姓名等手机文字输入场景。

## 方法详解

### 整体框架

FSboard 的核心贡献是数据集本身。工作流程为：(1) 设计面向手机文字输入场景的短语分布；(2) 通过 DPAN（Deaf Professional Arts Network）招募 147 位以 ASL 为主语言的聋人参与者；(3) 用 Pixel 4A 前置摄像头（1944×2592, 30fps）在多样环境下录制一只手的指拼视频；(4) 通过多轮自举（bootstrapping）方法清洗数据边界；(5) 划分为无重叠的训练/验证/测试集。基线模型采用 MediaPipe Holistic 提取 85 个 3D 手势关键点，线性投影后送入 ByT5-Small (300M) 字符级编码器-解码器模型进行序列到序列的指拼识别。

### 关键设计

1. **多领域短语设计**:

    - 功能：构建贴近真实手机文字输入场景的短语分布
    - 核心思路：包含五类短语——MacKenzie 经典文本输入评测短语（500 个，被多名签名者重复录制作为 sanity check）、随机生成的 URL（从网页爬虫提取真实 URL 部分后随机组合）、随机美国街道地址（基于 Census Bureau TIGER 数据）、随机人名（最常见 1000 名/姓组合）、随机电话号码（包含美国和国际格式）
    - 设计动机：手机文字输入不仅仅是普通句子，地址、姓名、URL 等是聋人社区特别期待的应用场景（如在 Google Maps 中输入地址）。数字签名虽不完全等同指拼，但在实际场景中与指拼密切相关

2. **自举数据清洗流程**:

    - 功能：修正由数据采集 App bug 导致的片段边界不准确问题
    - 核心思路：先用 YouTube 手语视频预训练 ByT5，然后 5 折交叉验证方式对 FSboard 进行预测——模型预测与标注一致的片段标记为"干净"，不一致的标为"噪声"。重复此过程三次，每次只在"干净"数据上训练。之后再辅以手动编辑和参与者特定规则进一步清洗
    - 设计动机：由于采集 App 的"错误"按钮bug，时间戳标注与实际指拼不匹配非常普遍，人工标注成本太高，自举方法利用模型自身能力逐步筛选干净数据

3. **基线模型：MediaPipe + ByT5**:

    - 功能：提供指拼识别基线性能
    - 核心思路：用 MediaPipe Holistic 在 30Hz 下提取 85 个 3D 关键点（手 + 姿态 + 面部），线性投影到 ByT5-Small 的编码器输入空间，每帧一个 soft token。最长 256 帧输入、256 字符输出，beam search (beam=5) 解码。在 32 TPUv3 上训练 200K 步
    - 设计动机：关键点输入比直接处理视频轻量很多，适合未来在手机端部署。ByT5 字符级模型天然适合指拼（逐字母识别），预训练知识带来巨大提升

### 损失函数 / 训练策略

标准编码器-解码器序列到序列训练，使用 Adafactor 优化器，学习率 0.001，batch size 64。在验证集 CER 上选择最优 checkpoint。

## 实验关键数据

### 主实验

| 模型/配置 | CER↓ | Top-1 Accuracy↑ |
|-----------|------|-----------------|
| **ByT5-Small + MediaPipe (30Hz)** | **11.1%** | **52.9%** |
| Kaggle 竞赛最佳 | 16.4% | - |
| ChicagoFSWild+ 基线 | 37.7% | - |
| ChicagoFSWild+ 人类表现 | 13.9% | - |

### 消融实验

| 配置 | CER↓ | Top-1 Accuracy↑ | 说明 |
|------|------|-----------------|------|
| ByT5 预训练 (300M) | 11.1% | 52.9% | 完整基线 |
| ByT5 从头训练 | 33.8% | 17.9% | 预训练知识贡献 22.7% CER |
| ByT5 Base (580M) | 13.3% | 49.1% | 更大模型反而过拟合 |
| 30Hz → 15Hz | 11.8% | 51.8% | 帧率减半 CER 仅涨 0.7% |
| 30Hz → 5Hz | 20.0% | 33.4% | 帧率太低急剧恶化 |
| 去掉 Face 关键点 | 12.0% | 50.6% | 面部唇读线索有小幅帮助 |
| 去掉 Face + Pose | 12.5% | 49.7% | 仅用手部关键点仍表现良好 |

### 关键发现

- **预训练是最关键因素**：ByT5 预训练 vs 从头训练的 CER 差距为 22.7%（11.1% vs 33.8%），且收敛速度快约 7×
- **帧率降低到 15Hz 几乎无损**（11.1% → 11.8%），对设备端实时部署非常有利
- **面部/姿态关键点贡献较小**（去掉后只涨 1.4% CER），说明指拼识别的核心信息确实在手部，面部可能只提供了辅助的唇读线索
- 基线已**超越 ChicagoFSWild+ 的人类表现**（11.1% < 13.9%），但这主要因为 FSboard 的任务设定是孤立指拼而非从连续签名中裁剪

## 亮点与洞察

- **社区驱动的数据集设计**是本文最重要的理念：3 位作者是聋人社区成员，选题、试点测试和招募全程有社区参与，数据集聚焦于真实需求（手机文字输入）而非学术炫技。这种"先解决实际需求再推进更大目标"的增量式参与方法值得所有 HCI/AI 研究借鉴。
- **MacKenzie 短语的指拼速度远超手机打字**（65 wpm vs 36 wpm），部分签名者超 100 wpm，这一数据强有力地支撑了指拼键盘的应用价值。
- **自举数据清洗流程**巧妙地将模型自身当做标注工具，虽然不如人工标注精确但成本极低且可迭代改善，适用于所有远程众包数据收集场景。

## 局限与展望

- 数据集短语分布较窄——只覆盖工程地址、URL、姓名等几类，缺乏自由文本和日常对话内容
- 大写/标点处理不够明确——指拼中大小写在少数场景才有区分，但文字输入需要，需要更明确的采集协议
- 基于 MediaPipe 的关键点提取在遮挡和极端角度下可能不可靠，未来应探索直接视频建模
- 测试集仍是与训练集使用相同合成语法的短语，独立的真实查询测试集更能反映实际性能
- 签名者数量（147 人）远少于 ChicagoFSWild+（260 人），种族/性别多样性虽合理但仍有偏斜（男性表现不足）

## 相关工作与启发

- **vs ChicagoFSWild+**: ChicagoFSWild+ 从 YouTube 爬取低分辨率视频，平均序列长度仅 5.5 字符；FSboard 使用高分辨率手机自拍，平均序列长度 21.2 字符。规模差异 10×+
- **vs PopSign**: PopSign 是孤立手语识别的教育游戏数据集，词汇量 250 个手语，与指拼序列识别任务不同但共享 Pixel 4A 自拍采集方式
- **vs ASL Citizen**: ASL Citizen 聚焦字典检索设置下的手语识别（2731 个手语），使用 top-N 检索而非序列转录。FSboard 的字符级序列到序列任务更具挑战性

## 评分

- 新颖性: ⭐⭐⭐ 核心贡献是数据集而非方法创新，但数据集的规模和设计理念本身就是重要贡献
- 实验充分度: ⭐⭐⭐⭐ 提供了全面的消融（帧率、关键点组件、模型大小、预训练），定性分析也充分
- 写作质量: ⭐⭐⭐⭐⭐ 论文对社区背景、伦理考量、数据集设计动机的阐述非常详尽和负责
- 价值: ⭐⭐⭐⭐ 为 ASL 指拼识别和辅助技术提供了宝贵资源，社区驱动的方法论有示范意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Scaling Large Motion Models with Million-Level Human Motions](../../ICML2025/human_understanding/scaling_large_motion_models_with_million-level_human_motions.md)
- [\[CVPR 2026\] OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis](../../CVPR2026/human_understanding/openfs_multi-hand-capable_fingerspelling_recognition_with_implicit_signing-hand_.md)
- [\[CVPR 2025\] GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior](gaussianip_identity-preserving_realistic_3d_human_generation_via_human-centric_d.md)
- [\[CVPR 2025\] Probabilistic Prompt Distribution Learning for Animal Pose Estimation](probabilistic_prompt_distribution_learning_for_animal_pose_estimation.md)
- [\[CVPR 2025\] Quaffure: Real-Time Quasi-Static Neural Hair Simulation](quaffure_real-time_quasi-static_neural_hair_simulation.md)

</div>

<!-- RELATED:END -->
