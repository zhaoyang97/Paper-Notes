---
title: >-
  [论文解读] STING-BEE: Towards Vision-Language Model for Real-World X-ray Baggage Security Inspection
description: >-
  [CVPR 2025][多模态VLM][X射线安检] 构建了首个多模态X射线行李安全数据集**STCray**（46,642张图像-描述对，21类威胁含IED和3D打印枪），设计**STING协议**系统生成领域感知的高质量描述，并训练领域特化VLM **STING-BEE**，在场景理解、威胁定位、视觉定地和VQA四项任务上建立新基线，并展现SOTA跨域泛化能力。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "X射线安检"
  - "领域特化VLM"
  - "多模态数据集"
  - "威胁检测"
  - "视觉定地"
---

# STING-BEE: Towards Vision-Language Model for Real-World X-ray Baggage Security Inspection

**会议**: CVPR 2025  
**arXiv**: [2504.02823](https://arxiv.org/abs/2504.02823)  
**代码**: [https://divs1159.github.io/STING-BEE/](https://divs1159.github.io/STING-BEE/)  
**领域**: 多模态VLM  
**关键词**: X射线安检, 领域特化VLM, 多模态数据集, 威胁检测, 视觉定地

## 一句话总结

构建了首个多模态X射线行李安全数据集**STCray**（46,642张图像-描述对，21类威胁含IED和3D打印枪），设计**STING协议**系统生成领域感知的高质量描述，并训练领域特化VLM **STING-BEE**，在场景理解、威胁定位、视觉定地和VQA四项任务上建立新基线，并展现SOTA跨域泛化能力。

## 研究背景与动机

随着全球航空客运量激增，行李安检面临巨大压力。X射线行李监控是安检核心环节，但传统方法存在多重挑战：

**领域现状**：现有计算机辅助筛查（CAS）系统主要依赖边缘检测、轮廓映射和注意力机制等视觉方法，局限于**封闭集预定义类别**，无法泛化到新型威胁。

**现有痛点**：
1. **数据集缺陷**——现有X射线数据集（SIXray、OPIXray、PIDRay等）全部是单模态的，缺乏文本描述；简单的类别标签无法表达复杂场景中的遮挡关系和隐藏策略
2. **泛化能力差**——训练在手枪上的模型可能无法检测步枪（形态差异）或3D打印枪（色彩差异）；跨设备域偏移进一步恶化检测效果
3. **VLM无法直接使用**——GPT-4、Gemini等通用VLM在X射线图像上严重幻觉，将行李扫描误解为医学图像，完全无法识别威胁物品

**核心矛盾**：VLM具备复杂场景理解和零样本泛化的潜力，但缺乏X射线领域的多模态数据；而现有数据集仅有简单标签，无法支撑VLM训练。

**切入角度**：从数据出发——设计系统化的威胁隐藏摄录协议（STING Protocol），生成高质量的领域感知描述，构建首个X射线多模态数据集，然后在此基础上训练领域特化VLM。

## 方法详解

### 整体框架

STING-BEE包含三个核心贡献：(1) STING协议驱动的数据集构建流程——系统变化威胁物品的位置、角度、遮挡程度来生成详尽描述；(2) STCray数据集——46,642张图像跨21个威胁类别，含IED和3D打印枪；(3) STING-BEE模型——基于LLaVA架构的领域特化VLM，通过多任务指令微调和视觉定地指令微调两阶段训练，支持场景理解、指代表达、视觉定地和VQA四类任务。

### 关键设计

1. **STING协议（Strategic Threat ConcealING Protocol）**
    - **功能**：系统化生成X射线行李扫描图像及其高质量领域感知描述
    - **核心思路**：
     - 选定行李类型（行李箱、背包、运动包等）和威胁类别
     - 系统变化威胁物品的**位置**（中央/角落/侧面）和**姿态**（平放/倾斜/竖立）
     - 逐步增加**遮挡复杂度**：从无遮挡→少量杂物→中等→高度杂波→刻意隐藏
     - 每个遮挡层级搭配不同密度和材质的遮挡物品，模拟真实走私策略
     - 描述详细记录：威胁位置、方向、材质特性、遮挡物品、遮挡程度
    - **设计动机**：通用VLM（GPT-4、Gemini）在X射线图像上产生大量幻觉和错误描述，不能直接用于生成训练数据。STING协议通过结构化扫描流程从源头确保描述准确性，而非依赖模型生成

2. **任务特化token设计（Task-Identification Tokens）**
    - **功能**：让单一模型区分不同粒度的视觉-语言任务，无需为每个任务训练单独模型
    - **核心思路**：引入两个特殊token：
     - `[refer]`——触发指代威胁定位任务，模型输出归一化边界框坐标
     - `[grounding]`——触发视觉定地任务，模型输出交织的描述文本和空间坐标
     - 场景理解和VQA不需要特殊token，仅需图像级理解
    - **设计动机**：X射线安检需要从粗到细的多层次理解——先识别有无威胁（分类），再定位在哪（定位），再描述详情（描述）。任务token让模型在不同粒度间灵活切换

3. **两阶段多任务指令微调训练策略**
    - **功能**：让模型逐步获得X射线领域知识和视觉定地能力
    - **核心思路**：
     - **阶段1：多任务威胁指令微调**——使用120,190条指令训练数据（含场景理解、指代表达、VQA），让模型建立对X射线数据的基础理解
     - **阶段2：威胁视觉定地指令微调**——加入29,444条视觉定地指令，训练模型生成交织边界框坐标的描述性响应
     - 使用LoRA微调LLM，同时训练MLP投影器，冻结视觉编码器
    - **设计动机**：先建立领域理解再学习细粒度定地，避免多任务冲突。LoRA微调保留通用语言能力的同时适配领域特性

### 损失函数 / 训练策略

基于LLaVA框架的标准自回归语言建模损失。图像编码器使用CLIP预训练的ViT-L/14，语言模型使用Vicuna。LoRA用于高效微调LLM参数。

## 实验关键数据

### VQA性能（In-domain，7类问题）

| 模型 | 实例定位 | 复杂推理 | 实例识别 | 计数 | 误导性 | 属性 | 交互 | 总体 |
|------|---------|---------|---------|------|-------|------|------|------|
| Florence-2 | 30.11 | 37.50 | 39.84 | 29.95 | 21.16 | 35.80 | 29.12 | 32.27 |
| LLaVA 1.5 | 29.73 | 56.67 | 74.04 | 34.24 | 13.76 | 40.85 | 24.03 | 41.94 |
| **STING-BEE** | **49.22** | **79.21** | **80.04** | **45.24** | **27.76** | **52.85** | **35.03** | **52.81** |

### STCray数据集规模

| 特征 | 数值 |
|------|------|
| 总图像数 | 46,642 |
| 威胁类别数 | 21 |
| 威胁实例数 | 57,218 |
| 训练集 | 30,044 |
| 测试集 | 16,598 |
| 单威胁图像 | 36,438 |
| 多威胁图像 | 9,255 |

### 与现有数据集对比

STCray是唯一同时满足以下条件的数据集：
- ✓ 多模态（图像-描述对）
- ✓ 战略隐藏（系统化遮挡）
- ✓ 新型威胁（IED、3D打印枪）
- ✓ 零样本能力

### 关键发现

1. STING-BEE在总体VQA上比LLaVA 1.5高10.87个百分点（52.81 vs 41.94），尤其在复杂推理（+22.54%）和实例定位（+19.49%）上优势巨大
2. 通用VLM（GPT-4、Gemini）在X射线图像上完全无法识别威胁物品，甚至将行李扫描误识别为医学影像
3. STING-BEE在跨域设置下展现SOTA泛化能力，说明STING协议生成的数据确实帮助模型学到了X射线的领域知识

## 亮点与洞察

1. **数据驱动而非模型驱动**——通过精心设计的数据采集协议（STING Protocol）解决核心瓶颈（缺乏多模态X射线数据），而非复杂的模型架构创新
2. **真实威胁物品数据**——包含IED、3D打印枪等真实安全威胁，这在之前的数据集中从未出现，极大提升了实际应用价值
3. **端到端多任务统一**——用任务token在单一模型中统一了分类、定位、发现和对话四种任务，避免了多模型维护的复杂性
4. **巨大的数据集构建投入**——团队花费约3,109小时（约130天全职）构建STCray数据集，体现了这一领域数据瓶颈的严重性

## 局限性

1. 数据集虽然规模不小但类别分布不均衡（如Explosive有6,491实例，而Shaving Razor仅1,284实例）
2. 基于LLaVA架构的模型在X射线这种特殊视觉域中可能存在视觉编码器预训练偏差（CLIP在自然图像上预训练）
3. VQA总体准确率仅52.81%，距离实际安检部署还有较大差距
4. 数据采集依赖真实X射线扫描仪，难以大规模扩展到更多威胁类型和场景

## 相关工作与启发

- **X射线安检数据集**: 从GDXray（3类）到PIDRay（13类），本文将类别推进到21类并首次引入多模态标注，为该领域从封闭集向开放集范式转变奠定数据基础
- **领域特化VLM**: 类似医学领域的Med-PaLM，本文展示了VLM在专业视觉领域（X射线安检）需要领域特化数据和训练
- **启发**: 当通用大模型在特定领域完全失败时，解决路径是"数据先行"——设计系统化的数据采集协议，而非尝试修改模型架构

## 评分

⭐⭐⭐⭐ — 工作量巨大（3,109小时数据构建），数据集贡献突出（首个X射线多模态数据集），实际应用价值高，但模型创新有限（基本是LLaVA+LoRA），VQA绝对性能仍不够高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VisionArena: 230K Real World User-VLM Conversations with Preference Labels](visionarena_230k_real_world_user-vlm_conversations_with_preference_labels.md)
- [\[ICCV 2025\] AdvDreamer Unveils: Are Vision-Language Models Truly Ready for Real-World 3D Variations?](../../ICCV2025/multimodal_vlm/advdreamer_unveils_are_visionlanguage_models_truly_ready_for.md)
- [\[ACL 2025\] REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](../../ACL2025/multimodal_vlm/real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)
- [\[ICLR 2026\] Can Vision-Language Models Answer Face to Face Questions in the Real-World?](../../ICLR2026/multimodal_vlm/can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)
- [\[CVPR 2025\] Vision-Language Model IP Protection via Prompt-based Learning](vision-language_model_ip_protection_via_prompt-based_learning.md)

</div>

<!-- RELATED:END -->
