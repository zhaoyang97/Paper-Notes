# DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection

**会议**: NeurIPS 2025  
**arXiv**: [2503.09271](https://arxiv.org/abs/2503.09271)  
**代码**: [https://aimagelab.github.io/DitHub/](https://aimagelab.github.io/DitHub/)  
**领域**: 目标检测  
**关键词**: open-vocabulary detection, incremental learning, modular deep learning, LoRA, adaptation modules, version control  

## 一句话总结
提出 DitHub，借鉴版本控制系统（Git）思想构建开放词汇目标检测的模块化适配框架——将不同领域的高效适配模块（LoRA）作为"分支"管理，支持按需获取（fetch）和合并（merge），在 ODinW-13 上达到 SOTA，首次系统性研究目标检测中适配模块的组合特性。

## 背景与动机
开放词汇目标检测器（如 Grounding DINO）可通过文本提示泛化到任意类别，但面对稀有类或需要在多个专业领域上工作时仍需适配。现有增量学习方法：
- **单体式适配**（single set of weights）：在整个模型上微调，不同领域间会遗忘
- **LoRA 微调**：高效但不同领域的 LoRA 模块相互独立，未研究如何组合多个领域的知识

关键问题：能否像 Git 管理代码一样管理 LoRA 适配模块——各领域独立开发（branch），需要时合并（merge）？

## 核心问题
如何在开放词汇检测中**增量地**适配新领域/新类别，同时保持对已学领域的能力、支持跨领域组合，且不需要存储所有历史训练数据？

## 方法详解

### 整体框架
DitHub 维护一个适配模块库，每个模块是一个轻量级 LoRA。三个核心操作类比 Git：
- **Branch**: 为新领域创建新的 LoRA 适配模块并独立训练
- **Fetch**: 需要在新领域检测时，从库中检索相关模块
- **Merge**: 将多个模块的权重组合，实现跨领域知识融合

### 关键设计
1. **模块化适配**: 每个领域/任务对应一个独立的 LoRA 模块，训练时冻结基础模型（Grounding DINO），仅更新 LoRA 参数。低参数量（每个模块 ~0.5M 参数），存储高效。

2. **模块组合策略**: 系统性研究了多种 LoRA 合并方法在检测任务中的效果：
   - 简单平均（averaging）
   - Task Arithmetic（任务算术加权）
   - TIES-Merging（修剪+符号投票+合并）
   - 基于注意力/相似度的自适应加权

3. **增量学习协议**: 定义了类别重现（class reappearance）的评估场景——同一类别在不同增量阶段重复出现，更贴近真实世界。提出 ODinW-O 新基准来评估此能力。

### 损失函数 / 训练策略
基于 Grounding DINO 的标准检测损失（Focal Loss + L1 + GIoU），每个领域独立训练 LoRA，推理时按需合并。

## 实验关键数据

| 方法 | ODinW-13 AP | 类型 |
|------|-----------|------|
| Grounding DINO (zero-shot) | ~55 | 基线 |
| Full fine-tuning | ~63 | 单体 |
| CL 方法 (EWC, LwF) | ~58-60 | 增量 |
| **DitHub** | **SOTA** | 模块化 |

首次系统性证明：(1) LoRA 模块在检测任务中具有可组合性，(2) 模块化方法在增量开放词汇检测中优于传统持续学习方法。

### 消融实验要点
- 不同合并策略的效果比较：Task Arithmetic 因检测任务的特殊性（类别非均匀分布）表现不同于 LLM 任务
- 模块数量增长对合并质量的影响：适中数量（~5-8）效果最优
- 类别重现场景下 DitHub 的优势尤为明显——同一类别的不同领域 LoRA 能互补

## 亮点
- **Git 版本控制的类比**非常直觉——分支/获取/合并的概念清晰易懂
- 首次在目标检测领域系统性研究适配模块的组合特性
- 提出 ODinW-O 基准，填补了类别重现评估的空白
- 模块化架构天然支持隐私保护——不同机构的数据可以独立训练 LoRA 再合并

## 局限性 / 可改进方向
- 仅基于 Grounding DINO，未测试其他开放词汇检测器（如 YOLO-World）
- 模块合并策略在某些领域组合下可能产生冲突
- 未探索自动化的模块选择机制（当前需手动指定要 fetch 的模块）
- ODinW 数据集规模相对有限，大规模场景验证不足

## 与相关工作的对比
- **vs 传统 CL（EWC、LwF）**: 传统方法通过正则化防遗忘但性能仍下降；DitHub 用独立模块完全避免遗忘
- **vs LoRAHub（NLP）**: LoRAHub 在 LLM 中验证 LoRA 组合，DitHub 首次将此思路引入目标检测
- **vs CQ-DINO（同批次笔记）**: CQ-DINO 解决大词汇量检测的梯度问题；DitHub 解决多领域增量适配问题，两者互补

## 启发与关联
- 模块化适配思想可迁移到分割（如为每个领域训练 SAM LoRA 再合并）
- 类比版本控制的框架设计可用于其他模型定制场景（如个性化 VLM）
- 可与 CQ-DINO 结合：CQ-DINO 解决单次大词汇量，DitHub 解决增量扩展词汇量

## 评分
- 新颖性: ⭐⭐⭐⭐ Git 类比新颖，但 LoRA 组合在 NLP 中已有探索
- 实验充分度: ⭐⭐⭐⭐ ODinW-13 + ODinW-O 两个基准，多种合并策略对比
- 写作质量: ⭐⭐⭐⭐ Git 类比使论文概念清晰易读
- 价值: ⭐⭐⭐⭐ 模块化增量检测是实际部署的真实需求
