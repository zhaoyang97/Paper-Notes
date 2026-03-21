# CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models

**会议**: CVPR 2025  
**arXiv**: [2603.12989](https://arxiv.org/abs/2603.12989)  
**代码**: 待确认  
**领域**: 多模态VLM / AI安全  
**关键词**: 后门攻击, LVLM防御, 注意力机制, test-time defense, visual token pruning

## 一句话总结
CleanSight 发现 LVLM 后门攻击的机制不在像素层面而在注意力层面——触发器通过"注意力窃取"（trigger token 抢夺 text token 的注意力）来激活后门，据此提出了一种免训练、即插即用的 test-time 防御方法：通过检测跨模态注意力比例异常来识别中毒输入，再通过剪枝高注意力视觉 token 来中和后门，ASR 降至接近 0% 且几乎不影响模型性能。

## 研究背景与动机
1. **领域现状**：LVLM（如 LLaVA）通过微调适配器适应下游任务，但微调阶段容易被后门攻击——攻击者在训练数据中注入带触发器的样本。
2. **现有痛点**：现有防御方法要么需要用干净数据重训练（计算昂贵、降低下游性能），要么在像素空间扰动输入（如图像变换），但**像素扰动对 LVLM 后门几乎无效**。
3. **核心矛盾**：与从头训练的 CLIP 不同，LVLM 的后门关联不绑定在低层视觉特征上，而是隐藏在跨模态注意力交互模式中——像素扰动碰不到注意力层面的异常。
4. **本文要解决什么？** 如何在不修改模型参数的情况下，在 test time 检测并中和 LVLM 后门？
5. **切入角度**：作者发现关键现象——后门激活的机理是"attention stealing"：中毒输入的视觉 token 在 LVLM 中间层异常抢夺文本 token 的注意力权重，压制指令遵循能力。
6. **核心idea一句话**：后门在注意力而非像素 → 检测注意力比例异常 + 剪枝高注意力视觉 token = 免训练 test-time 防御。

## 方法详解

### 整体框架
CleanSight 分两步工作：
1. **检测（Detection）**：在选定的中间跨模态融合层中，计算每个 attention head 的视觉-文本注意力比率，与干净参考分布比较（whitened $\ell_2$ 距离），判断输入是否中毒
2. **净化（Purification）**：对标记为中毒的输入，聚合各 head 中异常高注意力的视觉 token，剪枝这些 token，阻止它们在后续层和解码过程中"窃取"注意力

### 关键设计

1. **Attention Stealing 机制发现**:
   - 做什么：揭示 LVLM 后门激活的真实机理
   - 核心思路：正常输入中，视觉 token 在中间层的注意力远低于文本 token；中毒输入中，触发器区域的视觉 token 注意力急剧上升，文本 token 注意力对应下降——"窃取"了文本的注意力
   - 设计动机：这解释了为什么像素扰动无效（触发器模式仍在）但注意力扰动有效（均匀化注意力即可消除后门）

2. **基于注意力比率的检测**:
   - 做什么：用 head-specific 的视觉-文本注意力比率来检测中毒输入
   - 核心思路：在选定的中间层 $\ell$ 中，对每个 head $h$ 计算视觉注意力占总注意力的比例，组成 attention ratio vector $\mathbf{r}$。用少量干净样本估计 clean distribution $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，检测时用 whitened $\ell_2$ distance $d = \|(\mathbf{r} - \boldsymbol{\mu}) \boldsymbol{\Sigma}^{-1/2}\|_2$ 与阈值比较
   - 设计动机：head-specific 比率比全局平均更敏感，whitened distance 处理不同 head 的方差差异

3. **视觉 token 剪枝净化**:
   - 做什么：识别并移除"窃取注意力"的触发器 token
   - 核心思路：在所有选定 head 中，收集注意力值超过阈值（如 top-k 或百分位）的视觉 token 索引，取交集或并集后剪枝这些 token（直接从 KV cache 中移除），后续层不再看到这些 token
   - 设计动机：精准移除触发器关联 token 而非均匀扰动，最大限度保留干净语义

### 损失函数 / 训练策略
完全免训练。仅需少量干净样本（~100）估计 attention ratio 的参考分布。

## 实验关键数据

### 主实验（VQA + Image Captioning）

| 防御方法 | BadNet ASR↓ | Blended ASR↓ | ISSBA ASR↓ | WaNet ASR↓ | TrojVLM ASR↓ | VLOOD ASR↓ |
|---------|------------|-------------|-----------|-----------|-------------|-----------|
| No defense | 100.00 | 100.00 | 99.22 | 99.61 | 100.00 | 100.00 |
| ST defense | 85.55 | 98.05 | 67.19 | 53.91 | 77.73 | 82.42 |
| BDMAE | 88.28 | 100.00 | 100.00 | 99.22 | 80.86 | 86.33 |
| ZIP | 80.47 | 84.77 | 74.22 | 7.03 | 85.94 | 95.31 |
| **CleanSight** | **0** | **0** | **0** | **0** | **3.14** | **0** |

Clean Utility（VQAv2 准确率）在 CleanSight 下保持 62-68%，与无防御时基本持平。

### 消融实验

| 组件 | ASR↓ | 说明 |
|------|------|------|
| Detection only (flag + no action) | n/a | 仅检测不净化 |
| Pruning without detection | ASR↓ 但 CU↓ | 所有输入都剪枝损害正常性能 |
| Detection + Pruning (CleanSight) | ~0% | 检测+净化结合最优 |
| 均匀注意力扰动 | ~0% ASR 但 CU↓ | 有效但损害干净输入 |

### 关键发现
- **像素扰动对 LVLM 后门几乎无效**：ST defense 和 Blur 在多数攻击下 ASR 仍 >80%，而注意力扰动在强度为 1 时就消除后门
- **CleanSight 在 6 种攻击类型上 ASR 接近 0%**：显著优于所有 baseline（ZIP 最强但 ASR 仍 55-95%）
- 注意力窃取现象在多种触发器类型（patch/global/WaNet/ISSBA）中一致出现
- 剪枝的关键层是中间融合层（而非最深或最浅层）

## 亮点与洞察
- **机制发现比方法更重要**："Attention Stealing" 的发现深刻揭示了 LVLM 后门的工作原理，为该领域提供了新的理解框架
- **免训练+即插即用的实用性**：不修改模型参数、不需要重训练、只需前向传播中的轻量干预，部署成本极低
- **意外联系**：视觉 token pruning 原本用于加速推理（FastV, LLaVA-PruMerge），本文发现它还能增强后门安全性——加速和安全可以同时实现
- **可迁移思路**：attention ratio 分析可以用于检测其他类型的 LVLM 异常行为（对抗样本、输入污染等）

## 局限性 / 可改进方向
- 需要少量干净样本估计参考分布——虽然数量极少（~100），但在零样本场景下仍需探索
- 检测阈值的设定需要调优，不同模型/攻击可能需要不同阈值
- 仅在 LLaVA 系列验证，其他 LVLM 架构（如 Qwen-VL、InternVL）的泛化性待测
- 自适应攻击者可能设计不依赖注意力窃取的新型后门
- TrojVLM 类语义保留攻击的 ASR 降至 3-5% 但未完全消除，因其后门行为与正常任务目标紧密纠缠

## 相关工作与启发
- **vs ST defense**: 在像素空间做空间变换（旋转/翻转），对 LVLM 后门无效（ASR>80%），因为后门不在像素层
- **vs BDMAE**: 用 MAE 重建来净化输入，对部分攻击有效但不稳定
- **vs ZIP**: 通过零阶优化在输入空间搜索净化方向，对 WaNet 有效但其他攻击仍高 ASR
- **vs FastV**: FastV 剪枝的是低注意力 token（加速），CleanSight 剪枝的是高注意力异常 token（安全）——两者互补
- **vs BDMAE/SampDetox**: 基于 MAE/扩散模型的生成式净化方法在 LVLM 上效果不稳定，因后门不在像素层面

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 注意力窃取机制的发现非常深刻且启发性强
- 实验充分度: ⭐⭐⭐⭐⭐ 6种攻击×多个数据集×多个baseline，消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链清晰：观察→机制→方法→验证
- 价值: ⭐⭐⭐⭐⭐ 首个LVLM test-time后门防御，实用性极强
