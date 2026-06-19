---
title: >-
  [论文解读] CICA: Coupling Confidence-Aware Pretraining with Confidence-Informed Attention for Robust Multimodal Sentiment Analysis
description: >-
  [CVPR 2026][多模态VLM][多模态情感分析] CICA 让每个单模态编码器在预训练时学会"自评"信号可靠性（输出置信度 $s_m$ 和不确定度 $u_m$），再用这两个信号去调制一个置信度感知注意力的输出，从而在文本/视觉/语音相互矛盾或缺失时自适应地放大可信模态、压制噪声模态，在 MOSI/MOSEI/CH-SIMS/CH-SIMSv2 四个基准上刷新 SOTA。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "多模态情感分析"
  - "置信度感知"
  - "可靠性建模"
  - "自适应融合"
  - "模态冲突"
---

# CICA: Coupling Confidence-Aware Pretraining with Confidence-Informed Attention for Robust Multimodal Sentiment Analysis

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_CICA_Coupling_Confidence-Aware_Pretraining_with_Confidence-Informed_Attention_for_Robust_Multimodal_CVPR_2026_paper.html)  
**代码**: 无（未公开）  
**领域**: 多模态VLM  
**关键词**: 多模态情感分析, 置信度感知, 可靠性建模, 自适应融合, 模态冲突  

## 一句话总结
CICA 让每个单模态编码器在预训练时学会"自评"信号可靠性（输出置信度 $s_m$ 和不确定度 $u_m$），再用这两个信号去调制一个置信度感知注意力的输出，从而在文本/视觉/语音相互矛盾或缺失时自适应地放大可信模态、压制噪声模态，在 MOSI/MOSEI/CH-SIMS/CH-SIMSv2 四个基准上刷新 SOTA。

## 研究背景与动机

**领域现状**：多模态情感分析（MSA）要联合建模语言、视觉、语音三路信号来推断情感。主流做法从早期的张量融合（TFN、LMF）发展到基于 Transformer 的跨模态注意力（MulT），近年又出现了各种"引导式/非对称"融合（ALMT 假设文本主导、KuDA/CLGSI 用动态引导）。

**现有痛点**：真实数据几乎不可能干净、时间对齐。常见的是模态之间互相冲突——比如一个人脸上在笑（视觉强正），语气却很沮丧（语音负），文本中性，而 ground truth 跟随语音（-1.33），多数模型却被视觉的"笑"误导。它们用的是**统一注意力**，默认所有模态同等可信，内部根本没有一个机制去判断"这一条样本里到底哪个模态该说了算"。

**核心矛盾**：融合质量取决于"每个模态此刻有多可信"，但现有方法把可靠性当成全局常量，缺乏**样本级、模态级**的可靠性度量。已有的不确定性建模（证据深度学习、贝叶斯近似）大多是事后挂在决策层；三支决策类方法（如 3WD-DRT）只预测单一置信分 $s_m$ 然后做离散分区缩放，信号过于粗糙。

**本文目标**：构建一个"先感知、再决策"（perceive-and-decide）的框架——模型必须**先量化每个模态对当前样本的可信度，再决定怎么组合**。

**核心 idea**：把"自评置信度"和"融合时的决策"直接耦合起来：预训练阶段让编码器自己估出可靠性（置信 $s_m$ + 不确定 $u_m$），融合阶段把这两个信号当作连续调制因子作用到注意力输出上，可信且一致才放大贡献。

## 方法详解

### 整体框架
CICA 把"感知信号质量"和"决策怎么融合"拆成两个协同阶段。**Phase 1（CAP，置信度感知预训练）**单独训练每个单模态编码器 $E_m$（$m\in\{T,V,A\}$），让它不仅输出表示 $H_m$，还输出该样本的置信分 $s_m$ 与不确定度 $u_m$——这就是"自我感知"。**Phase 2（CIF，置信度引导融合）**冻结这些编码器，训练一个融合模块：先用置信度感知注意力（CIA）感知特征的内在结构质量，再把 CAP 给出的可靠性信号 $(s_m,u_m)$ 作为外部调制因子作用上去，最后用一个互信息对比保持（MCP）损失防止某个模态独占融合（"模态坍塌"）。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    X["文本 / 视觉 / 语音<br/>三路原始输入"] --> CAP
    subgraph CAP["Phase 1 · 置信度感知预训练 CAP"]
        direction TB
        E["单模态编码器 + MixDomainAdapter<br/>输出 H_m"] --> H["三个辅助头<br/>任务 / 置信 s_m / 不确定 u_m"]
    end
    CAP -->|冻结编码器, 传出 H_m 与 (s_m,u_m)| CIA["置信度感知注意力<br/>Q·K + S_mod 感知内在质量"]
    CIA --> MOD["可靠性调制耦合<br/>r_m=ReLU(1+s_m-u_m), z×r_m"]
    MOD --> MCP["互信息对比保持 MCP<br/>防模态坍塌"]
    MCP --> Y["情感预测 ŷ"]
```

### 关键设计

**1. CAP 置信度感知预训练：让编码器学会"自评"，而非只学特征**

针对"模型没有内部机制判断哪个模态可信"这个痛点，CAP 把每个单模态编码器 $E_m$ 训练成一个"感知器"：$(H_m, s_m, u_m)=E_m(X_m)$。编码器用 Transformer 主干，内部的 MixDomainAdapter 从最后一层抽任务相关特征 $h_{\text{spci}}$、从中间层抽域相关特征 $h_{\text{domain}}$，拼成 $H_m=\text{Concat}(h_{\text{spci}},h_{\text{domain}})$。在此之上挂三个辅助头，用联合目标预训练：

$$\mathcal{L}_{\text{CAP}}=\mathcal{L}_{\text{task\_pre}}+\lambda_{\text{CA}}\mathcal{L}_{\text{CA}}+\lambda_{\text{uncert}}\mathcal{L}_{\text{uncert}}$$

任务头 $C_{\text{pred}}$ 预测单模态情感、用 MSE 监督；置信头 $C_{\text{CA}}$ 预测标量 $s_m\in[0,1]$；不确定头 $C_{\text{uncert}}$ 预测有界绝对误差 $u_m\in[0,1]$。这一步的关键是它**显式校准了编码器自身的不确定性**：与 3WD-DRT 只给单个置信分不同，CICA 同时建模"自适应置信"和"任务不确定"，从而能把认知不确定（epistemic）和偶然不确定（aleatoric）拆开。

**2. Adaptive CA 损失：用可学习的分段惩罚把置信分标定成"真·可靠性"**

直接训练置信分很别扭：分类损失（如 BCE）会错误地把它当二分类，而静态回归损失 $L=(s_m-1)^2$ 又太死板（无脑逼所有样本都自信）。作者提出一个**可学习的标定函数** $\mathcal{L}_{\text{CA}}$，用一组参数 $\theta_{\text{CA}}=\{\hat\alpha,\hat\beta,\hat w_{\text{high}},\hat w_{\text{mid}},\hat w_{\text{low}}\}$ 自适应地定义两条可靠性边界（$0<\beta<\alpha<1$，由 sigmoid 约束 $\beta=\sigma(\hat\beta),\ \alpha=\beta+(1-\beta)\cdot\sigma(\hat\alpha)$）和三档惩罚权重（Softplus 保正）。分段权重函数为：

$$W(s_m,\theta_{\text{CA}})=\begin{cases}w_{\text{low}} & s_m\ge\alpha\\ w_{\text{mid}} & \beta\le s_m<\alpha\\ w_{\text{high}} & s_m<\beta\end{cases}$$

损失则是加权的"鼓励高置信"项 $\mathcal{L}_{\text{CA}}=\frac{1}{B}\sum_i W(s_{m,i},\theta_{\text{CA}})\cdot(s_{m,i}-1)^2$。不确定头的目标用 detach 后的误差给定 $u_{\text{target}}=\tanh(|y_m-\hat y_m^{\text{detach}}|)$，再用 MSE 拟合。⚠️ $\theta_{\text{CA}}$ 的完整参数化与论证在原文附录 H.1，这里给出的边界/权重形式以原文为准。这样置信分就不是拍脑袋的标量，而是被一条**可学习的标定曲线**约束出来的可靠性度量。

**3. CIA 置信度感知注意力 + 可靠性调制：把"内在质量感知"和"外部可靠性"解耦再耦合**

作者的核心观察是：标准注意力 $Q\cdot K^\top$ 只建模 query 和 key 的**相关性**，却完全忽略 key $K_m$ 本身的**内在质量**。CIA 因此在 softmax 内部加一个内在结构调制项 $S_{\text{mod}}$，它捕捉 key 的局部依赖 $N_m$ 和 token 级显著性 $\rho_m$，产出结构感知表示：

$$z_{\text{struct},m}=\text{softmax}\!\left(\frac{QK_m^\top}{\sqrt{d_k}}+S_{\text{mod}}\right)V_m$$

这是"感知内在质量"。然后把 CAP 算出的两路可靠性 $(s_m,u_m)$ 通过投影 $g(\cdot)$ 合成统一可靠性分 $r_m=g(s_m,u_m)=\text{ReLU}(1+s_m-u_m)$——只有置信强（$s_m$ 大）且不确定低（$u_m$ 小）时 $r_m$ 才大。最后用 $r_m$ 沿序列和特征维广播，连续调制结构表示：$z_{\text{CIF},m}=z_{\text{struct},m}\times r_m$。被判为不可靠的模态（$r_m\approx0$）贡献被自然压低。这一步是全框架的"耦合点"：把预训练阶段的"感知"和融合阶段的"决策"用一个乘法门显式接通，比 3WD-DRT 仅靠置信缩放更严格（要"自信且一致"才放大）。

**4. MCP 互信息对比保持损失：防止融合被单一模态独占**

把可靠性门做强之后会有新风险——某个模态（尤其文本）可能直接独占融合，导致"模态坍塌"，丢掉其它模态的独特信息。MCP 受对比预测编码（CPC）/NCE 启发，约束最终融合表示 $z_{\text{final}}$ 与每个单模态池化表示 $h_m=\text{MeanPool}(H_m)$ 保持对应：$\mathcal{L}_{\text{MCP}}=\sum_{m\in\{T,V,A\}}\mathcal{L}_{\text{NCE},m}(z_{\text{final}},h_m)$。这样即使在压制某个"骗人"的模态时，融合表示仍保留所有来源的足够信息。微调阶段的总目标为 $\mathcal{L}_{\text{Total}}=\mathcal{L}_{\text{task}}+\lambda_{\text{mcp}}\mathcal{L}_{\text{MCP}}$，主任务用 L1 损失 $\mathcal{L}_{\text{task}}=\frac{1}{B}\sum_i|y_i-\hat y_i|$，$\lambda_{\text{mcp}}=0.1$。

### 损失函数 / 训练策略
两阶段训练：先用 $\mathcal{L}_{\text{CAP}}$ 预训练单模态编码器（含任务/置信/不确定三头），冻结后再用 $\mathcal{L}_{\text{Total}}=\mathcal{L}_{\text{task}}+0.1\cdot\mathcal{L}_{\text{MCP}}$ 微调 CIF 融合模块。原文只在正文给出 $\lambda_{\text{mcp}}=0.1$，其余优化器/学习率/epoch 等配置在附录 C、D。

## 实验关键数据

### 主实验
四个基准（英文 MOSI/MOSEI，中文 CH-SIMS/CH-SIMSv2）全面刷新 SOTA。$\Delta$ 行为相对最强基线的提升：

| 数据集 | MAE↓ | Corr↑ | Acc-7/3↑ | Acc-2↑ | 对比最强基线 |
|--------|------|-------|----------|--------|--------------|
| MOSI | **0.630** | **0.855** | 49.56 | 88.19/90.24 | MAE ↓0.071, Corr ↑0.060 |
| MOSEI | **0.489** | **0.856** | 55.29 | 84.72/90.18 | MAE ↓0.029, Corr ↑0.077 |
| CH-SIMS | **0.378** | **0.754** | 76.37 | 86.00 | Corr ↑0.127, Acc-3 ↑9.73 |
| CH-SIMSv2 | **0.245** | **0.842** | 80.56 | 85.98 | Acc-3 ↑4.35, Corr ↑0.078 |

提升在交互更复杂的 CH-SIMS 上尤其夸张（Corr +0.127、Acc-3 +9.73）。

### 消融实验
在 MOSI/MOSEI 上逐项移除组件（括号为相对 Full 的掉幅）：

| 配置 | MOSI Corr↑ | MOSI F1(non-0)↑ | MOSEI F1(non-0)↑ | 说明 |
|------|-----------|-----------------|------------------|------|
| Full | 0.855 | 90.23 | 90.16 | 完整模型 |
| (A) w/o CAP | 0.791 (↓0.064) | 85.05 (↓5.18) | 83.19 (↓6.97) | 去掉置信感知预训练 |
| (B) w/o CIF | 0.812 (↓0.043) | 86.13 (↓4.10) | 83.90 (↓6.26) | 去掉置信注意力融合 |
| (D) w/o Coupling | 0.831 (↓0.024) | 87.02 (↓3.21) | 84.13 (↓6.03) | CAP/CIF 不耦合，独立用 |
| (C) w/o MCP | 0.847 (↓0.008) | 87.66 (↓2.57) | 84.71 (↓5.45) | 去掉防坍塌损失 |
| (E) w/o S_mod | 0.840 (↓0.015) | 89.15 (↓1.08) | 88.05 (↓2.11) | 去掉内在结构调制项 |

### 关键发现
- **CAP 和 CIF 是两根顶梁柱**：分别去掉它们掉点最多（MOSEI F1 non-0 各掉 6.97/6.26），证实"感知"和"决策注意力"缺一不可。
- **耦合本身有价值**：即使 CAP、CIF 都在、只是不显式耦合（D），MOSI Corr 仍掉 0.024——说明两个模块各自独立还不够，关键在于把感知信号接进融合决策。
- **MCP 和 $S_{\text{mod}}$ 是稳定器**：掉幅小但一致，主要贡献跨模态平衡与注意力稳定性。
- **鲁棒性是最大卖点**：注入高斯噪声（方差 0.2–0.8）后性能平滑衰减——MOSEI 轻度音频噪声下 Corr 仅 0.856→0.843，说明 CAP 检测到污染、CIF 下调其权重；缺模态时，MOSEI 去掉视觉（T+A）几乎不掉（MAE 0.483 vs 0.489），因为 CAP 把缺失模态标成 $s_V=0$、CIF 自动调整融合。
- ⚠️ 一个值得注意的现象：MOSI 上纯文本（T-only）Corr 0.857 还略高于 T+V+A 的 0.855，因为 MOSI 高度文本主导，0.857 就是文本上限；0.002 的差距来自 MCP 的轻度融合正则，作者认为这点代价换来跨基准的稳健性是值得的。

## 亮点与洞察
- **"感知-决策"显式耦合**：大多数不确定性方法把可靠性事后挂在决策层，CICA 把它前移到预训练、再用乘法门接回融合，整条链路"先量化可信度再决定怎么用"，思路干净且可解释。
- **置信分用可学习分段损失标定**：不是拍一个标量，而是用 $\theta_{\text{CA}}$ 学出可靠性边界，避免了 BCE/静态回归的两个极端，这个"learnable calibration"trick 可迁移到任何需要自评置信度的任务。
- **$r_m=\text{ReLU}(1+s_m-u_m)$ 的双信号门**：要"自信"且"低不确定"才放大，比单一置信缩放更严格，天然处理了缺模态（$s\approx0$ 时直接归零）。
- **MCP 防模态坍塌**：把可靠性门做强后用对比损失兜底保留各模态信息，是"压制噪声"和"不丢信息"之间一个聪明的平衡补丁。

## 局限与展望
- 代码未公开，$S_{\text{mod}}$（局部依赖 $N_m$、显著性 $\rho_m$）和 $\theta_{\text{CA}}$ 的具体实现都压在附录，复现门槛偏高。
- 两阶段、冻结编码器再训融合的流程较重；置信/不确定双头 + 三损失预训练带来额外开销，论文未报训练成本对比。
- MOSI 上多模态甚至略逊于纯文本，说明在文本极度主导的数据上多模态收益有限——框架更像是"在该用多模态时不被噪声带偏"，而非在文本足够时还能涨点。
- 自评可靠性的标定质量依赖标签质量；在弱标注或域偏移场景下 $s_m/u_m$ 是否仍准确，论文未充分检验。

## 相关工作与启发
- **vs 3WD-DRT（三支决策）**: 它只预测单个置信分 $s_m$ 再离散分区缩放；CICA 同时建模置信 $s_m$ 和不确定 $u_m$，用连续的 $r_m$ 门控，"自信且一致才放大"，更稳更可解释，四基准全面超越。
- **vs ALMT / KuDA（引导式融合）**: 它们靠"假设文本主导"或外部知识做动态引导，缺乏原则性的样本级可靠性度量；CICA 的可靠性是编码器自评出来的，不依赖先验假设。
- **vs MMIM / MISA（表示增强类）**: 它们改进表示或最大化互信息，CICA 借用互信息思想（MCP）只是辅助目标，主线是置信驱动的自适应融合。

## 评分
- 新颖性: ⭐⭐⭐⭐ "感知-决策"显式耦合 + 可学习置信标定，思路新颖且贴合冲突/缺模态痛点。
- 实验充分度: ⭐⭐⭐⭐⭐ 四基准 SOTA + 细致消融 + 噪声/缺模态鲁棒性双重压力测试。
- 写作质量: ⭐⭐⭐⭐ 主线清晰、公式完整，但 $S_{\text{mod}}$ 与 CA 损失参数化大量压进附录。
- 价值: ⭐⭐⭐⭐ 鲁棒多模态融合的可复用范式，置信标定与双信号门可迁移到其它多模态任务。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis](enhance-then-balance_modality_collaboration_for_robust_multimodal_sentiment_anal.md)
- [\[CVPR 2026\] Conflict-Aware Adaptive Cross-Reconstruction for Multimodal Sentiment Analysis](conflict-aware_adaptive_cross-reconstruction_for_multimodal_sentiment_analysis.md)
- [\[CVPR 2026\] Factorize, Reconstruct, Enhance: A Unified Framework for Multimodal Sentiment Analysis](factorize_reconstruct_enhance_a_unified_framework_for_multimodal_sentiment_analy.md)
- [\[CVPR 2026\] Linking Perception, Confidence and Accuracy in MLLMs](linking_perception_confidence_and_accuracy_in_mllms.md)
- [\[CVPR 2026\] Prototype-as-Prompt: Multimodal Sentiment Prototypes Endowing Large Language Models the Capability to Perform Multimodal Sentiment Analysis](prototype-as-prompt_multimodal_sentiment_prototypes_endowing_large_language_mode.md)

</div>

<!-- RELATED:END -->
