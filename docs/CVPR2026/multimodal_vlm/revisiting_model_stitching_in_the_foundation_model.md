# Revisiting Model Stitching In the Foundation Model Era

**会议**: CVPR 2026  
**arXiv**: [2603.12433](https://arxiv.org/abs/2603.12433)  
**代码**: 无  
**领域**: 多模态VLM / 模型融合  
**关键词**: 模型拼接, 视觉基础模型, 表示兼容性, VFM Stitch Tree, CLIP, DINOv2  

## 一句话总结
系统研究异质视觉基础模型(CLIP/DINOv2/SigLIP2/DINOv3)之间的"可拼接性"，发现通过Final Feature Matching预训练stitch层可实现可靠拼接，且拼接模型一致超越self-stitch基线，并提出VFM Stitch Tree(VST)在仅4.3%额外开销下恢复45%的多VFM性能增益。

## 背景与动机
不同VFM(CLIP、DINOv2、SigLIP2)使用不同数据、目标和模态组合训练，但它们的内部表示是否兼容？模型拼接(model stitching)——将一个模型的前层连接到另一个模型的后层——是探索这一问题的有力工具。先前研究表明在同一数据集上训练的小模型可以拼接，但VFM时代的异质模型能否拼接还不清楚。更重要的是，如果可以拼接，能否从分析工具升级为实用技术——解决多模态LLM中使用多个VFM带来的线性计算/内存开销？

## 核心问题
(1) 异质VFM是否可拼接？ (2) 如何训练stitch层才能使拼接有效？ (3) 拼接的增益来自额外容量还是真正的知识互补？ (4) 能否利用可拼接性减少多VFM系统的计算开销？

## 方法详解

### 整体框架
冻结source VFM前n层和target VFM后(N-n)层，中间插入可训练stitch层。三种训练策略：Layer Feature Matching(LFM, 匹配stitch点特征)、Final Feature Matching(FFM, 匹配最终层特征)、Task Loss Training(TLT, 直接优化下游任务损失)。最优方案：两阶段——先FFM预训练再TLT微调。

### 关键设计
1. **Final Feature Matching (FFM)**: 关键发现——LFM虽然在stitch点处距离极小(10^-3量级)，但最终特征距离很大（误差累积放大）。FFM直接匹配倒数第二层特征，同时隐式保持了stitch点的局部对齐。这解决了shallow stitch的核心困境。

2. **Self-Stitch基线**: 在同一模型内插入相同stitch层（如DINOv2→DINOv2），严格控制额外容量的影响。跨VFM拼接一致超越self-stitch基线+2.3%~2.6%，证明增益来自真正的知识互补而非模块容量。

3. **VFM Stitch Tree (VST)**: 多个VFM共享前层（如CLIP和DINOv2共享前14层），仅后层分叉保持各自特化。VST-22（共22层仅1层特化）用4.3%额外代价恢复45%增益；VST-14（14层共享9层特化）用39%代价恢复84%增益。

### 损失函数 / 训练策略
- FFM: 无标签，匹配$\|T_\phi^N(S(R_\theta^n(x))) - T_\phi^N(R_\phi^n(x))\|_2^2$
- TLT: 标准交叉熵损失
- 两阶段：先FFM预训练 → 再TLT微调
- AdamW, 100 epochs, early stopping patience 5

## 实验关键数据
**DINOv2↔SigLIP2拼接 (fMoW分类, layer 22)：**

| 配置 | Accuracy |
|------|----------|
| DINOv2 linear probe | 46.7% |
| SigLIP2 linear probe | 53.5% |
| DINOv2→DINOv2 self-stitch | 69.9% |
| SigLIP2→SigLIP2 self-stitch | 68.9% |
| DINOv2→SigLIP2 (FFM+TLT) | **71.8%** |
| SigLIP2→DINOv2 (FFM+TLT) | **72.2%** |

跨VFM拼接一致超越双方self-stitch基线。

**VFM Stitch Tree (MoF-LLaVA)：**

| 配置 | 额外开销 | Normalized Gain |
|------|---------|-----------------|
| Full (双VFM) | 100% | 100% |
| VST-14 | 39% | **84.2%** |
| VST-22 | 4.3% | **45.5%** |

### 消融实验要点
- FFM初始化对shallow stitch至关重要：layer 2处TLT单独仅25.1%，加FFM预训练达51.7%
- MLP stitch层优于Linear和LoRA——LoRA表达力更强反而性能更低，可能因为"控制失match"减少了互补融合
- CLIP作为source时拼接效果差（太弱会丢失关键信息），但作为target时效果好
- 跨4种VFM（CLIP/DINOv2/SigLIP2/DINOv3）、4个数据集、分类+分割任务一致成立

## 亮点
- **将model stitching从分析工具升级为实用技术**，VST的accuracy-efficiency knob设计极有实用价值
- Self-stitch基线设计严谨——直接回答"增益是否来自知识互补"这一关键问题
- FFM的insight简洁有力：stitch点的局部特征匹配不如最终特征匹配
- LoRA反而不如MLP的反直觉发现——"过于精确的匹配反而限制互补融合"
- 预测分析(Tab.10)展示拼接模型"rescue"(双方都错但拼接正确)远多于"interference"

## 局限性 / 可改进方向
- VST仅在MoF-LLaVA+VQAv2/MME上做了初步验证，需在更多MLLM任务上验证
- 当两个VFM能力差距过大时（如CLIP作source），拼接失效
- stitch层训练需要额外数据和计算，实际部署中的overhead需考虑
- 未探索动态stitch——根据输入自适应选择stitch点

## 与相关工作的对比
- **原始Model Stitching (Bansal et al.)**: 只在同数据集小模型上验证，本文扩展到异质VFM
- **SN-Net (Pan et al.)**: 从训练阶段设计可拼接网络，本文是post-hoc拼接独立训练的VFM
- **Model Soups/TIES-Merging**: 在权重空间合并模型，本文在激活空间通过stitch层连接，保持各模型独立性
- **Cambrian-1**: 直接使用4个VFM（300%额外开销），VST可将其降至117%

## 启发与关联
- "异质VFM可拼接"意味着不同训练范式的表示在深层趋于兼容——这对理解表示学习有理论价值
- VST可直接应用于任何使用多VFM的系统（如自动驾驶感知），减少部署成本
- "FFM优于LFM"的insight可推广到任何需要特征蒸馏/对齐的场景——匹配最终表示比匹配中间表示更稳定

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将stitching从分析工具升级为实用系统的视角新颖，VST设计有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 4种VFM、4个数据集、分类+分割、self-stitch控制实验、预测分析、stitch层类型对比，极其全面
- 写作质量: ⭐⭐⭐⭐⭐ 以问题为导向的叙事结构（"Are VFMs stitchable?"）清晰有力
- 价值: ⭐⭐⭐⭐⭐ 对多VFM系统的计算效率有直接实用价值，理论insight也有参考意义
