# CDG: Guiding Diffusion Models with Semantically Degraded Conditions

**会议**: CVPR 2026  
**arXiv**: [2603.10780]([https://arxiv.org/abs/2603.10780](https://arxiv.org/abs/2603.10780))  
**代码**: 有 ([https://github.com/Ming-321/Classifier-Degradation-Guidance](https://github.com/Ming-321/Classifier-Degradation-Guidance))  
**领域**: 扩散模型 / 图像生成 / 组合生成  
**关键词**: CFG替代, 条件退化引导, 组合语义, text-to-image, 语义控制  

## 一句话总结
提出CDG替代CFG——用语义退化条件替代空null prompt作为负面引导，将引导信号从粗粒度"好vs空"变为精细"好vs差一点"，在SD3/FLUX/Qwen-Image上显著提升组合精度，零额外计算。

## 背景与动机
Classifier-Free Guidance (CFG)是现代T2I扩散模型的基石——通过对比有条件生成和无条件生成(null prompt ∅)来增强条件信号。但CFG的null prompt是**语义空洞**的，生成的引导信号容易产生**几何纠缠**——对于"一只猫在桌子左边，一条狗在右边"这样的组合prompt，CFG无法精确区分空间语义，经常把属性搞混。根本原因是"好vs空"的对比太粗粒度了，就像考试只有"满分vs零分"的对比。

## 核心问题
如何构造更好的负面样本来改进扩散模型的引导信号？CFG的null prompt丢失了所有语义信息，导致引导信号模糊。理想的负面样本应该"差一点但不是完全没有"——这样对比才能产生精细的语义调整信号，就像"满分卷子vs90分卷子"的对比比"满分vs白卷"能发现更精准的错误。

## 方法详解

### 整体框架
CDG在推理时替代CFG的null prompt流程。不使用空条件∅，而是构造一个**语义退化条件c_deg**——部分破坏原始条件中的语义内容，但保留结构信息。引导公式变为：ε_guided = ε(c) + s·(ε(c) - ε(c_deg))，其中c_deg是"差一点"的条件。

### 关键设计
1. **Content vs Context Token发现**: 分析Transformer文本编码器中token的功能角色，发现token分为两类：**content token**（编码物体语义，如"cat""red"）和**context-aggregating token**（捕获全局上下文，如位置和关系）。这个发现是构造退化条件的基础。
2. **选择性语义退化**: 只退化content token——打乱/模糊物体语义词的表示，保持context token不变。这样c_deg仍有"大致结构"但丢失了"具体内容"。对比原始条件和退化条件，引导信号就能精确指向每个物体的具体语义，而不是泛泛的"有内容vs没内容"。
3. **零额外模型/训练**: 整个过程不需要外部模型，不需要训练，只是在推理时对文本编码做一次选择性退化操作。计算开销可忽略不计。

### 损失函数 / 训练策略
免训练。所有操作在推理时通过文本token的功能分类和选择性退化完成。

## 实验关键数据
| 架构 | 指标 | CDG | CFG | 效果 |
|------|------|-----|-----|------|
| SD3 | T2I对齐 | 显著提升 | baseline | 组合精度改善 |
| FLUX | T2I对齐 | 显著提升 | baseline | 一致改善 |
| Qwen-Image | T2I对齐 | 显著提升 | baseline | 跨架构有效 |

### 消融实验要点
- 只退化content token效果最好——退化context token反而有害
- 退化程度有sweet spot——太轻则引导信号弱，太重则接近null prompt
- 在所有测试架构上一致有效——说明方法的通用性

## 亮点 / 我学到了什么
- 🔥🔥 **挑战了CFG的核心假设** — null prompt不是最佳负面样本！这个质疑非常根本
- "好vs差一点"比"好vs空"更有信息量——这个原则可以推广到所有contrast-based方法（对比学习、DPO等）
- Content vs context token的功能分化是一个有价值的发现——可以用于其他需要选择性操作文本表示的任务
- **即插即用** = 现有所有扩散模型都能立即受益

## 局限性 / 可改进方向
- 退化程度是固定的，能否根据prompt复杂度自适应调整？
- Content/context token的分类是否在所有文本编码器中一致？
- 能否推广到视频生成的temporal consistency改进？
- → 与 `causal_diffusion.md` 的因果推理思路有潜在关联

## 与相关工作的对比
- **vs CFG**: CDG是CFG的直接升级——保持同样的推理框架，只替换负面样本构造方式。更精细的引导信号=更好的组合控制
- **vs Attend-and-Excite**: A&E通过注意力操作改进生成，CDG通过引导信号改进——两者正交且可结合
- **vs Composable Diffusion**: 组合扩散用多个条件做分解生成，CDG在单条件引导层面就改善了——更轻量

## 与我的研究方向的关联
- "更好的负面样本"原则适用于DPO/PPO的奖励设计
- Content/context token分化的发现对VLM的prompt engineering有价值
- 免训练+即插即用特别适合资源有限的研究场景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 质疑CFG的核心假设并提出优雅替代方案
- 实验充分度: ⭐⭐⭐⭐ 跨三种架构验证，但缺少更多量化指标
- 写作质量: ⭐⭐⭐⭐⭐ Motivation清晰，"good vs almost good"的比喻直觉
- 对我的价值: ⭐⭐⭐⭐ "更好的负面样本"原则可迁移到其他对比方法
