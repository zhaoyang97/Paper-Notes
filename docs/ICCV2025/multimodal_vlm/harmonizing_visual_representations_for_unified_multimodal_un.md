# Harmonizing Visual Representations for Unified Multimodal Understanding and Generation

**会议**: ICCV 2025  
**arXiv**: [2503.21979](https://arxiv.org/abs/2503.21979)  
**代码**: [https://github.com/wusize/Harmon](https://github.com/wusize/Harmon)  
**领域**: 多模态VLM / 统一理解与生成  
**关键词**: unified multimodal, MAR encoder, visual tokenization, understanding+generation, autoregressive  

## 一句话总结
发现Masked Autoregressive (MAR)模型的编码器同时具备优秀的语义理解能力和生成能力，基于此提出Harmon框架——用共享的MAR编码器统一视觉理解和生成任务，通过三阶段渐进训练在生成benchmark上达SOTA同时在理解benchmark上匹配专用语义编码器方法。

## 背景与动机
统一视觉理解和生成是多模态研究的重要目标，但两个任务对视觉表示的需求存在内在矛盾：理解需要高级语义特征，生成需要低级像素/纹理信息。现有统一方法（如使用VQ或VAE统一视觉表示）通常优先保留图像的内在特征（有利生成），牺牲了语义信息（损害理解）。Janus等方法为此引入双编码器——一个用于理解、一个用于生成——但增加了复杂性。

## 核心问题
能否找到一种视觉表示，同时满足理解（高级语义）和生成（低级保真度）的需求，不需要双编码器？

## 方法详解

### 整体框架
Harmon是一个统一的自回归框架，使用单一的MAR（Masked Autoregressive）编码器同时处理理解和生成。三阶段训练：(1) MAR编码器预训练（mask-and-reconstruct）→ (2) 理解能力对齐（连接LLM进行VQA等任务微调）→ (3) 生成能力微调（保持理解同时优化生成质量）。

### 关键设计
1. **MAR编码器的双重能力发现**：通过对MAR编码器representation的系统分析，发现其具有：(a) 优秀的linear probing准确率（表示语义丰富）——因为mask-and-reconstruct的预训练任务要求模型理解全局语义来重建被mask的区域；(b) 精准的视觉概念特征响应——特征激活与图像中的物体/概念位置精确对应。这些发现表明MAR编码器天然具备理解+生成的双重潜力。

2. **共享编码器统一两任务**：不像Janus需要理解编码器+生成编码器，Harmon仅用一个MAR编码器——理解时提取特征给LLM，生成时提供token给autoregressive decoder。共享编码器意味着理解和生成可以互相促进——更好的语义理解有助于生成更准确的内容。

3. **三阶段渐进训练**：Stage 1训练MAR编码器建立基础表示；Stage 2冻结编码器连接LLM优化理解（VQA/caption等）；Stage 3联合优化理解和生成，确保两个能力不互相干扰。

### 损失函数 / 训练策略
MAR预训练loss + LLM instruction tuning loss + autoregressive生成loss，三阶段渐进。

## 实验关键数据
- **生成**：在GenEval、MJHQ30K和WISE benchmark上达到SOTA
- **理解**：在标准VQA/理解benchmark上匹配使用专用语义编码器（如CLIP-ViT）的方法（如Janus）
- 单一MAR编码器同时实现两个任务的competitive性能——验证了representation harmonization的可行性
- 对比消融表明三阶段训练中每阶段都有不可替代的贡献

### 消融实验要点
- MAR vs VQ tokenizer vs VAE：MAR在语义理解指标（linear probe）上显著优于VQ和VAE
- 共享编码器 vs 双编码器：共享方案在参数更少的同时性能相当
- 三阶段训练缺任何一阶段都会影响最终性能

## 亮点
- **MAR编码器的双重能力发现**是核心insight：MIM预训练的编码器不仅能重建图像（适合生成），还学到了丰富的语义（适合理解）
- **单编码器统一两任务**：比Janus的双编码器方案更优雅且参数更少
- **与Scaling Laws for NMM的发现互相印证**：NMM证明了early-fusion（无专用视觉编码器）可行，Harmon证明了单编码器可以同时服务理解和生成
- 来自NTU Chen Change Loy组，质量有保障

## 局限性 / 可改进方向
- MAR编码器的分辨率可能限制生成质量
- 理解性能虽然匹配Janus，但尚未超越使用大型CLIP编码器的方法
- 三阶段训练相对复杂
- 未探索视频理解和生成的统一

## 与相关工作的对比
- **vs. Janus/Janus-Pro**：Janus用双编码器解耦理解和生成；Harmon证明单个MAR编码器就够——更简洁
- **vs. Show-o/MUSE-VL**：这些也是统一模型但用VQ tokenizer，语义表示不够好；Harmon发现MAR编码器在语义和生成上都更优
- **vs. EVEv2**：EVEv2解决encoder-free VLM的理解问题；Harmon进一步统一了理解+生成

## 启发与关联
- MAR编码器的发现可能影响视觉tokenizer的设计方向——MIM预训练可能比VQ更适合统一表示
- 与REPA-E的VAE改善发现类似——扩散训练可以改善VAE，这里MIM预训练也能让编码器兼顾语义和重建

## 评分
- 新颖性: ⭐⭐⭐⭐ MAR编码器双重能力的发现有价值，单编码器统一方案优雅
- 实验充分度: ⭐⭐⭐⭐ 理解+生成双维度评估，与Janus等SOTA对比
- 写作质量: ⭐⭐⭐⭐ 分析+方法+实验逻辑清晰
- 价值: ⭐⭐⭐⭐ 为统一多模态模型提供了新的视觉表示选择
