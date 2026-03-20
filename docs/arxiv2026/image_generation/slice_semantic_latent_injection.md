# SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking

**会议**: 投稿中  
**arXiv**: [2603.12749](https://arxiv.org/abs/2603.12749)  
**代码**: 未提及  
**领域**: 扩散模型  
**关键词**: slice, semantic, latent, injection, compartmentalized  

## 一句话总结
为了解决这个限制并提供值得信赖的语义感知水印，我们提出通过 $\underline{\textbf{C}}$ompartmentalized $\underline{\textbf{E}}$mbedding $\underline{\textbf{S}}$emantic $\underline{\textbf{L}}$atent $\underline{\textbf{I}}$njection ($\textbf{切片}$)。
## 核心问题
然而，它们对单一全局语义绑定的依赖使它们容易受到本地化但全局一致的语义编辑的影响。
## 关键方法
1. 为了解决这个限制并提供值得信赖的语义感知水印，我们提出通过 $\underline{\textbf{C}}$ompartmentalized $\underline{\textbf{E}}$mbedding $\underline{\textbf{S}}$emantic $\underline{\textbf{L}}$atent $\underline{\textbf{I}}$njection ($\textbf{切片}$)
2. 最近的语义感知水印方法通过对图像语义进行条件验证来提高鲁棒性
3. 然而，它们对单一全局语义绑定的依赖使它们容易受到本地化但全局一致的语义编辑的影响
4. 我们的框架将图像语义解耦为四个语义因素（主题、环境、动作和细节），并将它们精确地锚定到初始高斯噪声中的不同区域
## 亮点 / 我学到了什么
- 实验结果表明，针对高级语义引导再生攻击，SLICE 的性能显着优于现有基线，在保持图像质量和语义保真度的同时大幅降低攻击成功率
## 局限性 / 可改进方向
- 需要详细阅读原文以确认具体局限
- 潜在扩展方向待进一步分析

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 详见原文 | 详见原文 | — | — | — |

## 与我的研究方向的关联
- `causal_diffusion`
- `fractal_diffusion_design`
- `process_aware_alignment`

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐
