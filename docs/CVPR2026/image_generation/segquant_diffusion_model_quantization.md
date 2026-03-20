# SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models

**会议**: CVPR 2026  
**arXiv**: [2507.14811](https://arxiv.org/abs/2507.14811)  
**代码**: [https://github.com/OptiSys-ZJU/segquant](https://github.com/OptiSys-ZJU/segquant) (有)  
**领域**: 模型压缩 / 扩散模型  
**关键词**: 后训练量化, 语义感知分段, 极性保持, 部署友好, DiT量化  

## 一句话总结
提出 SegQuant，一个面向部署的扩散模型后训练量化框架，通过基于计算图静态分析的语义感知分段量化（SegLinear）和硬件原生的双尺度极性保持量化（DualScale），在 SD3.5、FLUX、SDXL 上实现跨架构通用的高保真 W8A8/W4A8 量化，同时保持与 TensorRT 等工业推理引擎的兼容性。

## 背景与动机
扩散模型计算密集，量化是降低推理开销的有效手段。后训练量化（PTQ）因不需重训练而最具实用性，但现有扩散模型 PTQ 方法存在"编译器鸿沟"（Compiler Gap）：

1. **架构特定启发式**：如 Q-Diffusion 手工编写 UNet skip-connection 的分割规则，无法推广到 DiT
2. **运行时动态依赖**：如 PTQ4DiT 依赖时间步变化的激活值分布（salient channels），与现代静态图编译器不兼容
3. **极性不对称问题**：DiT 使用 SiLU/GELU 激活函数，输出分布正负极不对称（负值范围窄但语义信息丰富），标准量化过度压缩负值导致高频细节丢失

## 核心问题
如何设计一个既高性能又编译器原生的通用量化框架，解决扩散模型中的语义异质性和激活极性不对称问题？

## 方法详解

### 整体框架
SegQuant 是一个模块化的自顶向下框架，由四个可替换组件构成：Optimizer（如 SmoothQuant/SVDQuant）、Calibrator（如 GPTQ/AMax）、SegLinear（语义分段）和 DualScale（极性保持）。用户可以自由替换 Optimizer 和 Calibrator，SegLinear 和 DualScale 作为核心增强模块插入。

### 关键设计

1. **SegLinear（语义感知分段量化）**: 核心观察——DiT 中的 AdaNorm、TimeEmbedding 等模块的线性层处理语义异质的输入，如 chunk/split 操作后的不同语义分支（时间信息 vs 潜在表示）。SegLinear 通过 torch.fx 静态计算图分析，自动检测 chunk/split（输出分段）和 concat/stack/reshape（输入分段）模式，将权重矩阵和激活按语义边界分割后独立量化。例如 SD3.5 的 DiT.0.norm1 层被自动推断为 6 个 chunk，分段量化的 F-norm 误差从 0.70 降至 0.54。关键优势：完全基于静态图分析，无需运行时数据，与 AI 编译器天然兼容。

2. **DualScale（双尺度极性保持量化）**: 专门解决 SiLU/GELU 的正负极性不对称问题。将激活矩阵分解为正部分 $X^+ = \max(X, 0)$ 和负部分 $X^- = \min(X, 0)$，各自用独立的量化尺度 $s^+, s^-$，然后分别做矩阵乘法再线性组合：$Y \approx s^+ s_w (X^+ W) + s^- s_w (X^- W)$。看似需要两次 GEMM，但实际通过 CUTLASS 的 BatchedGEMM 在一次 kernel launch 中并行执行两路计算，在融合 epilogue 中完成组合。相比非对称量化，DualScale 无需零点修正，更简洁高效。

3. **静态图模式检测算法**: 用 torch.fx 符号追踪 + shape propagation 获取完整计算图，自动遍历查找权重分段（Linear → chunk/split）和输入分段（concat/stack/reshape → Linear）模式，分段大小从算子参数直接推断。

### 损失函数 / 训练策略
- 完全免训练的 PTQ 方法
- SmoothQuant 的迁移强度 α 逐层搜索（0.0-1.0，步长 0.1），选择最小化 MSE 的值
- 校准集：256 张图（SD3/SDXL），64 张（FLUX 8-bit），32 张（FLUX 4-bit）
- 8-bit 用 per-tensor 方案，4-bit 用 per-channel 权重 + per-token 动态激活

## 实验关键数据

### MJHQ-30K 主实验（SD3.5 DiT W8A8）

| 方法 | FID ↓ | Image Reward ↑ | LPIPS ↓ | PSNR ↑ | SSIM ↑ |
|------|-------|---------------|---------|--------|--------|
| PTQD | 较差 | 较差 | 较差 | 较差 | 较差 |
| PTQ4DiT | 中等 | 中等 | 中等 | 中等 | 中等 |
| Smooth+ | 中等 | 中等 | 中等 | 中等 | 中等 |
| **SegQuant-G** | **最优** | **最优** | **最优** | **最优** | **最优** |

SegQuant 在 SD3.5、FLUX-DiT、SDXL-UNet 三种架构上一致性领先。

### 消融实验（SD3.5 W8A8, MJHQ）

| 方法 | FID ↓ | IR ↑ | LPIPS ↓ | PSNR ↑ |
|------|-------|------|---------|--------|
| Baseline (SmoothQuant) | 23.35 | 0.877 | 0.419 | 11.93 |
| + SegLinear | 23.36 | 0.899 | 0.395 | 12.03 |
| + DualScale | 22.61 | 0.909 | 0.401 | 12.14 |
| + 两者 | **22.54** | **0.952** | **0.377** | **12.50** |

两个模块互补，组合效果最优。

### 单层误差分析

| 层 | 方法 | 无SegLinear | 有SegLinear |
|----|------|-----------|-----------|
| DiT.0.norm1 | GPTQ | 0.835 | **0.444** |
| DiT.11.norm1_context | GPTQ | 3.017 | **1.761** |

SegLinear 在敏感层上大幅降低量化误差。

### 效率分析
- SegQuant 仅对 12%-29% 的层应用 DualScale，额外内存开销仅 3.4-5.6 MB
- DualScale kernel 推理速度优于 PTQ4ViT 的 AdaNorm 量化方案
- 校准耗时：SD3 约 2.5 小时（单 L20 GPU，256 张校准图）

## 亮点
- **编译器友好**：完全基于静态图分析，不依赖运行时数据，可直接集成到 TensorRT 等部署管线
- **跨架构通用**：同一框架适用于 DiT（SD3.5、FLUX）和 UNet（SDXL），无需手工规则
- **语义分段自动化**：Q-Diffusion 需要手工为 UNet skip-connection 编写规则，SegLinear 通过图分析自动发现所有需要分段的层
- **DualScale 硬件原生**：利用 BatchedGEMM 将两次 GEMM 融合为一次 kernel launch，避免自定义数据格式或非标准 kernel

## 局限性 / 可改进方向
- **校准成本**：超参搜索（α 步长 0.1）需 ~25 小时，虽然是一次性开销但仍较高
- **FLUX 内存限制**：48GB GPU 需要 swap-in/swap-out 策略进行校准
- **未涉及视频扩散模型**：虽然方法理论上通用，但未在视频生成模型上验证
- **4-bit 精度下仍有较大质量损失**：W4A8 场景虽然优于基线但与 FP16 差距仍明显
- 定量结果中具体数值因 HTML 渲染问题（\cellcolorlightgray 标记）未完全可读

## 与相关工作的对比
- **vs Q-Diffusion**: Q-Diffusion 手工设计 UNet skip-connection 分割规则，属于架构特定启发式；SegLinear 通过计算图自动分析实现通用化
- **vs PTQ4DiT**: PTQ4DiT 依赖运行时时间步变化的激活统计，与静态图编译器不兼容；SegQuant 完全基于静态结构
- **vs SVDQuant**: SVDQuant 用低秩分解处理离群值，SegQuant 可将其作为 Optimizer 组件集成使用（实际在 4-bit 实验中就是用 SVDQuant 作为 Optimizer）

## 启发与关联
- SegLinear 的计算图语义分析方法可以推广到 VLM 模型的量化——VLM 中文本和视觉 token 同样有语义异质性
- DualScale 的极性分解思想可能对其他使用 SiLU/GELU 的大模型（如 LLM）也有价值
- 整体模块化设计（Optimizer + Calibrator + 增强模块）值得在量化框架设计中借鉴

## 评分
- 新颖性: ⭐⭐⭐⭐ 语义分段和极性保持的设计有洞察力，但总体是工程导向的框架创新
- 实验充分度: ⭐⭐⭐⭐⭐ 三种架构、三个数据集、多种精度设置、详细消融和实际效率分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机明确，术语统一
- 价值: ⭐⭐⭐⭐ 解决了扩散模型量化的工程化部署痛点，实用性强
