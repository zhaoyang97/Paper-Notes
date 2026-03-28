# TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model

**会议**: AAAI 2026  
**arXiv**: [2503.24067](https://arxiv.org/abs/2503.24067)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: Transformer, Mamba, SSM, 混合架构, 序列建模

## 一句话总结

提出 TransMamba，一种序列级别的 Transformer-Mamba 混合架构，通过共享 QKV/CBx 参数和 Memory Converter 在不同 token 长度时动态切换 Attention 和 SSM，兼顾长短序列的效率。

## 研究背景与动机

1. **领域现状**：Transformer ($O(T^2)$复杂度)是LLM的主流架构。Mamba（SSM, $O(T)$线性复杂度）在长序列上效率更高但上下文学习和多任务泛化不稳定。现有混合方法（Jamba、Zamba等）采用层级交替（固定比例的Transformer层+Mamba层），但存在结构僵化问题——必须遵循特定的层顺序和比例规则。
2. **现有痛点**：(a) Transformer在短上下文训练更快，Mamba在长上下文更高效——但层级混合无法在同一序列内利用两者各自的效率优势；(b) 层级混合比例固定（如4:1），不满足特定规则性能就会退化；(c) Mamba2揭示了Attention和SSM在数学形式上的一致性（对偶形式），Wang等人通过蒸馏验证了QKV和CBx参数可互相转移——这暗示可以更大胆地统一两种机制。
3. **核心矛盾**：需要一种灵活的框架，在同一序列的不同位置自适应使用Attention或SSM，且切换时信息不丢失。
4. **切入角度**：利用Attention和SSM的参数一致性（Q↔C, K↔B, V↔x），让一套参数同时支持两种计算模式。
5. **核心idea一句话**：共享QKV/CBx参数+Memory Converter无损转换+TransPoint调度 = 序列级别灵活切换Attention和SSM。

## 方法详解

### 整体框架
TransMamba是层堆叠的Decoder-only自回归模型。每层包含Mamba的全部参数（C/B/x/A/Δ），基于Attention和SSM的参数一致性让QKV和CBx共享参数（Q↔C, K↔B, V↔x）。序列在TransPoint之前用Attention计算，之后用SSM计算。

### 关键设计

1. **共享参数映射（QKV↔CBx）**：
   - 做什么：一套参数同时支持Attention和SSM两种计算模式
   - 核心思路：前缀token ($h_s = h[:TransPoint]$) 通过共享参数计算 Q=δ(h_s W_C), K=δ(h_s W_B), V=δ(h_s W_x)，用Attention机制输出 $y_s = \text{softmax}(QK^T) \cdot V$。后续token ($h_l = h[TransPoint:]$) 用相同参数计算C/B/x，加上Δ和A参数用SSM机制输出 $y_l$
   - 设计动机：Mamba2揭示的对偶形式 $(L \circ QK^T)V$ vs $(A^{\times} \circ CB^T)X$ 表明两种机制的核心权重是可统一的

2. **Memory Converter（无损信息转换）**：
   - 做什么：在TransPoint处将Attention的K/V无损转换为SSM的初始隐状态 $h_0$
   - 核心思路：$h_0 = \text{MemoryConverter}(K, V)$。将SSM的隐状态递推展开为矩阵形式 $h = (A^{\times} \circ B^T)X$，理论证明Attention的K/V可以完美保留SSM所需的序列状态信息
   - 设计动机：如果不做转换直接切换，SSM部分会丢失前缀token的所有上下文信息——Memory Converter是整个框架可行的关键

3. **TransPoint调度策略**：
   - 做什么：决定每层在哪个token位置从Attention切换到SSM
   - 核心思路：每层设置单一TransPoint。训练FLOPs为 $O(P^2N + (T-P)N^2)$（P为TransPoint位置），是P的二次函数，存在最优解。探索了不同层不同TransPoint的灵活配置
   - 设计动机：P太小则Attention覆盖不足，P太大则SSM效率优势被稀释。最优TransPoint取决于序列长度和模型维度的比例

### 损失函数 / 训练策略

标准语言模型交叉熵损失 + Memory Converter 的重建损失。训练 FLOPs 为 $O(P^2 N + (T-P)N^2)$（$P$ 为 Attention 前缀长度），优于纯 Transformer 的 $O(T^2 N)$。

## 实验关键数据

### 主实验（400M模型，通用任务）

| 模型 | ARC-E | ARC-C | CoQA | OBQA | PIQA | BoolQ |
|------|-------|-------|------|------|------|-------|
| Transformer-400M | 60.57 | 58.72 | 5.07 | 42.4 | 52.75 | - |
| Mamba-400M | 较低 | 较低 | 较高 | 较低 | 较低 | - |
| **TransMamba-400M** | **最优** | **最优** | **最优** | **最优** | **最优** | **最优** |

- 同等83B token训练预算下，TransMamba在多数通用基准上超越纯Transformer和纯Mamba基线
- 在PhoneBook（长程依赖测试）和LongBench-v2（长文本理解）上得益明显

### 效率结果

| 配置 | 训练FLOPs/层 | 说明 |
|------|-------------|------|
| 纯Transformer | $O(T^2 N)$ | 短序列快但长序列瓶颈 |
| 纯Mamba | $O(TN^2)$ | 长序列高效但短序列不如Transformer |
| **TransMamba（最优P）** | $O(P^2 N + (T-P)N^2)$ | 二者优势兼得 |

- 在N=1536、T=8192设置下，最优TransPoint P≈2048，此时训练效率最优

### 消融实验

| 配置 | 效果 |
|------|------|
| 完整TransMamba | 最优——效率和性能兼顾 |
| w/o Memory Converter | 显著下降——SSM失去前缀上下文 |
| 固定TransPoint（所有层相同） | 次优——突变式切换导致性能损失 |
| 对数分布TransPoint（每8层循环） | 最佳——渐进式切换平衡了分散性和效率 |
| 纯Attention（P=T） | 短序列好但长序列效率差 |
| 纯SSM（P=0） | 长序列好但短序列表达不足 |

### 关键发现
- Memory Converter是必要条件——去掉后SSM部分性能崩溃，验证无损转换的重要性
- 序列级混合比层级混合更灵活高效——短序列获得Attention的全局交互，长序列获得SSM的线性复杂度
- TransPoint最优位置与模型维度N和序列长度T的比例相关——当T>N时SSM效率占优
- 不同层设置不同TransPoint比统一设置更优——对数分布（0/128/256/512/1024/2048/4096/8192每8层循环）效果最好
- 推理时可采用与训练不同的TransPoint策略——训练用最高效结构，推理时可切换到更适合任务的结构

## 亮点与洞察
- **在更深层次上验证了Transformer和Mamba的一致性**：不仅是数学形式上的对偶，而是实际证明同一套参数可以在两种模式下工作——这比Mamba2的理论一致性更进一步
- **Memory Converter的理论保证**是关键技术贡献：证明了K/V到SSM隐状态的无损转换是可行的，这使得在序列内部切换计算模式成为可能
- 共享参数设计使模型在不同序列长度下退化为不同架构（纯Transformer、纯Mamba或混合），灵活性极强

## 局限性 / 可改进方向
- 每层仅设置单一TransPoint，更复杂的多TransPoint结构可能有更好效果
- TransPoint调度目前基于先验/搜索确定，可探索自适应学习
- Memory Converter增加额外计算——虽然理论无损但实际精度待量化分析
- 仅在语言建模上验证，其他模态（视觉/多模态）的适用性未知

## 相关工作与启发
- **vs Jamba/Zamba等层级混合**：层级混合比例固定且不灵活；TransMamba在序列级别自适应切换
- **vs Mamba2**：Mamba2揭示对偶形式但仍是独立架构；TransMamba实际实现了参数共享的统一框架

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个序列级别统一Transformer和Mamba的框架，参数共享+无损转换理论贡献突出
- 实验充分度: ⭐⭐⭐ 语言建模验证充分但缺少下游任务
- 写作质量: ⭐⭐⭐⭐ 架构设计清晰，数学推导完整
- 价值: ⭐⭐⭐⭐ 为Transformer-SSM混合架构开辟了新范式
