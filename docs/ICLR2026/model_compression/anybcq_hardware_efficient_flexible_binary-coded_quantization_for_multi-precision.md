# AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

**会议**: ICLR 2026  
**arXiv**: [2510.10467](https://arxiv.org/abs/2510.10467)  
**代码**: [https://github.com/naver-aics/anybcq](https://github.com/naver-aics/anybcq)  
**领域**: 模型压缩 / LLM量化  
**关键词**: 二进制编码量化, 多精度推理, bit-plane操作, LLM部署, CUDA内核  

## 一句话总结
提出AnyBCQ，基于二进制编码量化(BCQ)的多精度LLM量化框架，通过渐进式精度扩展（冻结已有bit-plane+添加残差bit-plane）支持单个模型在2-4bit之间动态切换，专设CUDA内核直接在bit-plane级别计算避免查表/转置开销，在2-bit下准确率大幅超越Any-Precision LLM（MMLU 35.3% vs 24.7%），吞吐量最高达到FP16的3.0x。

## 研究背景与动机

1. **领域现状**：多精度LLM模型允许单一模型在运行时根据SLO动态选择精度。Any-Precision LLM是当前SOTA，但依赖非均匀量化（clustering-based），需要查表和bit-transpose操作。
2. **现有痛点**：(a) 非均匀量化无法直接在bit-plane上计算，需要额外的转置+查表开销；(b) 现有多精度方法在极低bit（如2-bit）时准确率崩溃，实际可用范围局限于3-4bit；(c) 存储多个独立精度模型的内存开销大（LLaMA-3.1-8B需9.85GB vs AnyBCQ的4.99GB）。
3. **核心矛盾**：非均匀量化表达力强但不适合硬件加速（需查表），BCQ天然适合硬件但固定精度。
4. **本文要解决什么**：将BCQ扩展到多精度设置，保持硬件友好性的同时支持动态精度切换。
5. **切入角度**：BCQ将权重表示为二进制bit-plane的线性组合 $\hat{W} = \sum_i \alpha_i B_i$，p-bit推理恰好对应p个bit-plane计算——天然支持多精度。
6. **核心idea一句话**：冻结低精度bit-plane→从残差添加新bit-plane→仅优化缩放因子，实现渐进式精度扩展。

## 方法详解

### 整体框架
AnyBCQ分两部分：(1) 离线量化——从基础精度$p_L$开始greedy初始化BCQ，逐步扩展到目标精度$p_H$，每步冻结已有binary codes只优化缩放因子；(2) 在线推理——专用CUDA内核按需加载bit-plane，直接用加减法计算（无需查表），支持per-request精度选择。

### 关键设计

1. **渐进式精度扩展（Progressive Precision Expansion）**
   - 做什么：从2-bit BCQ起步，逐步扩展到3-bit、4-bit
   - 核心思路：基础精度$p_L$: greedy初始化 → alternating refinement（LS求解α + BS更新B）。扩展到$p+1$: 冻结$B_1,...,B_p$ → 新bit-plane $B_{p+1} = \text{sign}(R_p)$（残差的符号）→ 仅用LS优化新精度的全部缩放因子$\{\alpha_i^{p+1}\}_{i=1}^{p+1}$
   - 设计动机：共享binary codes大幅节省存储（binary占主导地位），LLaMA-3.1-8B从9.85GB降至4.99GB（-49%）。精度单调递增保证

2. **直接bit-plane运算的CUDA内核**
   - 做什么：避免bit-transpose和centroid lookup
   - 核心思路：每次加载一个bit-plane→$B_i \in \{-1,+1\}$所以GEMM退化为加减法→用LUT-GEMM缓存常见结果→乘以$\alpha_i$后累加为partial sum→p个bit-plane完成后输出
   - 设计动机：非均匀量化需要bit-transpose(O(MKp)) + centroid lookup(O(MK))，BCQ直接计算省去这两步。低精度时内存带宽按比例减少（3-bit只加载3个plane而非4个再丢弃1个）

### 训练策略
- 512序列C4校准数据，10 epochs MRE优化
- 非对称BCQ + group-wise量化（g=128）
- 初始化20轮alternating refinement

## 实验关键数据

### 主实验（LLaMA-3.1-8B）

| 方法 | 2-bit MMLU | 3-bit MMLU | 4-bit MMLU |
|------|-----------|-----------|------------|
| AWQ | 24.12 | 47.28 | 60.49 |
| Any-Precision LLM | 24.66 | 55.53 | **64.04** |
| ShiftAddLLM | 24.83 | 56.53 | 63.50 |
| **AnyBCQ (Multi)** | **35.32** | 58.28 | 63.53 |
| FP16 | 65.02 | - | - |

AnyBCQ在2-bit下MMLU超越对手10+个百分点。4-bit下与Any-Precision LLM接近。

### 吞吐量
- 相比FP16: 最高3.0x加速
- 相比Any-Precision LLM: 最高1.2x加速
- 动态精度切换开销可忽略

### 关键发现
- 2-bit是区分度最大的区间——AnyBCQ的BCQ方案远优于非均匀量化
- Multi-prec vs Fixed-prec差距在3-4bit时出现——共享binary constraint压缩了优化空间
- 4-bit时各方法差距收敛——量化误差已经很小

## 亮点与洞察
- **BCQ天然适合多精度**的洞察是本文核心——p-bit计算=p个bit-plane加法，这使得BCQ在多精度场景下成为唯一不需要查表的方案
- **内存节省49%**（vs多模型方案）同时保持准确率，实用性极强
- 在2-bit这个传统"死区"（MMLU 24%→35%）取得了显著进步

## 局限性 / 可改进方向
- 4-bit时准确率略低于Any-Precision LLM——BCQ的表达力限制在高精度时显现
- 仅在LLaMA-3.1-8B上验证，更大模型待测试
- 渐进扩展中binary codes一旦冻结无法修正——早期的错误会传播到高精度
- 仅考虑weight-only量化，activation量化未涉及

## 相关工作与启发
- **vs Any-Precision LLM**: 非均匀量化表达力更强但不适合硬件；BCQ牺牲少量高精度准确率换取巨大的硬件效率优势和低bit性能
- **vs ShiftAddLLM**: 同为BCQ方法但ShiftAddLLM只支持固定精度，AnyBCQ扩展到多精度
- **vs GPTQ/AWQ**: 均匀量化在2-bit下完全崩溃，BCQ的二进制bit-plane结构更鲁棒

## 评分
- 新颖性: ⭐⭐⭐⭐ BCQ→多精度的扩展思路自然但有效，CUDA内核设计有工程创新
- 实验充分度: ⭐⭐⭐⭐ 多基准+吞吐量+消融完整，但仅一个模型
- 写作质量: ⭐⭐⭐⭐ Figure 1-3的对比图非常直观
- 价值: ⭐⭐⭐⭐⭐ 填补了BCQ在多精度LLM部署中的空白，实用性强
