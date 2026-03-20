# ATTS: Asynchronous Test-Time Scaling via Conformal Prediction

**会议**: ICLR 2026  
**arXiv**: [2509.15148](https://arxiv.org/abs/2509.15148)  
**代码**: [https://github.com/menik1126/Asynchronous-Test-Time-Scaling](https://github.com/menik1126/Asynchronous-Test-Time-Scaling)  
**领域**: LLM推理  
**关键词**: test-time scaling, speculative decoding, conformal prediction, asynchronous inference, rejection sampling  

## 一句话总结
提出 ATTS，一个基于 conformal prediction 的异步 test-time scaling 框架，通过将 rejection sampling 重构为假设检验过程来消除同步开销，在 MATH/AIME 等数学推理任务上实现最高 56.7x 加速和 4.14x 吞吐量提升，且无精度损失；1.5B/70B 的 draft/target 组合可达到 o3-mini (high) 的 AIME 水平。

## 研究背景与动机
1. **领域现状**：Test-time scaling（推理时增加计算预算）通过顺序缩放（更长推理链）和并行缩放（更多采样）显著提升 LLM 推理能力。Speculative decoding 是加速推理的自然选择（小模型生成、大模型验证）。
2. **现有痛点**：当 speculative decoding 遇到 test-time scaling 时面临两个瓶颈：(1) **内存瓶颈**——高并发采样时 KV cache 爆炸，GPU 内存溢出；(2) **同步开销**——rejection sampling 需要对所有候选进行全局排名或 softmax 归一化，随采样轮数指数增长的同步等待时间成为主要瓶颈。
3. **核心矛盾**：高效的 test-time scaling 需要同时沿并行和顺序维度缩放，但全局同步排名和归一化操作使得异步执行不可行——所有候选必须等待其他候选完成才能进行排名。
4. **本文要解决什么？** 如何在保持统计保证的前提下，消除 test-time scaling 中 rejection sampling 的同步瓶颈？
5. **切入角度**：引入 conformal prediction 构建 prediction set，用 p-value 替代归一化 softmax 分数做序数分类，使每个候选可以独立判断接受/拒绝，无需等待全局排名。
6. **核心idea一句话**：用 conformal prediction 的 p-value 替代全局排名实现异步 rejection sampling，消除 test-time scaling 的同步瓶颈。

## 方法详解

### 整体框架
ATTS 是一个三阶段 pipeline：(1) Draft 模型并行生成 m 个候选推理链；(2) 用 conformal p-value 异步判断每个候选是否落在 prediction set 内（接受/拒绝），无需等待所有候选；(3) 被接受的候选由 Target 模型继续生成。增加轮数实现顺序缩放，增加候选数实现并行缩放。

### 关键设计

1. **异步算术强度分析**:
   - 做什么：定义异步算术强度 $r = T_c / (T_m + T_s) \approx T_c / T_s$ 来量化性能瓶颈
   - 核心思路：传统算术强度只考虑计算和内存访问，但在 test-time scaling 中同步开销 $T_s$ 远大于内存访问时间 $T_m$，成为真正瓶颈。随采样数增加，$r$ 下降，说明同步是主要瓶颈
   - 设计动机：为异步设计提供理论动机和量化工具

2. **基于 Conformal Prediction 的序数分类**:
   - 做什么：将全局排名问题转化为独立的假设检验
   - 核心思路：对每个候选计算 conformal p-value $p_\xi^k$（基于非归一化的 conformity score $s_\xi^k = -\ell(X_\xi, \hat{Y}_\xi^k)$），与阈值 $\alpha$ 比较即可判断接受/拒绝。关键在于 p-value 不需要全局归一化——只需将当前 score 与 calibration set 中的历史 scores 比较即可
   - 设计动机：避免 softmax 归一化和全局排名的同步需求，每个候选可以独立异步评估
   - 统计保证：给出边际覆盖和条件覆盖的两种保证——$\mathbb{P}(y \in C_\alpha(Y)) \geq 1 - \alpha$

3. **在线校准 + Budget 预测**:
   - 做什么：在无 held-out 数据的 test-time 环境下动态维护 calibration set
   - 核心思路：用 memory bank 存储历史采样的 scores，随测试进行持续更新。rejection rate 由 $\alpha$ 精确控制——prediction set 大小恰好等于预定义的 budget $B$，避免 GPU OOM
   - 设计动机：test-time scaling 没有预留的校准集，必须在线积累

### 损失函数 / 训练策略
无需训练（training-free, lossless）。ATTS 完全工作在推理时，不修改模型权重。

## 实验关键数据

### 主实验（跨不同 Draft-Target 模型家族）

| Dataset | Draft Model | Target Model | Accuracy | Mar Speedup | Con Speedup |
|---------|------------|-------------|----------|-------------|-------------|
| MATH100 | Qwen2.5-7B-Inst | QwQ-32B | 96.0% (=TM) | **7.19x** | 5.35x |
| AIME24 | Qwen2.5-7B-Inst | QwQ-32B | 46.7% | **5.71x** | 10.10x |
| AIME25 | Qwen2.5-7B-Inst | QwQ-32B | 40.0% | **14.50x** | 12.82x |
| AMC23 | Qwen2.5-7B-Inst | QwQ-32B | 76.0% | **10.42x** | 8.20x |

### 大规模缩放结果

| 配置 | 说明 |
|------|------|
| 最高 56.7x 加速 | test-time scaling 场景下 |
| 4.14x 吞吐量提升 | 同时顺序+并行缩放 |
| 1.5B/70B draft/target | 达到 o3-mini (high) 的 AIME 水平 |
| Rejection rate 控制准确 | 与预设 $\alpha$ 高度一致 |

### 关键发现
- **跨家族 draft-target 组合有效**：即使 draft 和 target 来自不同模型家族（Qwen → QwQ, Llama → QwQ），ATTS 仍能提供有效的加速
- 红色标记的结果表示"无损加速"——加速后精度等于或超过 target model baseline
- 异步方案在采样数较多时优势明显——同步开销是指数增长的，而异步是常数
- 条件覆盖（per-instance 保证）通常比边际覆盖更保守但更可靠，对不同场景需权衡

## 亮点与洞察
- **conformal prediction 在 LLM 推理加速中的创新应用**：将统计学的 conformal prediction 引入 speculative decoding，用假设检验替代全局排名，是一个优雅的理论-工程结合
- **异步算术强度指标**：提供了量化 test-time scaling 瓶颈的新工具，可用于指导系统设计
- **"无损加速"的工程实用性**：training-free、model-agnostic、有统计保证，可直接部署

## 局限性 / 可改进方向
- 在线校准需要积累足够的历史 scores，冷启动阶段可能不够准确
- 精度在某些 draft-target 组合下不如 target model baseline（尤其是弱 draft model），说明 draft 质量仍很重要
- 仅在数学推理任务上验证，对开放式生成任务的适用性未知
- 需要同时部署 draft 和 target model，对 GPU 资源有额外需求

## 相关工作与启发
- **vs 标准 Speculative Decoding**: ATTS 将 speculative decoding 从 token-level 扩展到 chain-level（整条推理链），并解决了 test-time scaling 场景特有的同步瓶颈
- **vs BoN (Best-of-N)**: BoN 需要 N 条完整推理链全部生成完才能选择，ATTS 可以异步逐步筛选，大幅降低延迟
- **vs TPT 早停方法**: 早停可能剪掉正确推理路径，ATTS 通过 conformal 保证不丢失高质量候选

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ conformal prediction + 异步 test-time scaling 的结合非常新颖
- 实验充分度: ⭐⭐⭐⭐ 多 benchmark、多 draft-target 组合、加速和精度双指标
- 写作质量: ⭐⭐⭐⭐ 理论推导扎实，系统分析清晰
- 价值: ⭐⭐⭐⭐⭐ 为 test-time scaling 的高效部署提供了实用框架
