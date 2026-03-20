# AdmTree: Compressing Lengthy Context with Adaptive Semantic Trees

**会议**: NeurIPS 2025  
**arXiv**: [2512.04550](https://arxiv.org/abs/2512.04550)  
**代码**: 无（论文中未提供链接）  
**领域**: LLM效率 / 模型压缩  
**关键词**: 上下文压缩, 语义树, gist token, 层次化压缩, 长上下文  

## 一句话总结
提出 AdmTree——一种自适应层次化上下文压缩框架,通过信息密度驱动的动态分段构建叶 gist token，再用二叉语义树底向上聚合实现多粒度语义保留，解决了显式方法丢失局部细节和隐式方法位置偏差的双重问题,在 LongBench 上比 SOTA 基线 Activation Beacon 高 10%+。

## 研究背景与动机

1. **领域现状**：长上下文处理因 self-attention 的二次复杂度面临计算瓶颈。上下文压缩分为：显式方法（直接删除不重要文本）和隐式方法（将上下文编码为紧凑向量/gist token）。
2. **现有痛点**：(a) **显式方法**（如 LongLLMLingua）保留全局语义但破坏局部连贯性——摘要粒度越细性能越差；(b) **隐式方法**（如 AutoCompressor, ICAE）存在位置偏差——中间和早期信息被遗忘（"lost in the middle"）；(c) 递归压缩（如 Activation Beacon）用固定分段不考虑信息密度差异，且单向压缩导致信息逐步退化。
3. **核心矛盾**：如何同时保留全局+局部、均匀跨位置的语义信息？
4. **切入角度**：借鉴认知科学中的层次化信息处理——用树结构在广度（跨上下文覆盖）和深度（细粒度）间取得平衡。
5. **核心 idea 一句话**：信息密度自适应分段 + 二叉语义树层次聚合 + 双向注意力消除位置偏差。

## 方法详解

### 整体框架
长文本 $\mathbf{X}$ → 按信息密度自适应分段 → 每段后插入一个 gist token（叶节点） → 叶 gist token 底向上两两聚合构建二叉语义树 $\mathcal{T}$ → 基于树结构的压缩编码 → 条件生成。LLM 骨干冻结，仅训练 gist 注意力头 + gist embedding + 聚合器。

### 关键设计

1. **自适应叶 Gist Token 构建**:
   - 做什么：根据信息密度动态分配 gist token 预算
   - 核心思路：先均匀切分为初始段，计算每段的信息含量分数 $\text{Score}(\mathbf{X}_i) = \text{PPL} \cdot \exp(-\lambda \cdot \text{Entropy})$，高分段分配更多 gist token（更细粒度），低分段更少
   - 分配规则：top 25% 段得 $n/\tau$ 个 gist，middle 25% 得 $n/2\tau$，bottom 50% 得 $n/4\tau$（全局压缩率不变）
   - 设计动机：信息密集区域需要更多"存储空间"，信息稀疏区域可以更激进压缩

2. **语义二叉树构建**:
   - 做什么：将叶 gist token 自底向上聚合为层次化语义表示
   - 核心思路：每两个叶节点通过聚合函数生成父节点 $h_v = \text{Agg}(\{h_u | u \in C_v\})$，聚合函数是单层 self-attention + 平均
   - 增量构建：处理新子段时，仅需重用已有树 $\mathcal{T}_{<k}$ 并插入新叶节点
   - 设计动机：(a) 树结构使后续节点的信息可以双向影响前面节点（消除单向退化）；(b) 多层级自然提供全局到局部的语义

3. **基于树的压缩编码**:
   - 做什么：编码当前子段时利用语义树作为上下文
   - 核心思路：双分支注意力——text token 用原始 LLM 的 attention heads，gist token 用新训练的 attention heads ($W_q^{gt}, W_k^{gt}, W_v^{gt}$)，计算后拼合做联合 self-attention
   - 树节点按"左到右、底到上"顺序展平插入序列
   - 注意力范围：gist token 可注意到前面所有树节点和当前段的 text token

### 训练策略
- LLM 骨干（如 LLaMA-2-7B-Chat）完全冻结
- 仅训练：gist 注意力头 $\theta_{gt\_attn}$, gist embedding $\theta_{gt\_emb}$, 聚合器 $\theta_{agg}$
- 损失：标准 next-token prediction（条件于树和局部上下文）
- 压缩率：4K-8K ×2, 8K-16K ×4, 16K-32K ×8

## 实验关键数据

### 主实验 (LongBench, LLaMA-2-7B)

| 方法 | SingleDoc | MultiDoc | Summ. | FewShot | Code | AVG |
|------|-----------|----------|-------|---------|------|-----|
| Original LLM | 24.7 | 22.4 | 24.6 | 63.2 | 57.7 | 37.2 |
| AutoCompressor | 差 | 差 | 差 | 差 | 差 | 低 |
| ICAE | 差 | 差 | 差 | 差 | 差 | 低 |
| Activation Beacon | 好 | 好 | 好 | 好 | 好 | ~42 |
| **AdmTree** | **最佳** | **最佳** | **最佳** | **最佳** | **最佳** | **~52+** |

### 关键对比

| 对比维度 | AdmTree 优势 | 说明 |
|---------|-------------|------|
| vs Activation Beacon | +10% avg | LLaMA-2-7B 上 |
| vs Activation Beacon | +5% avg | Qwen-2-7B 上 |
| QA 任务最大提升 | +20 points | 多文档 QA 上 |
| 延迟 | 与递归方法相当 | 聚合器开销极小 |

### 关键发现
- **QA 任务提升最大**：因为 QA 需要同时保留全局位置信息和局部细节，AdmTree 的树结构正好匹配这一需求
- **自适应分配 vs 均匀分配**：自适应 gist 分配在高信息密度区域提供更多细节保留
- **双向聚合消除位置偏差**：Self-attention 聚合允许后续信息影响前序节点，解决 "lost in middle"
- **可解释性**：树结构的注意力可视化可以展示信息如何在不同粒度间流动

## 亮点与洞察
- **认知科学启发的层次化设计**：模拟人类信息处理的层次性——先记住细节，再逐步抽象。树结构天然提供多粒度语义
- **信息密度驱动的动态分段**：PPL × exp(-λ·Entropy) 作为信息含量指标简单有效，高信息段自动获得更多 gist token
- **LLM 骨干完全冻结**：参数效率极高，仅训练 gist 相关的少量参数
- **增量构建**：语义树可随新上下文到达而动态更新，适合流式长文本处理

## 局限性 / 可改进方向
- **树深度有限**：二叉树的深度为 $\log_2 M$，超长上下文可能需要很深的树
- **分段评分函数简单**：PPL+Entropy 的评分可能不适合所有语义维度
- **仅在 2 个 LLM 上验证**：LLaMA-2-7B 和 Qwen-2-7B，更大模型和不同架构的泛化性未知
- **改进方向**：(1) 非二叉树结构（根据语义自适应度数）；(2) 更复杂的信息密度度量；(3) 与 KV cache 压缩方法结合

## 相关工作与启发
- **vs Activation Beacon**：Beacon 用固定大小递归压缩（线性），AdmTree 用自适应树结构（层次化），在语义保留上全面优于 Beacon
- **vs AutoCompressor/ICAE**：这些方法用固定数量 gist token，无法适应不同长度和复杂度，AdmTree 的自适应分配解决了这个问题
- **vs SnapKV**：SnapKV 是 KV cache 驱逐方法，与上下文压缩正交，AdmTree 是在输入端做压缩
- **vs LongLLMLingua**：显式方法删除文本，AdmTree 是隐式方法压缩为向量，信息保留更完整

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 语义树压缩框架新颖，信息密度自适应+层次聚合的组合有原创性
- 实验充分度: ⭐⭐⭐⭐ LongBench 全面评估，两个 LLM，多类型任务对比
- 写作质量: ⭐⭐⭐⭐ 前置实验（Figure 1）动机分析到位，方法描述系统完整
- 价值: ⭐⭐⭐⭐⭐ 大幅超越 SOTA（+10%），解决了长上下文压缩的核心问题，实用性强
