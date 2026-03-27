# Mozart: Modularized and Efficient MoE Training on 3.5D Wafer-Scale Chiplet Architectures

**会议**: NeurIPS 2025
**arXiv**: [2603.07006](https://arxiv.org/abs/2603.07006)
**代码**: 无
**领域**: LLM效率 / 硬件协同设计
**关键词**: MoE, chiplet architecture, expert parallelism, wafer-scale, algorithm-hardware co-design

## 一句话总结
提出 Mozart 算法-硬件协同设计框架，通过专家聚类分配、细粒度流式调度和 3.5D 晶粒架构（NoP-Tree + 分层存储），在三个 MoE-LLM 上实现 1.9× 以上的训练加速。

## 研究背景与动机
1. **领域现状**：MoE 架构以稀疏激活实现高效扩展（如 Mixtral-8x7B、DeepSeek-MoE），但其稀疏性给硬件部署带来内存局部性差、通信开销大、计算资源利用不均等挑战。
2. **现有痛点**：(1) 现有芯片方案多为子晶圆级设计，不支持晶圆级集成；(2) 采用粗粒度静态工作负载划分，假设密集均匀计算，对 MoE 的动态稀疏不友好；(3) Expert parallelism 中 all-to-all 通信是关键瓶颈。
3. **核心矛盾**：MoE 的逻辑模块化与硬件的物理模块化之间缺乏对齐——频繁共激活的专家可能被分配到远距离的计算单元上。
4. **本文要解决什么？** 设计匹配 MoE 模块化特性的晶粒架构和调度算法，减少通信开销并提高资源利用率。
5. **切入角度**：类比人脑的模块化组织——专门模块处理不同任务、相邻区域低延迟协调。利用专家激活的先验分析（激活频率 + 共激活模式）来指导专家-晶粒映射。
6. **核心idea一句话**：基于专家共激活先验，将频繁共激活的专家聚类到同一晶粒组，配合 3.5D NoP-Tree 拓扑和流式调度实现高效 MoE 训练。

## 方法详解

### 整体框架
两层优化：(1) 算法层——先分析路由策略获取专家激活先验，再进行专家聚类+分配+细粒度调度；(2) 架构层——3.5D 晶粒系统（3D logic-on-memory 堆叠 + 2D NoP-Tree 互连 + 两级存储）。

### 关键设计

1. **专家聚类与分配（Expert Clustering & Allocation）**:
   - 做什么：两阶段方法——先将频繁共激活的专家聚类（基于共激活矩阵 $\mathcal{C}$），再将聚类分配到晶粒组以均衡负载
   - 核心思路：聚类用 farthest point sampling 最大化组间距离；分配用二进制整数规划最小化组间负载不均
   - 设计动机：共激活专家在同一晶粒上，只需发送一份 token 副本（而非 k 份），直接减少 all-to-all 通信量 $\mathcal{C}_\mathcal{T}$

2. **细粒度流式调度（Fine-grained Streaming）**:
   - 做什么：通过 token 和 expert 的流式处理实现通信-计算重叠
   - 核心思路：将 expert 权重的 DRAM→SRAM 加载与 token 计算交错执行，避免一次性加载所有 expert 权重
   - 设计动机：MoE 每次只激活 k 个 expert，大量 expert 权重闲置，流式加载减少峰值内存需求

3. **3.5D 晶粒架构**:
   - 做什么：设计 attention 晶粒（central dispatchers）+ expert 晶粒（leaves）的 NoP-Tree 拓扑
   - 核心思路：3D 集成（compute die + SRAM die via hybrid bonding）提供低延迟本地激活缓存；2D NoP-Tree 实现网络内 MoE aggregation（switch 节点做消息聚合）
   - 设计动机：attention 和 MoE 的访存模式截然不同——attention 计算密集，MoE 通信密集，异构晶粒设计匹配这一差异

## 实验关键数据

### 对比 baseline
| MoE 模型 | 加速比 |
|----------|--------|
| Mixtral-8x7B | >1.9× |
| DeepSeek-MoE | >1.9× |
| 第三个模型 | >1.9× |

### 关键发现
- Expert 参数占 MoE-LLM 总参数的 90%+，但激活模式高度不均匀
- 共激活聚类可将 all-to-all 通信量降低 30-40%
- 流式调度实现了 80%+ 的通信-计算重叠

## 亮点与洞察
- **人脑类比的设计理念**：将神经科学的模块化理论映射到硬件设计，逻辑模块化（MoE experts）与物理模块化（chiplets）的对齐思路新颖
- **先验驱动的优化**：利用预训练模型在指令调优数据上的路由统计来指导后训练部署，这种"先分析再优化"的策略实用性强

## 局限性 / 可改进方向
- **仅关注 post-training**：预训练阶段的路由模式可能不稳定，先验可能不适用
- **硬件方案为仿真验证**：未在真实晶粒上测试
- **二进制整数规划的可扩展性**：更多专家（如 DeepSeek-V3 的 256 experts）时求解可能变慢

## 相关工作与启发
- **vs FRED（Rashidi et al. 2024）**: FRED 是晶圆级 LLM 训练但用粗粒度静态划分；Mozart 引入 MoE 感知的细粒度调度
- **vs Cambricon-LLM**: 仅做推理且不考虑晶圆级集成；Mozart 面向训练且支持晶圆级

## 评分
- 新颖性: ⭐⭐⭐⭐ 算法-硬件协同设计 + MoE 感知的晶粒优化
- 实验充分度: ⭐⭐⭐ 三个模型但仅仿真验证
- 写作质量: ⭐⭐⭐⭐ 图示清晰，设计动机说明充分
- 价值: ⭐⭐⭐⭐ 对 MoE 硬件部署有重要指导意义
