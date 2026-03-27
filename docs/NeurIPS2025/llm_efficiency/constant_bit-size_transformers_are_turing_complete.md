# Constant Bit-Size Transformers Are Turing Complete

**会议**: NeurIPS 2025
**arXiv**: [2506.12027](https://arxiv.org/abs/2506.12027)
**代码**: 无
**领域**: LLM理论 / 计算复杂性
**关键词**: Turing completeness, transformer theory, computational complexity, Post machines, context window

## 一句话总结
首次证明常数 bit 精度、固定参数数量的 Transformer（仅允许上下文窗口增长）是图灵完备的，并建立了精确的复杂度等价关系 WINDOW[s(n)] = SPACE[s(n)]，表明扩展上下文窗口——而非模型尺寸——已足以实现通用计算。

## 研究背景与动机
1. **领域现状**：基于 CoT 的 Transformer 已被证明是图灵完备的，但之前所有的证明都要求模型的 bit-size 随输入长度增长——要么精度需要 $O(\log t(n))$ 位，要么嵌入维度需要 $O(\log t(n))$。
2. **现有痛点**：bit-size 随输入增长意味着处理更长输入需要更大的模型，这对追求终身学习的 AGI 系统形成了原则性障碍——系统与不断增长的环境交互时，输入长度无界增长。
3. **核心矛盾**：扩大模型精度/参数是否是处理更长输入的必要条件？
4. **本文要解决什么？** 证明模型 bit-size 无需增长，仅扩展 context window 即可实现任意图灵机的模拟。
5. **切入角度**：Post Machine（一种基于队列的图灵等价模型）的行为与 Transformer 的 autoregressive decoding 具有天然的结构相似性——context window 可以被视为队列。
6. **核心idea一句话**：通过模拟 Post Machine（队列自动机）而非传统图灵机，构造常数精度单层 Transformer 来实现图灵完备性。

## 方法详解

### 整体框架
证明分为两个方向：(1) WINDOW[s(n)] ⊆ SPACE[s(n)]：TM 用 O(s(n)) 空间即可模拟 context window 为 s(n) 的 Transformer（直接构造，维护 s(n) 大小的 buffer）；(2) SPACE[s(n)] ⊆ WINDOW[s(n)]：将 TM → Post Machine → Transformer 逐步模拟。

### 关键设计

1. **Post Machine 作为中间桥梁**:
   - 做什么：将图灵机先转化为等价的 Post Machine（PM），PM 是配备队列的有限自动机
   - 核心思路：PM 的队列将 TM 磁带以循环方式存储——TM 头右移时，队列前端删除、尾端追加；TM 头左移时，队列循环旋转。使用"延迟追加"技巧避免预读下一个队列元素
   - 设计动机：PM 的行为（从队列前端读取、向尾端写入）天然对应 Transformer 的 autoregressive decoding（旧 token 滑出窗口、新 token 追加），这种对齐使模拟构造简洁

2. **常数精度单层 Transformer 构造**:
   - 做什么：用 s(n) 大小的 context window 作为 PM 的队列，构造一个单层、单头、hardmax attention 的 Transformer
   - 核心思路：词表 $\mathcal{V} = \Sigma \times Q$（每个 token 同时编码 tape symbol 和状态）；相对位置编码仅区分三种位置（当前位置、窗口最旧位置、中间位置）；attention 通过 hardmax 精确检索窗口中最旧的 token（队列前端）；FFN 实现 PM 的转移函数 $\delta$
   - 设计动机：hardmax + 位置编码使 attention 退化为精确的"查找最旧 token"操作，FFN 作为有限状态查找表即可实现转移——整个构造只需要常数精度和常数参数

3. **队列大小固定化**:
   - 做什么：将 PM 调整为队列大小始终等于 s(n)，确保 context window 大小固定
   - 核心思路：在输入右侧填充 # 符号至 s(n) 长度，每步前后两个头都向右移动一格
   - 设计动机：确保 Transformer 的 context window 大小与 PM 的队列大小精确匹配

### 复杂度分析
- 模拟 s(n) 空间的 TM 需要 s(n) 大小的 context window
- 每个 TM 步骤需要 O(s(n)) 个 CoT token（因 PM 模拟 TM 的一步需要遍历整个队列）
- 总 CoT 长度：$O(t(n) \cdot s(n))$

## 实验关键数据

### 理论结果对比
| 工作 | 精度 | 嵌入维度 | 窗口大小 | 每TM步CoT |
|------|------|----------|---------|-----------|
| Pérez et al. (2021) | $O(\log t(n))$ | $O(1)$ | $n+t(n)$ | 1 |
| Li et al. (2024) | $O(1)$ | $O(\log t(n))$ | $O(t(n)\log t(n))$ | $O(\log t(n))$ |
| **本文** | **$O(1)$** | **$O(1)$** | **$s(n)$** | **$s(n)$** |

### 关键推论
| 推论 | 说明 |
|------|------|
| WINDOW[poly(n)] = PSPACE | 多项式窗口足以解决所有 PSPACE 问题 |
| 窗口 vs 时间 | 窗口只需随空间复杂度（而非时间复杂度）增长 |
| 通用模拟 | 单个固定 Transformer 加载通用 TM 描述即可计算任何可计算函数 |

### 关键发现
- **窗口与空间复杂度等价**：这是最核心的结果——context window 的计算角色等同于 TM 的空间资源
- **空间 vs 时间的gap很大**：许多 PSPACE-complete 问题（SAT、Sokoban）空间只需多项式但时间可能需要指数级，意味着窗口需求远小于之前的构造
- **相对位置编码的统一性**：模拟不同长度输入时，只需按显式公式调整相对位置编码，其他参数不变

## 亮点与洞察
- **Post Machine 的精妙选择**：之前的构造直接模拟 TM 磁带需要 log 精度来编码磁带位置，而 PM 的队列结构使位置信息隐含在窗口顺序中，消除了对额外精度的需求
- **Attention 作为队列操作的新解读**：attention 机制不仅是统计聚合器，还可以作为队列结构上的离散计算操作——这为理解 Transformer 的推理能力提供了新视角
- **为"扩大窗口"策略的理论辩护**：工程界已经在朝着更长 context window 的方向发展（100K+ tokens），本文从计算理论角度证明了这一策略的合理性

## 局限性 / 可改进方向
- **模拟效率低**：每个 TM 步需要 O(s(n)) 个 CoT token，总模拟时间是 $O(t(n) \cdot s(n))$，比直接运行 TM 慢很多
- **纯存在性证明**：构造的 Transformer 参数是手工设计的，不涉及学习——无法保证实际训练的 Transformer 会学到类似的计算模式
- **hardmax 假设**：用 hardmax 代替 softmax，虽是标准理论简化但与实践有差距
- **未考虑有限精度浮点**：实际 Transformer 用 fp16/bf16，与理论中的"常数精度"含义不完全一致

## 相关工作与启发
- **vs Pérez et al. (2021)**: 他们需要 $O(\log t(n))$ 精度，本文用常数精度替代，代价是每步需要更多 CoT
- **vs Li et al. (2024)**: 他们用常数精度但需要 $O(\log t(n))$ 维度的嵌入，实际 bit-size 仍在增长
- **vs Yang et al. (2025) PENCIL**: PENCIL 也将窗口与空间复杂度关联，但仍需要增长的 bit-size；本文是真正的常数 bit-size
- **vs Schuurmans et al. (2024)**: 他们也用 Post Machine 思路但分析的是更受限的 Lag system，每步需要 $O(s(n)^3)$ CoT（本文只需 $O(s(n))$）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在常数 bit-size 下证明图灵完备性，WINDOW=SPACE 等价关系非常优雅
- 实验充分度: ⭐⭐ 纯理论工作，无实验
- 写作质量: ⭐⭐⭐⭐⭐ 证明结构清晰，motivation 和 remark 很好地连接了理论和实践意义
- 价值: ⭐⭐⭐⭐⭐ 对 Transformer 计算理论的里程碑式贡献，直接影响架构设计哲学
