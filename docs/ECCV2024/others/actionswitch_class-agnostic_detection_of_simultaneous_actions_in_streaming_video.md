# ActionSwitch: Class-agnostic Detection of Simultaneous Actions in Streaming Videos

**会议**: ECCV 2024  
**arXiv**: [2407.12987](https://arxiv.org/abs/2407.12987)  
**代码**: [https://github.com/hyolim-kang/ActionSwitch](https://github.com/hyolim-kang/ActionSwitch) (有)  
**领域**: 视频理解 / 在线时序动作定位  
**关键词**: Online Temporal Action Localization, Class-agnostic Detection, Overlapping Actions, Conservativeness Loss, Finite State Machine  

## 一句话总结
提出 ActionSwitch——首个无需类别信息即可检测流式视频中重叠动作实例的在线时序动作定位（On-TAL）框架，核心将多动作检测建模为有限状态机的状态分类问题，并辅以 conservativeness loss 减少碎片化误检，在 THUMOS14、FineAction、Epic-Kitchens 100 等数据集上在 OAD 扩展方法中达到 SOTA。

## 背景与动机
在线动作检测（OAD）只做逐帧分类，缺乏实例级理解；而在线时序动作定位（On-TAL）需要在视频流中实时确定每个动作实例的起止时间。现有 On-TAL 方法面临两个核心痛点：(1) **无法处理重叠动作**——class-agnostic OAD 模型只输出 0/1，多个同时发生的动作会被合并为一个实例；class-aware 方法虽然可以按类别分组，但遇到同类动作重叠时束手无策（如 Epic-Kitchens 中同类动作大量重叠），且类别数增多时阈值需手动设定，灵活性极差。(2) **碎片化检测**——逐帧决策的天然不稳定性导致一个长动作被切碎成多个短实例。此外，在开放世界场景中预定义所有动作类别本身就不现实，将提案生成与分类解耦是更合理的设计。

## 核心问题
如何在不依赖类别信息的前提下，实时检测流式视频中**同时发生的多个动作实例**（包括同类重叠），并有效抑制碎片化的误检？

## 方法详解
整体思路非常优雅：把多动作检测抽象为一台多开关机器——每个"开关"独立负责一路动作的检测，多个开关同时打开就代表多个动作同时发生。通过有限状态机将开关组合编码为状态标签，让 OAD 模型直接预测帧级状态，再从状态序列中解码出动作实例。

### 整体框架
- **输入**: 预提取的视频特征序列 $f_t \in \mathbb{R}^D$
- **核心模型**: State-emitting OAD 模型——单向 GRU 编码器 + 状态分类器（MLP + residual），输出 $S$ 个状态的概率分布 $p_t = \text{softmax}(\text{SC}(g_t))$
- **状态解码**: 用 $s_t = \arg\max(p_t)$ 得到当前帧状态，对比 $s_t$ 与 $s_{t-1}$ 即时判断动作的开始/结束
- **输出**: 在线累积的 class-agnostic 动作提案（起止时间），后续可接独立分类器获得类别标签
- **推理速度**: 得益于轻量 GRU + MLP 设计，**超过 500 fps**

### 关键设计
1. **有限状态机建模（ActionSwitch）**: 以 2-switch 为例，4 个状态分别表示"无动作"、"仅 switch1 激活"、"仅 switch2 激活"、"两个都激活"。状态转换天然对应动作边界。用 argmax 取代手动阈值，彻底避免了 class-aware 方法中随类别数增加而爆炸的阈值调参问题。同一类别的两个重叠动作可被不同 switch 分别捕获，这是 class-aware 方法做不到的。
2. **状态标签编码（Encode Action）**: 训练时将 GT 动作实例编码为帧级状态标签——假设 switch1 先激活、switch2 后激活以消除歧义。实际上状态标签就是各 switch ID 之和（如 switch1+switch2 激活时状态为 3）。
3. **状态解码（Decode State）**: 推理时维护状态历史队列，通过比较相邻帧状态变化即时推断动作起止。状态序列与动作实例一一对应，无需额外的 boundary-matching 模块，从根本上消除了边界匹配误差。

### 损失函数 / 训练策略
**Conservativeness Loss $\mathcal{L}_c$**——本文最精炼的设计：

$$\mathcal{L}_c(p_t, s_{t-1}) = \begin{cases} -\log(p_t[s_{t-1}]), & \text{if } \arg\max(p_t) \neq s_{t-1} \\ 0, & \text{otherwise} \end{cases}$$

仅在模型预测发生状态跳变时施加惩罚——要求模型在"改变主意"时更加慎重。用前一帧的预测 $s_{t-1}$ 作为伪标签，对跳变帧施加交叉熵。总损失：

$$\mathcal{L} = CE(p_t, y_t) + \alpha \cdot \mathcal{L}_c(p_t, s_{t-1})$$

实现极其简洁（5 行 PyTorch），无需任何架构修改，即插即用。$\alpha$ 控制保守程度：较大 $\alpha$ 提高 precision 但降低 recall。

**分类器**: 额外训练一个 vanilla Transformer 分类器，输入特征序列，预测类别标签和置信度，用于 mAP 评估。

## 实验关键数据

| 数据集 | 指标 | ActionSwitch | 之前最优 On-TAL | 提升 |
|--------|------|-------------|----------------|------|
| THUMOS14 | F1@0.5 | **53.2** | CAG-QIL 45.8 | +7.4 |
| THUMOS14 | Avg mAP | **40.3** | CAG-QIL 33.1 | +7.2 |
| FineAction | F1@0.5 | **19.44** | CAG-QIL 15.67 | +3.77 |
| FineAction | Avg mAP | **5.36** | CAG-QIL 4.45 | +0.91 |
| Epic-Kitchens 100 | F1@0.5 | **32.44** | OAT 27.58 | +4.86 |
| Epic-Kitchens 100 | mAP@0.5 | **3.597** | OAT 3.296 | +0.30 |
| MultiTHUMOS | F1 (3-switch) | **32.76** | OAT 29.63 | +3.13 |
| THUMOS14 ODAS | p-mAP@offset=1 | **33.06** | SimOn 31.45 | +1.61 |

注：OAT 属于 TAL-extension 方法（约束更宽松），ActionSwitch 在更严格的 OAD-extension 约束下仍全面领先。

### 消融实验要点
- **Switch 数量**: 从 1-switch 到 2-switch，recall 显著提升（THUMOS14: 59.49→63.96），说明额外 switch 确实捕获了重叠动作。在 MultiTHUMOS（dense overlap 数据集）上最优为 3-switch。
- **Conservativeness loss 权重 $\alpha$**: 存在精度-召回权衡，$\alpha=0.025$ 在多数数据集上达到最佳 F1。$\alpha=0$（无保守性损失）时 precision 很低（THUMOS14: 35.75），加入后提升至 47.73（$\alpha=0.025$）。
- **过大 $\alpha$ 的副作用**: $\alpha=0.05$ 时 recall 明显下降，模型过于保守而遗漏真实动作边界。
- **SimOn 在大类别数据集上崩溃**: Epic-Kitchens 上 SimOn 生成 137k 提案（GT 仅 9.6k），precision 仅 2.35%，class-aware 分组策略在类别多时彻底失效。

## 亮点
- **有限状态机的抽象极为巧妙**：把多 OAD 模型的互斥检测问题转化为单模型的状态分类问题，大幅简化了架构。状态序列与动作实例天然一一对应，无需 boundary-matching。
- **Conservativeness loss 实现极简但效果显著**：5 行代码，无架构修改，直接将"动作边界稀疏"的先验注入损失函数。这个设计思路有很强的通用性——任何需要时序平滑性的任务都可以借鉴。
- **Argmax 替代阈值设定**：彻底消除了手动调阈值的痛点，对类别数多的数据集尤其友好。
- **推理速度 >500fps**：轻量 GRU+MLP 设计使得实时应用完全可行。

## 局限性 / 可改进方向
- **Switch 数量需预设**: 需要根据数据集特性手动选择 switch 数量，无法自适应确定当前帧的重叠动作数。可以引入动态 switch 分配机制。
- **状态编码的歧义性**: 训练时假设 switch1 先激活、switch2 后激活来消除歧义，这在部分场景下可能引入系统偏差。
- **分类器与提案生成完全解耦**: 虽然这是本文的设计理念，但类别信息可能对提案质量有帮助，是否可以做弱利用值得探索。
- **离线方法差距仍大**: 与 ActionFormer 等离线方法相比 mAP 仍有明显差距（THUMOS14: 40.3 vs 62.6），在线约束本身就限制了性能上限。
- **仅用预提取特征**: 没有端到端训练视觉编码器，feature 质量成为瓶颈。

## 与相关工作的对比
- **vs CAG-QIL**: CAG-QIL 也是 class-agnostic OAD 扩展，但需要两阶段训练（先训练 OAD，再用 Q-imitation learning 训练分组模块），且无法处理重叠动作。ActionSwitch 用单模型状态分类一步到位，且保守性损失比 CAG-QIL 的分组模块更简洁有效。
- **vs SimOn**: SimOn 是 class-aware 方法，按类别独立分组。类别多时提案数爆炸（Epic-Kitchens 上 137k vs GT 9.6k），且无法检测同类重叠。ActionSwitch 完全不依赖类别信息，扩展性更好。
- **vs OAT**: OAT 属于 TAL-extension 方法，允许在动作结束时回溯确定开始时间（约束更宽松），因此在小数据集（THUMOS14）上 mAP 更高。但在大规模数据集上训练不稳定（FineAction 需降采样才能工作），且依赖类别信息和手动阈值。ActionSwitch 在更严格约束下仍有竞争力，且天然支持 ODAS 任务。

## 启发与关联
- **与流式视频场景图 idea 的关联**: [streaming_world_scene_graph](../../../ideas/video_understanding/20260316_streaming_world_scene_graph.md) 中的流式视频理解框架可以借鉴 ActionSwitch 的有限状态机思路——将场景图的状态转换也建模为有限状态机，用 conservativeness loss 确保场景图更新的时序平滑性。
- **Conservativeness loss 的迁移价值**: 适用于任何从帧级预测到实例级输出的场景，如视频目标分割中的 mask 连续性、视频问答中的注意力聚焦稳定性等。
- **Class-agnostic + 后分类的范式**: 与目标检测中 proposal → classify 的两阶段设计哲学一致，在 open-vocabulary 场景中尤其有价值，可与视频-语言模型结合实现开放词汇在线动作定位。

## 评分
- 新颖性: ⭐⭐⭐⭐ 有限状态机建模思路新颖且自然，conservativeness loss 简洁有效但整体贡献偏轻量
- 实验充分度: ⭐⭐⭐⭐ 覆盖 4 个数据集（THUMOS14/FineAction/Epic-Kitchens/MultiTHUMOS），消融充分，但缺少端到端实验
- 写作质量: ⭐⭐⭐⭐⭐ 写作逻辑清晰，图示直观，问题-方案-实验的推进非常流畅
- 价值: ⭐⭐⭐⭐ 为 On-TAL 建立了强 baseline，class-agnostic 范式对开放世界应用有启示，不过 On-TAL 本身是比较小众的方向
