# Group-in-Group Policy Optimization for LLM Agent Training

**会议**: NeurIPS 2025  
**arXiv**: [2505.10978](https://arxiv.org/abs/2505.10978)  
**代码**: [https://github.com/langfengQ/verl-agent](https://github.com/langfengQ/verl-agent)  
**领域**: LLM Agent / 强化学习  
**关键词**: GiGPO, credit assignment, anchor state grouping, multi-turn agent, GRPO  

## 一句话总结
GiGPO 通过在 GRPO 的 episode 级分组内嵌套 step 级分组（利用跨轨迹的重复环境状态作为 anchor state），实现了无需额外 rollout 和 critic 模型的细粒度 credit assignment，在 ALFWorld 上比 GRPO 提升 >12%，WebShop 上提升 >9%。

## 研究背景与动机

1. **领域现状**：Group-based RL（GRPO、RLOO）在单轮任务（数学推理、代码生成）上非常成功，但这些方法将整条轨迹视为一个整体计算 advantage，无法区分轨迹内各步骤的贡献。

2. **现有痛点**：LLM Agent 的交互跨越数十步、数万 token（如 ALFWorld 最多 50 步、20k+ token），reward 通常只在 episode 末端给出。GRPO 给同一 episode 内所有 token 赋予相同的 advantage，好的步骤和差的步骤被同等对待。PPO 虽有 step 级 advantage，但需要额外的 critic 网络，内存开销大。

3. **核心矛盾**：想要 step 级 credit assignment，最直接的方法是对每个状态额外 rollout 多个 action 形成对比组——但这需要大量额外 LLM forward passes，计算代价极高。

4. **本文要解决什么**：在保持 group-based RL 的 critic-free、低内存、稳定收敛优势的前提下，为多轮 Agent 训练引入细粒度 step 级 credit assignment。

5. **切入角度**：关键洞察——同一任务、相同初始状态下采样的 N 条轨迹中，很多环境状态会自然重复出现（如反复访问同一网页、重回同一房间）。这些重复状态可以免费构建 step 级对比组，无需额外 rollout。

6. **核心idea一句话**：将跨轨迹的重复环境状态作为 anchor state，用 hashmap 回溯性地构建 step 级分组，从而在不增加 rollout 的条件下获得"group-in-group"的双层 advantage 估计。

## 方法详解

### 整体框架
GiGPO 采用两级 advantage 估计：(1) Episode-level macro advantage $A_E$——标准 GRPO 的轨迹间对比；(2) Step-level micro advantage $A_S$——通过 anchor state grouping 构建的步骤间对比。最终 advantage 为二者加权和 $A = A_E + \omega \cdot A_S$。

### 关键设计

1. **Episode Relative Advantage $A_E$**：
   - 做什么：捕获轨迹整体质量
   - 核心思路：对 N 条轨迹的总 return $R(\tau_i) = \sum_t r_t^{(i)}$ 做标准化 $A_E(\tau_i) = \frac{R(\tau_i) - \text{mean}}{F_{\text{norm}}}$
   - 设计动机：提供稳定的全局训练信号，鼓励策略发展出连贯的轨迹级行为

2. **Anchor State Grouping**：
   - 做什么：免费构建 step 级对比组
   - 核心思路：找出所有轨迹中出现过的唯一环境状态集合 $\mathcal{U}$，对每个状态 $\tilde{s}$ 收集所有从该状态出发的 (action, return) 对形成 step 级 group $G_S(\tilde{s})$。实现上只需基于轻量级 hashmap key 匹配，不触发任何额外 LLM 推理
   - 设计动机：Agent 在探索中经常回到相同状态（重访网页、重回房间、重复搜索），这些自然重复提供了 step 级对比的免费数据

3. **Step Relative Advantage $A_S$**：
   - 做什么：评估同一状态下不同 action 的相对优劣
   - 核心思路：对 step 级 group 内每个 action 计算 discounted return $R_t^{(i)} = \sum_{k=t}^{T} \gamma^{k-t} r_k^{(i)}$，然后做组内标准化。例如在 WebShop 中，从相同搜索结果页点击不同商品，成功购买正确商品的 action 获得最高 $A_S$
   - 设计动机：相比只用即时 reward $r_t$，discounted return 能捕获 action 的长期影响

4. **Similarity-based Grouping（扩展）**：
   - 做什么：对环境状态无法精确匹配的场景（如 QA 任务中搜索结果略有不同），用最长匹配子序列相似度 >0.9 进行近似分组

### 损失函数 / 训练策略
- 标准 PPO-clip + KL 正则的目标函数，advantage 替换为 $A = A_E + \omega \cdot A_S$
- $\omega = 1$（无需调参），$\gamma = 0.95$，rollout group size $N = 8$
- $F_{\text{norm}}$ 可选 std（标准 GRPO）或 1（RLOO 无偏估计），依任务而定
- 基于 veRL 框架实现，step-wise multi-turn rollout 避免 context 爆炸

## 实验关键数据

### 主实验

| 方法 | ALFWorld (7B) | WebShop Score (7B) | WebShop Succ (7B) |
|---|---|---|---|
| GPT-4o | 48.0 | 31.8 | 23.7 |
| Gemini-2.5-Pro | 60.3 | 42.5 | 35.9 |
| ReAct (7B) | 31.2 | 46.2 | 19.5 |
| PPO (with critic, 7B) | 80.4 | 81.4 | 68.7 |
| GRPO (7B) | 77.6 | 79.3 | 66.1 |
| **GiGPO w/o std (7B)** | **90.2** | **86.2** | **75.2** |
| **GiGPO w/ std (7B)** | **90.8** | **84.4** | **72.8** |

GiGPO 超过 GRPO 12.6%（ALFWorld）和 9.1%（WebShop Succ），同时超过需要额外 critic 的 PPO。

### 消融实验

| 配置 | ALFWorld (1.5B) | WebShop Succ (1.5B) |
|---|---|---|
| GiGPO full (w/o std) | 86.1 | 67.4 |
| w/o $A_S$ (只有 episode 级) | 显著下降 | 显著下降 |
| w/o $A_E$ (只有 step 级) | 大幅下降 | 大幅下降 |

两级 advantage 都不可或缺。去掉 $A_E$ 失去全局信号导致策略不连贯，去掉 $A_S$ 在复杂任务（Cool、Pick2、WebShop）上掉点最严重。

### 关键发现
- **Step 级 group 覆盖率极高**：训练中仅 < 35% 的状态只出现 1 次，超过 65% 的状态跨轨迹重复出现，为 anchor state grouping 提供充足数据
- **训练动态可解释**：初期 group 大小分布右偏（大量重复循环），随训练进展收敛到 6-8（=group size），说明 agent 学会了避免死循环
- **计算开销极低**：anchor state grouping（hashmap）仅 0.01s/iter，step advantage 计算仅 0.53s/iter，总增量 < 0.002% 训练时间
- **与 DAPO 等正交可组合**：GiGPO + DAPO 的 dynamic sampling 在 WebShop 上达 75.0%，超过 DAPO 单独的 66.1%
- **VLM Agent 也有效**：在 Sokoban 和 EZPoints 视觉任务上同样提升

## 亮点与洞察
- **"免费午餐"设计**：跨轨迹重复状态本就存在于 rollout 数据中，GiGPO 只是用 hashmap 回收了这些免费信号，无需任何额外推理。这个思路可以迁移到任何存在状态重访的序列决策问题
- **层次化 advantage 的精巧平衡**：$A_E$ 提供方向（这条轨迹整体好不好），$A_S$ 提供细节（这个 action 具体好不好），两者缺一不可。这种"整体+局部"的 credit assignment 范式比纯 trajectory-level 或纯 step-level 都更有效

## 局限性 / 可改进方向
- 依赖状态精确匹配或高相似度匹配，在高度随机或连续状态空间的环境中 anchor state 可能难以找到
- 极端情况下（无状态重复）退化为纯 GRPO，这是安全的 fallback 而非失败
- $\omega$ 对不同任务可能有不同最优值，虽然实验显示在 [0.4, 1.2] 范围内较稳健

## 相关工作与启发
- **vs GRPO/RLOO**：这些方法在多轮 Agent 场景中直接将整条轨迹当成单轮回复处理，丢失了步骤间的区分度。GiGPO 在不增加成本的情况下恢复了这种区分度
- **vs PPO**：PPO 通过 critic 网络做 step 级 advantage 估计，但需要额外内存和训练。GiGPO 用数据结构（hashmap）替代了神经网络（critic），达到甚至超越 PPO 的效果
- **vs ArCHer / AgentQ**：这些方法使用额外的 value 网络或 MCTS 做 step 级 credit assignment，计算代价高。GiGPO 的 anchor state grouping 是一个优雅的低成本替代
- **vs RAGEN**：RAGEN 将整个交互历史拼接成 episode 级 response 用标准 GRPO 处理，在长 horizon 任务中不可扩展。GiGPO 的 step-wise 设计更可扩展

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Anchor state grouping 是一个极其简洁却有效的想法，将"group-based RL 无法做 step 级 credit assignment"的根本限制用零成本方案解决
- 实验充分度: ⭐⭐⭐⭐⭐ 1.5B/3B/7B 三规模，ALFWorld/WebShop/QA/VLM 四类任务，消融、训练动态、计算开销分析齐全
- 写作质量: ⭐⭐⭐⭐⭐ Figure 3 的 WebShop step-level group 直观展示非常出色，动机到方法到实验逻辑流畅
- 价值: ⭐⭐⭐⭐⭐ 开源的 verl-agent 框架、与现有 group-based RL 完全兼容的 plug-in 设计，对 LLM Agent 训练有直接推动作用
