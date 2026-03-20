# THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning

**会议**: ICLR 2026  
**arXiv**: [2509.13761](https://arxiv.org/abs/2509.13761)  
**代码**: [GitHub](https://github.com/JingMog/THOR)  
**领域**: LLM Reasoning / Mathematical Reasoning  
**关键词**: tool-integrated reasoning, hierarchical RL, GRPO, code generation, self-correction, mathematical reasoning  

## 一句话总结
提出 THOR，通过 TIRGen 数据构建管线 + 层次化强化学习（episode 级+step 级优化）+ 自修正推理机制，系统性解决 LLM 工具集成数学推理中的数据构建、细粒度优化和推理增强三大挑战。

## 背景与动机
1. LLM 作为概率 next-token 预测器在高精度任务（数值计算、符号操作）上天然不足
2. 工具集成推理（TIR）是克服此瓶颈的有力范式，但面临三大挑战：
   - **数据构建**: 用外部大模型 prompt 合成数据存在风格不匹配，对推理模型（如 DeepSeek-R1）效果差
   - **细粒度优化**: 现有 RL 方法仅做 episode 级优化，忽略中间代码步骤的细粒度更新，导致稀疏奖励问题
   - **推理增强**: 单 pass 推理忽略了工具即时反馈的修正作用
3. 纯 CoT 方法（搜索/RL）和纯代码方法各有局限，需将语义推理与精确执行结合
4. SFT 方法需大量高质量示范数据且泛化差；现有 RL 方法奖励过于稀疏
5. 关键洞察：**中间工具调用的成功是最终答案正确性的强预测因子**
6. START 等方法用规则注入代码 prompt，但难以准确定位插入位置

## 方法（框架/设计）
- **TIRGen 数据构建管线**:
  - Generator 生成自然语言推理步骤（限长 $L_{step}$）
  - Refiner 评估哪些步骤可转为代码，与代码解释器交互获取精确结果
  - 优势：数据与 Generator 策略对齐（Refiner 仅看单步不看全题）；降低对大模型依赖
  - 多阶段过滤：格式一致性 → 代码质量（需有库调用或控制流）→ 难度平衡
- **层次化 RL 训练**:
  - **Cold Start**: 用 $\mathcal{D}_{SFT}$ 做 SFT 建立工具调用基础模式
  - **Episode 级优化**: GRPO + 最终答案正确性奖励，过滤执行失败轨迹以避免不当惩罚
  - **Step 级优化**: 对执行失败的代码步进行回溯（保留推理前缀 $r_{pre}^t$，重新生成后缀+代码），以代码执行成功率为奖励
  - 总损失: $\mathcal{L} = \mathcal{L}^{epis} + \mathcal{L}^{step}$，两级均采用 GRPO + VAPO 式 NLL 损失
- **自修正推理机制**: 推理时代码执行失败则回溯到推理前缀重新生成，最多修正 $N_{corr}$ 次，增量计算成本极小

## 实验关键数据
| Benchmark | THOR 表现 |
|-----------|----------|
| MATH500 | SOTA (同规模模型) |
| AIME 2024/2025 | SOTA |
| AMC | SOTA |
| Minerva Math | SOTA |
| Olympiad Bench | SOTA |
| HumanEval/MBPP/LiveCodeBench | 代码任务也有一致提升 |

- 跨模型泛化：在推理模型和非推理模型上均有效
- 层次化 RL 相比纯 episode RL 显著提升，验证了 step 级优化的必要性
- 自修正机制在降低推理开销的同时提升鲁棒性
- 统计验证了"中间工具调用成功 ↔ 最终答案正确"的相关性

## 亮点
- 三位一体框架系统性解决 TIR 的数据/优化/推理三大挑战
- TIRGen 的 Generator-Refiner 分工设计巧妙：保持策略对齐+降低大模型依赖
- Step 级 RL 的回溯+重生成设计精确瞄准代码错误步，缓解长推理链稀疏奖励
- 自修正机制几乎无额外成本（仅重新生成后缀和动作）

## 局限性
- 缓存文件截断，未能看到完整实验数据表
- Step 级优化依赖代码执行是否成功作为信号，对逻辑正确但执行报错的情况可能产生误判
- 回溯机制的 $L_{suf}$ 和 $N_{corr}$ 需调参，未讨论敏感性
- TIRGen 的 Refiner 仍需一定代码能力的模型，适用范围需验证
- 仅关注数学推理，未验证在其他工具使用场景（如搜索、API 调用）的推广性

## 相关工作
- **TIR**: ToRA (Gou et al.)、START (Li et al.)、Toolformer、AIMO-2
- **数学 RL**: DAPO、VAPO、GRPO (DeepSeekMath)
- **推理模型**: DeepSeek-R1、Qwen3、Kimi k1.5
- **代码生成优化**: ToRL、ReTool、Agent-R

## 评分
- 新颖性: ⭐⭐⭐⭐ (层次化 RL 和 TIRGen 管线均有创新)
- 实验充分度: ⭐⭐⭐⭐ (多 benchmark SOTA，跨模型验证)
- 写作质量: ⭐⭐⭐⭐ (形式化严谨，算法描述详尽)
- 价值: ⭐⭐⭐⭐⭐ (TIR 领域的系统性贡献，实用性强)
