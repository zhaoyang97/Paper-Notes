# AceCoder: Acing Coder RL via Automated Test-Case Synthesis

**会议**: ACL 2025  
**arXiv**: [2502.01718](https://arxiv.org/abs/2502.01718)  
**代码**: https://tiger-ai-lab.github.io/AceCoder  
**领域**: 代码生成  
**关键词**: code generation, reinforcement learning, reward model, test case synthesis, R1-style training

## 一句话总结
构建 AceCode-87K（87K 编码题 + 138 万自动合成测试用例），训练代码专用 Reward Model（7B 超越 340B Nemotron），Best-of-N 提升 Llama-3.1-8B 平均 8.9 分，R1 风格从 base 直接 RL 仅 80 步 HumanEval+ 提升 22.5%。

## 研究背景与动机
1. **领域现状**：代码生成模型主要依赖 SFT 提升，RL 的潜力尚未被充分挖掘。数学领域的 RL 成功（如 DeepSeek-R1）暗示代码领域也可以受益。
2. **现有痛点**：(1) 代码评估需要执行测试用例（不像数学可直接字符串匹配），缺乏可靠奖励信号；(2) 大规模带测试用例的编码数据集稀缺——APPS/TACO 依赖人工标注，之前 RL 工作限于 <5K 样本。
3. **核心矛盾**：现有通用 Reward Model（如 Skywork）不能泛化到代码评估，而人工标注测试用例成本太高无法规模化。
4. **本文要解决什么**：构建自动化管线从代码数据集生成大规模可靠测试用例 → 训练代码专用 RM → 用 RM 做 Best-of-N 或 RL。
5. **切入角度**：用 GPT-4o-mini 批量"想象"测试用例 + 强代码模型执行过滤 = 低成本高质量大规模测试数据。
6. **核心 idea 一句话**：全自动测试用例合成 + 执行过滤 → 代码 RM 训练 → RL/Best-of-N，将代码生成的 RL 扩展到 87K 规模。

## 方法详解

### 整体框架
五阶段管线：(1) 从种子代码数据集用 GPT-4o-mini 生成 LeetCode 风格题目 + ~20 个测试用例；(2) 用 Qwen2.5-Coder-32B 生成解答并执行过滤不通过的测试用例；(3) 从过滤后数据构造偏好对（Bradley-Terry loss）；(4) 训练 AceCode-RM（7B/32B）；(5) 用 RM 做 Best-of-N 采样或 Reinforcement++ RL 训练。

### 关键设计

1. **自动测试用例合成 + 过滤 (AceCode-87K)**:
   - 做什么：从 124K Python 函数（Magicoder-Evol/OSS/StackPy）出发，GPT-4o-mini 重写为 LeetCode 风格 + 生成 ~20 个测试用例
   - 核心思路：用 Qwen2.5-Coder-32B 为每道题生成解答并执行测试用例，**去掉解答不通过的用例 + 去掉剩余用例 <5 个的题目**
   - 最终规模：87,149 题 + 1,380,000 清洗后测试用例（平均 15.87 个/题）
   - 质量验证：200 个人工检查样本中仅 3 个无效（1.5% 错误率）

2. **偏好对构造与 RM 训练**:
   - 做什么：对每道题采样 16 个程序，按测试通过率排序，构造偏好对
   - 核心思路：选择性配对——仅当 $s_i > s_j + 0.4$ 且 $s_i > 0.8$ 且 $s_j > 0$ 时才配对，确保高质量偏好信号
   - 训练：Bradley-Terry loss，Qwen2.5-Coder-7B-Instruct 为骨干，1 epoch，8×A100 24 小时
   - 结果：307,509 个有效偏好对来自 46,618 道题

3. **RL 训练 (Reinforcement++)**:
   - 做什么：用 AceCode-RM 或二值规则奖励（全部测试通过=1，否则=0）做 RL
   - 核心思路：Reinforcement++ 算法（比 PPO 更高效，不需要 value model）；只取 AceCode-87K 中最难的 25% 题目训练
   - R1 风格实验：从 Qwen2.5-Coder-7B-**base**（不经过 SFT）直接 RL，仅 80 步
   - 关键发现：**规则奖励比 RM 奖励更有效**——RM 训练时存在 reward hacking

### 损失函数 / 训练策略
RM 训练：标准 Bradley-Terry loss。RL 训练：Reinforcement++ 算法，rollout batch=256，8 samples/question，lr=5e-7，1 episode，8×H100 6 小时。

## 实验关键数据

### Best-of-N 采样结果 (Llama-3.1-8B-Instruct)

| 方法 | HumanEval | MBPP | BigCodeBench-C | LiveCodeBench | 平均 |
|------|-----------|------|----------------|---------------|------|
| Greedy | 68.9 | 67.2 | 38.5 | 18.0 | 40.9 |
| AceCode-RM-7B (Best-of-64) | 81.7 | 74.6 | 47.8 | 27.6 | 49.3 |
| **AceCode-RM-32B (Best-of-64)** | **85.4** | **72.0** | **48.5** | **31.0** | **49.8** |

### R1 风格 RL 训练 (从 base 模型直接 RL, 80 步)

| 配置 | HumanEval+ | MBPP+ | BigCodeBench-I |
|------|-----------|-------|----------------|
| Qwen2.5-Coder-7B-Base | 61.6 | 76.9 | 40.2 |
| + AceCoder-Rule (80 步) | **84.1 (+22.5)** | **82.3 (+5.4)** | **43.2 (+3.0)** |
| + AceCoder-RM (80 步) | 83.5 (+21.9) | 80.2 (+3.3) | 36.8 (-3.4) |

### 消融实验

| 配置 | HumanEval | MBPP | BigCodeBench-H | 平均 |
|------|-----------|------|----------------|------|
| RM w/o Filter | 73.8 | 73.3 | 17.6 | 45.2 |
| **RM w/ Filter** | **77.4** | **76.5** | **20.3** | **47.7** |

### 关键发现
- AceCode-RM-7B 在 RM Bench 编码类目上超越 340B Nemotron-Reward 7.5 分（66.9% vs 54.5%相当水平），7B 模型超越 50× 大的通用 RM
- Best-of-N 对弱模型提升巨大（Mistral +13 分），对强模型提升有限（Qwen-Coder +4 分）
- R1 风格训练仅 80 步（48 H100-hours）就将 HumanEval+ 提升 22.5%，效率惊人
- **规则奖励 > RM 奖励**用于 RL 训练——RM 在 RL 过程中存在 reward hacking
- 测试用例过滤提升 2.5 分平均，特别是在难题上（BigCodeBench-Hard +2.7）
- RM 骨干选择很重要：Qwen2.5-Coder 骨干比 Llama-3.1 骨干好 11.6 分（HumanEval）

## 亮点与洞察
- **全自动测试用例合成管线**是核心贡献：GPT-4o-mini 生成 + 强模型执行过滤 = 低成本大规模高质量，错误率仅 1.5%。这个管线可迁移到任何需要测试的代码 RL 场景。
- **R1 风格训练**的效率令人惊叹：从 base 模型直接 RL（不经过 SFT），80 步就达到接近 SFT + RL 的水平，挑战了"必须先 SFT 再 RL"的传统流程。
- **规则奖励优于 RM 奖励**的发现值得深思：说明 RM 在 RL 训练循环中容易被利用（reward hacking），简单但准确的奖励信号（测试全通过/不通过）更有效。

## 局限性 / 可改进方向
- 测试用例合成仍有噪声：通过所有测试 ≠ 程序正确，可能存在边界情况遗漏
- 在已经很强的模型上（Qwen2.5-Coder-7B-Instruct）RL 提升有限（+0.7 分均值）
- 只用了 GPT-4o-mini 合成测试，用更强模型（GPT-4）可能进一步提升质量
- 仅训练 1 个 episode，进一步的 RL 扩展策略未探索

## 相关工作与启发
- **vs Skywork-Reward**：通用 RM 在代码任务上严重不足，代码专用 RM 必要且有效
- **vs DeepSeek-R1**：借鉴其"从 base 直接 RL"的思路，证明在代码领域同样可行
- **vs APPS/TACO**：这些依赖人工标注测试用例，AceCoder 全自动合成，规模扩大 10 倍+
- 启发：测试用例的"合成+过滤"范式可推广到任何可执行验证的任务（如 SQL、数据分析）

## 评分
- 新颖性: ⭐⭐⭐⭐ 自动测试合成管线 + R1 风格代码 RL 有价值贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型 × 多基准 + Best-of-N + RL + R1 + 详细消融
- 写作质量: ⭐⭐⭐⭐ 管线描述清晰，数据详实
- 价值: ⭐⭐⭐⭐⭐ 首次将代码 RL 扩展到 87K 规模，实用性极强

## 亮点与洞察
- **全自动测试合成管线**解决了代码 RL 的奖励信号瓶颈——测试通过率作偏好信号比人工标注更客观。
- **R1 风格从 base 直接 RL 的成功**说明代码是 RL 理想领域——有天然可执行验证器。
- **AceCode-87K 数据集**本身是重要贡献——87K 题 + 138 万清洗后测试用例。

## 局限性 / 可改进方向
- 仅覆盖 Python。测试用例过滤后仍有残留问题。RL 训练对超参数敏感。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个全自动测试合成→RM→RL 代码管线
- 实验充分度: ⭐⭐⭐⭐⭐ Best-of-N + RL + R1-style + 多基准
- 写作质量: ⭐⭐⭐⭐ 管线描述完整，实验设计严谨
- 价值: ⭐⭐⭐⭐⭐ 释放了代码生成 RL 的巨大潜力
