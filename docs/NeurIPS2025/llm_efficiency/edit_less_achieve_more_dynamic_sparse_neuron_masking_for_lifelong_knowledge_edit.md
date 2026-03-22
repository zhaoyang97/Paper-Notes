<!-- 由 src/gen_stubs.py 自动生成 -->
# Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs

**会议**: NEURIPS2025  
**arXiv**: [2510.22139](https://arxiv.org/abs/2510.22139)  
**代码**: [LiuJinzhe-Keepgoing/NMKE](https://github.com/LiuJinzhe-Keepgoing/NMKE)  
**领域**: llm_efficiency  
**关键词**: knowledge editing, lifelong learning, sparse masking, neuron attribution, LLM  

## 一句话总结
提出 NMKE 框架，通过神经元级归因发现 knowledge-general 和 knowledge-specific 两类知识神经元，并结合熵引导的动态稀疏 mask，实现精准神经元级知识编辑，在 5000 步连续编辑后仍保持高编辑成功率和模型通用能力。

## 研究背景与动机
- 终身知识编辑（Lifelong Knowledge Editing）需要在不重训 LLM 的前提下持续更新过时知识，但现有方法随编辑次数增多误差不断累积
- 外部参数方法（GRACE、WISE）资源开销递增，编辑准确率逐步下降
- 内部参数方法（ROME、MEMIT、AlphaEdit）采用层级或参数块级别修改，不可避免地波及无关神经元，导致模型遗忘和能力崩塌
- AlphaEdit 作为 SOTA，在 1500 步后出现灾难性遗忘：ZsRE 编辑成功率下降 0.67，CounterFact 下降 0.78
- 根本原因：粗粒度参数更新在终身编辑中导致不相关神经元的累积破坏
- 缺乏对 FFN 层中不同功能神经元的精细理解，无法实现"只改该改的"

## 方法详解

### 神经元级归因（Neuron Attribution）
- 将 FFN 视为 key-value 关联记忆：第 i 个神经元的 key 为 W^in 的第 i 行，value 为 W^out 的第 i 列
- 通过扰动法量化每个神经元对目标 token 预测的贡献：放大第 i 个神经元输出 s^(i)，计算 log 概率增益 Imp^(i)
- 发现三类神经元：knowledge-general（跨任务稳定激活）、domain-specific（同领域激活）、task-specific（仅特定任务激活）
- 消融实验验证：mask 掉 top-10 knowledge-general 神经元 → 准确率从 37.5% 暴降至 4.17%；mask task-specific 仅降 2.04%

### 动态稀疏 Mask（Dynamic Sparse Masking）
- **Knowledge-General 神经元选择**：统计每个神经元在多个 prompt 上正向归因的次数 r^ge_i，次数越多越可能是通用神经元
- **Knowledge-Specific 神经元选择**：取每个神经元在各 prompt 上的最大归因分数 r^sp_i
- **熵引导动态比例**：
  - 通用神经元比例 ρ_ge 由归因分布的平均归一化熵 H_ge 决定（熵越高→激活越均匀→通用神经元越多）
  - 特定神经元比例 ρ_sp 由最大归因分布的熵 H_sp 决定
  - 实际中通过常数缩放因子 a_ge/a_sp 和偏置 b_ge/b_sp 进一步调节比例
- **Mask 生成**：取 r^ge ≥ τ_ge 或 r^sp ≥ τ_sp 的神经元并集构成二值 mask m^(l)
- **阈值确定**：τ_ge 为 r^ge 分布的 (1-ρ_ge) 分位数，τ_sp 类似
- 编辑时仅更新 mask 选中的神经元子集，遵循 AlphaEdit 的优化流程（null-space projection）

### 三种归因计算策略
- **MPC (Mean Prompt Contribution)**：使用编辑 prompt 的均值贡献
- **PSA (Prompt-Specific Attribution)**：使用特定 prompt 的归因
- **LPS (Layer-wise Prompt Selection)**：逐层选择最相关 prompt 的归因
- 三种策略性能接近，MPC 计算最快（~22s/edit），PSA/LPS 稍慢（~30s/edit）但效果略优

## 实验关键数据

### 终身知识编辑性能（LLaMA3-8B-Instruct）

| 方法 | ZsRE T=1000 (Rel./Gen./Loc.) | ZsRE T=2000 (Rel./Gen./Loc.) |
|---|---|---|
| FT | 0.13/0.10/0.02 | 0.07/0.06/0.01 |
| ROME | 0.02/0.01/0.02 | 0.01/0.01/0.02 |
| MEMIT | 0.04/0.04/0.03 | 0.03/0.04/0.03 |
| WISE | 0.41/0.39/- | 0.37/0.36/- |
| AlphaEdit | 0.93/0.84/0.58 | 0.32/0.28/0.06 |
| **NMKE** | **0.95/0.85/0.77** | **0.94/0.85/0.71** |

| 方法 | CounterFact T=1000 (Rel./Gen./Loc.) | CounterFact T=2000 (Rel./Gen./Loc.) |
|---|---|---|
| AlphaEdit | 0.99/0.76/0.32 | 0.22/0.13/0.04 |
| **NMKE** | **0.99/0.65/0.50** | **0.98/0.67/0.38** |

### 通用能力保持

| 通用能力 (LLaMA3-8B, ZsRE T=2000) | AlphaEdit | NMKE |
|---|---|---|
| MMLU | ~0.25 (崩塌) | **0.59** (5000步仍保持) |
| GSM8K | 0.00 (T≥1500) | 保持可用 |
| HumanEval | 0.00 (T≥1500) | 保持可用 |
| CommonsenseQA | 显著下降 | 基本保持 |
| BBH-Zeroshot | 显著下降 | 基本保持 |

### 编辑效率

| 方法 | 单步时间 (s) | T=2000 Rel. | T=2000 Loc. |
|---|---|---|---|
| MEMIT | 16.83 | 0.04 | 0.03 |
| AlphaEdit | 22.16 | 0.62 | 0.14 |
| NMKE (MPC) | 22.25 | 0.93 | 0.77 |
| NMKE (LPS) | 30.42 | 0.94 | 0.74 |

## 亮点
- 首次从神经元功能归因角度揭示 LLM 终身编辑退化的根本原因：粗粒度更新累积破坏无关神经元
- knowledge-general vs knowledge-specific 神经元的划分及消融验证直观有力
- 熵引导的动态比例机制优于固定比例，自适应不同 prompt batch 的激活模式
- 在 5000 步连续编辑后 MMLU 仍保持 0.59，远超所有 baseline（均崩塌至随机水平）
- t-SNE 可视化直观展示 NMKE 对参数分布的扰动远小于 AlphaEdit
- MPC 变体与 AlphaEdit 单步时间几乎相同（22.25s vs 22.16s），但效果大幅领先
- 四种神经元选择策略的消融表明各策略都能有效保持通用能力，验证了框架的鲁棒性
- 在 GPT2-XL 和 Qwen2.5-7B 上也进行了验证，展示跨模型迁移性

## 局限性 / 可改进方向
- 每步编辑需运行神经元归因，单步耗时 ~30s（AlphaEdit ~22s），编辑效率略低
- 常数缩放因子 a_ge/a_sp 和偏置 b_ge/b_sp 需要手动调参
- 实验主要在 LLaMA3-8B 和 GPT2-XL 上验证，对 70B+ 模型的可扩展性未知
- 局部性（Loc.）指标在 CounterFact 上虽优于 AlphaEdit 但绝对值仅 0.38，仍有改进空间
- 仅关注 FFN 层的神经元，未探索注意力层参数的精细编辑

## 评分
- 新颖性: ⭐⭐⭐⭐ (神经元功能分类 + 熵引导动态 mask 的组合有新意，但单个组件非全新)
- 实验充分度: ⭐⭐⭐⭐⭐ (ZsRE/CounterFact 双数据集，5000 步连续编辑，5 个通用能力 benchmark，多消融)
- 写作质量: ⭐⭐⭐⭐ (动机清晰，图示直观，公式推导完整)
- 价值: ⭐⭐⭐⭐ (终身编辑场景实用性强，但应用面相对窄于架构创新)
