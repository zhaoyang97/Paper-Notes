# GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training

**会议**: CVPR 2026  
**arXiv**: [2512.13043](https://arxiv.org/abs/2512.13043)  
**代码**: [https://github.com/weit123/GTR-Turbo](https://github.com/weit123/GTR-Turbo)  
**领域**: 多模态Agent / 强化学习 / 知识蒸馏  
**关键词**: VLM Agent, 多轮RL, 思维引导, 模型合并, 自进化训练  

## 一句话总结
提出GTR-Turbo——将RL训练过程中的历史checkpoint通过TIES合并为"免费教师"来引导后续RL，完全去除对GPT等昂贵外部模型的依赖，在Points24上胜率从3.5%(RL4VLM)提升至53.5%，同时训练时间减半、计算成本降低60%。

## 背景与动机
多轮视觉Agent的RL训练面临稀疏奖励和长期信用分配问题。GTR等先前方法引入强大教师模型（GPT-4o）在每步提供thought-level supervision来缓解"思维崩溃"(entropy collapse)，但代价巨大——15K步训练需86小时和$147的API费用。核心问题：**能否不依赖昂贵外部模型，让Agent自己提供思维引导？**

## 核心问题
如何以极低成本获得一个稳定且能力渐进增长的教师模型，来引导多轮VLM Agent的RL训练中的推理过程？

## 方法详解

### 整体框架
GTR-Turbo的核心洞察：**RL训练过程中产生的历史checkpoint合并后,天然构成一个比当前模型更强、更稳定的教师。** 每次RL更新后,将当前checkpoint加入buffer,用TIES合并所有历史checkpoint得到merged model作为教师。教师通过SFT(生成参考thought)或soft logit distillation(KL散度奖励)引导Agent的推理。

### 关键设计
1. **TIES模型合并作为教师构建**：简单平均(SMA)或指数移动平均(EMA)合并历史checkpoints，用TIES方法(Trim→选择符号→选择性平均)避免参数干扰。实验证实：合并模型在Points24上始终比任何单一checkpoint表现更好更稳定（Figure 2）。原因：合并在更平滑的loss surface上优化，同时保留历史经验。TIES vs 线性平均: TIES更好,但线性平均也work——说明合并教师的idea很robust。

2. **两种思维引导方式**：(a) **SFT引导**：教师自回归生成reference thought → SFT损失训练Agent对齐。保留了GTR的框架但省去了API调用。(b) **KL引导**（更高效）：仅用教师的forward pass计算logit → 计算Agent与教师的reverse KL → 作为token级辅助奖励加入PPO。优势：不需要自回归解码(快)、保留概率分布信息(rich)、鼓励exploration(宽松约束)。reverse KL的mode-seeking特性比forward KL更适合Agent场景。

3. **引导范围仅限thought tokens**：与GTR一致，仅引导reasoning tokens(thought)而不引导action tokens——因为引导action会限制Agent的探索自由度,而探索对自进化系统至关重要。

### 损失函数 / 训练策略
KL版本: $\max_\theta \mathbb{E}[\min(r \cdot A', \text{clip}(r) \cdot A')]$，其中$A' = A_{\pi_\theta} - \text{RevKL}(\pi_\theta, \pi_{merged})$（KL作为负reward），clip负KL值确保奖励有效。LoRA微调,2×40GB GPU (一个放合并教师,一个训练Agent)。Points24训练30K步,ALFWorld训练20K步。

## 实验关键数据
**Points24 (Qwen2.5-VL-7B):**

| 方法 | 成功率 | Episode Return | 训练时间 | 额外成本 |
|--------|------|------|------|------|
| RL4VLM | 3.5% | -13.3 | 86h | $0 |
| GTR (GPT-4o) | 44.5% | 0.53 | 191h | $308 |
| **GTR-Turbo (SFT)** | **48.0%** | **1.32** | **168h** | **$217** |
| **GTR-Turbo (KL)** | **53.5%** | **2.39** | **89h** | **$115** |

GTR-Turbo (KL): 成功率+9% vs GTR，训练时间减少54%，成本减少63%！

**ALFWorld**: GTR-Turbo(KL) 15% ≈ GTR 16%，但训练时间78h vs 164h。

**AitW (Qwen3-VL-8B)**: GTR-Turbo 80.2% vs PPO 75.0% vs DigiRL 71.9%。

### 消融实验要点
- **合并教师 >> 静态KL正则化**：用初始checkpoint做KL reference无法持续提升，合并checkpoint则持续改善
- **RL探索 >> 拒绝采样**：Rejection Sampling无法在Points24上生成正确轨迹,RL可以通过探索获得
- **引导thought only > 引导thought+action**：引导action限制探索,降低性能
- **Reverse KL > Forward KL > 无clip**：Reverse KL的mode-seeking特性更适合Agent; 无clip导致负KL值使Agent远离教师
- **SMA ≈ 平衡EMA(α=0.5)**：过高/过低α都不好,但方法对此不太敏感
- **合并频率鲁棒**：每1-10步合并一次效果类似
- **超过pass@32上限**：GTR-Turbo训练后Agent超过基础模型的pass@32 ceiling——获得了新能力

## 亮点
- **"合并checkpoint是免费教师"**这个洞察极其简洁有力——零额外训练成本、零API费用、零隐私风险
- KL引导只需single forward pass——比SFT引导(需自回归生成)更快,且信息更丰富(概率分布vs one-hot)
- 53.5% vs 3.5%的绝对提升——在Points24这种需要精确推理的任务上效果惊人
- 与Qwen3-VL-8B的兼容性验证——框架与最新模型家族无缝对接
- 在AitW(GUI任务)上也有效——不限于特定环境

## 局限性 / 可改进方向
- 当基础模型太弱(初始成功率<5%)时，自进化可能失败——需要起码的初始能力
- 仅在7-8B模型上验证——更大模型的效果待确认
- ALFWorld上的绝对成功率(15%)仍低——与更强的API教师(Qwen-72B)有差距
- 合并教师的能力上限受限于Agent自身——无法引入Agent本身不知道的外部知识
- 未探索与其他RL算法(GRPO/DAPO)的组合

## 与相关工作的对比
- **vs GTR (GPT-4o teacher)**：GTR需要昂贵API调用,GTR-Turbo用免费合并教师。性能持平甚至更好,成本减少60%
- **vs RL4VLM (无引导)**：RL4VLM在两个环境中都发生思维崩溃。GTR-Turbo通过合并教师有效防止崩溃
- **vs Rejection Sampling (自改进)**：拒绝采样依赖模型能生成正确轨迹——复杂任务中不可能。GTR-Turbo通过RL探索+教师引导获得正确行为
- **vs 静态KL正则化**：用固定初始模型做KL penalty无法随训练进展提升。合并教师持续进化

## 启发与关联
- **核心理念可推广**："合并历史checkpoint作为自进化教师"可应用于任何RL fine-tuning场景——不限于Agent,也适用于数学推理、代码生成等
- 与StaR(GUI Agent)互补——StaR解决toggle状态感知的reasoning chain设计,GTR-Turbo解决多轮RL的training stability,二者可组合
- 与`ideas/llm_nlp/20260317_vlm_self_evolve.md`直接相关——该idea探索VLM自进化,GTR-Turbo提供了一种具体且验证有效的自进化方案（合并教师而非重新生成数据）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "合并checkpoint是免费教师"的洞察简洁而深刻,将模型合并引入Agent RL训练是全新方向
- 实验充分度: ⭐⭐⭐⭐⭐ 2+任务环境(Points24/ALFWorld/AitW)、2个模型(Qwen2.5-VL/Qwen3-VL)、详尽消融(KL/合并方法/权重/引导范围)
- 写作质量: ⭐⭐⭐⭐⭐ 从GTR的pain point→合并教师洞察→SFT/KL两种引导→实验验证的逻辑极其流畅
- 价值: ⭐⭐⭐⭐⭐ 解决了Agent RL训练的实际成本瓶颈,53.5%成功率+60%成本降低=巨大实用价值
