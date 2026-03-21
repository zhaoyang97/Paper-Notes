# Boost Your Human Image Generation Model via Direct Preference Optimization

**会议**: CVPR 2025  
**arXiv**: [2405.20216](https://arxiv.org/abs/2405.20216)  
**代码**: 待确认  
**领域**: 对齐RLHF / 人像生成  
**关键词**: DPO, Human Image, Curriculum Learning, Statistics Matching, LoRA

## 一句话总结
提出 HG-DPO，以真实人像作为 DPO 的 winning image（而非生成图像对）+ 三阶段课程学习（Easy/Normal/Hard）渐进弥合生成-真实图像分布 gap + 统计匹配损失解决色偏，FID 从 37.34 降至 29.41（-21.4%），CI-Q 0.906→0.934，win-rate 超越 Diffusion-DPO 达 99.97%。

## 研究背景与动机
1. **领域现状**：人像生成是图像合成的重点，DPO 已被用于对齐扩散模型，但现有方法使用 AI 生成的 winning/losing 图像对。
2. **现有痛点**：(a) 生成图像作 winning image 有天花板；(b) 直接用真实图像会导致色偏和训练不稳定。
3. **核心矛盾**：真实图像是更好的对齐目标，但生成-真实图像分布 gap 导致直接使用时训练崩溃（Naive DPO FID 暴涨到 112.67）。
4. **本文要解决什么？** 弥合分布鸿沟，让 DPO 能以真实人像为优化目标。
5. **切入角度**：课程学习逐步缩小 gap + 统计匹配消色偏。
6. **核心idea一句话**：三阶段课程 DPO（Easy→Normal→Hard 渐进引入真实图像）+ 统计匹配损失。

## 方法详解

### 整体框架
基于 SD 1.5 + LoRA 微调（U-Net rank 8, text encoder rank 64），三阶段课程训练渐进将 winning image 从生成最优→中间域→真实图像。$\beta=2500$。训练数据集包含 ~50K 真实人像（来自 LAION-Aesthetics 筛选）和对应 prompt。推理时直接使用训练好的 LoRA 权重，无额外开销。

### DPO 损失回顾
扩散模型 DPO 的损失函数为：
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \left(r(x_w, c) - r(x_l, c)\right)\right)\right]$$
其中 $r(x, c) = \log \frac{p_\theta(x|c)}{p_{\text{ref}}(x|c)}$ 为隐式奖励，$x_w, x_l$ 分别为 winning/losing image。HG-DPO 的创新在于 $x_w$ 的选择策略——从生成图像渐进到真实图像。

### 关键设计
1. **三阶段课程学习**：Easy（300K steps，生成图像间 DPO，N>2 图像池选优劣）→ Normal（20K，中间域过渡）→ Hard（20K，真实图像为 winning image）。Naive 直接用真实图像 FID 暴涨至 112.67，课程学习有效避免崩溃
   - Easy 阶段：对每个 prompt 生成 N=8 张图像，用 PickScore/HPSv2 打分，选最高和最低分构成偏好对。该阶段使模型学会区分好坏生成，为后续引入真实图像奠定基础
   - Normal 阶段：winning image 为 Easy 阶段最优生成与真实图像的插值混合（α=0.5 加权），losing image 仍为低分生成。渐进缩小生成域与真实域的分布距离
   - Hard 阶段：直接使用真实人像作为 winning image，losing image 为当前模型生成。此时模型已经适应了分布偏移，可以稳定训练
2. **统计匹配损失 $\mathcal{L}_{stat}$**：匹配生成与真实图像的 channel-wise 均值和标准差，消除色偏。公式为 $\mathcal{L}_{stat} = \sum_c \left|\mu_c^{gen} - \mu_c^{real}\right| + \left|\sigma_c^{gen} - \sigma_c^{real}\right|$，仅在 Hard 阶段启用。加入后 CI-Q 从 0.888→0.906
3. **偏好评估器**：Easy 阶段用奖励模型对每个 prompt 的多个生成图像排序，选最好/最差组成偏好对
4. **LoRA 配置**：U-Net rank=8（注意力层），text encoder rank=64（CLIP text encoder）。Text encoder LoRA 在 Hard 阶段尤为关键——它帮助模型适应真实图像的语义分布，去掉后 FID 从 29.41 升至 32.18

## 实验关键数据

### 主实验
| 方法 | P-Score | FID↓ | CI-Q | CI-S | ATHEC |
|------|:---:|:---:|:---:|:---:|:---:|
| Diffusion-DPO | 17.93 | 112.67 | 0.820 | 0.944 | 36.30 |
| AlignProp | 23.02 | 49.92 | 0.860 | 0.966 | 17.05 |
| Curriculum-DPO | 22.44 | 35.35 | 0.889 | 0.956 | 23.36 |
| **HG-DPO** | **22.60** | **29.41** | **0.934** | **0.986** | **29.41** |

Win-rate: vs Diffusion-DPO **99.97%**, vs Pick-a-Pic v2 **86.03%**

### 消融实验
| 配置 | FID | CI-Q | 说明 |
|------|:---:|:---:|------|
| Base | 37.34 | 0.906 | 基线 |
| Naive (直接真实) | 112.67 | 0.820 | 崩溃 |
| Easy only | 36.00 | 0.906 | 生成图像间 DPO |
| + Normal + Hard | 28.66 | 0.937 | 课程渐进 |
| **+ TE LoRA** | **29.41** | **0.934** | **完整 HG-DPO** |
| - $\mathcal{L}_{stat}$ | 31.52 | 0.888 | 无统计匹配 |
| Easy 只用2张 | 38.91 | 0.895 | N=2 不够 |

### 关键发现
- 课程 Hard 阶段是 FID 主要贡献（37→29），Easy 阶段是 P-Score 主要贡献
- N=8 图像池比 N=2 好得多——更多候选使偏好对的质量差距更明显
- 统计匹配损失消除了可见的色偏伪影（偏蓝/偏黄），对 CI-Q 提升 +0.018
- 支持个性化 T2I：HG-DPO + InstantBooth FID 29.30（vs 39.61），人脸相似度保持
- Hard 阶段仅需 20K steps（占总训练量 ~6%），但贡献了 FID 最大降幅

## 亮点与洞察
- **真实图像作 DPO 目标**是核心创新——打破了生成图像偏好对齐的天花板，传统 DPO 用 AI 生成的 winning image 有固有质量上限
- **课程学习弥合分布 gap**优雅实用——domain adaptation 思想在 DPO 中的应用，三阶段渐进的设计比直接 fine-tuning 更稳定
- **统计匹配损失的针对性设计**：色偏是生成→真实图像 DPO 的特有问题，channel-wise 统计对齐是简洁有效的解决方案
- **与个性化模型的兼容性**：HG-DPO 训练的 LoRA 可直接与 InstantBooth 等个性化方法叠加，说明学到的偏好知识是通用的

## 局限性 / 可改进方向
- 需要高质量真实人像数据集，训练成本不低（340K steps total），Easy 阶段需要大量 GPU 时间生成偏好对
- 仅验证了人像，可扩展到其他有丰富真实数据的领域（如风景、建筑、动物）
- DPO 的 $\beta=2500$ 极大，远高于 LLM 中的典型值（0.1-0.5），说明扩散模型 DPO 的超参数空间与 LLM 差异巨大，调参成本高
- Normal 阶段的插值策略（α=0.5）是启发式选择，自适应α可能更好
- 未测试更大的基础模型（如 SDXL、SD3），可扩展性存疑

## 相关工作与启发
- **vs Diffusion-DPO**: 直接用 DPO 人像上 FID 112，HG-DPO 课程学习解决。Diffusion-DPO 的失败说明生成-真实分布鸿沟是根本问题
- **vs Curriculum DPO**: 不同关注点——Curriculum DPO 关注样本难度排序，HG-DPO 关注 winning image 选择（从生成到真实的渐进）
- **vs AlignProp**: AlignProp 用可微奖励信号反传，FID 49.92 优于 Diffusion-DPO 但逊于 HG-DPO。AlignProp 的优势在于不需偏好对，但受限于奖励模型质量
- **vs RLHF for LLMs**: DPO 在扩散模型中的应用与 LLM 有本质差异——LLM 的 winning/losing 是文本序列，扩散模型是像素空间分布，后者的分布 gap 问题更严重

## 评分
- 新颖性: ⭐⭐⭐⭐ 真实图像 DPO + 课程 gap 弥合，思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 9 指标、10 方法对比、多 seed、PT2I 扩展
- 写作质量: ⭐⭐⭐⭐ 动机清晰消融充分
- 价值: ⭐⭐⭐⭐ 人像 DPO 的重要参考

## 补充说明
- 本文被分类到 llm_alignment，实际属于 image generation / diffusion model alignment 方向
- 课程学习三阶段的步数比例（300K:20K:20K）暗示 Easy 阶段是最耗时的——需要为每个 prompt 生成多张图像并用奖励模型排序
- $\beta=2500$ 的选择值得注意——如此大的 $\beta$ 意味着模型对偏好差异非常敏感，小的 $\beta$ 在扩散模型 DPO 中可能无法产生有意义的梯度信号
- 方法可直接扩展到 SDXL/SD3 等更大模型，只需调整 LoRA rank
- 真实图像数据集的质量和多样性对 Hard 阶段至关重要——低质量真实图像反而会降低生成质量

