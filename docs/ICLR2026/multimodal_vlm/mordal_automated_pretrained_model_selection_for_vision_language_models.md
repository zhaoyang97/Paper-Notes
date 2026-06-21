---
title: >-
  [论文解读] Mordal: Automated Pretrained Model Selection for Vision Language Models
description: >-
  [ICLR 2026][多模态VLM][VLM] Mordal 把"给定下游任务该挑哪个视觉编码器 + LLM 来拼 VLM"这件事自动化：先用表示相似度聚类砍候选数量，再用早停和缩放律预测砍单候选评估成本，用比网格搜索低 8.9–11.6× 的 GPU 时长找到最优组合。 领域现状：主流 VLM（LLaVA、InternV…
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "VLM"
  - "预训练模型选择"
  - "CKA 表示相似度"
  - "早停"
  - "缩放律预测"
---

# Mordal: Automated Pretrained Model Selection for Vision Language Models

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ZQiO12xlJq](https://openreview.net/forum?id=ZQiO12xlJq)  
**代码**: [https://github.com/SymbioticLab/Mordal](https://github.com/SymbioticLab/Mordal)  
**领域**: 多模态 / 视觉语言模型 / 模型选择 / AutoML  
**关键词**: VLM, 预训练模型选择, CKA 表示相似度, 早停, 缩放律预测  

## 一句话总结
Mordal 把"给定下游任务该挑哪个视觉编码器 + LLM 来拼 VLM"这件事自动化：先用表示相似度聚类砍候选数量，再用早停和缩放律预测砍单候选评估成本，用比网格搜索低 8.9–11.6× 的 GPU 时长找到最优组合。

## 研究背景与动机
**领域现状**：主流 VLM（LLaVA、InternVL、Qwen-VL）几乎都是"视觉编码器 + 特征投影器 + LLM"三段式结构，开发者要从 HuggingFace 上海量预训练模型里手动挑视觉编码器和 LLM 拼成 VLM。

**现有痛点**：选模型全靠人类直觉（挑最新的、最大的、最有名的），既不可靠也不可预测。论文用网格搜索实测发现——**没有银弹**：同一个 LLM 配不同视觉编码器、或同一编码器配不同 LLM，在 GQA / VizWiz / ChartQA / DocVQA / ScienceQA / AI2D 上各有胜负，Vicuna-1.5-7B 拼出来的 VLM 在某些任务甚至能压过 Llama-3-8B。

**核心矛盾**：现成的迁移性指标（EMMS、LogME、LEEP、NLEEP）是为分类/回归/纯语言任务设计的，靠零样本性能当模型质量代理。但 VLM 必须先做视觉-文本对齐——没训练好投影器，LLM 根本读不懂图像 embedding，零样本输出是随机噪声，迁移性指标在这里失效。而暴力网格搜索又贵到离谱：训一个 7B VLM 候选要 100+ GPU 时，49 个候选就是 5439 GPU 时，HuggingFace 上还有 15 万+ LLM 在不断新增。

**本文目标**：形式化"VLM 预训练模型选择问题"——给定对齐数据集和目标任务，找到对齐训练后在目标任务上表现最好的"视觉编码器 + LLM"组合。

**核心 idea**：[降搜索空间 + 降评估成本] 高效搜索必须同时优化两个正交方向——**减少要考虑的 VLM 候选数量**（相似模型性能相似 → 聚类后每簇只评一个），以及**缩短每个候选的评估时间**（早停淘汰弱者 + 缩放律从小样本外推全量性能）。

## 方法详解

### 整体框架
Mordal 是一条"先聚类剪候选、再高效评估剩余候选"的两段式流水线。第一段候选聚类（§3.1）用 CKA 表示相似度把视觉编码器和 LLM 各自分簇，再做簇间/簇内两步评估，把要训练的候选数压下来；第二段高效评估（§3.2）对仍需训练的候选用早停（SHA）和缩放律预测，从中间 checkpoint 复用、小样本外推，避免每个候选都训到全量数据。

```mermaid
flowchart LR
    A[预训练模型池<br/>7 视觉编码器 × 7 LLM] --> B[候选聚类<br/>CKA 相似度两步分簇]
    B --> C[簇间评估<br/>每簇选 medoid + 早停<br/>留 Top-K 簇]
    C --> D[簇内评估<br/>聚合 Top-K 簇候选<br/>缩放律预测全量性能]
    D --> E[最优 VLM 候选]
```

### 关键设计
**1. 表示相似度聚类：用 CKA 把"长得像的候选"归并掉**。Mordal 的出发点是"相似模型性能相似"，但难点在于——还没训练投影器时怎么衡量两个候选的相似度？参数相似度不考虑目标任务的数据分布，靠不住。Mordal 改用与任务相关的**表示相似度**，具体采用 CKA（centered kernel alignment）：$\mathrm{CKA}(K,L)=\frac{\mathrm{HSIC}(K,L)}{\sqrt{\mathrm{HSIC}(K,K)\cdot\mathrm{HSIC}(L,L)}}$，其中 $\mathrm{HSIC}(K,L)=\mathrm{Tr}(KHLH)$，$H=I-\frac{1}{n}\mathbf{1}\mathbf{1}^\top$ 是中心化矩阵。选 CKA 的两个理由：一是它能比较不同形状的表示（不同编码器输出维度不同，余弦相似度这类传统指标做不到），二是视觉表示常经 MLP 投影，而这种变换不破坏 CKA 的性质。实验证实在 ScienceQA 上 CLIP/SigLIP/DFN-CLIP 表示相似、性能也接近，验证了"表示相似 → 性能相似"的前提。

**2. 两步聚类：先聚视觉编码器、再在固定视觉表示下聚 LLM**。逐对算 CKA 很贵（K、L 都是批量数据的激活矩阵），所以 Mordal 拆成两步降成本。第一步对视觉编码器算 CKA、构造距离矩阵 $\mathrm{Dist}_{ve}$，按阈值 $t_{ve}$ 聚成视觉编码器簇 $C_{ve}$；第二步从每个视觉簇取 medoid 编码器生成固定图像 embedding，用一个预热过的投影器把形状对齐到 LLM 输入，再据此构造 LLM 的距离矩阵 $\mathrm{Dist}_{llm}$、按阈值 $t_{llm}$ 聚簇。最后对视觉簇和对应 LLM 簇做笛卡尔积得到候选簇。这样就避开了去算"视觉编码器本就不相似"的候选对之间的相似度——而那些恰恰是性能差异最大、最该被拆开的组合。

**3. 簇间/簇内两步评估 + 早停：先粗筛簇、再细选候选**。聚类后，簇间评估从每簇取 medoid 当代表，比较各簇性能、淘汰整体差的簇，留 Top-K 簇（K 用户定）。这一步用 Successive Halving（SHA）做早停激进剪枝：每轮给每个候选分配预算 $b$、评估后只保留 top $1/\eta$，下一轮把预算提到 $b\times\eta$，反复直到收敛或选出 Top-K，中间产生的 checkpoint 可被下游复用。簇间筛完后，簇内评估把 Top-K 簇里的候选聚到一起、回到候选粒度，训练后返回最优。关键经验是早停不能用到底——纯早停会误杀"前期慢热但后期强"的候选（如 AI2D 上的 SigLIP-Qwen），所以 Mordal 限制早停只用在簇间。

**4. 观测缩放律预测：从小样本外推全量性能，少训多估**。簇内评估如果每个候选都训全量数据仍然贵，于是 Mordal 引入缩放律预测。经典缩放律 $L(N,D)=\frac{a}{N^\alpha}+\frac{b}{D^\beta}+e$ 刻画参数量 $N$、数据量 $D$ 与性能的幂律关系；Mordal 的设定是 $N$ 固定、只变对齐数据量 $D$，并实测发现对齐数据量与任务误差在 log-log 下呈**对数线性**关系。于是算法从最大采样比 $R$（如 $\frac{1}{8}$）起，评估该比例训出的 checkpoint、记录 $(\log r,\log\mathrm{Err})$ 点，逐步把采样比降为 $r/u$ 并复用上一档的中间 checkpoint，攒够 $p$ 个点后拟合线性回归 $f_c$，用 $f_c(1)$ 外推全量性能而无需真正训到全量。早停和缩放律由此互补：早停管粗筛、缩放律管对有潜力候选的精确预测，二者都靠复用中间 checkpoint 省时。

## 实验关键数据
设置：7 数据集（Visual QA / Doc QA / Knowledge / 多学科 MMMU），7 视觉编码器 × 7 LLM = 49 候选，16× A40 GPU，LLaVA-1.5-Instruction 做对齐，LoRA 微调 LLM，两层 MLP 投影器。

### 主实验表格（搜索时间 + Top-1 命中）
网格搜索每任务需 **5439 GPU 时** 训完 49 候选。

| 任务 | 数据集 | LLaVA-1.5 基线 | Mordal 选中模型 | Mordal 时长(h) | 加速 | Top-1 分数 |
|------|--------|----------------|-----------------|----------------|------|------------|
| Visual QA | GQA | 61.5 | SigLIP-Vicuna | 483 | 11.2× | 66.4 |
| Visual QA | VizWiz | 41.2 | SigLIP-Mistral | 469 | 11.6× | 46.9 |
| Doc QA | ChartQA | 18.2 | CLIP-Qwen | 607 | 8.9× | 18.6 |
| Doc QA | DocVQA | 27.6 | SigLIP-Qwen | 593 | 9.2× | 28.5 |
| Knowledge | ScienceQA | 70.4 | SigLIP-Llama | 472 | 11.5× | 78.5 |
| Knowledge | AI2D | 54.8 | SigLIP-Qwen | 496 | 10.9× | 65.2 |
| 多学科 | MMMU | 35.3 | SigLIP-Llama | 503 | 10.8× | 36.6 |

七个任务里六个命中网格搜索的真·Top-1，且选中组合普遍优于默认 LLaVA-1.5-7B 结构。

### 排队质量对比（加权 Kendall's τ，等算力预算下）

| 数据集 | EMMS | LogME | LEEP | NLEEP | Mordal |
|--------|------|-------|------|-------|--------|
| GQA | 0.682 | -0.162 | 0.232 | 0.435 | **0.814** |
| VizWiz | 0.657 | 0.236 | 0.351 | 0.502 | **0.882** |
| ChartQA | 0.238 | -0.144 | -0.071 | 0.298 | **0.765** |
| DocVQA | 0.172 | 0.155 | 0.111 | 0.265 | **0.897** |
| ScienceQA | 0.770 | 0.269 | 0.344 | 0.562 | **0.960** |
| AI2D | 0.557 | 0.193 | 0.316 | 0.614 | **0.894** |
| MMMU | 0.526 | 0.101 | 0.245 | 0.220 | **0.875** |

Mordal 在全部 7 个数据集上 τ 都最高，平均比最强基线高约 69%。

### 消融实验（GQA，去掉某组件）

| 配置 | 说明 |
|------|------|
| w/o EE（去高效评估） | 仅靠簇间/簇内评估就大幅降时且 τ 仍高，验证聚类是核心 |
| w/o ES（去早停） | 时间上升 |
| w/o SP（去缩放律预测） | 早停用到底会误杀慢热强候选（AI2D 上 SigLIP-Qwen），τ 掉 |
| 敏感性 $t_{ve}$ | 0.5→τ 仅 0.52、446h；0.9→τ 0.86 但 1041h，阈值在"省时"和"准"间权衡 |

### 关键发现
- 对齐数据量与任务误差在 log-log 下对数线性，但只在样本量超过某阈值后才显现（与 Lin/Ruan 一致）。
- 早停适合粗筛、缩放律适合精测，二者分工才能在降时同时保住 τ。

## 亮点与洞察
- **问题定义本身是贡献**：第一次把"VLM 预训练组件选择"形式化为资源受限下预测对齐后性能的问题，并实证"没有银弹"。
- **用 CKA 绕过"必须先训对齐才能比较"的死结**：把跨形状、经 MLP 投影后仍稳健的表示相似度当聚类信号，是整套方法能省钱的根。
- **两正交方向拆解干净**：降候选数（聚类）和降单候选成本（早停+缩放律）正交，各自的中间 checkpoint 还能互相复用。

## 局限与展望
- 偶尔会漏掉最优组合——当强候选被归到弱簇里，或被早停过早淘汰（7 任务漏 1 个 Top-1 即此因）。
- 强依赖"表示相似 → 对齐后性能相似"的经验假设；CKA 相似但对齐后性能分叉的反例会破坏聚类有效性。
- 缩放律外推需要对数线性区间真实存在且样本量过阈值，对小数据/特殊任务可能不稳。
- 实验固定 LLaVA-1.5 结构、LoRA、两层 MLP 投影器和 7×7 模型池，更大搜索空间和更换投影器/训练范式下的可扩展性待验证。

## 相关工作与启发
- **模型选择/迁移性指标**：EMMS、LogME、LEEP、NLEEP 面向分类/回归，靠迁移性度量；本文指出它们在需对齐的 VLM 上失效，转而直接预测对齐后性能。
- **表示相似度**：CKA（Kornblith 2019）原用于比较网络各层表示，这里被借来做"未训练即可比较候选"的聚类信号。
- **早停 / HPO**：Successive Halving（Jamieson & Talwalkar 2016）的预算翻倍剪枝被用作簇间早停。
- **缩放律**：Kaplan/Hoffmann 的幂律和近期用于 LLM 模型选择的观测缩放律（Lin 2024、Ruan 2024），被改造成"固定 N、变 D"的对齐数据缩放预测。
- **启发**：这套"相似度聚类 + 早停 + 缩放外推"的省算力配方可迁移到其他需昂贵训练才能评估的组合搜索问题（如多模态融合结构搜索、模块化模型拼装）。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 首次形式化 VLM 预训练组件选择问题，并指出现有迁移性指标因缺对齐而失效，CKA 聚类的引入切中要害。
- **实验充分度**: ⭐⭐⭐⭐ — 7 数据集、49 候选网格搜索做 ground truth，与 4 个选择基线对比 + 完整消融/敏感性分析，证据扎实；唯模型池规模（7×7）和单一架构稍受限。
- **写作质量**: ⭐⭐⭐⭐ — 动机层层递进（无银弹→指标失效→暴搜太贵→两方向拆解），图表清晰，方法可复现。
- **价值**: ⭐⭐⭐⭐ — 把"拼 VLM 该选什么底座"从拍脑袋变成低成本可搜索，对工程落地和 AutoML-for-multimodal 都有实用意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Vision-Language Model Selection and Reuse for Downstream Adaptation](../../ICML2025/multimodal_vlm/vision-language_model_selection_and_reuse_for_downstream_adaptation.md)
- [\[ICLR 2026\] UniHM: Unified Dexterous Hand Manipulation with Vision Language Model](unihm_unified_dexterous_hand_manipulation_with_vision_language_model.md)
- [\[AAAI 2026\] PET2Rep: Towards Vision-Language Model-Driven Automated Radiology Report Generation for Positron Emission Tomography](../../AAAI2026/multimodal_vlm/pet2rep_towards_vision-language_model-drived_automated_radiology_report_generati.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[CVPR 2026\] TTL: Test-time Textual Learning for OOD Detection with Pretrained Vision-Language Models](../../CVPR2026/multimodal_vlm/ttl_test-time_textual_learning_for_ood_detection_with_pretrained_vision-language.md)

</div>

<!-- RELATED:END -->
