# ConsistCompose: Unified Multimodal Layout Control for Image Composition

**会议**: CVPR 2026  
**arXiv**: [2511.18333](https://arxiv.org/abs/2511.18333)  
**代码**: 无  
**领域**: 图像生成 / 布局控制  
**关键词**: 布局控制生成, 多实例图像合成, LELG, 坐标嵌入prompt, 身份保持

## 一句话总结
提出 ConsistCompose，通过将布局坐标直接嵌入语言prompt（LELG范式），在统一多模态框架中实现布局可控的多实例图像生成；构建340万样本的ConsistCompose3M数据集提供布局+身份监督；配合坐标感知CFG机制，在COCO-Position上实现布局IoU 7.2%提升和AP 13.7%提升，同时保持通用理解能力。

## 研究背景与动机

1. **领域现状**：统一多模态模型（如Bagel、OmniGen2）已能在单一架构中完成理解和生成，但主要聚焦于视觉理解（grounding），生成侧的布局精确控制仍然薄弱。
2. **现有痛点**：布局控制生成的现有方法存在根本性障碍——(a) 扩散模型方法（GLIGEN、InstanceDiffusion）依赖专门的布局-图像融合模块或区域感知U-Net，与Transformer生成框架不兼容；(b) 自回归模型（LayoutSAM、PlanGen）将布局作为独立模态处理，仅限于布局任务，无法兼顾视觉推理、理解等通用能力；(c) 多数方法只支持文本条件布局控制，不考虑更难的多参考图像身份保持场景。
3. **核心矛盾**：布局控制需要任务特定的分支/编码器，这与"统一"框架的理念相矛盾。如何在不引入额外架构模块的情况下实现精确布局控制？
4. **本文要解决什么？** 在统一多模态框架中同时支持：布局接地文本到图像生成、多参考身份一致的多实例合成、通用多模态理解——三者共用一个模型。
5. **切入角度**：布局本质上是一种可以用语言表达的信息。与其设计专门的空间编码器，不如把坐标编码为文本token，让Transformer通过语言理解自然学习空间接地。
6. **核心idea一句话**：语言即布局控制——将坐标嵌入prompt，让统一模型通过文本流学习空间布局，无需任何架构改动。

## 方法详解

### 整体框架
基于Bagel的MoT（Mixture of Transformers）架构，包含理解和生成两个Transformer专家。输入是带有坐标标注的文本prompt + 可选的参考图像，输出是满足布局约束的多实例图像。三大组件：(1) LELG范式将布局语义编码为文本token；(2) Coordinate-CFG在采样时增强空间控制；(3) ConsistCompose3M提供训练数据。

### 关键设计

1. **LELG范式 + 实例-坐标绑定Prompt (ICBP)**:
   - 做什么：将每个实例的bounding box直接插入其对应主语短语之后，形成统一的文本序列
   - 核心思路：对第 $i$ 个实例，归一化bbox $b_i = (x_1^i, y_1^i, x_2^i, y_2^i) \in [0,1]^4$ 用三位小数表示插入文本："a brown sofa <bbox>[0.123, 0.456, 0.789, 0.901]</bbox>"。这样坐标成为语言token的一部分，Transformer通过共享的self-attention自然学习实例-位置绑定。
   - 设计动机：(1) **零架构改动**——不需要布局编码器、ControlNet或额外attention模块；(2) **天然统一**——理解和生成共享同一个token空间，空间推理能力可以从理解任务迁移到生成；(3) 三位小数的离散化将连续空间映射到约1000³个离散位置，精度足够且与tokenizer兼容。
   - 与之前方法的区别：GLIGEN需要gated Transformer层，InstanceDiffusion需要多模态融合模块，CreatiLayout需要SiamLayout——都是架构层面的改动。LELG纯粹在输入层面解决问题。

2. **坐标感知Classifier-Free Guidance (Coordinate-CFG)**:
   - 做什么：在推理采样时，通过比较有/无坐标条件的预测速度差来增强空间控制
   - 核心思路：$\mathbf{v}_t^{\text{coord-cfg}} = \mathbf{v}_t^{\text{uncond}} + s_{\text{coord}}(\mathbf{v}_t^{\text{coord}} - \mathbf{v}_t^{\text{uncond}})$，$s_{\text{coord}}$ 控制空间引导强度。还引入了速度归一化 $\alpha = \|\mathbf{v}_t^{\text{base}}\| / \|\mathbf{v}_t^{\text{coord-cfg}}\|$ 防止引导幅度爆炸。
   - 设计动机：ICBP提供了空间信号，但模型可能不够"服从"。Coordinate-CFG类似于文本CFG的空间版本，显式放大坐标条件与无条件之间的差异，迫使生成更精确地遵循布局。实验显示增大 $s_{\text{coord}}$ 逐步提升位置准确性，过大时略微影响感知质量。

3. **ConsistCompose3M数据集**:
   - 做什么：提供340万样本的布局+身份监督训练数据
   - 核心思路：两个子集——(a) **T2I子集** (260万)：重新处理LayoutSAM数据，为每个实例用ICBP机制附加bbox坐标到caption中；(b) **参考条件子集** (80万)：复用Subjects200K和UNO的主体素材，在不同布局下重组为多主体场景，用CLIP/DINO相似度过滤保证身份一致性。
   - 设计动机：之前没有兼具布局标注和身份标注的大规模多实例生成数据集。缺乏数据是布局控制生成进展缓慢的重要原因。

### 训练策略
- **两阶段训练**：先做对齐阶段（混合通用理解数据+ConsistCompose3M注入布局意识），再做混合SFT阶段（联合训练理解/生成/编辑/多主体参考生成+ConsistCompose3M）
- **训练目标**：Flow Matching损失 $\mathcal{L}_{\text{FM}}$ + 语言模型损失 $\mathcal{L}_{\text{LM}}$ 的加权组合，无额外坐标回归损失——空间接地完全从语言流中隐式学习
- **高分辨率微调**：进一步平衡布局控制和通用图像生成性能

## 实验关键数据

### 主实验（COCO-Position）

| 方法 | Instance Success Avg↑ | Image Success Avg↑ | mIoU↑ | AP↑ | AP50↑ | AP75↑ |
|------|---------------------|-------------------|-------|-----|-------|-------|
| GLIGEN | 82.6 | 52.1 | 69.0 | 40.5 | 75.9 | 39.1 |
| InstanceDiffusion | 87.8 | 65.5 | 78.1 | 57.2 | 83.6 | 65.5 |
| MIGC++ | 86.8 | 63.4 | 74.9 | 48.3 | 79.2 | 52.6 |
| CreatiLayout | 74.0 | 42.5 | 64.9 | 32.4 | 61.1 | 31.6 |
| PlanGen | 82.5 | 50.3 | 66.2 | 31.9 | 74.0 | 21.5 |
| **ConsistCompose** | **92.6** | **76.1** | **85.3** | **70.9** | **89.1** | **76.9** |

- 相比最强基线InstanceDiffusion：布局mIoU +7.2%，AP +13.7%，Image Success Avg +10.6%

### 训练阶段消融

| 阶段 | Instance Success Avg | mIoU | AP |
|------|---------------------|------|-----|
| Alignment only | 88.4 | 79.1 | 58.3 |
| + Hybrid SFT | **92.6** | **85.3** | **70.9** |

### 关键发现
- **LELG有效性**：仅通过语言嵌入坐标（无额外架构），布局准确性即大幅超越所有专门设计的基线
- **通用能力保持**：在MMMU和MMBench上与Bagel backbone持平，说明布局控制训练不会损害通用理解
- **Coordinate-CFG的作用**：$s_{\text{coord}}$ 从1到3逐步提升位置精度，存在最优点（过大会略损质量）
- **两阶段训练必要**：Hybrid SFT阶段在Alignment基础上进一步提升AP 12.6%

## 亮点与洞察
- **LELG范式的简洁性**令人印象深刻：将"布局控制"这个看似需要专门模块的问题，化简为"在prompt中插入坐标"——零架构改动实现SOTA布局精度。这个设计思路暗示了一个更大的insight：很多看似需要专门模块的条件控制（深度、边缘、关键点），都可能被统一为语言接口的一部分。
- **Coordinate-CFG**巧妙地将CFG从"语义引导"扩展到"空间引导"，且独立于文本CFG工作，可以叠加使用。这个设计可以迁移到任何支持CFG的生成模型中。
- **数据集构建策略**值得借鉴：通过重新处理已有数据（LayoutSAM→T2I, Subjects200K→参考条件）构建新用途的数据集，高效利用已有资源。

## 局限性 / 可改进方向
- 三位小数的坐标离散化在高分辨率场景下可能精度不足（约0.1%图像宽度的误差）
- 当前只支持bounding box级别的布局控制，不支持更细粒度的mask、关键点或深度条件
- 依赖Bagel作为backbone，受限于其基础生成质量和训练规模
- 需要专门构建ConsistCompose3M数据集，数据准备成本不低
- 多实例场景中实例数较多时（如>6个），性能可能下降（COCO-Position测试最多6个实例）

## 相关工作与启发
- **vs GLIGEN [Li et al., 2023]**: GLIGEN用gated Transformer层引入bbox约束，是架构层面的改动。ConsistCompose的LELG范式更轻量且效果更好（AP +30.4%）
- **vs InstanceDiffusion [Wang et al., 2024]**: InstanceDiffusion通过多模态输入融合实现实例级控制，但仍是U-Net范式。ConsistCompose在Transformer生成范式下超越它
- **vs PlanGen [Gong et al., 2024]**: PlanGen先规划布局再生成图像，分两步走。ConsistCompose的端到端方式更统一且效果更好

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ LELG范式是布局控制生成的范式创新，用语言接口统一空间控制
- 实验充分度: ⭐⭐⭐⭐⭐ COCO-Position、MS-Bench、GenEval、MMMU、MMBench全面评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，技术细节充分
- 价值: ⭐⭐⭐⭐⭐ 为统一多模态模型的布局控制提供了简洁有效的解决方案
