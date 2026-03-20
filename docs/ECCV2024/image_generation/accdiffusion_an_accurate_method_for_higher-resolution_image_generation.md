# AccDiffusion: An Accurate Method for Higher-Resolution Image Generation

**会议**: ECCV 2024  
**arXiv**: [2407.10738](https://arxiv.org/abs/2407.10738)  
**代码**: [https://github.com/lzhxmu/AccDiffusion](https://github.com/lzhxmu/AccDiffusion) (有)  
**领域**: 图像生成 / 扩散模型 / 高分辨率生成  
**关键词**: 高分辨率图像生成, Patch-wise去噪, Cross-Attention引导, 目标重复消除, 免训练  

## 一句话总结
提出AccDiffusion，通过将全局文本prompt解耦为patch级别的内容感知prompt（利用cross-attention map判断每个词汇是否属于某patch），并引入带窗口交互的膨胀采样来改善全局一致性，在无需额外训练的情况下有效解决patch-wise高分辨率图像生成中的目标重复问题，在SDXL上实现了从2K到4K分辨率的无重复高质量图像外推。

## 背景与动机
Stable Diffusion等扩散模型训练成本极高（SD 1.5在256块A100训练20+天），因此训练分辨率通常被限制在512²（SD 1.5）或1024²（SDXL）。然而现实应用（如广告）对高分辨率图像有强需求。直接在高分辨率上推理会产生严重的目标重复和结构不一致问题。

现有方法分为两类：（1）**直接生成法**如Attn-SF和ScaleCrafter，修改注意力缩放因子或卷积感受野，但GPU显存随分辨率急剧增长且图像质量差；（2）**间接/Patch-wise生成法**如MultiDiffusion和DemoFusion，将高分辨率图像分割为多个patch独立去噪后融合，显存可控但产生严重的目标重复。DemoFusion虽通过残差连接和膨胀采样引入全局语义信息，部分缓解了重复问题，但在超高分辨率下仍然会生成小尺度的重复目标。

## 核心问题
**Patch-wise高分辨率生成中，为什么会产生小目标重复？如何彻底消除这一问题？**

论文通过深入的消融分析发现了根本原因：小目标重复是**相同文本prompt施加于所有patch**（倾向于在每个patch中重复生成目标）与**残差连接+膨胀采样提供的全局语义**（抑制重复生成）之间的**对抗性结果**。去掉prompt重复消失但细节退化，去掉全局信息则出现大目标重复。因此问题的关键在于：需要为每个patch提供更精确的、与其内容相匹配的prompt。

## 方法详解

### 整体框架
AccDiffusion基于DemoFusion的渐进式上采样流水线，分为两个阶段：
1. **Phase 1**: 在预训练分辨率（如1024²）生成低分辨率图像，同时采集cross-attention map
2. **Phase 2**: 渐进式上采样生成更高分辨率图像。在每个上采样阶段，先用patch-wise去噪生成局部细节，再用带窗口交互的膨胀采样增强全局一致性，最后通过残差连接注入低分辨率结构信息

核心创新在Phase 2中：用patch-content-aware prompt替代统一的prompt，用窗口交互改善膨胀采样质量。

### 关键设计

1. **Patch-Content-Aware Prompts（核心贡献）**: 利用Phase 1生成过程中U-Net的cross-attention map来自动判断每个词汇属于图像的哪个区域。具体做法是：对cross-attention map ℳ按列取均值作为阈值，将其二值化为mask ℬ；对二值mask进行形态学开操作（先腐蚀后膨胀）消除小连通域噪声；将mask上采样到高分辨率后，用滑动窗口切出每个patch对应的区域mask；根据每个词汇在patch区域内高响应区的占比是否超过阈值c来决定该词是否属于该patch的prompt。这样每个patch获得了内容匹配的子prompt，避免在不含某目标的patch中强行生成该目标。

2. **Dilated Sampling with Window Interaction**: DemoFusion的膨胀采样独立对每个采样子集去噪，导致全局语义信息不光滑、呈噪声状（不同子集间缺少交互）。AccDiffusion在每次去噪前，通过位置相关的双射函数(bijection function)在同一窗口内交换不同膨胀采样间的噪声，使它们在去噪过程中相互影响。去噪后再用逆映射恢复原始位置。这种窗口交互使膨胀采样产生的全局语义信息更加平滑连贯。

3. **自适应阈值设计**: 不同词汇的cross-attention map值域差异很大（如"Astronaut"的均值为0.13左右，"mars"为0.20），使用固定阈值会导致部分词汇被全部纳入或全部排除。因此采用每个词汇自身attention均值作为自适应阈值，确保每个词汇都有合理的高响应区域。

### 损失函数 / 训练策略
AccDiffusion是完全**免训练**的plug-and-play方法，无需任何微调或额外训练。它直接复用预训练的Stable Diffusion模型（SDXL/SD1.5/SD2.1），仅在推理时修改prompt分配和膨胀采样策略。超参数c=0.3控制词汇纳入patch prompt的阈值，膨胀采样权重η按余弦调度从1递减至0。

## 实验关键数据

| 分辨率 | 方法 | FID_r↓ | IS_r↑ | FID_c↓ | IS_c↑ | CLIP↑ | 时间 |
|--------|------|--------|-------|--------|-------|-------|------|
| 2048² (4×) | DemoFusion | 60.46 | 16.45 | 38.55 | 24.17 | 32.21 | 3min |
| 2048² (4×) | **AccDiffusion** | **59.63** | **16.48** | **38.36** | **24.62** | **32.79** | 3min |
| 3072² (9×) | DemoFusion | 62.43 | 16.41 | 47.45 | 20.42 | 32.25 | 11min |
| 3072² (9×) | **AccDiffusion** | **61.40** | **17.02** | **46.46** | **20.77** | **32.82** | 11min |
| 4096² (16×) | DemoFusion | 65.97 | 15.67 | 59.94 | 16.60 | 33.21 | 25min |
| 4096² (16×) | **AccDiffusion** | **63.89** | **16.05** | **58.51** | **16.72** | **33.79** | 26min |

在所有分辨率下，AccDiffusion在所有指标上均优于DemoFusion及其他方法（SDXL-DI、Attn-SF、MultiDiffusion、ScaleCrafter），且推理时间与DemoFusion几乎相同。

### 消融实验要点
- **两个核心模块互补**：去掉patch-content-aware prompt会产生大量重复小目标；去掉窗口交互则生成的小目标与图像语义不相关；同时去掉两者重复最严重；同时使用两者则完全消除重复
- **阈值c的敏感性**：c过小（如0.1）会将过多词汇纳入patch prompt，仍产生重复；c过大（如0.9）会过度简化prompt，导致细节退化。c=0.3是较好的平衡点，但可根据具体场景调节
- **自适应阈值优于固定阈值**：不同词汇的attention map值域差异大，使用均值自适应阈值比固定阈值更稳健

## 亮点
- **根因分析精准**: 通过消融实验深刻揭示了patch-wise生成中目标重复的根本原因——统一prompt在所有patch上的引导效应。这个发现本身就很有价值
- **无需外部模型**: 直接利用扩散模型自身的cross-attention map来确定patch内容，无需引入SAM等额外分割模型，优雅且高效
- **即插即用**: 完全免训练，适用于SD1.5、SD2.1、SDXL等多种stable diffusion变体
- **形态学操作去噪**: 巧妙利用数学形态学的开操作消除attention map中的小连通域噪声，是一个可复用的trick
- **窗口交互的双射设计**: 通过时间和位置相关的双射函数在膨胀采样间引入交互，既保持了理论可逆性又改善了全局一致性

## 局限性 / 可改进方向
- **推理延迟未改善**: 继承了DemoFusion的渐进式上采样和重叠patch去噪策略，推理时间随分辨率快速增长（4K需25min）
- **依赖预训练模型质量**: 高分辨率图像的保真度受限于底层diffusion模型的能力
- **极端高分辨率退化**: 超过6K（36×）时出现细节退化，说明cross-attention map引导的精度有上限
- **局部不合理内容**: 依赖LDM对裁剪图像的先验知识，在极端特写生成时可能产生不合理局部内容
- **未探索非重叠patch去噪**: 论文自身指出，非重叠patch-wise去噪是提高效率的潜在方向

## 与相关工作的对比
- **vs DemoFusion (CVPR 2024)**: 同为patch-wise方法，DemoFusion引入残差连接+膨胀采样但仍有小目标重复。AccDiffusion从根因出发，用patch-content-aware prompt彻底解决重复问题，且不增加推理时间
- **vs MultiDiffusion (ICML 2023)**: MultiDiffusion仅做基础的重叠patch融合，无全局语义信息，产生严重重复和变形。AccDiffusion在其基础上引入了更精细的prompt控制和全局信息增强
- **vs ScaleCrafter (ICLR 2024)**: 直接生成方法，修改卷积感受野，但GPU显存随分辨率快速增长且仍有结构变形问题。AccDiffusion通过patch-wise方式避免了显存瓶颈

## 启发与关联
- **Cross-Attention Map的利用方式值得借鉴**: 论文巧妙地将cross-attention map从Prompt-to-Prompt图像编辑领域迁移到高分辨率生成中，用于自动确定patch-prompt对应关系。这种利用扩散模型内部表征指导生成控制的思路有广泛适用性
- **与image_generation/ideas中的过程感知对齐(process_aware_alignment)有间接关联**: AccDiffusion通过分析扩散过程中间状态（attention map）来指导后续生成，这种"过程感知"的思路与过程感知对齐的理念有共通之处
- **形态学后处理trick可迁移**: 在attention map或feature map的二值化/阈值化场景中，数学形态学开操作去小连通域是一个通用且有效的后处理手段

## 评分
- 新颖性: ⭐⭐⭐⭐ 解决问题的思路清晰独到（从prompt层面根本解决重复），但整体框架仍基于DemoFusion
- 实验充分度: ⭐⭐⭐⭐ 消融实验充分揭示了各模块的贡献，定量+定性结合，但FID/IS指标难以反映重复程度
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，从问题分析到方案设计环环相扣，图示精美易懂
- 实用价值: ⭐⭐⭐⭐ 免训练即插即用，实际应用价值高，但推理速度仍是瓶颈
