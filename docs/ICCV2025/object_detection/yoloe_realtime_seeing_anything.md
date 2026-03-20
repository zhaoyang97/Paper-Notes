# YOLOE: Real-Time Seeing Anything

**会议**: ICCV 2025  
**arXiv**: [2503.07465](https://arxiv.org/abs/2503.07465)  
**代码**: [https://github.com/THU-MIG/yoloe](https://github.com/THU-MIG/yoloe)  
**领域**: 目标检测 / 实例分割 / 开放词汇  
**关键词**: open-vocabulary detection, YOLO, text/visual/prompt-free, re-parameterization, real-time  

## 一句话总结
提出YOLOE，在YOLO架构中统一支持文本提示、视觉提示和无提示三种开放场景的检测和分割，通过RepRTA（可重参数化区域-文本对齐）、SAVPE（语义激活视觉提示编码器）和LRPC（懒惰区域-提示对比）三个设计实现高效率高性能，以3x更少的训练成本在LVIS上超越YOLO-World v2。

## 背景与动机
YOLO系列虽然高效准确，但被限制在预定义类别中，无法适应开放场景。现有开放集方法（如GLIP、Grounding DINO、DINO-X）虽然支持多种prompt类型，但存在严重的效率问题：文本提示方法需要昂贵的跨模态融合，视觉提示方法依赖额外的视觉编码器或transformer，无提示方法依赖大语言模型生成类别名。目前缺乏一个单一模型能同时高效支持这三种提示方式。

## 核心问题
如何在保持YOLO级别的实时效率和简洁部署性的同时，让模型支持文本提示、视觉提示和无提示三种开放场景的检测与分割？

## 方法详解

### 整体框架
YOLOE基于标准YOLO架构（backbone + PAN + 回归头 + 分割头），将分类头替换为object embedding头，输出与prompt embedding对比得到类别标签。三种prompt分别由RepRTA、SAVPE和LRPC处理，最终都产生归一化的prompt embedding与anchor points的object embedding进行点积对比。

### 关键设计
1. **RepRTA（Re-parameterizable Region-Text Alignment）**：文本提示方案。用CLIP文本编码器预缓存文本嵌入（推理时不需要文本编码器），训练时通过一个轻量的辅助网络（单个SwiGLU FFN块）优化这些嵌入以改善视觉-语义对齐。关键创新在于推理时将辅助网络通过矩阵乘法与object embedding头的最后一层卷积重参数化合并，变回与标准YOLO完全相同的分类头结构，实现零推理开销。

2. **SAVPE（Semantic-Activated Visual Prompt Encoder）**：视觉提示方案。将感兴趣区域表示为mask，通过两个解耦的分支处理：(a) 语义分支从PAN多尺度特征中提取prompt无关的高维语义特征；(b) 激活分支在低维空间中融合visual prompt mask和图像特征，生成分组的prompt感知权重。两个分支通过分组聚合产生最终的prompt embedding，在几乎不增加计算的情况下编码视觉线索。

3. **LRPC（Lazy Region-Prompt Contrast）**：无提示方案。训练一个特殊的prompt embedding来检测"是否存在物体"（不区分类别），然后仅对被识别为物体的anchor points（而非全部8400个）进行类别检索，从内置的4585类大词汇表中匹配最相似的类名。相比用语言模型逐个生成类名（如GenerateU使用250M参数的FlanT5），这种检索方式省去了语言模型依赖，实现53x推理加速。

### 损失函数 / 训练策略
- 分类用BCE loss，回归用IoU + DFL loss，分割用mask BCE loss
- 分段训练：先训练text prompt 30 epochs → 冻结后训练SAVPE 2 epochs → 训练prompt-free专用embedding 1 epoch
- 分割mask由SAM-2.1生成伪标签（用GT bbox作prompt）
- 全局负样本字典策略替代空字符串负样本，提升0.9 AP

## 实验关键数据
| 模型 | Prompt | LVIS AP | APr | 训练时间 | FPS (T4/iPhone12) |
|------|--------|---------|-----|----------|------------|
| YOLO-Worldv2-S | T | 24.4 | 17.1 | 41.7h | 216/49 |
| **YOLOE-v8-S** | T/V | **27.9/26.2** | **22.3/21.3** | **12.0h** | **306/64** |
| YOLO-Worldv2-L | T | 35.5 | 25.6 | 80.0h | 80/22 |
| **YOLOE-v8-L** | T/V | **35.9/34.2** | **33.2/33.2** | **22.5h** | **103/27** |
| T-Rex2 | V | 37.4 | 29.9 | - | - |
| GenerateU (Swin-T) | Free | 26.8 | 20.0 | - | 0.48 |
| **YOLOE-v8-L** | Free | **27.2** | **23.5** | - | **25.3** |

- COCO迁移：YOLOE-v8-L full-tune 80 epochs → 53.0 APb / 42.7 APm，超越从零训300 epochs的YOLOv8-L (52.4/42.3)
- 分割zero-shot LVIS：YOLOE-v8-L达23.5 APm，超越fine-tuned YOLO-Worldv2-L的19.8 APm

### 消融实验要点
- RepRTA贡献最大：+2.3 AP，zero overhead
- 去除跨模态融合降低1.9 AP，但提速1.28x；用更强MobileCLIP补回1.5 AP
- SAVPE vs 简单mask pooling：+1.5 AP
- LRPC实现1.7x加速（v8-S），性能不降

## 亮点
- **Re-parameterization的巧妙应用**：训练时用辅助网络增强对齐，推理时无缝合并为标准YOLO结构，真正的"免费午餐"
- **统一三种prompt在一个模型中**：文本、视觉、无提示三种模式共享同一backbone和heads，只是prompt编码不同
- **训练极其经济**：总共只需33 epochs（30+2+1），8卡4090即可完成，比YOLO-World 3倍更快
- **实用性极强**：支持TensorRT和CoreML部署，手机端实时，对工业应用友好
- **检索代替生成**的无提示范式：避免语言模型依赖，53x加速

## 局限性 / 可改进方向
- 多任务（检测+分割）联合训练导致frequent类APf有0.9的下降
- 视觉提示仅训练2 epochs，冻结其他参数，可能限制visual prompt的上限
- 无提示模式依赖预定义的4585类词汇表，无法发现全新类别
- SAM-2.1生成的伪mask可能有噪声，影响分割精度

## 与相关工作的对比
- **vs. YOLO-World v2**：YOLOE去除了昂贵的跨模态融合，用RepRTA重参数化代替，3x更快训练，1.4x更快推理，+3.5 AP (S模型)
- **vs. DINO-X**：DINO-X也支持多prompt但计算量巨大不可部署在边缘设备，YOLOE保持YOLO级别效率
- **vs. T-Rex2**：YOLOE-v8-L在rare类上+3.3 APr，用更少训练数据和更少资源
- **vs. GenerateU**：同样prompt-free，6.3x更少参数，53x更快推理

## 启发与关联
- Re-parameterization思想可以迁移到其他需要推理时零开销的场景
- 检索式无提示检测范式比生成式更高效，值得在其他开放世界任务中探索
- visual prompt encoder的轻量设计对few-shot检测/分割等场景有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐ RepRTA是重参数化在开放检测中的新颖应用，三种prompt统一框架设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ LVIS/COCO全面评估，三种prompt模式各有详细消融，还有下游迁移实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，roadmap table展示从baseline到最终方案的每步贡献，非常直观
- 价值: ⭐⭐⭐⭐⭐ 实用性极强的开放检测方案，有望成为工业界首选的开放世界YOLO方案
