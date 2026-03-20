# Single Pixel Image Classification using an Ultrafast Digital Light Projector

**会议**: CVPR 2026  
**arXiv**: [2603.12036](https://arxiv.org/abs/2603.12036)  
**代码**: 无（数据已开源）  
**领域**: 计算成像 / 图像分类  
**关键词**: single pixel imaging, Hadamard patterns, microLED projector, extreme learning machine, compressed sensing  

## 一句话总结
利用microLED-on-CMOS超快光投影器(330kfps)进行单像素成像(SPI)，以12×12 Hadamard pattern照明MNIST数字并用单像素检测器采集时间序列，完全跳过图像重建，直接用ELM/DNN分类实测光信号，实现1.2kfps下>90%分类精度，二分类(异常检测)精度>99%。

## 背景与动机
传统相机在超高速场景和非可见光波段面临带宽限制。单像素成像(SPI)通过结构化照明+单点检测器替代面阵传感器，硬件简单且可工作在任意波段。但SPI的瓶颈在于pattern切换速度：DMD机械翻转极限~10^4 fps，限制了实时应用。最近microLED阵列作为pattern生成器，切换速度比DMD快~100倍。现有SPIC(单像素图像分类)工作多为仿真，本文在真实自由空间光学系统上验证了超快SPIC的可行性。

## 核心问题
能否在真实光学系统中实现kHz级帧率的图像分类，且不需要重建图像？如何在压缩感知(使用fewer patterns)的情况下维持分类精度？pattern选择策略对分类性能有何影响？

## 方法详解

### 整体框架
输入：DMD上显示的二值化MNIST图像 → microLED投影器依次投射288个Hadamard pattern（12×12基底的正/负互补对） → 单像素光电探测器(SiPM)采集每个pattern的叠加光强 → 实时示波器记录时间序列 → ML模型直接对时间序列分类（无需图像重建）。输出：数字类别(0-9)。

### 关键设计
1. **超快microLED光投影器**: 128×128有源矩阵microLED阵列，50μm间距，支持二值模式和5-bit灰度，全局快门模式330kfps切换。将12×12 Hadamard pattern映射到microLED阵列上照明DMD。关键优势是比DMD机械翻转快约30倍，实现亚毫秒级的完整Hadamard集合投射。

2. **Hadamard pattern压缩策略**: Had12共288个pattern（144基础pattern×正负互补对）。发现pattern按"sequency"(空间频率类比)排序后，低sequency(少空间翻转)的前1/4 pattern包含最多有用分类信息。具体分为Cat1(前44个，仅单轴变化)和Cat2(剩余，双向变化)。使用前1/4 pattern仍能维持~78%精度，同时带宽提升4倍达到4.8kHz。

3. **两种轻量分类模型**:
   - **ELM(极限学习机)**: 单隐层，输入权重随机固定不训练，只用岭回归求解输出权重(闭式解)。1000隐层神经元时多分类87.37%，二分类(one-vs-all)各类AUC均接近1.0(>99%)。推理31μs/样本。
   - **DNN**: 3层全连接+ReLU+Softmax，Adam优化，sparse categorical cross-entropy。使用完整Had12达到>90%精度。推理73μs/样本（比ELM慢2倍但精度更高）。

### 损失函数 / 训练策略
- ELM: 岭回归闭式解，α=1.0，无需迭代训练
- DNN: Adam + sparse categorical cross-entropy，300 epochs
- 噪声鲁棒性分析：对输入注入加性高斯白噪声，σ=0.5时精度仍>95%，σ=1.0时显著下降。重要发现：性能下降主因是结构信息缺失(压缩感知)而非等效SNR变化

## 实验关键数据
| 配置 | 精度 | 帧率 |
|------|------|------|
| 二值MNIST + DNN (数值仿真baseline) | 97.50% | - |
| 二值MNIST + ELM (数值仿真baseline) | 93.32% | - |
| 实验Had12完整 + DNN | >90% | 1.2 kHz |
| 实验Had12完整 + ELM(多分类) | 87.37% | 1.2 kHz |
| 实验Had12前1/4 + DNN | ~78% | 4.8 kHz |
| 实验Had12 + ELM(二分类/异常检测) | >99% AUC | 1.2 kHz |

### 消融实验要点
- **Pattern选择策略影响巨大**: 使用Had12的前n个(低sequency) >> 随机选择 >> 后n个(高sequency)，说明低频pattern对分类最重要
- **DNN学习曲线揭示信息瓶颈**: 使用fewer patterns时出现更长的vanishing gradient阶段（而非噪声导致的平滑下降），证明压缩感知下的性能退化本质是结构信息缺失
- **ELM隐层神经元数**: 100→1000精度稳步提升但趋于饱和，训练/测试精度差<1%表明无过拟合

## 亮点 / 我学到了什么
- **不重建直接分类的范式值得关注**: 完全绕过图像重建，将2D空间信息编码为1D时间序列后直接分类，信息保全由Hadamard正交基保证。这种"sensing即computing"的思路在边缘/光计算领域有潜力
- **Pattern的"频率排序"策略简单有效**: 类比Fourier分析，低sequency Hadamard pattern ≈ 低频分量，对粗粒度分类足够，高sequency pattern ≈ 高频细节，对简单任务冗余
- **ELM作为异常检测器极其轻量**: 闭式解训练+31μs推理，二分类AUC>99%，适合超快流水线上的go/no-go判断

## 局限性 / 可改进方向
- **仅在二值化MNIST上验证**: 28×28的二值手写数字是最简单的benchmark，实际机器视觉场景远更复杂。灰度/彩色图像、更大分辨率、自然场景的表现未知
- **12×12 Hadamard限制**: 受FPGA内存深度限制，空间分辨率极低(12×12)，实际应用需更高分辨率的pattern集
- **DMD切换仍是系统瓶颈**: microLED 330kfps但DMD物体切换仅32.5kHz，整体帧率受限于DMD而非光投影器
- **未与event camera对比**: 声称优于event camera但未做直接对比
- **实验setup依赖特定光路**: 自由空间光学系统，工程化部署还需集成化

## 与相关工作的对比
- **vs 传统SPI+分类**: 以往工作[15,16,17]多为仿真或低速硬件，本文首次在超快光学系统上实验验证
- **vs microLED模拟光计算[13,14]**: 这些工作将microLED用于模拟光学神经网络(矩阵-向量乘法)，本文用microLED做pattern投射+电子后处理，路线不同
- **vs event camera**: 都解决高速感知问题，但SPI可工作在任意波段(红外/THz)，event camera局限于可见/近红外

## 与我的研究方向的关联
- 计算成像+AI的交叉领域，与主流CV方向(VLM, 检测, 分割)距离较远
- "信号直接分类不需重建"的思路在压缩感知领域有广泛应用，可能启发视频理解中的token/帧压缩策略

## 评分
- 新颖性: ⭐⭐⭐ 单像素分类概念并非首创，本文核心贡献在硬件系统集成和实验验证
- 实验充分度: ⭐⭐⭐⭐ 多种pattern策略、两种模型、噪声分析、学习曲线分析都很系统
- 写作质量: ⭐⭐⭐⭐ 清晰易读，图表直观，实验设置描述详细
- 对我的价值: ⭐⭐ 有趣但与主流CV方向距离较远，Hadamard压缩策略有一定启发
