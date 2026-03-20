<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**📷 CVPR2026** · 共 **3** 篇

**[MAD-Avatar: Motion-Aware Animatable Gaussian Avatars Deblurring](motionaware_animatable_gaussian_avatars_deblurring.md)**

:   首次实现从模糊视频直接重建清晰可驱动3D高斯人体avatar：提出3D感知的物理模糊形成模型(将模糊分解为子帧SMPL运动+canonical 3DGS)，用B-spline插值+位姿变形网络建模子帧运动，帧间正则化解决运动方向歧义，在合成和真实数据集上大幅超越"2D去模糊+3DGS"两阶段方案(PSNR提升约2.5dB)。

**[POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](polishing_the_sky_widefield_and_highdynamic_range.md)**

:   POLISH++在POLISH框架基础上引入分块训练+拼接策略和arcsinh非线性变换，解决了射电干涉成像中宽视场（万级像素）和高动态范围（$10^4$-$10^6$）两大实际部署难题，在T-RECS仿真数据上大幅超越CLEAN方法的源探测精度，且能超分辨恢复PSF尺度附近的强引力透镜系统，有望将DSA巡天的透镜发现数量提升约10倍。

**[RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](rawdomain_degradation_models_smartphone_sr.md)**

:   通过对不同智能手机传感器进行设备特定的退化标定（模糊PSF和噪声模型），将公开渲染图像逆处理（unprocess）到各手机的RAW域来生成逼真的训练对，训练的RAW-to-RGB SR模型在未见设备上的真实数据上显著优于使用任意退化参数的基线。
