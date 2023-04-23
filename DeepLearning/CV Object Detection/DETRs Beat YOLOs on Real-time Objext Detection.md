# 论文信息
- 时间：2023.4.17 提交
- 期刊：CVPR
- 网络/算法名称：RT-DERT
- 意义：实时检测Transformer(RT-DETR)
- 作者：Wenyu Lv, Shangliang Xu, Yian Zhao, Guanzhong Wang, Jinman Wei, Cheng Cui, Yuning Du, Qingqing Dang, Yi Liu
- 实验环境：
- 数据集：



> **前言** 本文首先分析了现代实时目标检测器中NMS对推理速度的影响，并建立了端到端的速度基准。为了避免NMS引起的推理延迟，作者提出了一种实时检测Transformer（RT-DETR），这是第一个实时端到端目标检测器。具体而言，设计了一种高效的混合编码器，通过解耦尺度内交互和跨尺度融合来高效处理多尺度特征，并提出了IoU感知的查询选择，以提高目标查询的初始化。此外，本文提出的检测器支持通过使用不同的解码器层来灵活调整推理速度，而不需要重新训练，这有助于实时目标检测器的实际应用。
> 
> RTDETR-L在COCO val2017上实现了53.0%的AP，在T4 GPU上实现了114 FPS，而RT-DETR-X实现了54.8%的AP和74 FPS，在速度和精度方面都优于相同规模的所有YOLO检测器。

  

## 1、简介

目标检测是一项基本的视觉任务，涉及识别和定位图像中的目标。现代目标检测器有两种典型的体系结构：

- 基于CNN
    
- 基于Transformer
    

在过去的几年里，人们对基于CNN的目标检测器进行了广泛的研究。这些检测器的架构已经从最初的两阶段发展到一阶段，并且出现了两种检测范式，Anchor-Base和Anchor-Free。这些研究在检测速度和准确性方面都取得了重大进展。

基于Transformer的目标检测器（DETR）由于消除了各种手工设计的组件，如非最大值抑制（NMS），自提出以来，受到了学术界的广泛关注。该架构极大地简化了目标检测的流水线，实现了端到端的目标检测。

实时目标检测是一个重要的研究领域，具有广泛的应用，如目标跟踪、视频监控、自动驾驶等。现有的实时检测器通常采用基于CNN的架构，在检测速度和准确性方面实现了合理的权衡。然而，这些实时检测器通常需要NMS进行后处理，这通常难以优化并且不够鲁棒，导致检测器的推理速度延迟。

最近，由于研究人员在加速训练收敛和降低优化难度方面的努力，基于Transformer的检测器取得了显著的性能。然而，DETR的高计算成本问题尚未得到有效解决，这限制了DETR的实际应用，并导致无法充分利用其优势。这意味着，尽管简化了目标检测流水线，但由于模型本身的计算成本高，很难实现实时目标检测。

上述问题自然启发考虑是否可以将DETR扩展到实时场景，充分利用端到端检测器来避免NMS对实时检测器造成的延迟。为了实现上述目标，作者重新思考了DETR，并对其关键组件进行了详细的分析和实验，以减少不必要的计算冗余。

具体而言，作者发现，尽管多尺度特征的引入有利于加速训练收敛和提高性能，但它也会导致编码器中序列长度的显著增加。因此，由于计算成本高，Transformer编码器成为模型的计算瓶颈。为了实现实时目标检测，设计了一种高效的混合编码器来取代原来的Transformer编码器。通过解耦多尺度特征的尺度内交互和尺度间融合，编码器可以有效地处理不同尺度的特征。

此外，先前的工作表明，解码器的目标查询初始化方案对检测性能至关重要。为了进一步提高性能，作者提出了IoU-Aware的查询选择，它通过在训练期间提供IoU约束来向解码器提供更高质量的初始目标查询。

此外，作者提出的检测器支持通过使用不同的解码器层来灵活调整推理速度，而不需要重新训练，这得益于DETR架构中解码器的设计，并有助于实时检测器的实际应用。

本文提出了一种实时检测Transformer（RT-DETR），这是第一个实时基于Transformer的端到端目标检测器。RT-DETR不仅在精度和速度上优于目前最先进的实时检测器，而且不需要后处理，因此检测器的推理速度不会延迟并保持稳定，充分利用了端到端检测流水线的优势。

RT-DETR-L在COCO val2017上实现了53.0%的AP，在NVIDIA Tesla T4 GPU上实现了114 FPS，而RT-DETR-X实现了54.8%的AP和74 FPS，在速度和精度方面都优于相同规模的所有YOLO检测器。因此，RT-DETR成为了一种用于实时目标检测的新的SOTA，如图1所示。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgnvbAOHskO98LgK5ndgAVtBoLTopicdXib25pQF5M5wjuzz6b9caBKEtETvu0e8wl57NrPCMdYgbeVg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

此外，提出的RT-DETR-R50实现了53.1%的AP和108 FPS，而RT-DETR-R101实现了54.3%的AP和74 FPS。其中，RT-DETR50在准确度上优于DINO-Deformable-DETR-R50 2.2%的AP（53.1%的AP对50.9%的AP），在FPS（108 FPS对5 FPS）上优于DINO-Deformable-DETR-R5约21倍。

本文的主要贡献总结如下：

1. 提出了第一个实时端到端目标检测器，它不仅在准确性和速度上优于当前的实时检测器，而且不需要后处理，因此推理速度不延迟，保持稳定；
    
2. 详细分析了NMS对实时检测器的影响，并从后处理的角度得出了关于基于CNN的实时检测器的结论；
    
3. 提出的IoU-Aware查询选择在我们的模型中显示出优异的性能改进，这为改进目标查询的初始化方案提供了新的线索；
    
4. 本文的工作为端到端检测器的实时实现提供了一个可行的解决方案，并且所提出的检测器可以通过使用不同的解码器层来灵活地调整模型大小和推理速度，而不需要重新训练。
    

## 2、相关方法

### 2.1、实时目标检测器

经过多年的不断发展，YOLO系列已成为实时目标检测器的代名词，大致可分为两类：

- Anchor-Base
    
- Anchor-Free
    

从这些检测器的性能来看，Anchor不再是制约YOLO发展的主要因素。然而，上述检测器产生了许多冗余的边界框，需要在后处理阶段使用NMS来过滤掉它们。

不幸的是，这会导致性能瓶颈，NMS的超参数对检测器的准确性和速度有很大影响。作者认为这与实时目标检测器的设计理念不兼容。

### 2.2、端到端目标检测器

端到端目标检测器以其流线型管道而闻名。Carion等人首先提出了基于Transformer的端到端目标检测器，称为DETR（DEtection Transformer）。它因其独特的特点而备受关注。特别地，DETR消除了传统检测流水线中手工设计的Anchor和NMS组件。相反，它采用二分匹配，并直接预测一对一的对象集。通过采用这种策略，DETR简化了检测管道，缓解了NMS带来的性能瓶颈。

尽管DETR具有明显的优势，但它存在两个主要问题：

- 训练收敛缓慢
    
- 查询难以优化
    

已经提出了许多DETR变体来解决这些问题。具体而言，Deformable DETR通过提高注意力机制的效率，加速了多尺度特征的训练收敛。Conditional DETR和Anchor DETR降低了查询的优化难度。DAB-DETR引入4D参考点，并逐层迭代优化预测框。DN-DETR通过引入查询去噪来加速训练收敛。DINO以之前的作品为基础，取得了最先进的成果。

尽管正在不断改进DETR的组件，但本文的目标不仅是进一步提高模型的性能，而且是创建一个实时的端到端目标检测器。

### 2.3、目标检测的多尺度特征

现代目标检测器已经证明了利用多尺度特征来提高性能的重要性，尤其是对于小物体。FPN引入了一种融合相邻尺度特征的特征金字塔网络。随后的工作扩展和增强了这种结构，并被广泛用于实时目标检测器。Zhu等人首先在DETR中引入了多尺度特征，提高了性能和收敛速度，但这也导致了DETR计算成本的显著增加。

尽管Deformable Attention制在一定程度上减轻了计算成本，但多尺度特征的结合仍然会导致较高的计算负担。为了解决这个问题，一些工作试图设计计算高效的DETR。Effificient DETR通过初始化具有密集先验的目标查询来减少编码器和解码器层的数量。Sparse DETR选择性地更新期望被解码器引用的编码器token，从而减少计算开销。Lite DETR通过以交错方式降低低级别特征的更新频率来提高编码器的效率。尽管这些研究降低了DETR的计算成本，但这些工作的目标并不是将DETR作为一种实时检测器来推广。

## 3、检测器端到端速度

### 3.1、分析NMS

NMS是检测中广泛采用的后处理算法，用于消除检测器输出的重叠预测框。NMS中需要2个超参数：得分阈值和IoU阈值。

特别地，分数低于分数阈值的预测框被直接过滤掉，并且每当2个预测框的IoU超过IoU阈值时，分数较低的框将被丢弃。重复执行此过程，直到每个类别的所有框都已处理完毕。因此，NMS的执行时间主要取决于输入预测框的数量和两个超参数。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgnvbAOHskO98LgK5ndgAVtBwbCVNsOImrE1zWk5Crdwgb9SYRiawULD8OjmL1xe8aUC9U8FEkpGFsg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

为了验证这一观点，作者利用YOLOv5和YOLOv8进行实验。首先计算在输出框被相同输入图像的不同得分阈值滤波后剩余的预测框的数量。采样了0.001到0.25的一些分数作为阈值，对两个检测器的剩余预测框进行计数，并将其绘制成直方图，直观地反映了NMS易受其超参数的影响，如图2所示。

此外，以YOLOv8为例，评估了不同NMS超参数下COCO val2017的模型准确性和NMS操作的执行时间。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgnvbAOHskO98LgK5ndgAVtB9PmXiaAyKY7DvREFNHI9nxLvRIetHoWka6iaqJkqr9XmCMDAc6OopQaQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

注意，在实验中采用的NMS后处理操作是指TensorRT efficientNMSPlugin，它涉及多个CUDA内核，包括EfficientNMSFilter、RadixSort、EfficientNMS等，作者只报告了EfficientNMS内核的执行时间。在T4 GPU上测试了速度，上述实验中的输入图像和预处理是一致的。使用的超参数和相应的结果如表1所示。

### 3.2、端到端速度基准

为了能够公平地比较各种实时检测器的端到端推理速度，作者建立了一个端到端速度测试基准。考虑到NMS的执行时间可能会受到输入图像的影响，有必要选择一个基准数据集，并计算多个图像的平均执行时间。该基准采用COCO val2017作为默认数据集，为需要后处理的实时检测器添加了TensorRT的NMS后处理插件。

具体来说，根据基准数据集上相应精度的超参数测试检测器的平均推理时间，不包括IO和内存复制操作。利用该基准测试T4 GPU上基于锚的检测器YOLOv5和YOLOv7以及Anchor-Free检测器PP-YOLOE、YOLOv6和YOLOv8的端到端速度。

测试结果如表2所示。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgnvbAOHskO98LgK5ndgAVtB5TP1eyMlvQjzBibsGC0rCS3ribZ9qLYlgrEoKZWibDh94ia5329iawD96vA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

根据结果得出结论，对于需要NMS后处理的实时检测器，Anchor-Free检测器在同等精度上优于Anchor-Base的检测器，因为前者的后处理时间明显少于后者，这在以前的工作中被忽略了。这种现象的原因是，Anchor-Base的检测器比Anchor-Free的检测器产生更多的预测框（在测试的检测器中是3倍多）。

## 4、The Real-time DETR

### 4.1、方法概览

所提出的RT-DETR由Backbone、混合编码器和带有辅助预测头的Transformer解码器组成。模型体系结构的概述如图3所示。

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

具体来说：

- 首先，利用Backbone的最后3个阶段的输出特征作为编码器的输入；
    
- 然后，混合编码器通过尺度内交互和跨尺度融合将多尺度特征转换为一系列图像特征（如第4.2节所述）；
    
- 随后，采用IoU-Aware查询选择从编码器输出序列中选择固定数量的图像特征，作为解码器的初始目标查询；
    
- 最后，具有辅助预测头的解码器迭代地优化对象查询以生成框和置信度得分。
    

### 4.2、高效混合编码器

#### 1、计算瓶颈分析

为了加速训练收敛并提高性能，Zhu等人建议引入多尺度特征，并提出Deformable Attention机制以减少计算。然而，尽管注意力机制的改进减少了计算开销，但输入序列长度的急剧增加仍然导致编码器成为计算瓶颈，阻碍了DETR的实时实现。

如 所述，编码器占GFLOP的49%，但在Deformable DETR中仅占AP的11%。为了克服这一障碍，作者分析了多尺度Transformer编码器中存在的计算冗余，并设计了一组变体，以证明尺度内和尺度间特征的同时交互在计算上是低效的。

从包含关于图像中的对象的丰富语义信息的低级特征中提取高级特征。直观地说，对连接的多尺度特征进行特征交互是多余的。如图5所示，为了验证这一观点，作者重新思考编码器结构，并设计了一系列具有不同编码器的变体。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgnvbAOHskO98LgK5ndgAVtBk947LqpUuLr1QQrTgIdOHvtwficXfEIibJ9lhEVPDicYGqRxwV4jvaeicQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

该组变体通过将多尺度特征交互解耦为尺度内交互和跨尺度融合的两步操作，逐步提高模型精度，同时显著降低计算成本。首先删除了DINO-R50中的多尺度变换编码器作为基线A。接下来，插入不同形式的编码器，以产生基于基线A的一系列变体，具体如下：

1. A→ B：变体B插入一个单尺度Transformer编码器，该编码器使用一层Transformer Block。每个尺度的特征共享编码器，用于尺度内特征交互，然后连接输出的多尺度特征。
    
2. B→ C：变体C引入了基于B的跨尺度特征融合，并将连接的多尺度特征输入编码器以执行特征交互。
    
3. C→ D：变体D解耦了多尺度特征的尺度内交互和跨尺度融合。首先，使用单尺度Transformer编码器进行尺度内交互，然后使用类PANet结构进行跨尺度融合。
    
4. D→ E：变体E进一步优化了基于D的多尺度特征的尺度内交互和跨尺度融合，采用了设计的高效混合编码器。
    

#### 2、Hybrid design

基于上述分析，作者重新思考了编码器的结构，并提出了一种新的高效混合编码器。如图3所示，所提出的编码器由两个模块组成，即基于注意力的尺度内特征交互（AIFI）模块和基于神经网络的跨尺度特征融合模块（CCFM）。

AIFI进一步减少了基于变体D的计算冗余，变体D仅在上执行尺度内交互。作者认为，将自注意力操作应用于具有更丰富语义概念的高级特征可以捕捉图像中概念实体之间的联系，这有助于后续模块对图像中目标的检测和识别。

同时，由于缺乏语义概念以及与高级特征的交互存在重复和混淆的风险，较低级别特征的尺度内交互是不必要的。为了验证这一观点，只对变体D中的进行了尺度内相互作用，实验结果见表3，见行。与变体D相比，显著降低了延迟（快35%），但提高了准确性（AP高0.4%）。这一结论对实时检测器的设计至关重要。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgnvbAOHskO98LgK5ndgAVtBlZWdytIhzqdZRFw3SzxfYWJllFqOibuGWonNqibEiakmmsSSJWWRMbrdw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

CCFM也基于变体D进行了优化，在融合路径中插入了几个由卷积层组成的融合块。融合块的作用是将相邻的特征融合成一个新的特征，其结构如图4所示。融合块包含N个RepBlock，两个路径输出通过元素相加进行融合。

可以将这个过程表述如下：

![图片](https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgnvbAOHskO98LgK5ndgAVtBUxDHwyN7HyibxLbYPEU5ARSyLbc5ZNuRxr4tW230bOeR5rsxNlCIrxg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

其中表示多头自注意力，表示将特征的形状恢复到与相同的形状，这是的inverse操作。

### 4.3、IoU-Aware查询选择

DETR中的目标查询是一组可学习的嵌入，这些嵌入由解码器优化，并由预测头映射到分类分数和边界框。然而，这些目标查询很难解释和优化，因为它们没有明确的物理意义。后续工作改进了目标查询的初始化，并将其扩展到内容查询和位置查询（Anchor点）。其中，Effificient detr、Dino以及Deformable detr都提出了查询选择方案，它们的共同点是利用分类得分从编码器中选择Top-K个特征来初始化目标查询（或仅位置查询）。然而，由于分类得分和位置置信度的分布不一致，一些预测框具有高分类得分，但不接近GT框，这导致选择了分类得分高、IoU得分低的框，而分类得分低、IoU分数高的框被丢弃。这会削弱探测器的性能。

为了解决这个问题，作者提出了IoU-Aware查询选择，通过约束模型在训练期间为具有高IoU分数的特征产生高分类分数，并为具有低IoU得分的特征产生低分类分数。因此，与模型根据分类得分选择的Top-K个编码器特征相对应的预测框具有高分类得分和高IoU得分。

将检测器的优化目标重新表述如下：

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

其中和表示预测和GT，和。和分别表示类别和边界框。将IoU分数引入分类分支的目标函数（类似于VFL），以实现对正样本分类和定位的一致性约束。

#### 有效性分析

为了分析所提出的IoU感知查询选择的有效性，在val2017上可视化了查询选择所选择的编码器特征的分类分数和IoU分数，如图6所示。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgnvbAOHskO98LgK5ndgAVtBRRF6nl8ID9emwnAAHJx5uTbQVLDysMhCC9ReOYL5VfhzuVnTKIWuibg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

具体来说，首先根据分类得分选择前K个（在实验中K=300）编码器特征，然后可视化分类得分大于0.5的散点图。红点和蓝点是根据分别应用普通查询选择和IoU感知查询选择训练的模型计算的。点越靠近图的右上角，对应特征的质量就越高，即分类标签和边界框更有可能描述图像中的真实对象。

根据可视化结果发现最引人注目的特征是大量蓝色点集中在图的右上角，而红色点集中在右下角。这表明，使用IoU感知查询选择训练的模型可以产生更多高质量的编码器特征。

此外，还定量分析了这两类点的分布特征。图中蓝色点比红色点多138%，即分类得分小于或等于0.5的红色点更多，这可以被视为低质量特征。然后，分析分类得分大于0.5的特征的IoU得分，发现IoU得分大于0.5时，蓝色点比红色点多120%。

定量结果进一步表明，IoU感知查询选择可以为对象查询提供更多具有准确分类（高分类分数）和精确定位（高IoU分数）的编码器特征，从而提高检测器的准确性。

### 4.4、Scaled RT-DETR

为了提供RT-DETR的可扩展版本，将ResNet网替换为HGNetv2。使用depth multiplier和width multiplier将Backbone和混合编码器一起缩放。因此，得到了具有不同数量的参数和FPS的RT-DETR的两个版本。

对于混合编码器，通过分别调整CCFM中RepBlock的数量和编码器的嵌入维度来控制depth multiplier和width multiplier。值得注意的是，提出的不同规模的RT-DETR保持了同质解码器，这有助于使用高精度大型DETR模型对光检测器进行蒸馏。这将是一个可探索的未来方向。

## 5、实验

### 5.1、与SOTA比较

![图片](https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgnvbAOHskO98LgK5ndgAVtBibgJkpUm6QJYcPm6Mh5NyiaGABrunL2icYxJXXXsE5qD4wVafFibYJiawHw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

### 5.2、混合编码器的消融实验研究

![图片](https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgnvbAOHskO98LgK5ndgAVtBfprZsjmLSrGtb0QHq8Kck4AFhxVN8kc3vfNkc8Vjt5BHHI33rDuXjg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

### 5.3、IoU感知查询选择的消融研究

![图片](https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgnvbAOHskO98LgK5ndgAVtBibfgZtFBJibdvpvMdg5UDfjiagw8aoMjY1jIqf3JJFbbsR5cwPxibJKa7w/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

### 5.4、解码器的消融研究

![图片](https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgnvbAOHskO98LgK5ndgAVtBtnobUzHapUiblibTIg3B8v8Vh8SYVFSiafdWqmvlBibRibd3ydaN8gV9ORA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)