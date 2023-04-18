# 目标检测网络综述

该综述吸收于B站UP主[霹雳吧啦Wz](https://space.bilibili.com/18161609)([GitHub](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing))和B站UP主[Tsiu-Hinghiok](https://space.bilibili.com/111391089)的内容学习而来

## 目录

- [R-CNN](#r-cnn)
- [Fast R-CNN](#fast-r-cnn)
- [Faster R-CNN](#faster-r-cnn)
- [FPN](#fpn)
- [SSD](#ssd)
- [Mask R-CNN](#mask-r-cnn)
- [RetinaNet](#retinanet)
- [YOLOv1](#yolov1)
- [YOLOv2](#yolov2)
- [YOLOv3](#yolov3)
- [FCOS](#fcos)
- [YOLOv3 SPP](#yolov3-spp)
- [YOLOv4](#yolov4)
- [YOLOv5](#yolov5)
- [YOLOX](#yolox)
- [YOLOR](#yolor)


# R-CNN
- [ppt](../ppt/pytorch_object_detection/R-CNN.pdf)
## 1. 优势/历史地位
- R-CNN可以说是利用深度学习进行目标检测的开山之作。作者Ross Girshick多次在PASCAL VOC的目标检测竞赛中折桂，曾在2010年带领团队获得终身成就奖。

## 2. 算法流程
- 一共分为4个步骤
    1. 一张图像生成1K~2K个候选区域(使用Selective Search方法) 

    1. 对每个候选区域，使用深度网络提取特征

    1. 特征送入每一类的SVM 分类器，判别是否属于该类

    1. 使用回归器精细修正候选框位置
        ![R-CNN1.png](../pictures/R-CNN1.png)

- 框架
    ![R-CNN Construct.png](../pictures/R-CNN%20Construct.png)

## 3. 缺点
1. 测试速度慢：
    - 测试一张图片约53s(CPU)。用Selective Search算法提取候选框用时约2秒，一张图像内候选框之间存在大量重叠，提取特征操作冗余。

2. 训练速度慢：
    - 过程及其繁琐

3. 训练所需空间大：
    - 对于SVM和bbox回归训练，需要从每个图像中的每个目标候选框提取特征，并写入磁盘。对于非常深的网络，如VGG16，从VOC07训练集上的5k图像上提取的特征需要数百GB的存储空间。


# Fast R-CNN
- [ppt](../ppt/pytorch_object_detection/Fast_R-CNN.pdf)
## 1. 优势/历史地位
- Fast R-CNN是作者Ross Girshick继R-CNN后的又一力作。同样使用VGG16作为网络的backbone，与R-CNN相比训练时间快9倍，测试推理时间快213倍，准确率从62%提升至66%(再Pascal VOC数据集上)。

- R-CNN是将每个锚框都扔进CNN中，这里进行了多次重复计算；而Fast R-CNN直接将图像扔入CNN中，进行特征提取，再将锚框映射到特征图上

- R-CNN的CNN和SVM需要单独训练，Fast R-CNN让CNN和下游分类任务一块训练，可以进行统一的梯度下降

## 2. 算法流程
- 一共分为3个步骤
    1. 一张图像生成1K~2K个候选区域(使用Selective Search方法)

    1. 将图像输入网络得到相应的特征图，将SS算法生成的候选框投影到特征图上获得相应的特征矩阵

    1. 将每个特征矩阵通过ROI pooling层缩放到7x7大小的特征图，接着将特征图展平通过一系列全连接层得到预测结果
        ![Fast R-CNN1.png](../pictures/Fast%20R-CNN1.png)

- 框架
    ![Fast R-CNN Construct.png](../pictures/Fast%20R-CNN%20Construct.png)

## 3. 缺点
1. 测试速度没有非常优秀：
    - 除去SS算法生成锚框的时间，测试一张图片约2s(CPU)；涵盖锚框生成时间，用时2s左右。


# Faster R-CNN
- [ppt](../ppt/pytorch_object_detection/Faster_R-CNN.pdf)
## 1. 优势/历史地位
- Faster R-CNN是作者Ross Girshick继Fast R-CNN后的又一力作。同样使用VGG16作为网络的backbone，推理速度在GPU上达到5fps(包括候选区域的生成)，准确率也有进一步的提升。在2015年的ILSVRC以及COCO竞赛中获得多个项目的第一名。

- 核心是RPN网络，实现了生成锚框、特征提取和分类网络的同一训练，替代了SS算法。可以认为Faster R-CNN就是RPN+除去SS的Fast R-CNN

- 将Fast R-CNN检测图片的速度从除去SS算法生成锚框的时间，检测图片需要0点几秒，实现了整个网络检测图片只需要0点几秒


## 2. 算法流程
- 一共分为3个步骤
    1. 将图像输入网络得到相应的特征

    1. 使用RPN结构生成候选框，将RPN生成的候选框投影到特征图上获得相应的特征矩阵

    1. 将每个特征矩阵通过ROI pooling层缩放到7x7大小的特征图，接着将特征图展平通过一系列全连接层得到预测结果
        ![Faster R-CNN2.png](../pictures/Faster%20R-CNN2.png)

- 框架
    ![Faster R-CNN Construct.png](../pictures/Faster%20R-CNN%20Construct.png)

- 注意：
    1. 锚框(anchor)是在特征图上生成的；候选框(bounding box)是在原图上生成的。
    2. 提议框(proposal)是锚框(anchor)进行学习、偏移后得到的

## 3. 缺点
1. 仍然是two-stage的：
    - two-stage还是太慢了
    - RPN一次预测，Fast R-CNN还有一次预测

1. 对小目标检测效果很差
    - 因为是对一个尺度的特征层进行检测

# FPN
- [ppt](../ppt/pytorch_object_detection/fpn.pdf)
## 1. 优势/历史地位
![FPN1.png](../pictures/FPN1.png)

## 2. 算法流程
1. 特征图像处理类型
    ![FPN2.png](../pictures/FPN2.png)

    - 生成不同尺度的特征图像
        - (a)的效率很低
        - (b)是标准的Faster R-CNN的流程，对小目标预测效果不好
        - (c)是SSD的类型
        - (d)FPN结构

1. 融合过程
    ![FPN3.png](../pictures/FPN3.png)

    - 下采样都是2的整数倍
    - 1x1卷积核的目的就是调整不同特征图的channel
        - 通常越小的特征图的channel越多
        - 原论文中1x1的卷积核的个数为256，即最终得到的特征图的channel都等于256
    - 上采样也是2倍
    - 算法就是nearest neighbour upsampling

1. 网络结构
    ![FPN4.png](../pictures/FPN4.png)

    - 不同尺寸的预测特征层预测不同大小的proposal
    - 通常尺寸大的特征图检测小目标(尺寸小的anchors)，尺寸小的特征图检测大目标(尺寸大的anchors)

1. RPN的proposal映射到不同的预测特征层
    ![FPN5.png](../pictures/FPN5.png)
    
    - k：2，3，4，5(对应P2 P3 P4 P5)
    - k0 = 4
    - w, h：RPN预测得到的proposal在原图上的宽度和高度

    - 映射方法Level Mapper
        ![FPN6.png](../pictures/FPN6.png)

## 3. 缺点


# SSD
- [ppt](../ppt/pytorch_object_detection/SSD.pdf)
## 1. 优势/历史地位
- 借助VGG的框架在不同特征尺度上预测不同尺度的目标
    
- one-stage，但是还没完全超过two-stage
    

## 2. 算法流程
- VGG+多尺度预测
    ![SSD2.png](../pictures/SSD2.png)

- $(c+4)\times k$ 卷积核需要参与预测
    - c个类别
    - 4个边界框回归参数
    - k个边界框
    ![SSD1.png](../pictures/SSD1.png)

    - 注意：
        - Faster R-CNN中是预测4c个，是因为对每个类别预测4个边界框回归参数
        - SSD对每个Default box只生成4个边界框参数，不关注属于哪个类别的

- Ground Truth：IOU匹配
    - 负样本算highest confidence loss
    - 负样本是正样本三倍（Hard Negative Mining）


## 3. 缺点

# Mask R-CNN
- [ppt](../ppt/pytorch_object_detection/mask_r-cnn.pdf)

![Mask R-CNN7.png](../pictures/Mask%20R-CNN7.png)

- 图像分类：返回类别及概率
- 目标检测：返回类别和检测框
- 语义分割：返回分割后的物体(像素分类)
- 实例分割：语义分割只能分割出类别，但是实例分割可以将同一类别的不同物体进行分割


## 1. 优势/历史地位
![Mask R-CNN8.png](../pictures/Mask%20R-CNN8.png)

## 2. 算法流程
1. Mask R-CNN & Faster R-CNN
    ![Mask R-CNN9.png](../pictures/Mask%20R-CNN9.png)

    - Faster R-CNN源码也是RoIAlign，而不是RoIpooling

    - Mask 分支如下：
        ![Mask R-CNN10.png](../pictures/Mask%20R-CNN10.png)

1. RoIAlign
    - 为什么将RoIpooling替换成RoIAlign
        - 因为RoIpooling涉及两次取整操作，两次取整肯定会导致定位的偏差
        - 论文称这个现象为misalignment
        ![Mask R-CNN11.png](../pictures/Mask%20R-CNN11.png)

        - 用RoIAlign定位更加准确

    - RoIpooling
        ![Mask R-CNN12.png](../pictures/Mask%20R-CNN12.png)

        - 第一次取整：特征图相对于原图的步距，需要做除法
        - 第二次取整：特征图可能不会被均分

    - RoIAlign
        ![Mask R-CNN13.png](../pictures/Mask%20R-CNN13.png)
        
        - 第一步不取整
        - 第二部均分，依靠sampling ratio设置每个均分区域有 $(sampling ratio)^{2}$ 个采样点
            - 当采用多个采样点的时候，每个子区域的输出取所有采样点的均值

        ![Mask R-CNN14.png](../pictures/Mask%20R-CNN14.png)

        - 通过双线性插值去计算采样点的数值
        - 0.3125是大矩形框的左上角点坐标
        - 3.875是大矩形框的右下角点坐标
        - u是橙色点距离左侧最近黑边的距离
        - v是橙色点距离右侧最近黑边的距离

        ![Mask R-CNN15.png](../pictures/Mask%20R-CNN15.png)

        ![Mask R-CNN16.png](../pictures/Mask%20R-CNN16.png)

        ![Mask R-CNN17.png](../pictures/Mask%20R-CNN17.png)

        ![Mask R-CNN18.png](../pictures/Mask%20R-CNN18.png)

1. Mask分支(FCN)
    ![Mask R-CNN19.png](../pictures/Mask%20R-CNN19.png)

    - 预测器的RoIAlign和mask的RoIAlign是不一样的，不共用

    ![Mask R-CNN20.png](../pictures/Mask%20R-CNN20.png)

    ![Mask R-CNN21.png](../pictures/Mask%20R-CNN21.png)

    - 之前讲过的FCN对每个像素，每个类别都会去预测一个概率分数。对每个像素沿channel方向做一个softmax处理。那么通过softmax处理就知道每个像素，归属每个类别的分数
    - 这里在每个Mask分支都会预测一个蒙版。但是，我们不会针对每一个数据沿channel方向做softmax方向处理，而是根据Fast R-CNN分支，预测该目标的类别信息，这样类别与类别之间不存在竞争关系

    ![Mask R-CNN22.png](../pictures/Mask%20R-CNN22.png)

    - RPN提供的边界框很多，相当于提供给Mask分支很多训练样本，且都是于GT有很大交集的
    - 类似随机裁剪数据增强的效果
    - 但是最终预测就采用Fast R-CNN的结果，是为了得到最准确的效果
        - 因为Fast R-CNN里面还有nms，可以进一步滤除

1. Mask R-CNN损失
    ![Mask R-CNN23.png](../pictures/Mask%20R-CNN23.png)

1. Mask分支损失
    ![Mask R-CNN24.png](../pictures/Mask%20R-CNN24.png)

    - 在RPN筛选的时候，知道输入的GT label，然后在计算损失的时候，把对应label的Mask拿出来
    - 虽然不做softmax处理了，但是还是对mask做了sigmoid，让预测值的范围在0~1之间
    - 由于RPN还知道proposal的具体位置，就可以让gt在原图上裁剪对应位置，缩放到和mask一样的大小，用来计算loss
    - 在gt mask中，对应目标区域的数值等于1，对应背景区域的数值是等于0的

1. Mask分支预测使用
    ![Mask R-CNN25.png](../pictures/Mask%20R-CNN25.png)


## 3. 缺点


# YOLOv1
- [ppt](../ppt/pytorch_object_detection/yolov1.pdf)
## 1. 优势/历史地位
- 比Faster R-CNN快，45FPS

## 2. 算法流程
- B个bounding box(默认是两个)，同时附带一个confidence，以及类别。总共就是 $[4(位置参数)+1(confidence)]*B+20(类别数)$ 
- 坐标x,y是相对预测目标的cell的坐标，范围在0~1之间；w,h是相对整张图片的大小，范围也是0~1之间
- confidence：预测目标与真实目标的交并比
- bounding box的损失计算的时候开根号了。因为大目标和小目标偏移同样的距离，大目标的IOU受影响明显小于小目标

## 3. 缺点
- 对群体小目标检测效果很差。因为YOLOv1对bounding box进行预测，且每个cell只关注两个框，还都是预测同一个类别
- 目标出现新的尺寸，效果也会很差(scale能力不强)
- 出错的原因大概率都是因为定位不准
- 比Faster R-CNN准确率低，只有63.4mAP
- 比同年的SSD的输入图像尺寸、速度和精度都低

# YOLOv2
- [ppt](../ppt/pytorch_object_detection/yolov2.pdf)
## 1. 优势/历史地位
- 回归了R-CNN和SSD的anchor检测方式
- mAP和FPS都非常高

## 2. 算法流程
- 做了7种的尝试：
    - Batch Normalization
        - 有帮助训练收敛
        - 正则化操作
        - 替代drop out
        - 可以提升两个百分点
    - High Resolution Classifier
        - 此前都是224x224，作者使用448x448
        - 可以提升4个百分点
    - Convolutional With Anchor Boxes
        - 简化边界框预测
        - mAP下降0.3%，但是召回率增加了7%
    - Dimension Clusters
        - 此前没说box的尺寸是怎么选取的，用k-means生成priors
        - 好的priors对网络训练分厂有帮助
    - Direct location prediction
        - 直接基于anchor训练，模型会不稳定，特别是训练的前期。大部分不稳定因素都来源于预测anchor的中心坐标。由于公式并没有限制 $t_ {x}和t_ {y}$ 的值，中心坐标加上回归参数可能出现在图像的任意位置。用sigmoid函数对 $t_ {x}和t_ {y}$ 加以限制在[0,1]之间，保证预测框的稳定
    - Fine-Grained Features
        - 在预测特征图上，结合一些更底层的信息。底层信息包含更多的图像细节，图像细节在检测小目标的时候用处很大。高、低层细节融合，进行细节补充(passthrough layer)
    - Multi-Scale Training
        - 为了提高YOLOv2的鲁棒性，作者将图像缩放到不同尺度。每迭代10个batch，就将网络的缩放因子进行随机改变(32的整数倍)

- backbone：Darknet-19
    - 预测数量：$[4(位置参数)+1(confidence)+20(类别数)]*5(一共5个这样的box)$ 
## 3. 缺点


# RetinaNet
- [ppt](../ppt/pytorch_object_detection/retinanet.pdf)
## 1. 优势/历史地位
- one-stage网络首次超越two-stage
- 论文名称：Focal Loss for Dense Object Detection
- RetinaNet的对比效果
![RetinaNet2.png](../pictures/RetinaNet2.png)

- 很明显，RetinaNet的效果远远好于two-stage和现有的one-stage

## 2. 算法流程
![RetinaNet1.png](../pictures/RetinaNet1.png)

1. 采用FPN结构
    ![RetinaNet3.png](../pictures/RetinaNet3.png)
    
    - 注意：在原论文中P6是在C5的基础上生成的，这里是根据pytorch官方提供的实现方式绘制的
    - 而且，FPN会在C2位置生成P2，但是RetinaNet没有，原论文说，P2会占用更多的计算资源
    - 使用了3个scale，3个ratio，共9组anchor template

1. 预测器部分
    ![RetinaNet4.png](../pictures/RetinaNet4.png)

    - 之前的FPN和Faster R-CNN是类似的，是two-stage网络
        - 首先会根据RPN生成proposal
        - 再通过Fast R-CNN生成最终的预测参数

    - RetinaNet是one-stage网络，所以直接使用了一个 **权值共享** 的预测头，就是不同特征层的权值都共享
    - 预测器有两个分支：
        - 类别分支，但是不包含背景类别
        - 目标框参数

1. 正负样本
    ![RetinaNet5.png](../pictures/RetinaNet5.png)

    1. $IoU >= 0.5$ , 正样本
    2. $IoU < 0.4$ , 负样本
    3. $IoU \in [0.4, 0.5)$ , 舍弃

1. 损失计算
    ![RetinaNet6.png](../pictures/RetinaNet6.png)

    - 核心是Focal Loss
    - 详细内容在YOLOv3 SPP中讲解了

- 其余内容与R-CNN差不多，熟练掌握Faster R-CNN代码
## 3. 缺点

# YOLOv3
- [ppt](../ppt/pytorch_object_detection/yolov3.pdf)
## 1. 优势/历史地位
- 内容很少，主要是整合了当前主流网络的优势
- YOLOv3的速度是非常快的，但是mAP其实不是特别出色，没有RetinaNet那么好
    - 在COCO AP IOU=0.5位置上精度还是不错的
## 2. 算法流程
- 修改了backbone
    - YOLOv2使用了DarkNet-19，YOLOv3使用了DarkNet-53
    ![YOLOv3 comparison backbone.png](../pictures/YOLOv3%20comparison%20backbone.png)
    - DarkNet-53没有pooling层，全部用卷积层下采样
    - 速度快的原因是DarkNet-53的通道数比ResNet少，意味着卷积核个数少
- Conv + BN + LeakyReLU
- 每一个方框都是一个残差结构
- YOLOv3结构在论文中非常模糊
1. 输入图像416x416
2. 3个预测特征层
    - 每个特征层上是3个尺度
    - $N \times N \times [3 \times (4+1+80)]$
        - N是预测特征层的尺寸
        - 每个cell预测3种尺度
        - 4个box偏移参数
        - 1个confidence score
        - COCO上80个类别
3. 尺寸较小的特征图会进行上采样，并与尺寸大一倍的特征图在通道数的维度上进行拼接
    - 每个尺度的特征图都是通过最后一个1x1的卷积层进行预测
4. anchor机制
![YOLOv23.png](../pictures/YOLOv23.png)
- 与faster R-CNN和SSD不一样的地方：
    - 前者是目标中心点的参数相对于cell的中心点
    - YOLOv2，v3是相对于cell的左上角点
    - $\sigma(x) = Sigmoid(x)$
5. 正负样本匹配
    - 每个GT都会分配一个bounding box prior
    - 寻找于gt重合度最高的bounding box代替gt
        - 如果这个bounding box重合度不是最高的，但是又大于我们设定的阈值，会被丢弃
        - 剩下的就被认为是负样本
        - 对于没有被分配到gt的bounding box是没有位置和类别损失的，只有confidence(objectness)
    - 按照上述原论文的说明，会导致正样本非常的少，因此找一个优化方案：
        - gt于anchor模板左上角重合去做IOU
        - 设置一个IOU阈值，认为大于阈值的都是正样本
        - GT中心在哪一个cell里面，这个cell里面对应IOU超过阈值的anchor template就是正样本
6. 损失计算(论文没给，UP自己写的)
    - 置信度损失
        - logistic regression：实际一般都是二值交叉熵损失(Binary Cross Entropy)
    - 分类损失
        - 二值交叉熵损失(Binary Cross Entropy)
        - 问题：每个预测值是用Sigmoid处理的，是互不干扰的，会导致类别的概率之和不为1
    - 定位损失(了解即可，后面的网络就改了)
        - sum of squared error loss


## 3. 缺点
- 检测精度没完全赶上R-CNN系列


# FCOS
- [ppt](../ppt/pytorch_object_detection/FCOS.pdf)
## 1. 优势/历史地位
- Anchor Free
    - 之前的网络都是anchor based
    - 基于生成好的anchor，去预测它的偏移和倍率系数
    - 预测l, r, t, b
    ![FCOS anchor free.png](../pictures/FCOS%20anchor%20free.png)

- One-Stage
- FCN-based

## 2. 算法流程
1. 前言
![FCOS1.png](../pictures/FCOS1.png)

1. FCOS网络结构
    ![FCOS2.png](../pictures/FCOS2.png)

    - 20年的版本是把Center-ness和Regression放在了一个分支

    ![FCOS3.png](../pictures/FCOS3.png)

    ![FCOS4.png](../pictures/FCOS4.png)

    - 5个预测特征层共用同一个Head
    - 注意Regression部分，正常应该预测 $4 \times num_anchors$ 组参数，但是由于anchor free，不依赖anchors的尺寸，所以只预测4个

    ![FCOS5.png](../pictures/FCOS5.png)

    - centerness是反映了当前预测点，对于目标中心的远近程度
    - 热度图中，蓝色代表数值0，红色代表数值为1
    - 加上了centerness分支有助于提高mAP

1. 正负样本的匹配
    ![FCOS6.png](../pictures/FCOS6.png)

    - 之前的都是通过GT和Anchor box做IoU找正样本，并设定阈值
    - 现在是anhor free，没有anchors，就没办法使用之前的方法
    - 19年说的是，只要预测点落入GT box中，就是正样本
    - 20年认为，落入sub-box才叫正样本
        - $c_ {x}$就是中心点坐标
        - s是特征图相对于原图的步距
        - r是超参数

    ![FCOS7.png](../pictures/FCOS7.png)

    - 同时落入多个相交区域怎么办？
        - 默认分配给面积最小的GT Box
        - 但是这并不是一个很好的解决方法，通过引入FPN结构处理


1. 损失计算
    ![FCOS8.png](../pictures/FCOS8.png)

1. Ambiguity问题
    ![FCOS9.png](../pictures/FCOS9.png)

    - 尺寸更大的特征图，适合预测小目标
    - 尺度更小的特征图，适合预测大目标

1. Assigning objects to FPN
    ![FCOS10.png](../pictures/FCOS10.png)

    - $l^{*}, t^{*}, r^{*}, b^{*}$ 是相对于于预测中心点，到GT box边界的左侧距离，上侧距离，右侧距离和下面距离
    - $m_ {i}$ 是作者预先给的一套阈值
        - 例如：在P3特征图上 $m_ {2} < 3 < m_ {3}$ 视为正样本

## 3. 缺点


# YOLOv3 SPP
- [ppt](../ppt/pytorch_object_detection/yolov3spp.pdf)
## 1. 优势/历史地位
![YOLOv3 SPP compare.png](../pictures/YOLOv3%20SPP%20compare.png)

- u版代码，作者做了非常多的trick
- Focal loss作者说效果不好，没启用
## 2. 算法流程
1. Mosacic图像增强
    - 实现方法
        - 将多张图片拼接在一起，看作一张图像输入网络
    - 优势：
        1. 增加数据的多样性
        1. 增加目标个数
        1. BN能一次性统计多张图片的参数

1. SPP模块
![YOLOv3 SPP.png](../pictures/YOLOv3%20SPP.png)
    - 注意：
        - 这里的SPP和SPPnet中的SPP(Spatial Pyramid Pooling)结构不一样
        - 只是借鉴
    - 在第一个预测特征图之前，拆开了Convolution Set，加入了SPP模块
    - 在多个通路都添加SPP的模块，只有在大图像的情况下效果才比较好，mAP会增加，但是速度变慢

1. CIoU Loss
    - IoU Loss -> GIoU Loss -> DIoU Loss -> CIoU Loss
    1. IoU Loss
        - IoU不同的情况下，L2损失可能相同
        - 优点：
            1. 能够更好的反应重合成都
            1. 具有尺度不变性
        - 缺点：
            1. 当不相交时loss为0
            1. 收敛慢
            1. 回归的不够准确

    1. GIoU Loss
        - 优点：
            1. IoU为0，也有损失

        - 缺点：
            1. 两个框尺寸相同，且x或y方向重合，那么会退化成IoU
            1. 收敛慢
            1. 回归的不够准确
    1. DIoU Loss
        - 优点：
            1. 更快的收敛
            1. 更高的定位精度
            1. 相较于前两者，最大的特色是：
                - 能反映出gt和bbox的位置关系带来的损失差距
        - 缺点：
            1. 考虑的信息还是不够周全

    1. CIoU Loss
        - 优点：
            - 考虑全面
                - 重叠面积(IoU)
                - 中心点距离(DIoU)
                - 长宽比(全新的内容)

1. Focal Loss
    - 针对One-Stage目标检测模型，都会面临一个Class Imbalance问题，就是正负样本不平衡
        - two-stage没有这么严重是因为，在第二阶段之前，第一阶段会把检测框的数量压得很小，就不会出现one-stage几十个正样本面对上万个负样本的情况

    - 正样本高权重少数量，也填充不了负样本低权重大数量的情况。这种现象叫degenerate models
        - 此前通过hard negative mining是实现了筛选对训练有帮助的，大损失的负样本，但是不如Focal Loss

    - 比正常的Cross Entropy Loss引入 $\alpha$ 和 $\gamma$ 两个超参数
        - $\alpha$用于平衡正负样本的权重，但是不能区分那些是容易的样例，那些是复杂的样例
        - $\gamma$所在的损失负责降低简单样本的损失权重，这样我们就可以聚焦于训练复杂的负样本
    - 但是实验证明，敏感的Focal Loss同样容易受到噪音的干扰，因此数据集尽量不要出现标注错误的现象

## 3. 缺点
- 检测精度没完全赶上R-CNN系列



# YOLOv4
- [ppt](../ppt/pytorch_object_detection/yolov4.pdf)
## 1. 优势/历史地位
- 不是原作者的工作了
- mAP提升10%
- FPS提升12%
- 和YOLOv3提升并不大

## 2. 算法流程
1. 网络结构
![YOLOv4 structure drawio.png](../pictures/YOLOv4%20structure%20drawio.png)

- Backbone: CSPDarknet53
- Neck: SPP, PAN
- Head:YOLOv3

    1. CSPDarkNet53
        - Strengthening learning ability of a CNN
        - Removing computational bottlenecks
        - Reducing memory costs

        ![YOLOv4 CSPDenseNet structure.png](../pictures/YOLOv4%20CSPDenseNet%20structure.png)

        ![YOLOv4 CSPDenseNet structure by code.png](../pictures/YOLOv4%20CSPDenseNet%20structure%20by%20code.png)

        ![YOLOv4 CSPDenseNet structure all.png](../pictures/YOLOv4%20CSPDenseNet%20structure%20all.png)

    1. SPP
        ![YOLOv4 CSPDenseNet SPP.png](../pictures/YOLOv4%20CSPDenseNet%20SPP.png)

    1. PAN(Path Aggregation Network)
        - PAN其实就是在Upsampling之后，再做一个downsampling
        - 对之前的PAN做的更改是
            - 把融合过程的addition变成了concatenation
        ![YOLOv4 CSPDenseNet PAN.png](../pictures/YOLOv4%20CSPDenseNet%20PAN.png)


2. 优化策略
- Eliminate grid sensitivity
- Mosaic data augmentation
- IoU threshold(match positive sample)
- Optimizered Anchors
- CIoU

    1. Eliminate grid sensitivity
        ![YOLOv4 Eliminate grid sensitivity1.png](../pictures/YOLOv4%20Eliminate%20grid%20sensitivity1.png)

        ![YOLOv4 Eliminate grid sensitivity2.png](../pictures/YOLOv4%20Eliminate%20grid%20sensitivity2.png)

        - Sigmoid的问题：当gt box的中心点坐标在grid cell的左上角点时，需要预测的参数 $t_ {x}, t_ {y}$ 都是0，但是Sigmoid要在负无穷的情况才能趋于0，这恶鬼条件非常难以达到
        - 解决方法：引入缩放因子

    1. Mosaic data augmentation
        ![YOLOv4 Mosaic data augmentation.png](../pictures/YOLOv4%20Mosaic%20data%20augmentation.png)

    1. IoU threshold(match positive sample)
        ![YOLOv4 IoU threshold1.png](../pictures/YOLOv4%20IoU%20threshold1.png)

        ![YOLOv4 IoU threshold2.png](../pictures/YOLOv4%20IoU%20threshold2.png)

        ![YOLOv4 IoU threshold3.png](../pictures/YOLOv4%20IoU%20threshold3.png)

        - 原作者的意思是，首先取大于阈值的anchor模板，且只取最大的
            - 但是这样的话，正样本数量就太少了
        - 所以现在普遍的做法是，取所有大于阈值的anchor模板，都当成正样本，并匹配上对应的gt
        - 相当于从gt和模板的一对一映射，变成了gt对anchor是一对多
        - YOLOv4取阈值算IoU的步骤都一样，但是取正样本的时候，领域的grid cell对应的AT(anchor template)也被认为是正样本
            - 只会取上下左右四个方向的grid cell，不取左上、左下、右上、右下四个方向

    1. Optimizered Anchors
        ![YOLOv4 Optimizered Anchors.png](../pictures/YOLOv4%20Optimizered%20Anchors.png)

        - YOLOv3 的尺寸是通过聚类得到的
        - YOLOv4针对512x512优化了一下
        - 但是YOLOv5用的还是YOLOv3的anchor尺寸

    1. CIoU
        ![YOLOv4 CIoU.png](../pictures/YOLOv4%20CIoU.png)

        - 和YOLOv3 SPP一样
    
## 3. 缺点


# YOLOv5
- [ppt](../ppt/pytorch_object_detection/yolov5.pdf)

## 1. 优势/历史地位
![YOLOv5 ability.png](../pictures/YOLOv5%20ability.png)

- 距离YOLOv4出来很近
- 迭代版本很多
- YOLOv5根据大小升序分为(n, s, m, l, x)，图像尺寸640x640，最大下采样32倍，预测特征层3层
    - (n6, s6, m6, l6, x6)，图像尺寸1280x1280，下采样率64倍，预测特征层有4层

## 2. 算法流程
- 绘制的是l大小模型的图
![YOLOv5 directory.png](../pictures/YOLOv5%20directory.png)


1. 网络结构
    - Backbone：New CSP Darknet53
    - Neck：SPPF，New CSP-PAN
    - Head：YOLOv Head

    - 补充：
        - 将6.1之前的Focus模块替换成了6行的普通卷积层。两者功能相同，但后者效率更高
        ![YOLOv5 focus.png](../pictures/YOLOv5%20focus.png)

        - SPP -> SPPF：结果等价，效率更高，UP的实验说快了两倍左右
        ![YOLOv5 SPPF.png](../pictures/YOLOv5%20SPPF.png)

2. 数据增强
    1. Mosaic
    2. Copy Paste
        - 注意：必须有对应实例的标签，不然没法启用
        - 本质就是将不同实例拼接到不同图像上，相当于机器PS
    3. Random affine
        - 随机仿射变换
        - 随机旋转、平移、缩放、错切
    4. MixUP
        - 两张图片按透明程度混合成一张新的图片

    5. Albumentations(一个包)
        - 滤波、直方图均衡化以及改变图片质量等等

    6. Augment HSV(Hue, Saturation, Value)
        - 调色度、饱和度和明度

    7. Random horizontal flip
        - 按一定比例，随机的水平翻转


3. 训练策略(策略很多，UP只罗列了一部分)
    1. Multi-scale training(0.5~1.5x)
        - 一定都是32的整数倍

    1. Auto Anchor(For training custom data)
        - 根据自定义数据集，重新自动聚类生成新的anchor template的大小

    1. Warmup and Cosine LR scheduler
        - 训练初期，让学习率从一个很小的值，慢慢增长到我们设定的值
        - cosine的形式慢慢降低学习率

    1. EMA(Exponential Moving Average)
        - 学习变量加了一个动量，让训练更加平滑

    1. Mixed precision(含有TensorCores的GPU才支持混合精度训练)
        - 混合精度训练
        - 减少GPU显存占用
        - 加速网络训练

    1. Evolve hyper-parameters(炼丹)
        - 建议使用默认


4. 损失计算
    1. Classes loss，分类损失，采用的是BCE loss，注意只计算正样本的分类损失。
    2. Objectness loss，obj损失，采用的依然是BCE loss，注意这里的obj指的是网络预测的目标边界框与GT Box的CIoU。这里计算的是所有样本的obj损失。
        - 很多人之前实现YOLOv3和YOLOv4，把obj设置为，有目标为1，无目标为0
        - 但是在YOLOv5中，obj是CIoU
    3. Location loss，定位损失，采用的是CIoU loss，注意只计算正样本的定位损失。
    4. 平衡不同尺度损失
        - 针对三个预测特征层（P3, P4, P5）上的obj损失采用不同的权重

5. 消除Grid敏感度
    - 和YOLOv4的[算法流程 -> 优化策略 -> Eliminate grid sensitivity]差不多
    - 将计算公式革新了
        ![YOLOv5 GRID.png](../pictures/YOLOv5%20GRID.png)
        - 指数不受限，很容易出现指数爆炸的情况

6. 匹配正样本
    - 计算gt和at的长宽比值 -> 计算比例差异，越接近于1，差异越小 -> 找到宽度/高度差异最大的比值
    - 差异小于阈值则匹配成功
        ![YOLOv5 find at.png](../pictures/YOLOv5%20find%20at.png)

    - 和YOLOv4一样去扩充正样本

## 3. 缺点



# YOLOX
- [ppt](../ppt/pytorch_object_detection/YOLOX.pdf)

## 1. 优势/历史地位
- 借鉴于FCOS
- 与之前的网络最大的区别就是Anchor-Free
- 解耦检测头：decoupled detection head
- 更先进的正负样本匹配：advanced label assigning strategy(SimOTA)
- 获得了Streaming Perception Challenge的第一名
## 2. 算法流程
- 整体论文结构
![YOLOX introduction.png](../pictures/YOLOX%20introduction.png)

1. 前言
    - 主要对比YOLOv5
    - 数据集分辨率很高的话，建议使用YOLOv5，应为YOLOX也只是640x640
    ![YOLOX effect.png](../pictures/YOLOX%20effect.png)

1. YOLOX网络结构
    - 使用网络结构(YOLOX-L)绘制的图
    ![YOLOX structure1.png](../pictures/YOLOX%20structure1.png)

    - YOLOX是基于YOLOv5的v5.0构建的，网络结构到PAN之前都一样，只有Head不一样(上面的YOLOv5是v6.1，和v5.0还有出入)
    - 区别：
        1. Focus -> 6x6的卷积(原理一样)
        1. YOLOv5是SPPF，但是YOLOX是SPP，而且YOLOX的摆放位置和YOLOv5也是一样的
        1. YOLOv5的检测头是1x1的卷积层，在YOLOX中改成如下的形式：
            ![YOLOX structure2.png](../pictures/YOLOX%20structure2.png)
            
            - 作者认为YOLOv5这么做是一个耦合的检测头，耦合的检测头对网络是有害的。但是如果换成解耦的检测头，可以加速收敛，提升AP
            - 检测类别和检测定位以及obj的卷积层是分开的。检测三个项目的检测头是参数不共享的，而且不同的预测特征层的检测头参数也是不共享的。FCOS是共享的

1. Anchor-Free
    ![YOLOX Anchor Free.png](../pictures/YOLOX%20Anchor%20Free.png)
    
    - 这里预测的 $x_ {center}, y_{center}, w, h$ 都是在预测特征层上的尺度，再恢复到原图上还要计算缩放问题
    - 仔细看这个公式，之前的YOLO公式是要乘上对应ancher的尺寸，这里公式里不再使用anchor尺寸了，所以是anchor free

1. 损失计算
    ![YOLOX Loss.png](../pictures/YOLOX%20Loss.png)

1. 正负样本匹配SimOTA
    - 论文消融实验都是和YOLOv3做对比

    ![YOLOX SimOTA1.png](../pictures/YOLOX%20SimOTA1.png)

    ![YOLOX SimOTA2.png](../pictures/YOLOX%20SimOTA2.png)

    ![YOLOX SimOTA3.png](../pictures/YOLOX%20SimOTA3.png)

    - 在FCOS网络中，落入sub-box中的所有anchor point视为正样本，除此之外都是负样本
    - 在YOLOX中也是做了一个预筛选，首先找在GT box或者fixed center area(类似sub-box)范围之内的anchor point(fixed center area由一个参数，center_radius=2.5)
    - 可以将YOLOX中的点细分为两个部分：
        1. 既落入GT box，又落入fixed center box
        2. 除了上面之外的点
    - 从损失(cost)公式中，可以看到，前两项是正常的分类损失和定位损失，后一项就是除了GT box和fixed center box交集区域以外的点，给了一个很大的权重，迫使降低这个部分的错误率

    ![YOLOX SimOTA4.png](../pictures/YOLOX%20SimOTA4.png)

    - 筛选IoU最大的10个，或者更少的anchors

    ![YOLOX SimOTA5.png](../pictures/YOLOX%20SimOTA5.png)

    - dynamic_ks代表论文中的Dynamic k Estimation Stragegy，意思是，每个GT分配的正样本的个数不一样，需要动态计算
    - 计算方法就是：对GT分配的正样本的IoU矩阵，对IoU的值进行求和，再向下取正

    ![YOLOX SimOTA6.png](../pictures/YOLOX%20SimOTA6.png)

    - 根据dynamic_ks确定anchors的最终个数，根据cost的升序排列，选最小的dynamic_ks个anchors

    ![YOLOX SimOTA7.png](../pictures/YOLOX%20SimOTA7.png)

    - 如果出现一个anchor被分配给了多个GT，那就看它跟哪个GT的cost最小，将其分配给对应的GT
        - 注意：这一步是在确定了每个GT最小的dynamic_ks个anchors，意味着冲突竞争中失败的GT们，最终获得的anchors数量会减少

    ![YOLOX SimOTA8.png](../pictures/YOLOX%20SimOTA8.png)

## 3. 缺点


# YOLOR

- 专注于某个感官的时候，其他感官的感受性可能会降低

## 1. 面临的挑战
1. 一般训练中的问题
    ![YOLOR notation](../pictures/YOLOR%20notation.png)

    - 训练网络的过程大致如下：
        ![YOLOR general learning process](../pictures/YOLOR%20general%20learning%20process.png)

    - 我们对事物的关注点如下：
        ![YOLOR attention map](../pictures/YOLOR%20attention%20map.png)

    - 为什么我们会产生这样的原因呢？
        ![YOLOR Formula](../pictures/YOLOR%20Formula.png)

    - 学习的时候，其实我们只学习不同类别之间有区分的地方，相似的地方我们不关注。就像下面的activation map上，猫和狗的身体在map上都不显现，因为靠猫和狗的身体无法帮助我们区分猫还是狗
        ![YOLOR activation map](../pictures/YOLOR%20activation%20map.png)

    - 这也就带来一个问题：就像下面的皮卡丘，它们相似的部分我们不关注。这就会导致，它改变了颜色，改变了公母，改变了帽子，我们都不会关注，因为我们忽略它们形状类似的部分。
    - 导致这件事情的原因就是，在训练网络的Formula里的error，我们只把它们当成一个简单的error数值，却没去考虑它和原图有这个不同的点具体是什么？没有追根溯源
        ![YOLOR Limitation](../pictures/YOLOR%20Limitation.png)

1. 多任务训练的问题
    - 最简单的想法就是，一个任务训练疑个model。但是这会消耗大量的资源，而且最后的结果也并不一定是最好的。
        ![YOLOR one model for one task](../pictures/YOLOR%20one%20model%20for%20one%20task.png)
    
    - 现在我们希望，所有的任务共用一个网络，就是backbone，这样我们可以在real time的时间里，做出相应的结果
    - 但是会发现，有的任务采取这种方法，效果还不错。但是有的任务就训练不起来
    - 这是因为不同任务需要的特征不一样，而且对特征的需求可能是冲突的。比如对目标检测任务检测宝可梦，希望宝可梦们的特征尽可能的相似；而对宝可梦的性别分析，可能要分析尾巴的花纹。那么这对特征提取就会有一定的冲突
        ![YOLOR shared backbone](../pictures/YOLOR%20shared%20backbone.png)

    - 现在的一些解决方案是上述两种方案的折中：训练多个特征提取器，但是之间互相share一些weights，不共享的权重，用于提取特别需要的特征。
    - 但是问题是，怎么有效率的去甄别要共享哪些权重呢？
        ![YOLOR soft parameter sharing](../pictures/YOLOR%20soft%20parameter%20sharing.png)

## 2. 解决方案
1. Manifold Learning
    - 在高维度上评估距离是不可靠的。就像第二幅图，可能红色之间的距离，甚至比红色和蓝色之间的距离都要远
    - 但是如果降维到低维度上，就可以更好的使用距离error方法
        ![YOLOR manifold](../pictures/YOLOR%20manifold.png)

    - 常用的Manifold Learning方法是t-SNE方法
    - 找到一个合适的Manifold Learning的方法是很重要的
        ![YOLOR t-SNE](../pictures/YOLOR%20t-SNE.png)

    - 来看一个例子
    - x轴上代表了狗的姿势，y轴上代表了狗的种类
        ![YOLOR reduce manifold space of the representation1](../pictures/YOLOR%20reduce%20manifold%20space%20of%20the%20representation1.png)

    - 通过reduce维度，我们可以将复杂的问题投影到低维度上，变成一个简单的问题来处理
    - 如果reduce其中一个dimension的话，可以只提取一个种类的但是不同姿势的狗
        ![YOLOR reduce manifold space of the representation2](../pictures/YOLOR%20reduce%20manifold%20space%20of%20the%20representation2.png)

    - 如果reduce另一个dimension的话，可以只提取一个姿势的但是不同种类的狗
        ![YOLOR reduce manifold space of the representation3](../pictures/YOLOR%20reduce%20manifold%20space%20of%20the%20representation3.png)
        

1. Model the Error Term
    - 对error建模，让模型知道为什么error了
    - 希望我们的任务输出，在high dimension上，每个维度上的数据是有关联性的

    - 此前我们是把属于同一类的error，映射到低维度的时候，都压缩成一个类别，压缩之后就丢失了这个类别的属性信息了，也没办法进一步知道它们具体error的点
        ![YOLOR minimize the error term](../pictures/YOLOR%20minimize%20the%20error%20term.png)

    - 现在我们不对维度进行压缩了，我们映射到更高的维度，寻找一个新的方式进行映射压缩，这个方式压缩后，可以反映出为什么会产生error
        ![YOLOR relax the error term](../pictures/YOLOR%20relax%20the%20error%20term.png)

    - 那么我们就要对error进行一个建模
    - 获得了error在高维的投影，我们就可以根据error种类的需要，去做不同的Manifold，获得对应压缩后的结果
        ![YOLOR model the error term](../pictures/YOLOR%20model%20the%20error%20term.png)

    - 在结合这些explicit和implicit上面，可以有很多运算操作：addition, multiplication, concatenation
        ![YOLOR operation](../pictures/YOLOR%20operation.png)

1. Disentangle the Representation of Input and Tasks
    - 根据输入和任务去做裁剪
    - 相同的输入，但是在不同的想法下，是有不同的答案的
        ![YOLOR observation](../pictures/YOLOR%20observation.png)

    - 需要找到只跟输入有关，但是跟任务无关的 $P(x)$
    - 需要找到只跟任务有关，但是跟输入无关的 $P(c)$
    - 最好还能找到基于任务的输入的关系，这样就能解释为什么通过这个输入能够得到这样的输出
        ![YOLOR posterior](../pictures/YOLOR%20posterior.png)

## 3. YOLOR for Object Detection
1. YOLOR
    - 中间的Analyzer是只跟Input有关
    - 根据输入可以得到一定的explicit Knowledge
    - 以及一些网络中没有输入的Implicit Kownledge
    - 通过Discriminator用来分辨任务种类
        ![YOLOR YOLOR](../pictures/YOLOR%20YOLOR.png)

1. Explicit Kownledge
    ![YOLOR explicit kownledge](../pictures/YOLOR%20explicit%20kownledge.png)

1. Implicit Kownledge
    - 在论文中，我们更关注Implicit Kownledge
    ![YOLOR our focus](../pictures/YOLOR%20our%20focus.png)

    - 举一个之前在object detection上常见的问题
    - 高分辨率的图像，所包含的信息是很多的
    - 低分辨率的图像，包含信息很少
        ![YOLOR kernel space alignment1](../pictures/YOLOR%20kernel%20space%20alignment1.png)

    - 但是我们很容易把多个分辨率的图像，都reduce到信息最少的低分辨率图像上
    - 原因就是之前的目标检测error都压缩了很多信息，所以即使高分辨率图像包含了大量信息，但是由于低分辨率图像上没有，取交集之后，就相当于映射到了低分辨率图像的信息集上
        ![YOLOR kernel space alignment2](../pictures/YOLOR%20kernel%20space%20alignment2.png)

    - 近几年非常流行的FPN网络就在解决这个问题
    - 不同分支可以对不同物件分析不同的信息
        ![YOLOR kernel space alignment3](../pictures/YOLOR%20kernel%20space%20alignment3.png)

    - 但是这样会导致，不同分支产生的特征彼此之间很难去映射
    - 大的特征图信息多，小的特征图信息少，映射很困难
        ![YOLOR kernel space alignment4](../pictures/YOLOR%20kernel%20space%20alignment4.png)

    - 这时候Implicit Knowledge的作用就显现出来了
    - Implicit Knowledge的加入，对原始特征图上的特征都做了不同程度的偏移，从而使多个特征图放在一块的时候可以进行一个比较
        ![YOLOR kernel space alignment5](../pictures/YOLOR%20kernel%20space%20alignment5.png)

## 4. 实验的结果和结论
1. YOLOR + YOLO
    - combine explicit knowledge and implicit knowledge
    - addition：把不同特征通过加法做结合
    - multiplication：类似attention的机制
    - concatenation：类似给定一个条件去做condition的运算
        ![YOLOR combine explicit knowledge and implicit knowledge](../pictures/YOLOR%20combine%20explicit%20knowledge%20and%20implicit%20knowledge.png)
    
    - 对最后的特征图使用加法或者乘法
        ![YOLOR implicit representation1](../pictures/YOLOR%20implicit%20representation1.png)

    - 最后得到的一些准度的结果
        ![YOLOR performance1](../pictures/YOLOR%20performance1.png)
    
    - 很显然，implicit knowledge在初始化为1附近采样的情况下，能够根据anchors的尺寸，学习到周期性的内容，也能根据数据集中每个类别样本数量的多少做出调整
        ![YOLOR physical meaning](../pictures/YOLOR%20physical%20meaning.png)

    - 在中间层添加了implicit knowledge后，也会产生数值上的一些区分
        ![YOLOR implicit representation2](../pictures/YOLOR%20implicit%20representation2.png)

    - feature special alignment效果还是不错的
    - 实验结果是，略微增加了参数量(基数很大，百分比很小)，但是提高了0.5%的精度
        ![YOLOR performance2](../pictures/YOLOR%20performance2.png)
        
    - 做了很对不同的implicit knowledge model
    - Neural network：认为得到的z中的每个维度彼此是关联的
    - 矩阵分解：认为得到的z中的每个维度彼此是独立的，但是每个维度也有很多不同的变因导致最后的结果，通过乘以一个权重c，进行每个维度的加权和
    - 结果是，不管采用哪种方法，最后都是提升
        ![YOLOR model explicit knowledge and implicit knowledge](../pictures/YOLOR%20model%20explicit%20knowledge%20and%20implicit%20knowledge.png)

    - 最后提升了88%的速度和3.8%的精度


1. YOLOR + Multiple Tasks
    ![YOLOR Faster R-CNN](../pictures/YOLOR%20Faster%20R-CNN.png)

    ![YOLOR Mask R-CNN](../pictures/YOLOR%20Mask%20R-CNN.png)

    ![YOLOR ATSS](../pictures/YOLOR%20ATSS.png)

    ![YOLOR FCOS](../pictures/YOLOR%20FCOS.png)

    ![YOLOR sparse R-CNN](../pictures/YOLOR%20sparse%20R-CNN.png)

    ![YOLOR multiple task performance](../pictures/YOLOR%20multiple%20task%20performance.png)

## 5. Q&A
- $z$ 是implicit的部分
- $x$ 是explicit的部分
- $z$ 是加在channel轴上
- $z$ 是被训练出来的
- 矩阵分解的方法中， $c$ 是coeffience
- 比attention based方法，参数量更少，但是效果更好


# 题目

## 1. 优势/历史地位

## 2. 算法流程

## 3. 缺点