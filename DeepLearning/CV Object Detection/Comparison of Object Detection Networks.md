# 目录

- [R-CNN](#r-cnn)
- [Fast R-CNN](#fast-r-cnn)
- [Faster R-CNN](#faster-r-cnn)
- [SSD](#ssd)
- [RetinaNet](#retinanet)
- [YOLOv1](#yolov1)
- [YOLOv2](#yolov2)
- [YOLOv3](#yolov3)
- [YOLOv3 SPP](#yolov3-spp)


# R-CNN

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

# SSD

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


# RetinaNet

## 1. 优势/历史地位
- one-stage网络首次超越two-stage
- 论文名称：Focal Loss for Dense Object Detection

## 2. 算法流程
![RatinaNet1.png](../pictures/RatinaNet1.png)

- 正负样本
    1. $IoU >= 0.5$ , 正样本
    2. $IoU < 0.4$ , 负样本
    3. $IoU \in [0.4, 0.5)$ , 舍弃

- 其余内容与R-CNN差不多，熟练掌握Faster R-CNN代码
## 3. 缺点

# YOLOv1

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


# YOLOv3

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

# YOLOv3 SPP

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
        - two-stage没有这么严重是因为，在第二阶段之前，第一阶段会把检测狂的数量压得很小，就不会出现one-stage几十个正样本面对上万个负样本的情况

    - 正样本高权重少数量，也填充不了负样本低权重大数量的情况。这种现象叫degenerate models
        - 此前通过hard negative mining是实现了筛选对训练有帮助的，大损失的负样本，但是不如Focal Loss

    - 比正常的Cross Entropy Loss引入 $\alpha$ 和 $\gamma$ 两个超参数
        - $\alpha$用于平衡正负样本的权重，但是不能区分那些是容易的样例，那些是复杂的样例
        - $\gamma$所在的损失负责降低简单样本的损失权重，这样我们就可以聚焦于训练复杂的负样本
    - 但是实验证明，敏感的Focal Loss同样容易受到噪音的干扰，因此数据集尽量不要出现标注错误的现象

## 3. 缺点




# 题目

## 1. 优势/历史地位

## 2. 算法流程

## 3. 缺点