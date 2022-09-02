# 论文信息
- 时间：2017
- 期刊：CVPR
- 网络名称： MobileNet
- 意义：适合终端设备的小CNN
- 作者：Andrew G. Howard；Menglong Zhu；Bo Chen；Dmitry Kalenichenko；Weijun Wang；Tobias Weyand；Marco Andreetto；Hartwig Adam；Google
- 实验环境：
- 数据集： ILSVRC 2012

# 一、解决的问题
1. >We present a class of efficient models called MobileNets for mobile and embedded vision applications
2. >The general trendhas been to make deeper and more complicated networks in order to achieve higher accuracy
3. >Many papers on small networks focus only on size but do not consider speed
4. >Although the base MobileNet architecture is already small and low latency, many times a specific use case or application may require the model to be smaller and faster

# 二、做出的创新
- MobileNet模型基于深度可分离卷积，这是一种因式卷积，它将标准卷积分解为深度卷积和1×1卷积，称为逐点卷积

![Depthwise Separable Convolution](../pictures/Depthwise%20Separable%20Convolution.png)

![Depthwise Separable Convolution vs Standard convolutional](../pictures/Depthwise%20Separable%20Conv%20vs%20standard.png)

![MobileNet Body Architecture](../pictures/MobileNet%20Body%20Architecture.png)

# 三、设计的模型
1. >All layers are followed by a batchnorm and ReLU nonlinearity with the exception
of the final fully connected layer which has no nonlinearity and feeds into a softmax layer for classification
2. >Down sampling is handled with strided convolution in the depthwise convolutions as well as in the first layer
3. >A final average pooling reduces the spatial resolution to 1 before the fully
connected layer
4. >Counting depthwise and pointwise convolutions as separate layers, MobileNet has 28 layers

# 四、实验结果
## 1、比之前模型的优势
- MobileNet uses 3 × 3 depthwise separable convolutions which uses between 8 to 9 times less computation than standard convolutions at only a small reduction in accuracy
- MobileNet算法和正常的卷积比起来，准确率上差距很小
- 对于缩小网络来看，在参数量近似的情况下，缩减网络的宽度，比缩减网络的深度对准确率影响更小

## 2、有优势的原因
1. >contrary to training large models we use less regularization and data augmentation techniques because small models have less trouble with overfitting
2. MobileNet模型基于深度可分离卷积，这是一种因式卷积，它将标准卷积分解为深度卷积和1×1卷积，称为逐点卷积，大大减小了参数量
3. 网络太窄了，低于0.5 rate之后，准确率就断崖式下降了
4. 输入分辨率的下降，导致准确率是平滑的下降
5. 比AlexNet小，准确率还高

## 3、改进空间

# 五、结论

## 1、模型是否解决了目标问题
- 达到了大幅度缩小模型的目的
## 2、模型是否遗留了问题
- 准确率有待提高
## 3、模型是否引入了新的问题
- 准确率还是有不小的下降
# 六、代码

# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论