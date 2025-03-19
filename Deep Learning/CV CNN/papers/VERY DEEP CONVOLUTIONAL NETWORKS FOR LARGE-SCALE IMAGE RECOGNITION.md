# 论文信息
- 时间：2015
- 期刊：ICLR
- 网络名称： VGG
- 意义：使用 3x3 卷积构造更深的网络
- 作者：Karen Simonyan & Andrew Zisserman Visual Geometry Group, Department of Engineering Science, University of Oxford
- 实验环境：four NVIDIA Titan Black GPUs, training a single net took 2–3 weeks depending on the architecture
- 数据集： ILSVRC-2012 dataset
- [返回上一层 README](../README.md)

# 一、解决的问题
1. 探索卷积神经网络的深度和它的准确率的关系
2. 继AlexNet的成果发展而来

# 二、做出的创新
1. 主要贡献：使用3x3卷积核构造更深的网络，深度可达16-19层
# 三、设计的模型
1. >the input to our ConvNets is a fixed-size 224 × 224 RGB image
2. 使用3x3卷积核
3. 有例外：
>In one of the configurations we also utilise 1 × 1 convolution filters
4. >The convolution stride is fixed to 1 pixel
5. >the padding is 1 pixel for 3 × 3 conv
6. > Spatial pooling is carried out by five max-pooling layers, which follow some of the conv. layers (not all the conv. layers are followed by max-pooling). Max-pooling is performed over a 2 × 2 pixel window, with stride 2.
7. >A stack of convolutional layers (which has a different depth in different architectures) is followed by three Fully-Connected (FC) layers: the first two have 4096 channels each, the third performs 1000-way ILSVRC classification and thus contains 1000 channels (one for each class). The final layer is the soft-max layer. The configuration of the fully connected layers is the same in all networks.
8. 激活函数全部是ReLU
9. >The batch size was set to 256, momentum to 0.9
10. >The learning rate was initially set to 10−2, and then decreased by a factor of 10 when the validation set accuracy stopped improving

# 四、实验结果
- 模型越深，准确率越高
- LRN没什么用
- 只有模型够深的时候，使用大数据集才会提高识别的准确率
- MULTI-SCALE EVALUATION 在模型越深的时候表现越好
- MULTI-CROP EVALUATION 在模型越深的时候表现越好
## 1、比之前模型的优势
1. 识别的更加准确，成为了ILSVRC classification and localisation tasks的SOTA
2. >Our best single-network performance on the validation set is 24.8%/7.5% top-1/top-5 error 
## 2、有优势的原因
1. 比起其他作者通过改进小的窗口和小的步长，VGG通过增加卷积层改进了深度
2. 优化了各个超参数
3. 能够实现增加深度的原因，是在所有的卷积层中都使用了3x3的卷积核

## 3、改进空间

# 五、结论

## 1、模型是否解决了目标问题
- >(up to 19 weight layers 确实提高了准确率，增加了模型的深度

## 2、模型是否遗留了问题
- 模型增大，training时间增长
## 3、模型是否引入了新的问题

# 六、代码

# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论
