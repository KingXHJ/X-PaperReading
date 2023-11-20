# 论文信息
- 时间：2020
- 期刊：PMLR
- 网络名称：EfficientNet
- 意义：通过架构搜索得到的CNN
- 作者：Mingxing Tan；Quoc V. Le；Google
- 实验环境：
- 数据集：

# 一、解决的问题
1. >balancing network depth, width, and resolution can lead to better performance
2. >In previous work, it is common to scale only one of the three dimensions – depth, width, and image size
3. >Is there a principled method to scale up ConvNets that can achieve better accuracy and efficiency?
4. >Depth:the accuracy gain of very deep network diminishes: for example, ResNet-1000 has similar accuracy as ResNet-101 even though it has much more lay

# 二、做出的创新
1. >Notably, the effectiveness of model scaling heavily depends on the baseline network; to go even further, we use neural architecture search to develop a new baseline network, and scale it up to obtain a family of models, called EfficientNets
2. 这篇文章其实就是探寻了输入图像的分辨率，神经网络的宽度和深度，对训练准确率的影响

# 三、设计的模型

![Model Scaling](../pictures/EfficientNet/Model%20Scaling.png)

![Efficient Equation1](../pictures/EfficientNet/EfficientNet%20Equation1.png)

![Efficient Equation2](../pictures/EfficientNet/EfficientNet%20Equation2.png)

![Efficient Equation3](../pictures/EfficientNet/EfficientNet%20Equation3.png)

-  STEP 1: we first fix φ = 1, assuming twice more resources available, and do a small grid search of α, β, γ based on Equation 2 and 3. In particular, we find
the best values for EfficientNet-B0 are α = 1.2, β =1.1, γ = 1.15, under constraint of α · β^2· γ^2 ≈ 2.
-  STEP 2: we then fix α, β, γ as constants and scale up baseline network with different φ using Equation 3, to obtain EfficientNet-B1 to B7

# 四、实验结果

## 1、比之前模型的优势
1. >In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy
on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. 
2. >Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters
3. >It is also well-recognized that bigger input image size will help accuracy with the overhead of more FLOPS
- >Observation 1 – Scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger models
- >Observation 2 – In order to pursue better accuracy and efficiency, it is critical to balance all dimensions of network width, depth, and resolution during ConvNet scaling
4.  Compared to other single-dimension scaling methods, our compound scaling method improves the accuracy on all these models, suggesting the effectiveness of our proposed scaling method for general existing ConvNets
## 2、有优势的原因
- 合理的配置了输入图片的分辨率、神经网络的深度和宽度
## 3、改进空间

# 五、结论

## 1、模型是否解决了目标问题

## 2、模型是否遗留了问题

## 3、模型是否引入了新的问题

# 六、代码

# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
- Depth:：直觉上越深的网络越能捕捉到更多的更复杂的特征，但是，事实证明，越深的网络会由于梯度消失，更加难以训练
- Width：通常用于较小的模型当中，
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论
