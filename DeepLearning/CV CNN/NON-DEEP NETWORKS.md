# 论文信息
- 时间：2021
- 期刊：CVPR
- 网络名称： Non-deep Networks (ParNet)
- 意义：让不深的网络也能在ImageNet刷到SOTA
- 作者：Ankit Goyal 1,2; Alexey Bochkovskiy 2; Jia Deng 1; Vladlen Koltun 2; 1 Princeton University; 2 Intel Labs
- 实验环境：single RTX 3090 using Pytorch 1.8.1 and CUDA 11.1
- 数据集：MS-COCO

# 一、解决的问题
1. >Is it possible to build high-performing “non-deep” neural networks? 
2. >A deeper network leads to more sequential processing and higher latency
3. >it is harder to parallelize and less suitable for applications that require fast responses
# 二、做出的创新
1. we use parallel subnetworks instead of stacking one layer after another
2. “embarrassingly parallel”
3. Skip-Squeeze-and-Excitation(SSE) which is based on the Squeeze-and-Excitation (SE)

# 三、设计的模型
![ParNet and ParNet block](../pictures/ParNet.png)
1. We replace the ReLU activation with SiLU which make the network more non-linearity
2. Skip-Squeeze-and-Excitation(SSE) won't increase depth
3. Downsampling and Fusion blocks
- In the Downsampling block, there is no skip connection; instead, we add a single-layered SE module parallel to the convolution layer
4. The Fusion block is similar to the Downsampling block but contains an extra concatenation layer. Because of concatenation, the input to the Fusion block has twice as many channels as a Downsampling block

# 四、实验结果

## 1、比之前模型的优势
1. We find that one can achieve surprisingly high performance with a depth of just 12
2. ParNet-XL performs comparably to ResNet101 and gets a top-5 accuracy of 94.13, in comparison to 94.68 achieved by ResNet101, while being 8 times shallower
3. we find that even on a single GPU, ParNet achieves higher speed than strong baselines
4. We find that ParNet performs competitively with state-of-the-art deep networks like ResNets and DenseNets while using a much lower depth and a comparable number of parameters
## 2、有优势的原因
1. 并行的结构，多GPU并行表现更好
2. 在保证精度的情况下，削减SE模块的深度
## 3、改进空间

# 五、结论

## 1、模型是否解决了目标问题
1. 确实做到了用12层的神经网络去和SOTA较量

## 2、模型是否遗留了问题
1. 但是仍然达不到最优秀
2. 实验仍然说明了增大参数、流的数量和输入分辨率有利于提高准确率
## 3、模型是否引入了新的问题
1.对硬件并行要求较高

# 六、代码

# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
- Scaling DNNs
- Shallow networks
- Multi-stream networks
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论