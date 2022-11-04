# 论文信息
- 时间：2018
- 期刊：NeurIPS
- 网络名称：GPipe
- 意义：流水线（Pipeline）并行	
- 作者：Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, Zhifeng Chen {huangyp,ylc,ankurbpn,orhanf,miachen,dehao hyouklee,jngiam,qvl,yonghui,zhifengc} @google.com
- 实验环境：
- 数据集：

# 名词解释
1. Google作者喜欢用加速器（accelerator）一词，指代TPU或者GPU，Google指TPU
2. 模型并行：把模型分成很多小块，让不同的小块放到不同的加速器上，但是如何分割网络是个问题
3. Lingvo：基于tensorflow上面的一个框架，特别针对变长输入的语言模型，可能和Keras比较像，追求可重复性，超参数通过代码写在里面
4. Bubble：空闲的时间
# 一、解决的问题
1. 利用流水线并行的技术，有效的训练巨大的神经网络、
    ![GPipe greater model better](../pictures/GPipe%20greater%20model%20better.png)
    - 可学习参数增多，精度越高
2. 神经网络变大了之后会超过单个加速器的容量，需要高带宽（GPU对内存带宽要求更高），高内存（显存），主要是成本问题
3. 只要模型都是一层一层叠加的，这个并行流水线加速是通用的

# 二、做出的创新
1. 把CPU的流水线思想用在GPU训练上
2. 提供两个技术：
    - Re-materialization：把一些中间结果丢掉，下次再用的时候重新计算，减少内存占用率
    - micro-batches：微批量，将一个小批量再切块，做到更小的尺度，带来流水线并行
# 三、设计的模型
1. 在Lingvo框架上实现的
2. Interface（接口）
    1. 认为一个神经网络可以定义成一个L层的序列，第i个层 $L_i$ 本质上就是一个前置的一个函数 $f_i$ ，就是给一个输入，怎么样计算输出，以及对应的可学习参数
    2. GPipe允许用户告诉说，这个层的计算开销是什么，比如，每一个层需要多少Flops
    3. 定义好任务后，要告诉把网络切成多少块，就是把该网络序列切成k个子序列，每个子序列叫做一个单元，假设 $p_k$ 对应第k个单元的话，那么它就对应当前单元的前置计算函数 $f_i$ 和对应的权重
    
3. Algorithm
    1. 把切的第k个单元，放在第k个加速器上面，模型并行和数据并行主要区别是，怎么切
        ![GPipe model parallel and data parallel](../pictures/GPipe%20model%20parallel%20and%20data%20parallel.png)
    2. 通常分布式系统认为数据并行更好，因为模型并行有个很致命的问题
        ![GPipe interface](../pictures/GPipe%20interface.png)
        - 从时间上和做单GPU算是没区别的，除了可以看作内存增大了，但是计算能力没有增强，不叫并行
    3. 将小批量再切成多个微批量，每次在一个微批量上做运算，模型并行+数据并行（数据之间没有相互依赖关系）
4. Performance Optimization
    1. 如果有批量归一化的时候，要做一点处理
        - 批量归一化的时候，要对每个批量算一次均值和方差
        - **Transformer没有这个问题，用了层的归一化，每一次算均值和方差是对每一个样本里面算的**
    2. 之前完成的*模型的切割*和*数据的切割*，现在还要看怎么节省内存
        1. 还有大量数据（每个层的中间输入，后面算梯度还要用到）要放在activation memory上
            ![GPipe activation](../pictures/GPipe%20activation.png)
            - 计算换空间
            - n：样本大小
            - d：隐藏层大小
            - l：层数
        2. Re-materialization：
            - 每个加速器维护一个单元，它只会存activation在边界处的地方，别的地方就不存了，现用现算，做forward，把之前的内存空间复杂度从 $O(N \times L)$ 减少到 $O(N + \frac{L}{K} \times \frac{N}{M})$
                ![GPipe space complex](../pictures/GPipe%20space%20complex.png)
            - 开销：
                -  Bubble time $O(\frac{K-1}{M+K-1})$ ，只要切的块： $M \geqslant 4 \times K $
                - 重新forward 
                - GPU之间通信
                - 会不会有GPU算的特别慢，别人都要等它，这就用到了一开始告诉GPipe计算开销的数据，网络越均匀越好
               
# 四、实验结果
1. 性能分析
    ![GPipe table1](../pictures/GPipe%20table1.png)
    - 第一个做的是变形虫，CNN网络
    - 第二个是transformer，是large
    - 第一个应该是写错了，都用的是TPU，一个是v2，一个是v3
    - 内容：
        - 第一行是超参数
        - 第二行是模型有多少可学习的参数
        - 第三行是整个模型参数所占用的内存
        - 第四行是最多的时候，中间变量占的内存
    - 应该是线性关系，但是因为CNN的每一个层，不断地把高和宽减低，通道数增加，所以它只保证每一层的计算差不多，但是内存使用率是不一样的，切的不会很均匀，增加卡的时候，中间可能有一个GPU，的内存会占的多一点，成为瓶颈；Transformer就会好很多，输出就是一个隐藏层（Dimension），隐藏通道数这个东西是不变的，Transformer层之间的计算开销和内存开销是差不多的，非常均匀，增长线性
    - 确实做到了增加GPU的时候，线性增加了模型的大小

    ![GPipe table2](../pictures/GPipe%20table2.png)
    - K：卡的数量
    - M：微批量
    - 真的取决于模型并行的时候，切的能不能均匀

    ![GPipe table3](../pictures/GPipe%20table3.png)
    - 应该是强制GPU传到CPU去做GPU的交互，证明模型并行比数据并行通信量少
2. 性能开销的一个分解
    1. 讲的原因是之前的结果都有一定的误导性
        - 因为为了支撑很大的模型，用了Re-materialization（把一些中间结果删掉，这样要重新算一次前置运算），那这样加速比都不太好看，没法跟别人比
        ![GPipe table4](../pictures/GPipe%20table4.png)

3. 实验都过时了，核心还是看这份工作是否真的有用
## 1、比之前模型的优势
1. 比同时期的PipeDream（来自微软，更系统化，更复杂的算法），更出名，应该是用的方法更简单
## 2、有优势的原因

## 3、改进空间

# 五、结论

## 1、模型是否解决了目标问题
- 高亮三个贡献：
    1. 性能：
        - 数据切开，提高了并行度，得到了近似线性的提升
        - 支持更大的模型，但是需要Re-materialization，需要额外的20%开销
    2. 灵活性（？也可能是局限性）
        - 支持任意一个串起来的神经网络（或者说只能支持这种网络）
        - 不串起来的->图神经网络就做不了
    3. 可靠性
        - 同步的梯度下降，和单卡一样
        - PipeDream用了异步，把性能做得更好，系统更难
        - 同步异步可能让算法更好，也可能不好
## 2、模型是否遗留了问题

## 3、模型是否引入了新的问题

# 六、代码

# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论
