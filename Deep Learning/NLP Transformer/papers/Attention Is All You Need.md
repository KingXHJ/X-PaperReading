# 论文信息
- 时间：2017
- 期刊：NeurIPS
- 网络名称： Transformer
- 意义：继MLP、CNN、RNN后的第四大类架构
- 作者：Ashish Vaswani∗, Google Brain; Noam Shazeer∗, Google Brain; Niki Parmar∗, Google Research; Jakob Uszkoreit∗, Google Research; Llion Jones∗, Google Research; Aidan N. Gomez∗ †, University of Toronto; Łukasz Kaiser∗, Google Brain; Illia Polosukhin∗ ‡
- 实验环境：8个P100 GPUs
- 数据集：
- [返回上一层 README](../README.md)

[PPT](../ppt/Transformer/TRM.pptx)

[Theory Notes](../ppt/Transformer/TRM.pdf)

[Code Anlysis](../ppt/Transformer/TRMCode.pptx)

[Code Anlysis Notes](../ppt/Transformer/TRMCode.pdf)

[Python Code](../code/Transformer/TRM.py)

[Interview Answers](../ppt/Transformer/Answers.pdf)

# 一、解决的问题
1. 之前使用RNN是个时序模型，是一个无法在时间上进行并行，后一个词依赖于前一个词的计算结果
2. 且RNN会丢弃历史信息
3. attention主要用在怎么把编码器的东西，有效的传给解码器
4. CNN需要很多层才能看到全部信息，但是CNN可以有很多通道
# 二、做出的创新
- 使用了一个encoder和一个decoder的时序模型（sequence model），在它们之间使用了一个叫注意力机制的东西（Attention）
1. 只使用Attention完成网络结果
2. 需要弥补CNN无法在第一层就看到全部信息的缺点，同时又要保留CNN能够具有多通道的优势
# 三、设计的模型
- Auto-regressive 上一时刻的输出是下一时刻的输入

![Transformer Architecture](../pictures/Transformer/Transformer%20architecture.png)

- 编码器输入：进入Embedding，把单词变成向量，并加入位置信息
- 解码器的输入：Outputs（shift right）在做预测的时候是没有输入的，其实就是解码器在之前时刻的输出
- 解码器多了一个Mask Multi-Head Attention

1. Encoder：
  - 两个超参数：几层？维度？
  - 一个multi-head attention和一个feed-forward network（本质MLP）
  - 每个子层用了残差连接

  ![Transformer layer norm](../pictures/Transformer/Transformer%20layer%20norm.png)

  - layer normalization：
    - batch noralization 是在每次的mini-batch里面把每个特征学习成均值为0，方差为1，对全局去算。相当于样本-特征矩阵中的一列（对一个特征做操作）
    - layer normalization 是把每个样本学习成均值为0，方差为1，对样本去算。相当于样本-特征矩阵中的一行（对一个样本做操作）
    - layer normalization 在时序用的多是因为样本的sequence长度不一样，对全局算的话，对新的预测样本来说，如果长度发生大的变化，那么之前算过的均值和方差就不会特别好使；但是layer norm对每个样本去算就会好很多。

2. Decoder：
  - 之前的两个层和Encoder都相同
  - 解码器是一个自回归，当前的输出与上一时刻及历史的输出有关，但是不可以看到未来的结果。但是在注意力机制里，每次能看到一个完整的输入，因此需要一个掩码机制

3. Attention：
  - 注意力函数是一个将query和一些key-value对，映射成一个output的一个函数
  - query、key、value和output均为向量
  - output是value的加权和，和value的维度相同
  - 加权和是通过value的key和query的相似度（compatibility function）计算得到的
  
  ![Transformer Attention calculate](../pictures/Transformer/Transformer%20Attention%20calculate.png)
  - q和k维度相同为 $d_k$ ，做内积，再除以长度开根号，最后通过softmax算出来权重（非负，加和为1）；v的维度是 $d_v$ ；完整计算：
  
  $$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$$
  
  - 和其他注意力机制的区别：
    - 一种叫加型的注意力机制，用于处理query不等长的情况；还用一种点积的注意力机制，和本文的类似，除了本文除以了 $\sqrt{d_k}$
    - 点乘比较简单
    - 除以 $\sqrt{d_k}$ ，在长度较小的时候没什么差距；在值比较大的时候，如果直接做softmax，由于较大的值之间差距大，做完后大的更靠近于1，小的更靠近于0，导致梯度很小，容易跑不动。因此要除以长度，避免这种i情况

![Transfomer Attention](../pictures/Transformer/Transformer%20Attention.png)
4. Multi-Head Attention
  - LEFT：mask会把未来的数据点换成非常大的一个负数，在进入softmax之后会变成零，就达到了mask的效果
  - RIGHT：
    - 先进入线性层，把vkq投影到较低的维度
    - 再做一个Scale Dot-Product Attention（LEFT picture），做h次
    - 输出向量全部合并在一起（concatenate）
    - 再做一个线性层回到原来的样子
  - 目的是：点积的Attention没什么好学的东西，但是Linear投影有参数是可以学习的，学习不同的投影方式，来匹配不同的相似函数。
  
  $$MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O$$
  
  $$where \quad head_i=Attention(QW^Q_i,KW^K_i,VW^V_i)$$
  
  $$Where \quad the \quad projections \quad are \quad parameter \quad matrices \quad W^Q_i\in \mathbb{R}^{d_{model}×d_k}, W^K_i\in \mathbb{R}^{d_{model}×d_k}, W^V_i\in \mathbb{R}^{d_{model}×d_v}andW^O\in \mathbb{R}^{hd_v×d_{model}}$$

5. Transformer如何使用注意力的：
  - Encoder（Multi-Head Attention）：
    - 1->3：说明qkv都是一个东西，所以是自注意力
    - 输出就是value的加权和，是输入的一个加权和
  - Decoder(Masked Multi-Head Attention):
    - 1->3：说明qkv都是一个东西，所以是自注意力
    - 输出就是value的加权和，是输入的一个加权和
    - 多加了Mask的成分，不让看后面的内容
  
  ![Transformer use Attention](../pictures/Transformer/Transformer%20use%20Attention.png)
  - Encoder和Decoder交会的地方：
    - k、v来自于编码器的输出，q来自于解码器
    - 解码器的输出会根据需求，挑在编码器中感兴趣的向量

![Transformer vs RNN](../pictures/Transformer/Transformer%20vs%20RNN.png)
6. point-wise feed-forward network
  - 本质就是MLP
  - 每个词是一个点，对每个词作用一次就是point-wise
  
  $$FFN(x)=max(0,xW_1+b_1)W_2+b_2$$
  
  $$d_{model}=512,\quad d_{ff}=2048$$

7. Embeddings and Softmax
  - Embedding(encoder decoder softmax的embedding权重相同，训练方便)就是去学习一个向量去表示一个词
  - 权重乘了一个 $\sqrt{d_{model}}$ ，因为学习到的参数都比较小，为了能和Position Encoder相加的时候，不会被忽略

8. Position Encoding
  - Attention是不会有时序信息的，所以要加入时序信息
  - 用512长度去表示一个位置
  
  $$PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})$$
  
  $$PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})$$

# 四、实验结果
- 模型并行度更好，并且用更少的时间来训练
## 1、比之前模型的优势

## 2、有优势的原因
![Transformer efficient](../pictures/Transformer/Transformer%20efficient.png)
- 可见Transformer的效率还是很高的

- 学习率是算出来的：

$$lrate = d_{model}^{-0.5} \cdot min(step_num^{-0.5},step_num \cdot warmup_steps^{-1.5})$$

- Residual dropout

- Label Smoothing

## 3、改进空间
- 可以用在除机器翻译和NLP之外的领域（作者预测了未来）
- Transformer对模型假设更少，因此需要更多的训练数据
# 五、结论

## 1、模型是否解决了目标问题

## 2、模型是否遗留了问题

## 3、模型是否引入了新的问题

# 六、代码

# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论

# 论文知识
- 星号是指作者贡献相同
