# 强化学习一小时完全入门

## 目录
- [视频课程](https://www.bilibili.com/video/BV13a4y1J7bw/?spm_id_from=333.999.0.0&vd_source=09eb8c9e7b3a221f6536a575e712dfa4)
- [视频代码](../code/One%20Hour%20of%20Reinforce%20Learning/One%20Hour%20of%20Reinforce%20Learning%20ooxx.py)
- [元素介绍](#元素介绍)
- [元素关系](#元素关系)
- [强化学习的特点](#强化学习的特点)
- [强化学习的核心问题](#强化学习的核心问题)
- [举例：多臂老虎机（K-armed Bandit）](#举例多臂老虎机k-armed-bandit)
- [几种算法](#几种算法)
- [误差的概念](#误差的概念)
    - [第一类公式推导](#第一类公式推导)
    - [第二类公式推导](#第二类公式推导)
- [推广：状态价值/状态行动价值——强化学习的核心](#推广状态价值状态行动价值强化学习的核心)
    - [状态行动价值函数](#状态行动价值函数)
    - [状态价值函数](#状态价值函数)
- [后果(Outcome/Afterstate)](#后果outcomeafterstate)
- [返回上一层 README](../README.md)

## 元素介绍

![](../pictures/One%20Hour%20of%20Reinforce%20Learning/Elemetns.jpg)

> 第一层结构：基本元素（Basic Elements）
>
> 第二层结构：主要元素（Main Elements）
>
> 第三层结构：核心元素（Core Elements）

*[跳转至目录](#目录)*

## 元素关系

![](../pictures/One%20Hour%20of%20Reinforce%20Learning/RL%20Beginner.png)

*[跳转至目录](#目录)*


## 强化学习的特点

1. Trial and Error（试错学习）
1. Delayed Reward（延迟奖励）

- 如何学习过去行动的价值？
    1. Credit Assignment
    1. Back Propagation
    - 举例：复盘法

*[跳转至目录](#目录)*

## 强化学习的核心问题

**如何平衡 Exploration（探索）& Exploitation（利用）？**

1. Exploration（探索）
    - 不完全相信学到的价值函数，不完全相信价值函数的结果，时不时突破既定策略，不采用理论最优解，尝试一些其他行动，优化价值函数

1. Exploitation（利用）
    - 完全相信学到的价值函数，完全相信价值函数的结果，按照既定策略，采用理论最优解，做出相应的行动

*[跳转至目录](#目录)*

## 举例：多臂老虎机（K-armed Bandit）

这里假设```K=2```，可以认为只有两台老虎机

> 这个例子的特点是：
> 1. 只有一个状态
> 1. 没有回报延迟，即时回报 


1. 基本元素（Basic Elements）
    1. Agent（玩家）：参与者
    1. Environment（环境）：K-armed Bandit
    1. Goal（目标）：挣到尽可能多的钱

1. 主要元素（Main Elements）
    1. State（状态）：多臂老虎机的状态只有一个
    1. Action（行动）：使用第几个摇臂
    1. Reward（回报）：每个摇臂都对应了一种数学分布

1. 核心元素（Core Elements）
    1. Value（价值）：将来能够获得所有奖励之和的期望值，这里采用 样本平均法（Sample-Average）
        - 样本平均法（Sample-Average）对某一个摇臂的价值预测（初始值不为0且计入平均计算）：$Q_ {n}=\frac{1}{n}(Q_ {1} + \sum^{n-1}_ {i=1}R_ {i})$
    1. Policy（策略）：这里采用 普通的Greedy+初始值不为0且计入平均计算
        - Greedy：选择期望最大的 $A_ {t}=\underset{a}{argmax}Q_ {t}(a)$
        1. 普通的Greedy，且初始值为0：开局即陷入某一种行动，无法自拔
        1. 普通的Greedy+强制做一遍所有行动&普通的Greedy+初始值不为0且不计入平均计算：强制做一遍所有行动
        1. 普通的Greedy+初始值不为0且计入平均计算：鼓励更多的探索行为
        1.  $\epsilon-Greedy$：探索进一步被增强

*[跳转至目录](#目录)*

## 几种算法
1. Finite Markov Decision Process
1. Dynamic Programming
1. Monte Carlo Methods
1. Temporal-Difference Learning

*[跳转至目录](#目录)*

## 误差的概念

强化学习本身是一种基于误差的学习方法

### 第一类公式推导

Sample-Average，**以初始值为0且不计入平均计算**，初始 $Q_ {1}$ 对之后的价值估计无影响（也可以有带初始值的算术平均）

$$Q_ {n+1}=\frac{1}{n}\sum^{n}_ {i=1}R_ {i}$$

其中：
- $Q_ {1}$：初始值
- $Q_ {n+1}$：采取n次行动之后，价值预测

$$
\begin{aligned}
&Q_ {n+1} = \frac{1}{n}\sum^{n}_ {i=1}R_ {i} \\
&Q_ {n} = \frac{1}{n-1}\sum^{n-1}_ {i=1}R_ {i} \\
\end{aligned}
$$

$$
\begin{aligned}
Q_ {n+1} &= \frac{1}{n}(R_ {n} + \sum^{n-1}_ {i=1}R_ {i}) \\
&= \frac{1}{n}R_ {n} + \frac{1}{n}\sum^{n-1}_ {i=1}R_ {i} \\
&= \frac{1}{n-1}\sum^{n-1}_ {i=1}R_ {i} + \frac{1}{n}R_ {n} + \frac{1}{n}\sum^{n-1}_ {i=1}R_ {i} - \frac{1}{n-1}\sum^{n-1}_ {i=1}R_ {i} \\
&= Q_ {n} + \frac{1}{n}R_ {n} - \frac{1}{n}Q_ {n} \\
&= Q_ {n} + \frac{1}{n}(R_ {n} - Q_ {n}) \\
\end{aligned}
$$


$$\underset{New Estimate}{Q_ {n+1}} = \underset{Old Estimate}{Q_ {n}} + \underset{Learning Rate}{\frac{1}{n}}\underset{Reward Prediction Error}{(R_ {n} - \underset{Estimate of R_ {n}}{Q_ {n}})}$$


### 第二类公式推导

Weighted-Average，初始 $Q_ {1}$ 对之后的估计有影响

$$
\begin{aligned}
&Q_ {n+1} = Q_ {n} + \alpha(R_ {n} - Q_ {n}) \\
&Q_ {n} = Q_ {n-1} + \alpha(R_ {n-1} - Q_ {n-1}) \\
\end{aligned}
$$

$$
\begin{aligned}
Q_ {n+1} &= Q_ {n-1} + \alpha(R_ {n-1} - Q_ {n-1}) + \alpha R_ {n} - \alpha Q_ {n-1} - \alpha^{2}(R_ {n-1} - Q_ {n-1})\\
&= (1 - \alpha)^{2}Q_ {n-1} + \alpha R_ {n} + \alpha(1 - \alpha)R_ {n-1} \\
\end{aligned}
$$

继续递推：
$$Q_ {n+1} = (1 - \alpha)^{n}Q_ {1} + \sum^{n}_ {i-1}\alpha(1 - \alpha)^{n-i}R_ {i}$$

*[跳转至目录](#目录)*

## 推广：状态价值/状态行动价值——强化学习的核心

流程：
$$
\begin{aligned}
S_ {t} &\to A_ {t} \to R_ {t} \\
&\swarrow \\
S_ {t+1} &\to A_ {t+1} \to R_ {t+1} \\
&\swarrow \\
\cdots &\quad \cdots \quad \cdots\\
S_ {T} &\to A_ {T} \to R_ {T} \\
&\swarrow \\
S_ {T+1} &\to A_ {T+1} \to R_ {T+1} \\
\end{aligned}
$$


令：
$$
\begin{aligned}
\sum^{T}_ {i=t}R_ {i} &= R_ {t} + \sum^{T}_ {i=t+1}R_ {i} \\
&\approx R_ {t} + Q(S_ {t+1}, A_ {t+1}) \\
\end{aligned}
$$


### 状态行动价值函数

$$
\begin{aligned}
\underset{New Estimate}{Q(S_ {t} ,A_ {t})} &\gets \underset{Old Estimate}{Q(S_ {t},A_ {t})} + \underset{Learning Rate}{\alpha}\underset{Error}{(\sum^{T}_ {i=t}R_ {i} - Q(S_ {t},A_ {t}))} \quad 蒙特卡洛方法(Monte Carlo Method)的雏形 \\
&\gets Q(S_ {t},A_ {t}) + \alpha(R_ {t} + Q(S_ {t+1},A_ {t+1}) - Q(S_ {t},A_ {t})) \quad 时序差分学习法(Temporal-Difference Learning)的雏形 \\
\end{aligned}
$$

### 状态价值函数

$$
\begin{aligned}
\underset{New Estimate}{Q(S_ {t})} &\gets \underset{Old Estimate}{Q(S_ {t})} + \underset{Learning Rate}{\alpha}\underset{Error}{(\sum^{T}_ {i=t}R_ {i} - Q(S_ {t}))} \quad 蒙特卡洛方法(Monte Carlo Method)的雏形 \\
&\gets Q(S_ {t}) + \alpha(R_ {t} + Q(S_ {t+1}) - Q(S_ {t})) \quad 时序差分学习法(Temporal-Difference Learning)的雏形 \\
\end{aligned}
$$

*[跳转至目录](#目录)*


## 后果(Outcome/Afterstate)

虽然状态——行动对可能不同，但导致的后果可能相同

*[跳转至目录](#目录)*