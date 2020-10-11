# Assignment 1

## Question 1

In many pattern classification problems, one has the option either to assign the pattern to one of $c$ classes, or to reject it as being unrecognizable. If the cost of rejection is not too high, rejection may be a desirable action.
Let
$$
\lambda\left(\alpha_{i} \mid w_{j}\right)=\left\{\begin{array}{ll}
0, & i=j \quad i, j=1, \ldots, c \\
\lambda_{r}, & i=c+1 \\
\lambda_{s}, & \text { otherwise }
\end{array}\right.
$$
where $\lambda_{r}$ is the loss incurred for choosing the $(c+1)$ the action, rejection, and $\lambda_{s}$ is the loss incurred for making a substitution error. Show that the minimum risk is obtained if we decide $w_{i}$ if $P\left(w_{i} \mid x\right) \geq P\left(w_{j} \mid x\right)$ for all $j$ and if $P\left(w_{i} \mid x\right) \geq 1-\frac{\lambda_{r}}{\lambda_{s}},$ and reject otherwise. What happens if $\lambda_{r}=0 ?$ And what happens if $\lambda_{r}>\lambda_{s} ?$

### Answer1:

(1 )对于$i=1,...,c$ 而言：
$$
\begin{aligned}
R\left(\alpha_{i} \mid \mathbf{x}\right) &=\sum_{j=1}^{c} \lambda\left(\alpha_{i} \mid \omega_{j}\right) P\left(\omega_{j} \mid \mathbf{x}\right) \\
&=\lambda_{s} \sum_{j=1, j \neq i}^{c} P\left(\omega_{j} \mid \mathbf{x}\right) \\
&=\lambda_{s}\left[1-P\left(\omega_{i} \mid \mathbf{x}\right)\right] \\
\end{aligned}
$$
对于$i=c+1$而言：
$$
R\left(\alpha_{c+1} \mid \mathbf{x}\right)=\lambda_{r}
$$
所以当选择$\omega_{i}$ 时，
$$
\begin{aligned}
R\left(\alpha_{i} \mid \mathbf{x}\right) &\leq R\left(\alpha_{c+1} \mid \mathbf{x}\right) \\
\\
\lambda s[1-P(w i \mid x)] & \leq \lambda r \\
1-P(w i \mid x) & \leq \frac{\lambda r}{\lambda s} \\
P(w i \mid x) & \geqslant 1-\frac{\lambda r}{\lambda s}
\end{aligned}
$$
因此当$R\left(\alpha_{i} \mid \mathbf{x}\right) \leq R\left(\alpha_{c+1} \mid \mathbf{x}\right)$即$P\left(\omega_{i} \mid \mathbf{x}\right) \geq 1-\frac{\lambda_{r}}{\lambda_{e}}$时选择$w i$ ，不满足则拒绝识别。

(2) 如果 $\lambda_{r}=0,$ 将一直拒识。

(3) 如果 $\lambda_{r}>\lambda_{s},$ 将永不拒识。



<div style="page-break-after: always;"></div>

## Question 2

Let $p\left(x \mid w_{i}\right) \sim \mathcal{N}\left(\mu_{i}, \sigma^{2}\right)$ for a two-category one-dimensional problem with $p\left(w_{1}\right)=p\left(w_{2}\right)=0.5$
(a) Show that the minimum probability of error is given by
$$
P_{e}=\frac{1}{\sqrt{2 \pi}} \int_{a}^{\infty} e^{-\frac{\mu^{2}}{2}} d \mu
$$
where $\alpha =\frac{\left|\mu_{1}-\mu_{2}\right|}{2 \sigma}$
(b) Use the inequality
$$
P_{e} \leq \frac{1}{\sqrt{2 \pi} a} e^{-\frac{a^{2}}{2}}
$$
to show that $P_{e}$ goes to zero as $\frac{\left|\mu_{1}-\mu_{2}\right|}{\sigma}$ goes to infinity.

### Answer2:

(a) 证明：

利用似然比计算决策面：对于一个二分类问题而言，如果$x$属于$w 1$ ，则似然比满足如下条件：
$$
\frac{p\left(x \mid \omega_{1}\right)}{p\left(x \mid \omega_{2}\right)}>\frac{P\left(\omega_{2}\right)}{P\left(\omega_{1}\right)}
$$
$\because P(w _1)=P(w _2)=1/2$  ，且 $$
P\left(x \mid \omega_{i}\right) \text { 服从 } N\left(\mu_{l}, \sigma^{2}\right), \text { 即 }
$$：
$$
P\left(x \mid \omega_{i}\right)=\frac{1}{\sqrt{2 \pi} \sigma} e^{-\frac{(x-\mu)^{2}}{2 \sigma}}
$$
则根据似然比决策，对于一个给定的特征$x$，判决为$w 1$ 时有：
$$
\begin{array}{c}
\frac{\frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2 \sigma^{2}}\left(x-\mu_{1}\right)^{2}}}{\frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2 \sigma^{2}}\left(x-\mu_{2}\right)^{2}}}>\frac{1 / 2}{1 / 2}

\\
{\frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2 \sigma^{2}}\left(x-\mu_{1}\right)^{2}}}>
\frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2 \sigma^{2}}\left(x-\mu_{2}\right)^{2}} 
\\
 
\left(x-\mu_{1}\right)^{2}>\left(x-\mu_{2}\right)^{2}\\
x^{2}-2 x \mu_{1}+\mu_{1}^{2}>x^{2}-2 x \mu_{2}+\mu_{2}^{2} \\
\left(\mu_{2}-\mu_{1}\right)\left(2 x-\left(\mu_{2}+\mu_{1}\right)\right)<0 \\
x<\frac{\left(\mu_{1}+\mu_{2}\right)}{2}, {x判决为 w _1} \\
x>\frac{\left(\mu_{1}+\mu_{2}\right)}{2}, {x判决为 w _2}
\end{array}
$$
则决策边界为：
$$
x = \frac {u _1 + u _2}{2}
$$
将二分类器划分为$R\ _1$  和$R\ _2$ 两个区域，则错误分类可能有以下两种形式出现：

1. 真实类别为$w _1$ 而被分为$R _2$  			2. 真实类别为$w _2$而被分为$R _1$

因此误差概率为：
$$
\begin{aligned}
P(\text {error}) &=P\left(\mathbf{x} \in \mathcal{R}_{2}, \omega_{1}\right)+P\left(\mathbf{x} \in \mathcal{R}_{1}, \omega_{2}\right) \\
&=P\left(\mathbf{x} \in \mathcal{R}_{2} \mid \omega_{1}\right) P\left(\omega_{1}\right)+P\left(\mathbf{x} \in \mathcal{R}_{1} \mid \omega_{2}\right) P\left(\omega_{2}\right) \\
&=\int_{\mathcal{R}_{2}} p\left(\mathbf{x} \mid \omega_{1}\right) P\left(\omega_{1}\right) d \mathbf{x}+\int_{\mathcal{R}_{1}} p\left(\mathbf{x} \mid \omega_{2}\right) P\left(\omega_{2}\right) d \mathbf{x}\\
&=\frac{1}{2} \cdot \frac{1}{\sqrt{2 \pi} \sigma}\left(\int_{\frac{\mu_{1}+\mu_{2}}{2}}^{\infty} e^{-\frac{\left(x-\mu_{1}\right)^{2}}{2 \sigma^{2}}} d x+\int_{-\infty}^{\frac{\mu_{1}+\mu_{2}}{2}} e^{-\frac{\left(x-\mu_{2}\right)^{2}}{2 \sigma^{2}}} d x\right)\\

& \text { 另 } \mu =\frac{x-\mu_{i}}{\sigma},i=1, 2\\
&将\frac{\mu _1+\mu _2}{2}带入\frac{x-\mu_{i}}{\sigma},i=1, 2, 则得新边界 \frac{\mu 1-\mu 2}{2\sigma }\\

&=\frac{1}{2} \cdot \frac{1}{\sqrt{2 \pi}}\left(\int_{\frac{\mu 1-\mu 2}{2\sigma }}^{\infty} e^{-\frac{u^{2}}{2}} d u+\int_{-\infty}^\frac{\mu 1-\mu 2}{2\sigma } e^{-\frac{u^{2}}{2}} d u\right)\\

&=\frac{1}{\sqrt{2 \pi}} \int_{|\frac{\mu_{2}-\mu_{1}}{2 \sigma}|}^{\infty} e^{-\frac{\mu ^{2}}{2}} d u\\
&=\frac{1}{\sqrt{2 \pi}} \int_{a}^{\infty} e^{-\frac{{\mu}^2}{2}} d u ,其中 \alpha =|\frac{\mu 2-\mu 1}{2 \sigma}|
\end{aligned}
$$
即，最小误差概率为：
$$
P_{e}=\frac{1}{\sqrt{2 \pi}} \int_{\alpha}^{\infty} e^{-\frac{\mu^{2}}{2}} d \mu\\
\alpha =\frac{\left|\mu_{1}-\mu_{2}\right|}{2 \sigma}
$$


(b) 证明：当$P _e$为0时，$|\frac{\mu 1+\mu 2}{\sigma}|$趋于无穷大
$$
\begin{aligned}
& \because P_{e}=\frac{1}{\sqrt{2 \pi}} \int_{\alpha}^{\infty} e^{-\frac{\mu^{2}}{2}} d \mu ，{且 }P_{e} \leq \frac{1}{\sqrt{2 \pi} a} e^{-\frac{a^{2}}{2}} \\
& \therefore P_{e}=\frac{1}{\sqrt{2 \pi}} \int_{\alpha}^{\infty} e^{-\frac{\mu^{2}}{2}} d \mu \leq \frac{1}{\sqrt{2 \pi} a} e^{-\frac{a^{2}}{2}}\\
& \because \lim _{a \rightarrow \infty} \frac{1}{a \sqrt{2 \pi}} e^{-\frac{1}{2} a^{2}} =0 \\ 
& \therefore  P_{e} =\frac{1}{\sqrt{2 \pi}} \int_{a}^{\infty} e^{-\frac{1}{2} a^{2}} d a \leq 0\\
& 此时积分无限趋近于0，则下界\alpha逼近于 \infty \\
& 当P _e为0时，|\frac{\mu 1+\mu 2}{\sigma}|趋于无穷大即为所证
\end{aligned}
$$

<div style="page-break-after: always;"></div>

## Question 3:

To classify a feature vector $x \in \mathcal{R}^{d}$ in a task of $c$ classes, we assume that for each class, **the prior is same** and the **class conditional probability density is a Gaussian distribution.**
(a) Write the mathematical form of the conditional probability density function

(b) Write the discriminant function of minimum error rate in the following two cases: (a) class covariance matrices are unequal;(b) class covariance matrices are same.

(c) For the quadratic discriminant function based on Gaussian probability density, it becomes incalculable when the covariance matrix is singular. Name two ways to overcome the singularity.

### Answer 3:

(a) 因为类条件概率密度服从高斯分布，所以条件概率密度函数的数学表达形式如下：
$$
p(\mathbf{x})=\frac{1}{(2 \pi)^{d / 2}|\mathbf{\Sigma}|^{1 / 2}} \exp \left[-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{t} \mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right]
$$
其中$x$是一个$d$维的列向量，$\mu $是$d$维的均值向量，$\Sigma $是$d\times d$的协方差矩阵，$|\Sigma|和\Sigma ^{-1}$分别是其行列式的值和逆，$(x-\mu)^t是(x-\mu)$的转置。



(b) 写出最小错误率的判别函数两种情况：（1）类协方差矩阵不相等；（2）类协方差矩阵相同。

最小误差概率判别函数可以简化为如下公式：
$$
g_{i}(x)=\ln p\left(x \mid \omega_{i}\right)+\ln p\left(\omega_{i}\right)
$$
$\because p(x|w _i)$是多元正态分布，则：
$$
g_{i}(x)=-\frac{1}{2}(x-\mu _i)^{t} \Sigma_{i}^{-1}(x-\mu _i)-\frac{d}{2} \ln 2 \pi-\frac{1}{2} \ln \left|\Sigma_{i}\right|+\ln P\left(\omega_{i}\right)
$$
**情况一：类协方差矩阵不相等**

$\frac{d}{2}\ln 2\pi$与$i$无关，是不关紧要的附加常量，可以被省略。因此可以将判别函数简化为如下形式:
$$
g_{i}(x)=-\frac{1}{2}(x-\mu _i)^{t} \Sigma_{i}^{-1}(x-\mu _i)-\frac{1}{2} \ln \left|\Sigma_{i}\right|+\ln P\left(\omega_{i}\right)
$$

$$
\begin{array}\\
令：\\
\mathbf{W}_{i}=-\frac{1}{2} \mathbf{\Sigma}_{i}^{-1}\\
\mathbf{w}_{i}=\mathbf{\Sigma}_{i}^{-1} \boldsymbol{\mu}_{i} \\
w_{i 0}=-\frac{1}{2} \boldsymbol{\mu}_{i}^{\prime} \mathbf{\Sigma}_{i}^{-1} \boldsymbol{\mu}_{i}-\frac{1}{2} \ln \left|\mathbf{\Sigma}_{i}\right|+\ln P\left(\omega_{i}\right)\\
则：\\
g_{i}(x)=x^{t} W_i x+w^{t}_i x+w_{i 0}
 \end{array}
$$

**情况二：类协方差相同**

$|\Sigma _i|和(d/2)\ln 2\pi$两项与$i$无关，是不关紧要的附加常量，可以被省略。且根据题目可知所有 c 类别的先验概率 $P\left(\omega_{i}\right)$ 都相同,那么 $\ln P\left(\omega_{i}\right)$ 项也可被省略。则判别函数可以被简化为如下形式：
$$
g_{i}(\mathbf{x})=-\frac{1}{2}\left(\mathbf{x}-\boldsymbol{\mu}_{i}\right)^{\prime} \mathbf{\Sigma}^{-1}\left(\mathbf{x}-\boldsymbol{\mu}_{i}\right)
$$

$$
\begin{array}\\
令：\\
\mathbf{w}_{i}=\mathbf{\Sigma}_{i}^{-1} \boldsymbol{\mu}_{i} \\
w_{i 0}=-\frac{1}{2} \boldsymbol{\mu}_{i}^{\prime} \mathbf{\Sigma}_{i}^{-1} \boldsymbol{\mu}_{i}\\
则：\\
g_{i}(\mathbf{x})=\mathbf{w}_{i}^{t} \mathbf{x}+w_{i 0}
 \end{array}
$$



(c)  基于高斯概率的二次判别函数：当协方差矩阵为奇异时，它变得无法计算，说出两种克服奇异性的方法。

克服奇异性的方法:

1. 降低维度，剔除特征为0得数据
2. 求矩阵得伪逆矩
3. 奇异值分解

<div style="page-break-after: always;"></div>

## Question 4:

Suppose we have two normal distributions with **the same covariance** but **different means**: $\mathcal{N}\left(\mu_{1}, \Sigma\right)$ and $\mathcal{N}\left(\mu_{2}, \Sigma\right) .$ In terms of their prior probabilities $P\left(w_{1}\right)$ and $P\left(w_{2}\right),$ state the condition that Bayes decision boundary does not pass between the two means.

### Answer 4:

因为两个分布均为高斯分布，根据贝叶斯决策该问题的边界是线性的，决策面的位置为：
$$
\begin{aligned}
\mathbf{w}^t&=\Sigma^{-1}(\mu_1-\mu2)\\
其中：\\
\mathbf{w} &=\mathbf{\Sigma}^{-1}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right) \\
\mathbf{x}_{o} &=\frac{1}{2}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)-\frac{\ln \left[P\left(\omega_{1}\right) / P\left(\omega_{2}\right)\right]}{\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)^{t} \mathbf{\Sigma}^{-1}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)
\end{aligned}
$$
贝叶斯决策边界不经过均值$\mu_1,\mu_2$，可以理解为决策面同向。即$\mathbf{w}^{t}\left(\boldsymbol{\mu}_{1}-\mathbf{x}_{o}\right)$ 和 $\mathbf{w}^{t}\left(\boldsymbol{\mu}_{2}-\mathbf{x}_{o}\right)$ 同方向：
$$
\mathbf{w}^{t}\left(\boldsymbol{\mu}_{1}-\mathbf{x}_{o}\right)>0 \quad \text { and } \quad \mathbf{w}^{t}\left(\boldsymbol{\mu}_{2}-\mathbf{x}_{o}\right)>0
$$
或者:
$$
\mathbf{w}^{t}\left(\boldsymbol{\mu}_{1}-\mathbf{x}_{o}\right)<0 \quad \text { and } \quad \mathbf{w}^{t}\left(\boldsymbol{\mu}_{2}-\mathbf{x}_{o}\right)<0
$$
将$\mathbf{w}$和$\mathbf{x}_{0}$带入,则可以将条件转变为如下：
$$
\begin{aligned}
\mathbf{w}^{t}\left(\boldsymbol{\mu}_{1}-\mathbf{x}_{o}\right) &=\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)^{t} \mathbf{\Sigma}^{-1}\left(\boldsymbol{\mu}_{1}-\frac{1}{2}\left(\boldsymbol{\mu}_{1}+\boldsymbol{\mu}_{2}\right)\right)-\ln \left[\frac{P\left(\omega_{1}\right)}{P\left(\omega_{2}\right)}\right] \\
&=\frac{1}{2}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)^{t} \mathbf{\Sigma}^{-1}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)-\ln \left[\frac{P\left(\omega_{1}\right)}{P\left(\omega_{2}\right)}\right] \\
\mathbf{w}^{t}\left(\boldsymbol{\mu}_{2}-\mathbf{x}_{0}\right) &=\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)^{t} \mathbf{\Sigma}^{-1}\left(\boldsymbol{\mu}_{1}-\frac{1}{2}\left(\boldsymbol{\mu}_{1}+\boldsymbol{\mu}_{2}\right)\right)-\ln \left[\frac{P\left(\omega_{1}\right)}{P\left(\omega_{2}\right)}\right] \\
&=-\frac{1}{2}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)^{t} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)-\ln \left[\frac{P\left(\omega_{1}\right)}{P\left(\omega_{2}\right)}\right]
\end{aligned}
$$
当属于$\mathbf{w}^{t}\left(\boldsymbol{\mu}_{1}-\mathbf{x}_{0}\right)>0 \text { and } \mathbf{w}^{t}\left(\boldsymbol{\mu}_{2}-\mathbf{x}_{o}\right)>0$的情况时：
$$
\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)^{t} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)>2 \ln \left[\frac{P\left(\omega_{1}\right)}{P\left(\omega_{2}\right)}\right]
$$

$$
\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)^{t} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)<-2 \ln \left[\frac{P\left(\omega_{1}\right)}{P\left(\omega_{2}\right)}\right]
$$

当属于$\mathbf{w}^{t}\left(\boldsymbol{\mu}_{1}-\mathbf{x}_{o}\right)<0 \text { and } \mathbf{w}^{t}\left(\boldsymbol{\mu}_{2}-\mathbf{x}_{o}\right)<0$的情况时：
$$
\begin{array}{c}
\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)^{t} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)<2 \ln \left[\frac{P\left(\omega_{1}\right)}{P\left(\omega_{2}\right)}\right] \text { and } \\
\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)^{t} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)>-2 \ln \left[\frac{P\left(\omega_{1}\right)}{P\left(\omega_{2}\right)}\right]
\end{array}
$$
根据上述两种情况，要使Bayes判定边界不在两个平均值之间通过的条件可以表述如下：
**情况1：**

 $P\left(\omega_{1}\right) \leq P\left(\omega_{2}\right)$时：

当 $\left(\mu_{1}-\mu_{2}\right)^{t} \Sigma^{-1}\left(\mu_{1}-\mu_{2}\right)<2 \ln \left[\frac{P\left(\omega_{1}\right)}{P\left(\omega_{2}\right)}\right]$ ， $\mathbf{w}^{t}\left(\boldsymbol{\mu}_{1}-\mathbf{x}_{o}\right)>0$ 且$\mathbf{w}^{t}\left(\boldsymbol{\mu}_{2}-\mathbf{x}_{o}\right)>0$时成立。

**情况2：**

$P\left(\omega_{1}\right)>P\left(\omega_{2}\right) .$时：

当$\left(\mu_{1}-\mu_{2}\right)^{t} \Sigma^{-1}\left(\mu_{1}-\mu_{2}\right)<2 \ln \left[\frac{P\left(\omega_{1}\right)}{P\left(\omega_{2}\right)}\right]$ ， $\mathbf{w}^{t}\left(\boldsymbol{\mu}_{1}-\mathbf{x}_{o}\right)<0$ 且 $\mathbf{w}^{t}\left(\boldsymbol{\mu}_{2}-\mathbf{x}_{o}\right)<0$时成立。

<div style="page-break-after: always;"></div>

## Question 5:

Maximum likelihood methods apply to estimate of prior probability as well. Let samples be drawn by successive, independent selections of a state of nature $w_{i}$ with unknown probability $P\left(w_{i}\right) .$ Let $z_{i k}=1$ if the state of nature for the $k$ -th sample is $w_{i}$ and $z_{i k}=0$ otherwise.
(a) Show that
$$
P\left(z_{i 1}, \ldots, z_{i n} \mid P\left(w_{i}\right)\right)=\prod_{k=1}^{n} P\left(w_{i}\right)^{z_{i k}}\left(1-P\left(w_{i}\right)\right)^{1-z_{i k}}
$$
(b) Show that the maximum likelihood estimate for $P\left(w_{i}\right)$ is
$$
\hat{P}\left(w_{i}\right)=\frac{1}{n} \sum_{k=1}^{n} z_{i k}
$$

### Answer 5：

(a)证： 

因为假设样本是连续独立地从自然状态$w_i$中抽取的，每一个自然状态的概率为$P(w_i)$，则：
$$
z_{i k}=\left\{\begin{array}{ll}
1 & \text {第 } k^{t h} \text {个样本的自然状态为 } \omega_{i} \\
0 & \text { 否则 }
\end{array}\right.
$$
通过以概率$P(\omega _i)$连续选择自然状态$\omega_i$来绘制样本,则
$$
\operatorname{Pr}\left[z_{i k}=1 \mid P\left(\omega_{i}\right)\right]=P\left(\omega_{i}\right)
$$
$$
\operatorname{Pr}\left[z_{i k}=0 \mid P\left(\omega_{i}\right)\right]=1-P\left(\omega_{i}\right)
$$
这两个方程可以统一为：
$$
P\left(z_{i k} \mid P\left(\omega_{i}\right)\right)=\left[P\left(\omega_{i}\right)\right]^{z_{i k}}\left[1-P\left(\omega_{i}\right)\right]^{1-z_{i k}}
$$
根据最大似然估计的基本原理得：
$$
\begin{aligned}
P\left(z_{i 1}, \cdots, z_{i n} \mid P\left(\omega_{i}\right)\right) &=\prod_{k=1}^{n} P\left(z_{i k} \mid P\left(\omega_{i}\right)\right) \\
&=\prod_{k=1}^{n}\left[P\left(\omega_{i}\right)\right]^{z_{i k}}\left[1-P\left(\omega_{i}\right)\right]^{1-z_{i k}}
\end{aligned}
$$


(b)证： 

 $P\left(\omega_{i}\right)$ 的对数似然函数为：
$$
\begin{aligned}
l\left(P\left(\omega_{i}\right)\right) &=\ln P\left(z_{i 1}, \cdots, z_{i n} \mid P\left(\omega_{i}\right)\right) \\
&=\ln \left[\prod_{k=1}^{n}\left[P\left(\omega_{i}\right)\right]^{z_{i k}}\left[1-P\left(\omega_{i}\right)\right]^{1-z_{i k}}\right] \\
&=\sum_{k=1}^{n}\left[z_{i k} \ln P\left(\omega_{i}\right)+\left(1-z_{i k}\right) \ln \left(1-P\left(\omega_{i}\right)\right)\right]
\end{aligned}
$$
对上面式子关于$P(w_i)$求导，得到了一组求解最大似然估计值$P(w_i)$的必要条件：
$$
\nabla_{P\left(\omega_{i}\right)} l\left(P\left(\omega_{i}\right)\right)=\frac{1}{P\left(\omega_{i}\right)} \sum_{k=1}^{n} z_{i k}-\frac{1}{1-P\left(\omega_{i}\right)} \sum_{k=1}^{n}\left(1-z_{i k}\right)=0
$$
求解上式：
$$
\left(1-\hat{P}\left(\omega_{i}\right)\right) \sum_{k=1}^{n} z_{i k}=\hat{P}\left(\omega_{i}\right) \sum_{k=1}^{n}\left(1-z_{i k}\right)
$$
$$
\sum_{k=1}^{n} z_{i k}=\hat{P}\left(\omega_{i}\right) \sum_{k=1}^{n} z_{i k}+n \hat{P}\left(\omega_{i}\right)-\hat{P}\left(\omega_{i}\right) \sum_{k=1}^{n} z_{i k}\\
\sum_{k=1}^{n} z_{i k}=n \hat{P}\left(\omega_{i}\right)
$$
$$
\hat{P}\left(\omega_{i}\right)=\frac{1}{n} \sum_{k=1}^{n} z_{i k}
$$
即为所证。

<div style="page-break-after: always;"></div>

## Question 6:

Let the sample mean $\hat{\mu}_{n}$ and the sample covariance matrix $C_{n}$ for a set of $n d$ -dimensional samples $x_{1}, \ldots, x_{n}$ be defined by
$$
\hat{\mu}_{n}=\frac{1}{n} \sum_{k=1}^{n} x_{k}, \quad C_{n}=\frac{1}{n-1} \sum_{k=1}^{n}\left(x_{k}-\hat{\mu}_{n}\right)\left(x_{k}-\hat{\mu}_{n}\right)^{T}
$$
(a) Show that alternative, recursive techniques for calculating $\hat{\mu}_{n}$ and $C_{n}$ based on the successive addition of new samples $x_{n+1}$ can be derived using the recursion relations
$$
\hat{\mu}_{n+1}=\hat{\mu}_{n}+\frac{1}{n+1}\left(x_{n+1}-\hat{\mu}_{n}\right)
$$
and
$$
C_{n+1}=\frac{n-1}{n} C_{n}+\frac{1}{n+1}\left(x_{n+1}-\hat{\mu}_{n}\right)\left(x_{n+1}-\hat{\mu}_{n}\right)^{T}
$$
(b) Discuss the computational complexity of finding $\hat{\mu}_{n}$ and $C_{n}$ by the recursive methods.



### Answer 6:

(a):

1. 由题目可知:

$$
\hat{\mu}_{n}=\frac{1}{n} \sum_{k=1}^{n} x_{k}
$$

则对于新样本$x_{n+1}$而言：
$$
\begin{aligned}
\mu_{n+1}^{\wedge} &=\frac{1}{n+1} \sum_{k=1}^{n+1} x_{k} \\
&=\frac{1}{n+1}\left(\sum_{k=1}^{n} x_{k}+x_{n+1}\right) \\
&=\frac{1}{n+1}\left(n \mu_{n}+x_{n+1}\right) \\
&=\frac{n}{n+1} \mu_{n}+\frac{1}{n+1} x_{n+1} \\
&=\frac{(n+1)-1}{n+1} \mu_{n}+\frac{1}{n+1} x_{n+1} \\
&=\mu_{n}+\frac{1}{n+1}\left(x_{n+1}-\mu_{n}\right)
\end{aligned}
$$
即可以基于新样本 $x_{n+1}$ 连续相加， 递归计算出 $\widehat{\mu}_{n+1}$。

2.根据题意可知：
$$
\quad C_{n}=\frac{1}{n-1} \sum_{k=1}^{n}\left(x_{k}-\hat{\mu}_{n}\right)\left(x_{k}-\hat{\mu}_{n}\right)^{T}
$$
则对于新样本$x_{n+1}$而言：
$$
\begin{aligned}
C_{n+1} &=\frac{1}{n} \sum_{k=1}^{n+1}\left(x_{k}-u_{n+1}^{\wedge}\right)\left(x_{k}-u_{n+1}^{\wedge}\right)^{T} \\
=& \frac{1}{n} \sum_{k=1}^{n+1}\left[x_{k}-\hat{u}_{n}-\frac{1}{n+1}\left(x_{n+1}-\hat{u}_{n}\right)\right]\left[x_{k}-\hat{u}_{n}-\frac{1}{n+1}\left(x_{n+1}-u_{n}\right)\right]^{T} \\

=& \frac{1}{n} \sum_{k=1}^{n}\left(x_{k}-\hat{u}_{n}\right)\left(x_{k}-\hat{u}_{n}\right)^{T}-\frac{1}{n(n+1)} \sum_{k=1}^{n}\left(x_{n+1}-\hat{u}_{n}\right)^{T}\left(x_{k}-\hat{u}_{n}\right)-\frac{1}{n(n+1)} \sum_{k=1}^{n}\left(x_{n+1}-\hat{u}_{n}\right)\left(x_{k}-\hat{u}_{n}\right)^{T} \\
&+\frac{1}{n(n+1)^{2}} \sum_{k=1}^{n}\left(x_{n+1}-\hat{u}_{n}\right)\left(x_{n+1}-\hat{u}_{n}\right)^{T}+\frac{n}{(n+1)^{2}}\left(x_{n+1}-\hat{u}_{n}\right)\left(x_{n+1}-\hat{u}_{n}\right)^{T} \\
\\
=& \frac{n-1}{n} C_{n}-\frac{1}{n(n+1)}\left[\sum_{k=1}^{n}\left(x_{n+1}-\hat{u}_{n}\right)^{T}\left(x_{k}-\hat{u}_{n}\right)-\sum_{k=1}^{n}\left(x_{n+1}-\hat{u}_{n}\right)\left(x_{k}-\hat{u}_{n}\right)^{T}\right]+\\
& \frac{1}{n+1}\left(x_{n+1}-\hat{u}_{n}\right)\left(x_{n+1}-\hat{u}_{n}\right)^{T} \\
=& \frac{n-1}{n} C_{n}-\frac{1}{n(n+1)}\left[\left(x_{n+1}-\hat{u}_{n}\right)^{T}\left(n u_{n}-n \hat{u}_{n}\right)-\sum_{k=1}^{n}\left(x_{n+1}-u_{n}\right)\left(n u_{n}-n \hat{u}_{n}\right)^{T}\right]+\\
& \frac{1}{n+1}\left(x_{n+1}-\hat{u}_{n}\right)\left(x_{n+1}-u_{n}\right)^{T} \\
=& \frac{n-1}{n} C_{n}+\frac{1}{n+1}\left(x_{n+1}-\hat{u}_{n}\right)\left(x_{n+1}-\hat{u}_{n}\right)^{T}
\end{aligned}
$$
即可以基于新样本 $x_{n+1}$ 连续相加， 递归计算出 $C_{n+1}$。



(b) 因为由已知条件可知：
$$
\hat{\mu}_{n}=\frac{1}{n} \sum_{k=1}^{n} x_{k}
$$
所以 $\hat{\mu}_{n}$ 的计算复杂度为 $O(n d)$
 因为由已知条件可知：
$$
C_{n}=\frac{1}{n-1} \sum_{k=1}^{n}\left(x_{k}-\hat{u}_{n}\right)\left(x_{k}-\hat{u}_{n}\right)^{T}
$$
所以 $\hat{\mu}_{n}$ 的计算复杂度为 $O\left(n d^{2}\right)$