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

### Question 3:

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

克服奇异性的方法: SVD 分解（奇异值分解）; QR 分解; 求伪逆



## Question 4:

Suppose we have two normal distributions with **the same covariance** but **different means**: $\mathcal{N}\left(\mu_{1}, \Sigma\right)$ and $\mathcal{N}\left(\mu_{2}, \Sigma\right) .$ In terms of their prior probabilities $P\left(w_{1}\right)$ and $P\left(w_{2}\right),$ state the condition that Bayes decision boundary does not pass between the two means.