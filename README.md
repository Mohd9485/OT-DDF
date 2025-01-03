# Optimal Transport Data-Driven Filters (OT-DDF)

This repository is by Mohammad Al-Jarrah, [Bamdad Hosseini](https://bamdadhosseini.org/), [Amirhossein Taghvaei](https://www.aa.washington.edu/facultyfinder/amir-taghvaei) and contains the Pytorch source code to reproduce the experiments in our 2024 paper [Data-Driven Approximation of Stationary Nonlinear Filters with Optimal Transport Maps](https://arxiv.org/abs/2403.15630). 

To illustrate the proposed OT-DDF algorithm in comparison with three other filters: the Ensemble Kalman Filter (EnKF),
Optimal transport particle filter (OTPF), and the sequential importance resampling (SIR) PF, we use the following two examples:

## Linear dynamics with linear and quadratic observation models
Consider

$$
\begin{aligned}
        X_{t} &= \begin{bmatrix}
        \alpha & \sqrt{1-\alpha^2}
        \\
        -\sqrt{1-\alpha^2} & \alpha
    \end{bmatrix}
    X_{t-1} + \sigma V_t\\
    Y_t &= h(X_t) + \sigma W_t
\end{aligned}
$$

for $t=1,2,\dots$ where $X_t\in \mathbb{R}^2,~ Y_t \in \mathbb{R},~ V_t$ and $W_t$ are i.i.d sequences of $2$-dimensional and one-dimensional standard Gaussian random variables, $\alpha=0.9$ and $\sigma^2=0.1$. Two observation functions are of interest:

$$
\begin{aligned}
    h(X_t)=X_t(1), \quad  h(X_t)=X_t(1)^2
\end{aligned}
$$

where $X_t(1)$ is the first component of the vector $X_t$. We refer to these observation models as linear and quadratic, respectively.

The following figure is related to the linear observation model $h(X_t)=X_t(1)$, where the right figures are averaged over 50 independent simulations:
<p align="center">
<img src="/images/X.png" width="1000" height="300">
</p>

The next figure is related to the quadratic observation model $h(X_t)=X_t(1)^2$, where the right figures are averaged over 10 independent simulations:
<p align="center">
<img src="/images/XX.png" width="1000" height="300">
</p>

## Lorenz 63
$$
\begin{aligned}
\begin{bmatrix}
    \dot{X}(1) \\ 
    \dot{X}(2) \\
    \dot{X}(3)
\end{bmatrix}
&= 
\begin{bmatrix}
    \sigma (X(2) - X(1)) \\
    X(1) (\rho - X(3)) - X(2) \\
    X(1)X(2) - \beta X(3)   
\end{bmatrix},\quad X_0 \sim \mathcal{N}(\mu_0,\sigma_0^2I_3),
\\
Y_t &= X_t(1) + \sigma_{obs}W_t,
\end{aligned}
$$

where $[X(1),X(2),X(3)]^\top$ are the variables representing the hidden states of the system, and $\sigma$, $\rho$, and $\beta$ are the model parameters. We choose $\sigma=10$, $\rho=28$, $\beta=8/3$, $\mu_0 = [0,0,0]^\top$, and $\sigma_{0}^2=10$. The observed noise $W$ is a $1$-dimensional standard Gaussian random variable with $\sigma_{obs}^2=10$.

The following figure is related to the Lorenz 63 example, where the right figures are averaged over 10 independent simulations:
<p align="center">
<img src="/images/L63.png" width="1000" height="300">
</p>



Please take a look at the paper for more details on this example. Also, please consider citing our paper if you find this repository useful for your publication.

```
@article{al2024data,
  title={Data-Driven Approximation of Stationary Nonlinear Filters with Optimal Transport Maps},
  author={Al-Jarrah, Mohammad and Hosseini, Bamdad and Taghvaei, Amirhossein},
  journal={arXiv preprint arXiv:2403.15630},
  year={2024}
}
```

## Setup
* Python/Numpy
* PyTorch

## Running the code and Regenerating data and figures.
1. Run the 'main.py' file to regenerate and save the date. The number of independent simulations 'AVG_SIM' is set to 1 but in the figures above we set it to 50,10,10 for the three figures, respectively. 
2. Use the file 'import_DATA.py' to import and plot all the desired figures.

Note: Unfortunately, we ran a random seed every time we ran the code, so we do not have a seed function to provide identical results to our paper, but the figure should be close enough.

