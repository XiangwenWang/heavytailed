# heavy_tailed
Perform distribution analysis on heavy-tailed distributed data

Raw data are expected to be **positive integers**. Maximum likelihood estimation (**MLE**) will be performed to fit the following models to the data:
 
* Exponential distribution
<!---$$P(x) \sim e^{-\lambda x}$$-->  
* Power-law distribution
<!---$$P(x) \sim x^{-\alpha}$$-->  
* Power-law distribution with exponential cutoff
<!---$$P(x) \sim x^{-\alpha}e^{-\lambda x}$$-->  
* Pairwise power-law distribution
<!---$$P(x) \sim \left\lbrace \begin{split} &x^{-\alpha}, \quad x <  x_\text{trans} \\
&x^{-\beta}, \quad x_\text{trans} \le x
\end{split}\right.$$-->   
* Poisson distribution
<!---$$P(x) \sim \mu^x / x!$$-->  
* Yule–Simon distribution
<!---$$P(x) \sim \Gamma(x) / \Gamma(x + \alpha)$$-->  
* Lognormal distribution
<!---$$\ln(x) \sim N(\mu, \sigma^2)$$-->  
* Truncated lognormal distribution
<!---$$\ln(x) \sim N(\mu, \sigma^2)$$ for $$x\le x_{m}$$-->  
* Shifted power-distribution with exponential cutoff
<!---$$P(x) \sim \frac{(x-\delta)^{-\alpha}}{\displaystyle 1+e^{\lambda (x-\beta)}}$$-->  
* Truncated shifted power-law distribution
<!---$$P(x) \sim \left\lbrace \begin{split}
&(\ x - \delta )^{-\alpha},\ & x < x_\text{max}, \\
&\zeta(\alpha,\ m_\text{max} -\delta),\ & x = x_\text{max},
\end{split}\right.$$-->  

<!---In above formulas, the normalization factors and the condition $$x\ge x_\text{min}$$ are omitted.-->

An optimizer based on sequential least squares programming (SLSQP) is applied to maximize the likelihood function. (Initially, it was based on L-BFGS-B, but L-BFGS-B cannot handle inequality constraints, which are used to avoid overflow.)

The model with minimum **AIC** (or say the largest Akaike weight) will be selected as the best-fitted model.

The analysis mainly focuses on the tails, and the start of the tail will be determined through minimizing the **K-S distance** between fitted models and the empirical distribution.

Example Usage:  
```python
from heavy_tailed import compare
compare.comparison('testdata/raw_25_bets.dat', xmin=25)
```

The MLE could be a non-convex function, therefore it is suggested to try different initial values (for distribution parameters) to avoid local minima.