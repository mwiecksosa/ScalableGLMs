# ScalableGLMs
# Reproduction of Scalable Approximations for Generalized Linear Models by Erdogdu et al. (2019) published in JMLR 2019

In practice, empirical risk is often minimized using an iterative stochastic optimization algorithm. However, when the number of observations grows much
larger than the dimension of the feature space, these iterative algorithms become intractable. In summary.pdf, I give an overview of the Mathematical derivation and experimental results of a scalable algorithm that approximates the population risk minimizer in generalized linear problems introduced by Erdogdu et al. (2019). I discuss three main topics: 
- First, I will show that the true minimizer of the population risk is approximately proportional to
the ordinary least squares estimator. 
- Second, I will offer a limited experimental comparison of
the proposed algorithm with a batch gradient descent algorithm. The experimental results are
mostly consistent with the results of Erdogdu et al. (2019), which claim that the accuracy of the
proposed algorithm is the same as the empirical risk minimizer. 
- Lastly, and most interestingly, by analyzing the proposed scalable algorithm, we see it is computationally cheaper than classical
batch methods such as gradient descent at least a factor of O(p).
