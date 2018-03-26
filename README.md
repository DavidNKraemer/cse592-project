This project seeks to implement Stochastic damped L-BFGS and SdLBFGS with
variance reduction as described in Wang, et al (2017) in TensorFlow, a modern
graph-based deep learning framework. Our goal is to proceed with the following
steps:

[Link.](https://epubs.siam.org/doi/abs/10.1137/15M1053141)

1. Implementing SdLBFGS(-VR) in simple Python, to understand the mechanics of the
algorithm,
2. Implementing SdLBFGS(-VR) in TensorFlow and prepare a pull request,
3. Performing empirical tests of the performance of SdLBFGS(-VR) against other
minimization techniques available in TensorFlow, such as stochastic gradient
descent.
