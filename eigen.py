import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# setup for figures
# fig, ax = plt.subplots()
# fig.set_size_inches(5, 5)
# sns.set()

# sample from multivariate normal
true_covMat = [[2,1],[1,1]]
x = np.random.multivariate_normal(mean=[0,0], cov=true_covMat, size=3000)

# compute statistics
meanVec = np.mean(x,axis=0)
centralized_x = x - meanVec
sample_covMat = np.matmul(centralized_x.transpose(), centralized_x)

# eigendecomposition
u,s,vh = np.linalg.svd(sample_covMat)
