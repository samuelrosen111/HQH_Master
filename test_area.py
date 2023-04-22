import numpy as np


lambda_param = 1
for i in range(0,100):
    print(np.random.exponential(scale=1/lambda_param))


