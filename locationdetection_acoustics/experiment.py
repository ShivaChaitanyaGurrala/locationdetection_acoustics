import numpy as np
import matplotlib.pyplot as plt
data = np.load("/home/shiva/Downloads/X_test.npy")

x = data[0]
plt.imsave("test_data.png",x)