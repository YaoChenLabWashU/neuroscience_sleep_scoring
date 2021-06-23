import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(pd)
print(np)

df = pd.read_csv('data_export.csv', sep=',')
#Short version to speed up debugging
#df = pd.read_csv('trimmed.csv', sep=',')


npy = df.to_numpy()

# Remove pandas weird subindexing
npy = npy[0]

print("Len of npy " + str(len(npy)))

nums = np.arange(0, len(npy)*(1/400), 1/400)

print(npy)
print(nums)

plt.figure(1)
plt.plot(nums, npy)
plt.ylabel('My Plot')

# Load npy plot too

data2 = np.load("downsampEMG_Acq3.npy")


time = np.arange(0, len(data2)*(1/400), (1/400))

print("data2: length of " + str(len(data2)) )
print(data2)

plt.figure(2)
plt.plot(time, data2)
plt.ylabel('My Plot')

plt.show()


