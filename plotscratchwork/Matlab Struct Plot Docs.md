# Matlab Struct Plot Docs



## Phase 1: Data Conversion/Cleaning

With Matlab, the .mat can easily be converted into a CSV that is non-framework specific and leaves it open for use in not just numpy, buit any other software

**If we need to automate this script, this might be able to be run on the command line or via a shell script**

Run this in Matlab to generate a CSV:

~~~matlab
%import .mat file, change this to match your filename
load 'data.mat'

%init a table, change "Ad_3" to match the name of your struct
T = struct2table(AD3_3,'AsArray',true)

% Get rid of unecessary rows
T = removevars(T, {'xscale', 'yscale', 'zscale', 'plot', 'UserData', 'note', 'timeStamp', 'holdUpdates', 'needsReplot'})

% Write to a CSV
writetable(T, 'data_export.csv')
~~~

Lingering Question: Can I make this executable in Python? Or can I at least make a shell script for this?

## Phase 2: Numpy array conversion using pandas and Plot

~~~ python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(pd)
print(np)

df = pd.read_csv('trimmed.csv', sep=',')

npy = df.to_numpy()

# Remove pandas weird subindexing
npy = npy[0]

print("Len of npy " + str(len(npy)))

#Make a matching x-axis. This will jump it in increments of 1*10^-5 but I don't know if this is correct since the struct doesn't seem to carry the data with it
nums = np.arange(0, len(npy)*1.0e-5, 1.0e-5)

print(npy)
print(nums)

plt.plot(nums, npy)
plt.ylabel('My Plot')
plt.show()
~~~



Lingering Questons: How can I get time for the matlab struct? 

