import numpy as np
import os
import sys

def prntnpy(path):
	data = np.load(path)

	print("# of Entries: " + str(len(data)) + "\n")

	np.savetxt("foo.csv", data, delimiter=",")

	# Or Either save it to the text file or something.

if __name__ == "__main__":
	args = sys.argv
	prntnpy(args[1])
