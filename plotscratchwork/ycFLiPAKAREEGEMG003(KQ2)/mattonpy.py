import scipy.io
import numpy

#Change filename here
data = scipy.io.loadmat('Ad3_3.mat')
R = numpy.array(data)




if len(data) > 0:
    filename = "export.npy"

    numpy.save(filename, R)

    print("Success! file is expored as " + filename)
else:
    print("Error with conversion")
