import matplotlib.pyplot as plt


labels = ['TensorFlow Softmax', 'C++ Softmax', 'TensorFlow Convolution 1D', 'C++ Convolution 1D']
gflops_values = [1.13493, 2.3231266, 7.19964, 3.007340]  


plt.figure(figsize=(10,6))
plt.bar(labels, gflops_values, color=['blue', 'orange', 'green', 'red'])


plt.xlabel('Operation')
plt.ylabel('GFLOPs')
plt.title('GFLOPs Comparison: Softmax and CNN 1D')

plt.show()
