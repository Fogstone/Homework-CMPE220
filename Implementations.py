import matplotlib.pyplot as plt


labels = ['Native', 'BLAS', 'MKL', 'CUDA']
values = [1122.143, 27314.150, 78663.37, 542600776.20]


plt.figure(figsize=(10,6))
plt.bar(labels, values, color=['blue', 'green', 'orange', 'red'])


plt.xlabel('Library')
plt.ylabel('GFlops')
plt.title('Execution Performance Comparison in Flops')
plt.yscale('log')  

plt.show()
