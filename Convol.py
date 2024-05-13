import numpy as np
from time import time

def custom_convolution_1d_numpy(array, mask):
  pad_width = len(mask) // 2
  padded_array = np.pad(array, pad_width, mode='constant')
  result = np.convolve(padded_array, mask, mode='valid')
  return result

def custom_benchmark_numpy(array, mask):
  start_time = time()
  result = custom_convolution_1d_numpy(array.copy(), mask.copy())
  end_time = time()
  execution_time = end_time - start_time
  flops = len(array) * len(mask)
  gflops = flops * 1e-9
  print("Custom NumPy Convolution Benchmark:")
  print("Execution Time (s):", execution_time)
  print("GFLOPs:", gflops)

array_size = 1 << 20
mask_size = 7
input_array = np.random.randint(0, 100, size=array_size)
custom_mask = np.random.randint(0, 10, size=mask_size)

result = custom_convolution_1d_numpy(input_array, custom_mask)
custom_benchmark_numpy(input_array, custom_mask)
