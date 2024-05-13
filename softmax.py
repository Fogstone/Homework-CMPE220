import tensorflow as tf
from timeit import default_timer

def custom_softmax(logits):
  return tf.nn.softmax(logits)

def custom_benchmark_softmax(batch_size, input_size):
  data = tf.random.normal(shape=(batch_size, input_size))
  custom_softmax(data)  # Warm-up run

  start_time = default_timer()

  for _ in range(10):
    _ = custom_softmax(data)

  end_time = default_timer()

  execution_time = (end_time - start_time) / 10

  flops = 2 * input_size * batch_size + input_size * batch_size * (tf.math.log(tf.cast(batch_size, tf.float32)) + 1)

  gflops = flops / (execution_time * 1e9)

  return execution_time, gflops

custom_batch_size = 1048576
custom_input_size = 100

custom_execution_time, custom_gflops = custom_benchmark_softmax(custom_batch_size, custom_input_size)

print("Custom Execution Time (s):", custom_execution_time)
print("Custom GFLOPs:", custom_gflops.numpy())
