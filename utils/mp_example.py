import multiprocessing as mp
import random
import string
import time

random.seed(123)

# Define an output queue
output = mp.Queue()

# define a example function
def cube(x):
  time.sleep(1)
  return (x, x**3)

pool = mp.Pool(processes=8)

res = []
for i in range(20):
  res.append(pool.apply_async(cube, args=(i,)))


for p in res:
  print (p.get())