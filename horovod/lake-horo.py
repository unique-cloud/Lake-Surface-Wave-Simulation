#!/bin/python

#Import libraries for simulation
import tensorflow as tf
import numpy as np
import horovod.tensorflow as hvd
import sys
import time

#Imports for visualization
import PIL.Image

def print_heatmap(filename, u):
    f = open(filename, mode='w')
    for x in range(u.shape[0]):
        for y in range(u.shape[1]):
            f.write("{:8f} {:8f} {:8f}\n".format(float(y)/u.shape[0], 1 - float(x)/u.shape[1], u[x, y]))
    f.close()

def DisplayArray(filename, a, fmt='png', rng=[0,1]):
  """Display an array as a picture."""
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  with open(filename,"w") as f:
      PIL.Image.fromarray(a).save(f, "png")

sess = tf.InteractiveSession()

# Computational Convenience Functions
def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """Compute the 2D laplacian of an array"""
#  5 point stencil #
  five_point = [[0.0, 1.0, 0.0],
                [1.0, -4., 1.0],
                [0.0, 1.0, 0.0]]

#  9 point stencil #
  nine_point = [[0.25, 1.0, 0.25],
                [1.00, -5., 1.00],
                [0.25, 1.0, 0.25]]
						   
#  13 point stencil #
  thirteen_point = [[0.0, 0.0, 0.0, 0.125, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.125, 0.25, 1.0, -5.5, 1.0, 0.25, 0.125],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.125, 0.0, 0.0, 0.0]]
                
  laplace_k = make_kernel(thirteen_point)
  return simple_conv(x, laplace_k)

# Define the PDE
if len(sys.argv) != 4:
	print "Usage:", sys.argv[0], "N npebs num_iter"
	sys.exit()
	
N = int(sys.argv[1])
npebs = int(sys.argv[2])
num_iter = int(sys.argv[3])

hvd.init()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())

#sess = tf.InteractiveSession() #CPU version
#sess = tf.InteractiveSession(config=config) #Use only for capability 3.0 GPU

# Initial Conditions -- some rain drops hit a pond

# Set everything to zero
u_init  = np.zeros([N+3, N], dtype=np.float32)
ut_init = np.zeros([N+3, N], dtype=np.float32)


# Some rain drops hit a pond at random points
for n in range(npebs):
  a = None
  if(hvd.rank() == 0):
      a = np.random.randint(0, N)
  else:
      a = np.random.randint(3, N+3)
  b = np.random.randint(0, N)
  u_init[a,b] = np.random.uniform()

# Parameters:
# eps -- time resolution
# damping -- wave damping
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

# Create variables for simulation state
U  = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

recv_buffer0 = None
if(hvd.rank() == 0):
    recv_buffer0 = U[N:N+3,:]
else:
    recv_buffer0 = tf.Variable(np.zeros([3, N], dtype=np.float32))
recv_buffer1 = None
if(hvd.rank() == 1):
    recv_buffer1 = U[0:3,:]
else:
    recv_buffer1 = tf.Variable(tf.Variable(np.zeros([3, N], dtype=np.float32)))
send_buffer0 = U[N-3:N,:]
send_buffer1 = U[3:6,:]

# Discretized PDE update rules
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

# Operation to update the state
step = tf.group(
  U.assign(U_),
  Ut.assign(Ut_))

#communicate
bcast = tf.group(
        tf.assign(recv_buffer1, hvd.broadcast(send_buffer0, 0)),
        tf.assign(recv_buffer0, hvd.broadcast(send_buffer1, 1)))
# Initialize state to initial conditions
tf.global_variables_initializer().run()

# Run num_iter steps of PDE
start = time.time()
for i in range(num_iter):
  bcast.run()
  # Step simulation
  step.run({eps: 0.06, damping: 0.03})

end = time.time()
print('Elapsed time: {} seconds'.format(end - start))  
result = None
if(hvd.rank == 0):
    result = U[0:N,:]
else:
    result = U[3:N+3,:]
#DisplayArray("lake_py_"+str(hvd.rank())+".png", result.eval(), rng=[-0.1, 0.1])
print_heatmap("lake_c_"+str(hvd.rank())+".dat", result.eval())
