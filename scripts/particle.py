# All the components for the particle filter are found here 
import numpy as np
from numpy.random import uniform
from numpy.random import randn
import scipy
import random

# Initial uniform distribution of particles
def create_uniform_particles(x_range,y_range,theta_range,N):
    particles = np.empty((N,3))
    particles[:,0] = uniform(x_range[0], x_range[1], size=N)
    particles[:,1] = uniform(y_range[0], y_range[1], size=N)
    particles[:,2] = uniform(theta_range[0], theta_range[1], size=N)
    particles[:,2] %= 2*np.pi       # Normalize wrt pi for heading
    return particles
# TEST
#print(create_uniform_particles((0,1),(0,1),(0,2*np.pi),4))

# Initial gaussian distribution of particles
def create_gaussian_particles(mean, std, N):
    particles = np.empty((N,3))
    particles[:,0] = mean[0] + (randn(N)*std[0])
    particles[:,1] = mean[1] + (randn(N)*std[1])
    particles[:,2] = mean[2] + (randn(N)*std[2])
    particles[:,2] %= 2*np.pi
    return particles
# TEST
#print(create_gaussian_particles((0.0,0.0,0.0),(0.3,0.3,0.1),10))

# ---------------------------- Predict Step
# Steps the motion model, which is the state transition of each particle
# with noise
# Nonlinear model - Input is in the shape (w,v)
def predict_nonlinear_planar(particles, u, std, dt=1.):
    N = len(particles)
    particles[:,2] += u[0] + (randn(N)*std[0])
    particles[:,2] %= 2*np.pi                  # Always normalize angular variables
    dist = (u[1]*dt) + (randn(N)*std[1])
    particles[:,0] += np.cos(particles[:,2])*dist
    particles[:,1] += np.sin(particles[:,2])*dist

# Linear model - Holonomic agent, inputs are directly velocities on the axes
def predict_linear_planar(particles, u, std, dt=1.):
    N = len(particles)
    particles[:,0] += u[0]*dt + (randn(N)*std[0])
    particles[:,1] += u[1]*dt + (randn(N)*std[1])

# Update step (for now related to landmark-based model, 
# to be decoupled for suiting more sensor models)
def update(particles,weights,z,R,landmarks):
    # For each landmark
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:,0:2] - landmark, axis=1)  # P(z|x)
        weights *= scipy.stats.norm(distance,R).pdf(z[i])               # P(z|x)*P(x)

    weights += 1.e-300  
    weights /= sum(weights)     # [P(z|x)*P(x)]/P(z)

# State estimation
# Exploits the weight update to compute an estimate of the real state
# In most cases a weighted sum is enough
def estimate(particles, weights):
    # Just tracking (x,y) position
    pos = particles[:,0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos-mean)**2, weights=weights, axis=0)
    return mean, var
# TEST
# particles = create_uniform_particles((0,1),(0,1),(0,5),1000)
# weights = np.array([1.0/1000.0]*1000)
# [mean,var] = estimate(particles,weights)

# Resampling step
# Used to avoid degenerancy of the particle set
def multinomial_resample(particles, weights):
    N = len(particles)
    # Cumulative sum to have a sorted array
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0  # For avoiding round-off
    indexes = np.searchsorted(cumulative_sum, randn(N))
    # Resampling
    particles[:] = particles[indexes]
    weights.fill(1.0/N)

# Takes indexes which are output of a resampling algorithm
def resample_from_index(particles,weights,indexes):
    # Only selected particles survive
    particles[:] = particles[indexes]
    # Weights are normalized again
    weights.resize(len(particles))
    weights.fill (1.0 / len(weights))

# Resampling trigger
# Many metrics can be adopted, the basic one is 
# computing the number of effective (i.e. useful) particles
def neff(weights):
    return 1.0/np.sum(np.square(weights))




