# Script for testing the basic particle filtering algorithm
# This does not require ros 
from particle import *
from matplotlib import pyplot as plt
import numpy as np
from numpy.random import seed
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats

# Resampling algorithms from filterpy
from filterpy.monte_carlo import systematic_resample

# Basic filtering test loop
def run_sim_pf(N, iters=30, sensor_std_err=0.1, do_plot=True, plot_particles=False,
            xlim=(0,2), ylim=(0,2), initial_x=None):
    landmarks = np.array([[-1, 2], [5, 10], [12,14], [18,21]])
    NL = len(landmarks)      
    plt.figure()

    # --------------------------------Initialize filter elements
    # Gaussian or Uniform
    if initial_x is not None:
        particles = create_gaussian_particles(
            mean=initial_x, std=(0.35, 0.4, np.pi/4), N=N)
    else:
        particles = create_uniform_particles((0,20), (0,20), (0, 6.28), N)
    # Initial normalized weights
    weights = np.ones(N) / N

    # Plot initial particles distribution
    # if plot_particles:
    #     alpha = .20
    #     if N > 5000:
    #         alpha *= np.sqrt(5000)/np.sqrt(N)           
    #     plt.scatter(particles[:, 0], particles[:, 1], 
    #                 alpha=alpha, color='g')

    xs = []                         # Particle set
    robot_pos = np.array([0., 0.])  # Initial robot position

    # Simulation and estimation loop
    for x in range(iters):
        # Evolve robot position
        robot_pos += (1, 1)

        # Measurements (distance from landmarks)
        zs = (norm(landmarks - robot_pos, axis=1) + (randn(NL) * sensor_std_err))

        # Evolve particles following the motion model
        predict_nonlinear_planar(particles, u=(0.00, 1.414), std=(.2, .05))
        
        # Update weights from measurements
        update_landmarks(particles, weights, z=zs, R=sensor_std_err, landmarks=landmarks)
        
        # Resample triggering
        if neff(weights) < N/2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
            #multinomial_resample(particles, weights)
            #assert np.allclose(weights, 1/N)

        # State estimation
        mu, var = estimate(particles, weights)
        xs.append(mu)

        if plot_particles:
            plt.scatter(particles[:, 0], particles[:, 1], 
                        color='k', marker=',', s=1)

        # Plot ground truth and estimate
        p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+',
                         color='b', s=180, lw=3)
        p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')

    # Plot data after loop
    xs = np.array(xs)
    plt.legend([p1,p2], ['Actual','PF'], loc=4, numpoints=1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    print('final position error, variance:\n\t', mu - np.array([iters, iters]), var)
    plt.show()

# Run simulation
seed(2)
run_sim_pf(N=5000,plot_particles=True,initial_x=(0.6,0.6, np.pi/4))