#!/usr/bin/env python

# Early test of usage of Particles python library in ROS for bayesian estimation
# The only version which can be used is the legacy 0.1 which supports python 2, mandatory for ROS

import warnings; warnings.simplefilter('ignore')  # Skip warnings, like the one for quasi-monte-carlo features 

from matplotlib import pyplot as plt
import numpy as np 
import seaborn as sb 
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from filterpy.monte_carlo import systematic_resample

# User defined
from mob_agent import MobAgent
from particle import *
from ros_utils import *

# Basic ROS publisher
def particle_pub():
    rospy.init_node('particle_publisher',anonymous=True)

    # Publishers
    pub_sim_rob = rospy.Publisher('mobile_agent', PoseStamped, queue_size=10)
    pub_particles = rospy.Publisher('particles', PoseArray, queue_size=20)
    pub_estimate = rospy.Publisher('state_estimate', PoseStamped, queue_size=10)

    # Subscribers
    # TODO

    rate = rospy.Rate(100)     # 100 Hz rate

    # Instance of simulated robot (constant velocity)
    mob_rob = MobAgent(0,0,0.01)
    vx = 0.1
    vy = 0.1

    num_particles = 1000

    # Create particle filter
    initial_x=(0.6,0.6, np.pi/4)
    particles = create_gaussian_particles(mean=initial_x, std=(0.35, 0.4, np.pi/4), N=num_particles)
    weights = np.ones(num_particles) / num_particles
    #print(particles)
    #fig, ax = plt.subplots()
    #ax.scatter(particles[:,0],particles[:,1],color='green',s=3)
    #ax.grid()
    #plt.show()
    #exit(0)

    # Estimate
    xs = []

    # For plots 
    ground_truth_lst = []
    meas_lst = []
    estimate_lst = []

    while not rospy.is_shutdown():
        # ----------------------- Agent Simulation
        # Evolve simulated agent
        mob_rob.step(vx,vy)

        # Get simulated noisy measurement
        [x_meas, y_meas] = mob_rob.state_meas()

        # Ground truth message
        p_real = build_rob_pose_msg(mob_rob)          
        pub_sim_rob.publish(p_real)

        # ------------------------ Prediction step
        predict_linear_planar(particles, u=(vx, vy), std=(.005,.005), dt=mob_rob.dt)

        # Particles message
        p_particles = build_particles_msg(particles)
        pub_particles.publish(p_particles)
        # -----------------------------------------

        # ------------------------ Correction step
        sensor_std_err = 0.05
        z = np.linalg.norm(np.array([x_meas,y_meas]))
        update_fullstate(particles, weights, z=z, R=sensor_std_err)
        # Resample triggering
        if neff(weights) < num_particles/2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
        #-----------------------------------------

        # --------------------------- Estimation
        mu, var = estimate(particles, weights)

        # Estimate message
        p_estimate = build_estimate_msg(mu)
        pub_estimate.publish(p_estimate)
        # --------------------------------------

        # Saving for plots
        ground_truth_lst.append([mob_rob.x,mob_rob.y])
        meas_lst.append([x_meas,y_meas])
        xs.append(mu)
        estimate_lst.append(mu)

        rate.sleep()
    # ----------------------------------------------------------------------

    print("Plotting results")
    # Post processing and plots when execution is over
    # Ground truth simulated robot and noisy measurements
    gt_plot = np.array(ground_truth_lst)
    meas_plot = np.array(meas_lst)
    estimate_plot = np.array(estimate_lst)
    fig, ax = plt.subplots()
    ax.plot(gt_plot[:,0],gt_plot[:,1],color='k',linewidth=2)
    ax.scatter(meas_plot[:,0],meas_plot[:,1],color='b',s=3)
    ax.plot(estimate_plot[:,0],estimate_plot[:,1],color='r',linewidth=2)
    ax.set(xlabel='x(m)', ylabel='y(m)',
       title='Ground truth robot pose')
    ax.legend(['Ground truth','Measurements'])
    ax.grid()
    plt.show()

# Run publisher node
if __name__ == '__main__':
    try:
        particle_pub()
    except rospy.ROSInterruptException:
        pass