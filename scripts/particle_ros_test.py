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

# Basic ROS publisher
def particle_pub():
    rospy.init_node('particle_publisher',anonymous=True)

    # Define publishers
    pub_sim_rob = rospy.Publisher('mobile_agent', PoseStamped, queue_size=10)
    pub_particles = rospy.Publisher('particles', PoseArray, queue_size=20)
    pub_estimate = rospy.Publisher('state_estimate', PoseStamped, queue_size=10)

    # Define subscribers
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

        # Fill the message
        p_real = PoseStamped()
        p_real.header.seq = 1
        p_real.header.frame_id = "map"
        p_real.header.stamp = rospy.Time.now()
        p_real.pose.position.x = mob_rob.x
        p_real.pose.position.y = mob_rob.y
        p_real.pose.position.z = 0
        p_real.pose.orientation.x = 0
        p_real.pose.orientation.y = 0
        p_real.pose.orientation.z = 0
        p_real.pose.orientation.w = 1                      
        pub_sim_rob.publish(p_real)

        # ------------------------ Prediction step
        predict_linear_planar(particles, u=(vx, vy), std=(.005,.005), dt=mob_rob.dt)
        # Message
        p_particles = PoseArray()
        p_particles.header.seq = 1
        p_particles.header.frame_id = "map"
        i = 0
        for p in particles:
            p_tmp = Pose()
            p_tmp.position.x = p[0]
            p_tmp.position.y = p[1]
            p_tmp.position.z = 0
            p_tmp.orientation.x = 0
            p_tmp.orientation.y = 0
            p_tmp.orientation.z = 0
            p_tmp.orientation.w = 1
            
            p_particles.poses.append(p_tmp)
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
        # Message
        p_estimate = PoseStamped()
        p_estimate.header.seq = 1
        p_estimate.header.frame_id = "map"
        p_estimate.header.stamp = rospy.Time.now()
        p_estimate.pose.position.x = mu[0]
        p_estimate.pose.position.y = mu[1]
        p_estimate.pose.position.z = 0
        p_estimate.pose.orientation.x = 0
        p_estimate.pose.orientation.y = 0
        p_estimate.pose.orientation.z = 0
        p_estimate.pose.orientation.w = 1

        pub_estimate.publish(p_estimate)
        xs.append(mu)
        estimate_lst.append(mu)
        # --------------------------------------

        # Saving for plots
        ground_truth_lst.append([mob_rob.x,mob_rob.y])
        meas_lst.append([x_meas,y_meas])

        rate.sleep()

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