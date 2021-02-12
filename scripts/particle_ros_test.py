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
import particles
from particles import distributions as dists
from particles import state_space_models as ssm
from particles.collectors import MomentsCollector  # In newer version is called 'Moments'
from particles import state_space_models as ssms   # For defining motion and measurement models from state space
from particles import distributions as dists

# User defined
from mob_agent import MobAgent

# Basic ROS publisher
def particle_pub():
    pub_sim_rob = rospy.Publisher('mobile_agent', PoseStamped, queue_size=10)
    rospy.init_node('particle_publisher',anonymous=True)

    rate = rospy.Rate(100)     # 100 Hz rate

    # Instance of simulated robot (constant velocity)
    mob_rob = MobAgent(0,0,0.01)
    vx = 0.1
    vy = 0.1


    # For plots 
    ground_truth_lst = []
    meas_lst = []

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

        # ------------------------ State Estimation

        # -----------------------------------------

        # Saving for plots
        ground_truth_lst.append([mob_rob.x,mob_rob.y])
        meas_lst.append([x_meas,y_meas])

        rate.sleep()

    print("Plotting results")
    # Post processing and plots when execution is over
    # Ground truth simulated robot and noisy measurements
    gt_plot = np.array(ground_truth_lst)
    meas_plot = np.array(meas_lst)
    fig, ax = plt.subplots()
    ax.plot(gt_plot[:,0],gt_plot[:,1],color='blue',linewidth=2)
    ax.scatter(meas_plot[:,0],meas_plot[:,1],color='green',s=3)
    ax.set(xlabel='x(m)', ylabel='y(m)',
       title='Ground truth robot pose')
    ax.grid()
    plt.show()

# Run publisher node
if __name__ == '__main__':
    try:
        particle_pub()
    except rospy.ROSInterruptException:
        pass