# Auxiliary functions for ROS-based architecture
# TODO Refactor on the message type
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
import rospy

def build_rob_pose_msg(mobile_robot):
    _msg = PoseStamped()
    _msg.header.seq = 1
    _msg.header.frame_id = "map"
    _msg.header.stamp = rospy.Time.now()
    _msg.pose.position.x = mobile_robot.x
    _msg.pose.position.y = mobile_robot.y
    _msg.pose.position.z = 0
    _msg.pose.orientation.x = 0
    _msg.pose.orientation.y = 0
    _msg.pose.orientation.z = 0
    _msg.pose.orientation.w = 1  
    return _msg  

def build_particles_msg(particles):
    _msg = PoseArray()
    _msg.header.seq = 1
    _msg.header.frame_id = "map"
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
        _msg.poses.append(p_tmp)
    return _msg

def build_estimate_msg(estimate):
    _msg = PoseStamped()
    _msg.header.seq = 1
    _msg.header.frame_id = "map"
    _msg.header.stamp = rospy.Time.now()
    _msg.pose.position.x = mu[0]
    _msg.pose.position.y = mu[1]
    _msg.pose.position.z = 0
    _msg.pose.orientation.x = 0
    _msg.pose.orientation.y = 0
    _msg.pose.orientation.z = 0
    _msg.pose.orientation.w = 1
    return _msg
