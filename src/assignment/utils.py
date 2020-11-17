from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import rospy
import tf


def msg_def_PoseStamped(pose, orient, frame_id='map'):
	msg = PoseStamped()
	msg.header.seq = 0
	msg.header.stamp = rospy.Time.now()
	msg.header.frame_id = frame_id
	msg.pose.position.x = pose[0]
	msg.pose.position.y = pose[1]
	msg.pose.position.z = pose[2]
	quaternion = tf.transformations.quaternion_from_euler(orient[0], orient[1], orient[2]) #1.57
	msg.pose.orientation.x = quaternion[0]
	msg.pose.orientation.y = quaternion[1]
	msg.pose.orientation.z = quaternion[2]
	msg.pose.orientation.w = quaternion[3]
	msg.header.seq += 1
	msg.header.stamp = rospy.Time.now()
	return msg

def publish_pose(pose, orient, topic_name):
	msg = msg_def_PoseStamped(pose, orient)
	pub = rospy.Publisher(topic_name, PoseStamped, queue_size=1)
	pub.publish(msg)

def publish_path(path_numpy, orient=[0,0,0,1], topic_name='path', limit=1000):
	path = Path()
	for pose in path_numpy.tolist()[:limit]:
		msg = msg_def_PoseStamped(pose, orient)
		path.header = msg.header
		path.poses.append(msg)
	pub = rospy.Publisher(topic_name, Path, queue_size=1)
	pub.publish(path)