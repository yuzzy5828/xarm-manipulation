#!/usr/bin/env python

# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters


import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

## END_SUB_TUTORIAL


def all_close(goal, actual, tolerance):
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


class GraspingXarm(object):

    def __init__(self):
        super(GraspingXarm, self).__init__()

        # # RealSenseカメラの設定
        # self.pipeline = rs.pipeline()
        # config = rs.config()
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # self.pipeline.start(config)

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)

        ## Instantiateprint(plan) a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        group_name = "xarm7"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        ## END_SUB_TUTORIAL

        ## BEGIN_SUB_TUTORIAL basic_info
        ##
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")
        ## END_SUB_TUTORIAL

        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

        # ペットボトル識別の初期化
        self.bottle_class_id = 39  # YOLOv5のCOCOデータセットでのボトルのクラスID
        self.confidence_threshold = 0.5  # 50%以上の信頼度

        # self.t_left = 0.55
        # self.t_right = 0.65
        # self.t_forward = 0.7
        # self.t_backward = 0.8

        self.t_x = 0.60  # アーム目標地点：x成分
        self.t_y = 0.75  # アーム目標地点：y成分
        self.t_z = 0.14  # アーム目標地点：z成分
        
        # self.x_center = 0
        # self.y_center = 0
        # self.image_height = 0
        # self.image_width = 0

        self.w_x = 1.0  # アームの進む向きを表すベクトル：x成分
        self.w_y = 1.0  # アームの進む向きを表すベクトル：y成分
        
        self.k = 0.3  #移動の大きさを調整する係数
        
        self.count = 0
        self.value = 0.14  # アームの移動量

        # ROSノードの初期化
        rospy.init_node("grasping_bottle", anonymous=True)

        # Publisher
        self.detection_result_pub = rospy.Publisher('/detection_result', Image, queue_size=10)

        # Subscriber
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 1.0).registerCallback(self.callback)

        self.bridge = CvBridge()
        self.rgb_image, self.depth_image = None, None

        self.model = YOLO('yolov10n.pt')

    def go_to_initial_pose(self, initial_state):
        self.move_group.set_joint_value_target(initial_state)
        self.move_group.set_max_velocity_scaling_factor(0.3)

        self.move_group.go(wait=True)

        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def searching_bottle(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            if self.rgb_image is None or self.depth_image is None:
                rate.sleep()
                continue

            # YOLOで物体検出
            results = self.model(self.rgb_image)
            detected_bottle = False
            tmp_image = self.rgb_image.copy()

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = box.conf[0]
                    if cls == self.bottle_class_id and conf > self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(tmp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        detected_bottle = True
                        x_center = (x1 + x2) // 2
                        y_center = (y1 + y2) // 2

                        # 深度情報の取得
                        z = self.depth_image[y_center, x_center] / 1000.0  # メートル単位に変換

                        image_height, image_width = self.rgb_image.shape[:2]

                        self.w_x = x_center/image_width - self.t_x
                        self.w_y = y_center/image_height - self.t_y
                        self.w_z = z - self.t_z

                        l_w = self.w_x**2 + self.w_y**2

                        if l_w > 0.0064:
                            self.move_xy()
                        else:
                            current_pose = self.move_group.get_current_pose().pose
                            return current_pose

                        # 検出結果を画像に描画
                        cv2.putText(tmp_image, f"{self.model.names[cls]}: {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    else:
                        pass
                        
            
            # 結果の表示（デバッグ用）
            cv2.imshow("YOLO Detection", tmp_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 検出結果の画像をROSトピックとして送信
            detection_image_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(tmp_image, cv2.COLOR_RGB2BGR), "bgr8")
            self.detection_result_pub.publish(detection_image_msg)

            if not detected_bottle:
                rospy.loginfo("Bottle not detected. Continuing search...")
                self.count += 1
                self.up_searching()

            rate.sleep()

        return None  # ボトルが見つからなかった場合

    def callback(self, data1, data2):
        cv_array = self.bridge.imgmsg_to_cv2(data1, 'bgr8')
        cv_array = cv2.cvtColor(cv_array, cv2.COLOR_BGR2RGB)
        self.rgb_image = cv_array

        cv_array = self.bridge.imgmsg_to_cv2(data2, 'passthrough')
        self.depth_image = cv_array
    
    def down_planning(self, current_pose):
        target_pose = geometry_msgs.msg.Pose()
        target_pose.orientation = current_pose.orientation
        target_pose.position.x = current_pose.position.x
        target_pose.position.y = current_pose.position.y
        target_pose.position.z = current_pose.position.z - self.value - self.count * 0.005

        self.move_group.set_pose_target(target_pose)
        
        self.move_group.set_max_velocity_scaling_factor(0.1)
        self.move_group.set_num_planning_attempts(10)
        self.move_group.set_planning_time(20.0)
        self.move_group.set_planner_id("RRTConnect")
        
        plan = self.move_group.plan()
        
        # planがタプルの場合plan = task.up_planning()
        if isinstance(plan, tuple):
            return plan[1]
        else:
            return plan

    def up_planning(self):
        current_pose = self.move_group.get_current_pose().pose
        target_pose = geometry_msgs.msg.Pose()
        target_pose.orientation = current_pose.orientation
        target_pose.position.x = current_pose.position.x
        target_pose.position.y = current_pose.position.y
        target_pose.position.z = current_pose.position.z + self.value

        self.move_group.set_pose_target(target_pose)
        
        self.move_group.set_max_velocity_scaling_factor(0.1)
        self.move_group.set_num_planning_attempts(10)
        self.move_group.set_planning_time(20.0)
        self.move_group.set_planner_id("RRTConnect")
        
        plan = self.move_group.plan()
        
        # planがタプルの場合、2番目の要素を返す
        if isinstance(plan, tuple):
            return plan[1]
        else:
            return plan
    
    def up_searching(self):
        current_pose = self.move_group.get_current_pose().pose
        target_pose = geometry_msgs.msg.Pose()
        target_pose.orientation = current_pose.orientation
        target_pose.position.x = current_pose.position.x
        target_pose.position.y = current_pose.position.y
        target_pose.position.z = current_pose.position.z + self.count * 0.005

        self.move_group.set_pose_target(target_pose)
        
        self.move_group.set_max_acceleration_scaling_factor(0.2)
        self.move_group.set_max_velocity_scaling_factor(0.1)
        self.move_group.set_num_planning_attempts(10)
        self.move_group.set_planning_time(20.0)
        self.move_group.set_planner_id("RRTConnect")
        
        self.move_group.go(wait=True)

        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def move_xy(self):
        current_pose = self.move_group.get_current_pose().pose
        target_pose = geometry_msgs.msg.Pose()
        
        target_pose.orientation.w = current_pose.orientation.w
        target_pose.orientation.x = current_pose.orientation.x
        target_pose.orientation.y = current_pose.orientation.y
        target_pose.orientation.z = current_pose.orientation.z

        target_pose.position.x = current_pose.position.x - self.k*self.w_y
        target_pose.position.y = current_pose.position.y - self.k*self.w_x
        target_pose.position.z = current_pose.position.z

        self.move_group.set_pose_target(target_pose)

        self.move_group.go(wait=True)

        self.move_group.stop()
        self.move_group.clear_pose_targets()
    
    def go_to_goal_pose(self, goal_state):
        self.move_group.set_joint_value_target(goal_state)
        self.move_group.set_max_velocity_scaling_factor(0.3)

        self.move_group.go(wait=True)

        self.move_group.stop()
        self.move_group.clear_pose_targets()
    
    def move_to_goal(self):
        try:
            waypoints = []
            posemiddle = geometry_msgs.msg.Pose()
            posemiddle.orientation = self.move_group.get_current_pose().pose.orientation
            posemiddle.position = self.move_group.get_current_pose().pose.position
            self.move_group.compute_cartesian_path(
                                            waypoints,   # waypoints to follow
                                            0.01,        # eef_step
                                            0.0)         # jump_threshold

            if fraction < 1.0:
                rospy.logwarn(f"Only {fraction:.2%} of the path was successfully planned.")
                user_input = input("Do you want to continue with partial path? (y/n): ")
                if user_input.lower() != 'y':
                    rospy.loginfo("Movement cancelled by user.")
                    return None

            success = self.move_group.execute(plan, wait=True)

            if success:
                rospy.loginfo("Movement executed successfully.")
                pose = self.move_group.get_current_pose().pose
                return pose
            else:
                rospy.logerr("Failed to execute the movement.")
                return None

        except moveit_commander.MoveItCommanderException as e:
            rospy.logerr(f"MoveIt commander error: {e}")
            return None
        except rospy.ROSInterruptException:
            rospy.logerr("ROS was interrupted. Movement cancelled.")
            return None
        except Exception as e:
            rospy.logerr(f"An unexpected error occurred: {e}")
            return None

    def open_grasping_hand(self):
        gripper_group = moveit_commander.MoveGroupCommander("xarm_gripper")
        gripper_group.set_named_target("open")
        gripper_group.go(wait=True)

    def close_grasping_hand(self):
        gripper_group = moveit_commander.MoveGroupCommander("xarm_gripper")
        gripper_group.set_named_target("close")
        gripper_group.go(wait=True)
    
    def execute_plan(self, plan):
        self.move_group.execute(plan, wait=True)

        self.move_group.stop()
        self.move_group.clear_pose_targets()

def main():
    initial_state = [0.2859010696411133, \
    -0.540014922618866, \
    -0.3745407462120056, \
    0.5726205706596375, \
    -0.21252599358558655, \
    1.0677554607391357, \
    0.04890037700533867]

    goal_state = [0.5129781365394592, 0.49402567744255066, -0.3729110658168793, 0.8349484205245972, 2.7422215938568115, 1.2920500040054321, -2.758296012878418]

    try:
        task = GraspingXarm()

        task.go_to_initial_pose(initial_state)

        pose = task.move_group.get_current_pose()

        task.searching_bottle()

        # # for temporary
        # current_pose = geometry_msgs.msg.Pose()
        # current_pose = pose.pose
        # current_pose.position.x = pose.pose.position.x + 0.05
        
        current_pose = task.searching_bottle()

        plan = task.down_planning(current_pose)

        task.open_grasping_hand()
        
        task.execute_plan(plan)

        task.close_grasping_hand()

        plan = task.up_planning()

        task.execute_plan(plan)

        current_pose = task.go_to_goal_pose(goal_state)

        # # for temporary
        # current_pose = task.move_group.get_current_pose().pose   

        task.open_grasping_hand()

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()