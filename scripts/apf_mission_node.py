#!/usr/bin/env python3

import rospy
import numpy as np
import yaml
import os
import random
from enum import Enum

from geometry_msgs.msg import PoseStamped, TwistStamped, Point, Quaternion
from sensor_msgs.msg import Image
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from cv_bridge import CvBridge

# Import the new core APF logic
from apf_core import APF_Core
# For robust coordinate transformations
import tf2_ros
from tf.transformations import quaternion_from_euler, euler_from_quaternion

class MissionState(Enum):
    STARTUP = 0
    RESPAWNING = 1
    ARMING = 2
    TAKEOFF = 3
    NAVIGATING = 4
    STUCK_ESCAPE = 5
    GOAL_HOVER = 6
    MISSION_COMPLETE = 7
    FAIL = 8

class APFMissionManager:
    def __init__(self):
        rospy.init_node('apf_mission_node')

        # --- Load Configuration ---
        config_path = rospy.get_param('~config_path', '')
        self.config = self.load_config(config_path)

        # --- Instantiate the APF Core Engine ---
        self.apf_engine = APF_Core(self.config)
        rospy.loginfo("[APF Mission] Core APF engine instantiated.")

        # --- ROS & System Setup ---
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.rate = rospy.Rate(self.config['system']['control_frequency'])

        # --- State Machine ---
        self.state = MissionState.STARTUP
        self.mission_count = 0

        # --- Drone State ---
        self.current_pose = None
        self.current_vel = None
        self.current_state_fcu = State()
        self.depth_image = None
        
        # --- Mission Variables ---
        self.goal_pos = None
        self.state_timer = rospy.Time.now()
        self.stuck_check_pos = None
        self.stuck_check_time = rospy.Time.now()
        
        # --- ROS Communication ---
        self.vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=1)
        rospy.Subscriber('/mavros/state', State, lambda msg: setattr(self, 'current_state_fcu', msg))
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, lambda msg: setattr(self, 'current_vel', np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])))
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback, queue_size=1, buff_size=2**24)

        rospy.loginfo("[APF Mission] Waiting for services...")
        rospy.wait_for_service('/mavros/cmd/arming'); self.arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        rospy.wait_for_service('/mavros/set_mode'); self.set_mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        rospy.wait_for_service('/gazebo/set_model_state'); self.set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        rospy.loginfo("[APF Mission] Services connected.")

    def load_config(self, path):
        # ... (Same robust config loading as your original script)
        # Using a simplified version here for clarity
        if not path or not os.path.exists(path):
            rospy.logfatal("[APF Mission] Config file not found! Shutting down.")
            rospy.signal_shutdown("Config needed.")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    # --- Callbacks ---
    def pose_callback(self, msg):
        self.current_pose = msg
        
    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Depth processing error: {e}")

    # --- Core Mission Loop ---
    def run(self):
        rospy.loginfo("[APF Mission] Starting mission manager.")
        while not rospy.is_shutdown():
            # This is the explicit state machine
            if self.state == MissionState.STARTUP:
                self.handle_startup()
            elif self.state == MissionState.RESPAWNING:
                self.handle_respawning()
            elif self.state == MissionState.ARMING:
                self.handle_arming()
            elif self.state == MissionState.TAKEOFF:
                self.handle_takeoff()
            elif self.state == MissionState.NAVIGATING:
                self.handle_navigating()
            elif self.state == MissionState.STUCK_ESCAPE:
                self.handle_stuck_escape()
            elif self.state == MissionState.GOAL_HOVER:
                self.handle_goal_hover()
            elif self.state == MissionState.MISSION_COMPLETE:
                self.handle_mission_complete()
            elif self.state == MissionState.FAIL:
                rospy.logerr_throttle(5.0, "Mission in FAIL state. Check logs.")
                self.publish_velocity(np.zeros(3)) # Failsafe

            self.rate.sleep()
        rospy.loginfo("[APF Mission] Mission manager shut down.")

    # --- State Handlers ---
    def handle_startup(self):
        rospy.loginfo_once("State: STARTUP. Waiting for first pose...")
        if self.current_pose is not None:
            self.transition_to(MissionState.RESPAWNING)
    
    def handle_mission_complete(self):
        if self.config['mission']['continuous_mode']:
            rospy.loginfo(f"Mission {self.mission_count} complete. Starting next mission.")
            self.transition_to(MissionState.RESPAWNING)
        else:
            rospy.loginfo(f"Mission {self.mission_count} complete. Continuous mode off. Shutting down.")
            rospy.signal_shutdown("Mission finished.")

    def handle_respawning(self):
        self.mission_count += 1
        rospy.loginfo(f"State: RESPAWNING for Mission {self.mission_count}")
        self.apf_engine.reset() # Reset calculator state
        
        # Respawn logic
        spawn_cfg = self.config['spawn_settings']
        state_msg = ModelState(model_name='iris')
        state_msg.pose.position.x = spawn_cfg['start_position']['x']
        state_msg.pose.position.y = spawn_cfg['start_position']['y']
        state_msg.pose.position.z = 0.1 # Spawn on the ground
        state_msg.pose.orientation.w = 1.0
        try:
            self.set_model_state_service(state_msg)
            rospy.sleep(spawn_cfg['respawn_delay'])
            self.goal_pos = self.sample_new_goal()
            self.transition_to(MissionState.ARMING)
        except rospy.ServiceException as e:
            rospy.logerr(f"Respawn failed: {e}")
            self.transition_to(MissionState.FAIL)
            
    def handle_arming(self):
        rospy.loginfo_once("State: ARMING")
        # Arming logic... (simplified for brevity)
        if not self.current_state_fcu.armed or self.current_state_fcu.mode != "OFFBOARD":
            self.arm_service(True)
            self.set_mode_service(custom_mode="OFFBOARD")
            self.publish_velocity(np.zeros(3)) # Stream setpoints
        else:
            rospy.loginfo("Drone is ARMED and in OFFBOARD mode.")
            self.transition_to(MissionState.TAKEOFF)
            
    def handle_takeoff(self):
        rospy.loginfo_once("State: TAKEOFF")
        takeoff_alt = self.config['altitude']['takeoff_height']
        pos = self.current_pose.pose.position
        error = takeoff_alt - pos.z
        
        if abs(error) < 0.15:
            rospy.loginfo("Takeoff complete.")
            self.transition_to(MissionState.NAVIGATING)
            return

        # Simple P-controller for takeoff
        vz_cmd = np.clip(error * 1.5, -0.5, 0.8)
        self.publish_velocity([0, 0, vz_cmd])

    def handle_navigating(self):
        rospy.loginfo_once("State: NAVIGATING")
        if self.check_if_goal_reached():
            self.transition_to(MissionState.GOAL_HOVER)
            return
        if self.check_if_stuck():
            self.transition_to(MissionState.STUCK_ESCAPE)
            return
        
        vel_cmd, _ = self.compute_apf_velocity(is_stuck=False)
        self.publish_velocity(vel_cmd)

    def handle_stuck_escape(self):
        rospy.logwarn_throttle(1.0, "State: STUCK_ESCAPE")
        if (rospy.Time.now() - self.state_timer).to_sec() > self.config['stuck_detection']['escape_duration']:
            rospy.loginfo("Escape maneuver finished. Returning to navigation.")
            self.transition_to(MissionState.NAVIGATING)
            return
            
        vel_cmd, _ = self.compute_apf_velocity(is_stuck=True)
        self.publish_velocity(vel_cmd)

    def handle_goal_hover(self):
        rospy.loginfo_once("State: GOAL_HOVER")
        self.publish_velocity(np.zeros(3))
        if (rospy.Time.now() - self.state_timer).to_sec() > self.config['goal_settings']['hover_duration']:
            self.transition_to(MissionState.MISSION_COMPLETE)
    
    # --- Helper Functions ---
    def transition_to(self, new_state):
        if self.state != new_state:
            rospy.loginfo(f"Transitioning from {self.state.name} -> {new_state.name}")
            self.state = new_state
            self.state_timer = rospy.Time.now()
            # Reset stuck checker on state changes out of NAVIGATING/STUCK_ESCAPE
            if new_state not in [MissionState.NAVIGATING, MissionState.STUCK_ESCAPE]:
                self.stuck_check_pos = None

    def compute_apf_velocity(self, is_stuck):
        """Gets forces from APF Core and applies system-level constraints."""
        pos = self.current_pose.pose.position
        current_pos_np = np.array([pos.x, pos.y, pos.z])
        
        # **ROBUSTNESS IMPROVEMENT**: Use tf2 to handle coordinate transformations
        try:
            # Transform from the camera's frame to the world frame ('map' or 'odom')
            transform = self.tf_buffer.lookup_transform('map', 'camera_depth_optical_frame', rospy.Time(0), rospy.Duration(0.1))
            
            # Get the obstacle points in the camera's frame from the core
            _, obstacle_vectors_cam = self.apf_engine.calculate_total_force(current_pos_np, self.goal_pos, self.depth_image, is_stuck)
            
            # Now transform them to the world frame
            obstacle_vectors_world = []
            for p_cam in obstacle_vectors_cam:
                p_world = self.transform_point(p_cam, transform)
                obstacle_vectors_world.append(p_world)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, f"TF transform error: {e}")
            obstacle_vectors_world = []

        # Now get the final force using world-frame obstacles
        total_force = self.apf_engine._calculate_attractive_force(current_pos_np, self.goal_pos) + \
                      self.apf_engine._calculate_repulsive_force(current_pos_np, obstacle_vectors_world)
        if is_stuck:
            total_force += self.apf_engine._calculate_tangential_escape_force(total_force)
            
        # Clamp velocity
        vel_mag = np.linalg.norm(total_force)
        vel_cap = self.config['velocity']['vel_cap']
        if vel_mag > vel_cap:
            total_force = (total_force / vel_mag) * vel_cap
        
        return total_force, obstacle_vectors_world

    def transform_point(self, point, transform):
        """Applies a TF transform to a 3D point."""
        q = transform.transform.rotation
        r_matrix = self.quaternion_to_rotation_matrix([q.x, q.y, q.z, q.w])
        t_vec = transform.transform.translation
        translation = np.array([t_vec.x, t_vec.y, t_vec.z])
        return np.dot(r_matrix, point) + translation

    def quaternion_to_rotation_matrix(self, q):
        x, y, z, w = q
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])

    def publish_velocity(self, vel_cmd):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = vel_cmd
        self.vel_pub.publish(msg)

    def check_if_stuck(self):
        stuck_cfg = self.config['stuck_detection']
        if self.stuck_check_pos is None:
            self.stuck_check_pos = self.current_pose.pose.position
            self.stuck_check_time = rospy.Time.now()
            return False

        if (rospy.Time.now() - self.stuck_check_time).to_sec() > stuck_cfg['stuck_time_threshold']:
            p1 = self.stuck_check_pos
            p2 = self.current_pose.pose.position
            dist_moved = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            
            self.stuck_check_pos = p2
            self.stuck_check_time = rospy.Time.now()
            
            if dist_moved < stuck_cfg['stuck_dist_threshold']:
                rospy.logwarn("Stuck condition detected!")
                return True
        return False

    def check_if_goal_reached(self):
        pos = self.current_pose.pose.position
        dist = np.linalg.norm(np.array([pos.x, pos.y, pos.z]) - self.goal_pos)
        return dist < self.config['goal_settings']['goal_reach_threshold']

    def sample_new_goal(self):
        area = self.config['goal_settings']['goal_area']
        x = random.uniform(area['x_min'], area['x_max'])
        y = random.uniform(area['y_min'], area['y_max'])
        z = self.config['altitude']['takeoff_height']
        return np.array([x, y, z])

if __name__ == '__main__':
    try:
        node = APFMissionManager()
        node.run()
    except rospy.ROSInterruptException:
        pass