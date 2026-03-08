#!/usr/bin/env python3

import rospy
import numpy as np
import random
import yaml
import os
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError 
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from scipy.ndimage import median_filter
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import cv2
import sys 

class APFDepthNavigator:
    def __init__(self, config_file_path_arg=None):
        rospy.init_node('apf_depth_nav_node')

        self.config = self.load_config(config_file_path_arg)
        self.load_parameters() 
        
        self.bridge = CvBridge()
        self.curr_pos = np.zeros(3)
        self.curr_vel = np.zeros(3)
        self.depth_image = None
        self.current_state = State()
        self.takeoff_complete = False
        self.navigating = False 

        self.obstacle_detected = False
        self.last_obstacle_time = rospy.Time.now()
        self.last_repulsive_force_raw = np.zeros(3) 

        self.last_movement_time = rospy.Time.now()
        self.last_progress_time = rospy.Time.now()
        self.last_progress_dist_to_target = float('inf')
        self.escape_mode = False
        self.escape_start_time = None
        self.escape_direction = None
        self.post_escape_boost_active = False 
        self.post_escape_boost_end_time = rospy.Time.now()
        
        # This is the ABSOLUTE position the drone will always try to respawn at.
        # It uses values loaded by self.load_parameters() from self.spawn_settings_config and self.takeoff_height
        self.fixed_spawn_pos = np.array([
            self.spawn_settings_config.get('start_position',{}).get('x', -12.0), # Default if key missing
            self.spawn_settings_config.get('start_position',{}).get('y', 0.0),
            self.spawn_settings_config.get('start_position',{}).get('z', self.takeoff_height) 
        ])
        
        self.start_pos_of_session = None 
        self.goal_pos = None 
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.logged_approaching_final_wp_for_current_target = False
        
        self.force_history = []
        self.mission_count = 0
        
        rospy.Subscriber('/mavros/state', State, self.state_callback)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, self.velocity_callback)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback, queue_size=1, buff_size=2**24)

        self.vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=5)
        
        rospy.loginfo("[APF] Waiting for MAVROS and Gazebo services...")
        try:
            rospy.wait_for_service('/mavros/cmd/arming', timeout=15)
            self.arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
            rospy.wait_for_service('/mavros/set_mode', timeout=15)
            self.set_mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)
            rospy.wait_for_service('/gazebo/set_model_state', timeout=15)
            self.set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState) 
            rospy.loginfo("[APF] MAVROS and Gazebo services connected.")
        except rospy.ROSException as e:
            rospy.logfatal(f"[APF] Critical service not available: {e}. Shutting down.")
            rospy.signal_shutdown("Critical services failed.")
            return

        self.rate = rospy.Rate(self.control_frequency) 
        rospy.loginfo("[APF] Navigator initialized successfully!")
        if self.data_collection_mode:
            rospy.logwarn("[APF] DATA COLLECTION MODE IS ACTIVE.")
        if self.debug_config.get('verbose_logging', False): 
            self.log_configuration()

    def load_config(self, config_file_path_arg):
        resolved_config_path = None
        if config_file_path_arg and os.path.exists(os.path.expanduser(config_file_path_arg)):
            resolved_config_path = os.path.expanduser(config_file_path_arg)
            rospy.loginfo(f"[APF] Using config file from argument: {resolved_config_path}")
        else:
            primary_path = '~/catkin_ws/src/apf_depth_nav_node/config/apf_config.yaml' 
            expanded_primary_path = os.path.expanduser(primary_path)
            if os.path.exists(expanded_primary_path):
                resolved_config_path = expanded_primary_path
                rospy.loginfo(f"[APF] Using config file from primary path: {resolved_config_path}")
            else: 
                rospy.logwarn(f"[APF] Primary config path {expanded_primary_path} not found.")
                script_dir = os.path.dirname(os.path.realpath(__file__))
                pkg_dir = os.path.abspath(os.path.join(script_dir, '..'))
                possible_paths = [
                    os.path.join(pkg_dir, 'config', 'apf_config.yaml'), 
                    os.path.join(script_dir, 'apf_config.yaml'),       
                    os.path.expanduser('~/apf_config.yaml'),
                    os.path.expanduser('~/.ros/apf_config.yaml')
                ]
                try:
                    import rospkg
                    rospack = rospkg.RosPack()
                    node_name_parts = rospy.get_name().split('/')
                    pkg_name_candidate = node_name_parts[1] if len(node_name_parts) > 1 and node_name_parts[0] == '' else node_name_parts[0]
                    pkg_path_from_rospkg = rospack.get_path(pkg_name_candidate)
                    possible_paths.insert(0, os.path.join(pkg_path_from_rospkg, 'config', 'apf_config.yaml'))
                except Exception as e: 
                    rospy.logdebug(f"[APF] rospkg failed or package name parsing issue for config search: {e}")
                for path in possible_paths:
                    if os.path.exists(path):
                        resolved_config_path = path
                        rospy.loginfo(f"[APF] Found config file at fallback path: {resolved_config_path}")
                        break
        
        final_config = self._get_internal_default_config() 
        if resolved_config_path:
            try:
                with open(resolved_config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config: 
                        rospy.loginfo(f"[APF] Successfully loaded YAML from: {resolved_config_path}")
                        def merge_dicts(base, new_data): 
                            for key, value in new_data.items():
                                if isinstance(value, dict) and isinstance(base.get(key), dict):
                                    base[key] = merge_dicts(base.get(key, {}), value)
                                else:
                                    base[key] = value
                            return base
                        final_config = merge_dicts(final_config, yaml_config)
                        rospy.loginfo("[APF] Merged YAML config with internal defaults.")
                    else:
                        rospy.logwarn(f"[APF] YAML config file {resolved_config_path} is empty. Using internal defaults.")
            except Exception as e:
                rospy.logerr(f"[APF] Failed to load or parse YAML config {resolved_config_path}: {e}. Using internal defaults.")
        else:
            rospy.logwarn("[APF] No valid config file path provided or found. Using internal defaults only.")
        return final_config

    def _get_internal_default_config(self): 
        # These match your latest YAML's structure and values as the new base defaults
        return {
            'force_field': {'k_att': 0.8, 'k_rep': 4.0, 'force_history_size': 3},
            'obstacle_detection': {'depth_threshold': 2.5, 'obs_radius': 1.5, 'camera': {'fx': 462.1, 'fy': 462.1, 'cx': 320.0, 'cy': 240.0}},
            'velocity': {'vel_cap': 1.5},
            'altitude': {'takeoff_height': 1.5, 'min_safe_altitude': 1.2, 'max_altitude': 2.5, 'low_altitude_gain': 2.0, 'high_altitude_gain': 1.0},
            'stuck_detection': {'stuck_speed_threshold': 0.15, 'stuck_time_threshold': 3.0, 'escape_duration': 1.5, 'escape_velocity': 0.7, 'escape_upward_bias': 0.5, 'post_escape_boost_duration': 2.0, 'post_escape_k_att_multiplier': 1.5},
            'waypoint_navigation': {'num_waypoints': 3, 'waypoint_reach_threshold': 1.0, 'waypoint_offset_range': 0.75, 'waypoint_vertical_offset_scale': 0.3},
            'goal_settings': {'goal_area': {'x_min': 16.0, 'x_max': 18.0, 'y_min': -3.0, 'y_max': 6.0}, 'altitude_variation': 0.4, 'goal_reach_threshold': 0.8, 'goal_speed_threshold': 0.25, 'hover_duration': 1.0, 'hover_duration_data_collection': 0.1},
            'spawn_settings': {'start_position': {'x': -9.0, 'y': 0.0, 'z': 1.5}, 'respawn_delay': 2.0}, # Matched your YAML for default z too
            'takeoff': {'gains': {'p_gain_large': 0.9, 'p_gain_small': 0.6}, 'max_takeoff_velocity': 1.5, 'normal_takeoff_velocity': 1.2, 'stability': {'altitude_error_threshold': 0.15, 'velocity_threshold': 0.1, 'stable_count_required': 25, 'stable_count_required_data_collection': 5}, 'max_timeout_cycles': 800, 'minimum_acceptable_altitude': 1.0},
            'system': {'control_frequency': 20, 'log_interval': 2.0, 'initial_setpoints': 100},
            'mission': {'continuous_mode': True, 'max_missions': 0, 'mission_timeout': 120, 'data_collection_mode': True},
            'debug': {'verbose_logging': False, 'visualize_forces': False, 'debug_log_interval': 5.0, 'log_stuck_detection': True},
            'safety': {'emergency_min_altitude': 0.5, 'emergency_max_altitude': 3.0, 'max_emergency_velocity': 2.0, 'enable_altitude_limits': True, 'enable_velocity_limits': True}, # Matched YAML for max_emergency_velocity
            'exploration_data_collection': {'enable_agitation': True, 'agitation_probability': 0.05, 'agitation_strength': 0.25, 'agitation_vertical_scale': 0.2}
        }

    def load_parameters(self):
        def get_param(ros_name, config_path_keys, default):
            if rospy.has_param(f"~{ros_name}"): return rospy.get_param(f"~{ros_name}")
            val = self.config
            try:
                for key in config_path_keys: val = val[key]
                return val
            except (KeyError, TypeError): return default

        self.k_att = get_param('k_att', ['force_field', 'k_att'], 0.8)
        self.k_rep = get_param('k_rep', ['force_field', 'k_rep'], 4.0)
        self.force_history_size = int(get_param('force_history_size', ['force_field', 'force_history_size'], 3))

        self.obs_config = self.config.get('obstacle_detection', {})
        self.depth_threshold = get_param('depth_threshold', ['obstacle_detection', 'depth_threshold'], 2.5)
        self.obs_radius = get_param('obs_radius', ['obstacle_detection', 'obs_radius'], 1.5)
        self.fx = get_param('fx', ['obstacle_detection', 'camera', 'fx'], 462.1)
        self.fy = get_param('fy', ['obstacle_detection', 'camera', 'fy'], 462.1)
        self.cx = get_param('cx', ['obstacle_detection', 'camera', 'cx'], 320.0)
        self.cy = get_param('cy', ['obstacle_detection', 'camera', 'cy'], 240.0)
        
        self.vel_cap = get_param('vel_cap', ['velocity', 'vel_cap'], 1.5)

        self.alt_config = self.config.get('altitude', {})
        self.takeoff_height = get_param('takeoff_height', ['altitude', 'takeoff_height'], 1.5)
        self.min_safe_altitude = get_param('min_safe_altitude', ['altitude', 'min_safe_altitude'], 1.2)
        self.max_altitude = get_param('max_altitude', ['altitude', 'max_altitude'], 2.5)
        self.low_altitude_gain = get_param('low_altitude_gain', ['altitude', 'low_altitude_gain'], 2.0)
        self.high_altitude_gain = get_param('high_altitude_gain', ['altitude', 'high_altitude_gain'], 1.0)
        
        self.stuck_config = self.config.get('stuck_detection', {})
        self.stuck_speed_threshold = get_param('stuck_speed_threshold', ['stuck_detection', 'stuck_speed_threshold'], 0.15)
        self.stuck_time_threshold = get_param('stuck_time_threshold', ['stuck_detection', 'stuck_time_threshold'], 3.0)
        self.escape_duration = get_param('escape_duration', ['stuck_detection', 'escape_duration'], 1.5)
        self.escape_velocity_mag = get_param('escape_velocity', ['stuck_detection', 'escape_velocity'], 0.7)
        self.escape_upward_bias = get_param('escape_upward_bias', ['stuck_detection', 'escape_upward_bias'], 0.5)
        # INTEGRATED: Load post_escape parameters
        self.post_escape_boost_duration = get_param('post_escape_boost_duration', ['stuck_detection', 'post_escape_boost_duration'], 2.0)
        self.post_escape_k_att_multiplier = get_param('post_escape_k_att_multiplier', ['stuck_detection', 'post_escape_k_att_multiplier'], 1.5)

        self.waypoint_config = self.config.get('waypoint_navigation', {})
        self.num_waypoints = int(get_param('num_waypoints', ['waypoint_navigation', 'num_waypoints'], 3))
        self.waypoint_reach_threshold = get_param('waypoint_reach_threshold', ['waypoint_navigation', 'waypoint_reach_threshold'], 1.0)
        self.waypoint_offset_range = get_param('waypoint_offset_range', ['waypoint_navigation', 'waypoint_offset_range'], 0.75)
        self.waypoint_vertical_scale = get_param('waypoint_vertical_offset_scale', ['waypoint_navigation', 'waypoint_vertical_offset_scale'], 0.3)
        
        self.goal_config = self.config.get('goal_settings', {})
        self.goal_area = self.goal_config.get('goal_area', {'x_min': 16.0, 'x_max': 18.0, 'y_min': -3.0, 'y_max': 6.0}) 
        self.altitude_variation = get_param('altitude_variation', ['goal_settings', 'altitude_variation'], 0.4)
        self.goal_reach_threshold = get_param('goal_reach_threshold', ['goal_settings', 'goal_reach_threshold'], 0.8)
        self.goal_speed_threshold = get_param('goal_speed_threshold', ['goal_settings', 'goal_speed_threshold'], 0.25)
        
        self.mission_config = self.config.get('mission', {})
        self.data_collection_mode = get_param('data_collection_mode', ['mission', 'data_collection_mode'], True)

        hover_dur_key = 'hover_duration_data_collection' # Specific key from your YAML
        default_hover_val = self.goal_config.get('hover_duration', 1.0) # Fallback to general hover if specific key missing
        if self.data_collection_mode:
            final_hover_val = self.goal_config.get(hover_dur_key, 0.1) # Use data collection hover if specified
        else:
            final_hover_val = self.goal_config.get('hover_duration', 1.0)
        self.hover_duration = rospy.Duration(final_hover_val)

        self.spawn_settings_config = self.config.get('spawn_settings', {}) # Renamed for clarity as it's a dict now
        self.respawn_delay = get_param('respawn_delay', ['spawn_settings', 'respawn_delay'], 2.0)
        
        self.takeoff_config = self.config.get('takeoff', {})
        self.takeoff_gains = self.takeoff_config.get('gains', {'p_gain_large': 0.9, 'p_gain_small': 0.6})
        self.takeoff_stability_config = self.takeoff_config.get('stability', {}) # Renamed for clarity
        
        stable_count_key = 'stable_count_required_data_collection' # Specific key
        default_stable_count_dc = self.takeoff_stability_config.get('stable_count_required_data_collection', 5) # Use default from YAML structure
        default_stable_count_normal = self.takeoff_stability_config.get('stable_count_required', 25)
        self.stable_count_req = int(get_param(stable_count_key if self.data_collection_mode else 'stable_count_required', 
                                             ['takeoff', 'stability', stable_count_key if self.data_collection_mode else 'stable_count_required'], 
                                             default_stable_count_dc if self.data_collection_mode else default_stable_count_normal))
        
        self.max_takeoff_vel = get_param('max_takeoff_velocity', ['takeoff', 'max_takeoff_velocity'], 1.5)
        self.normal_takeoff_vel = get_param('normal_takeoff_velocity', ['takeoff', 'normal_takeoff_velocity'], 1.2) 
        self.takeoff_alt_err_thresh = get_param('altitude_error_threshold', ['takeoff', 'stability', 'altitude_error_threshold'], 0.15)
        self.takeoff_vel_thresh = get_param('velocity_threshold', ['takeoff', 'stability', 'velocity_threshold'], 0.1)
        self.takeoff_max_timeout_cycles = int(get_param('max_timeout_cycles', ['takeoff', 'max_timeout_cycles'], 800))
        self.takeoff_min_alt = get_param('minimum_acceptable_altitude', ['takeoff', 'minimum_acceptable_altitude'], 1.0)

        self.system_config = self.config.get('system', {})
        self.control_frequency = int(get_param('control_frequency', ['system', 'control_frequency'], 20))
        self.log_interval = get_param('log_interval', ['system', 'log_interval'], 2.0)
        self.initial_setpoints_count = int(get_param('initial_setpoints', ['system', 'initial_setpoints'], 100))
        
        self.continuous_mode = get_param('continuous_mode', ['mission', 'continuous_mode'], True)
        self.max_missions = int(get_param('max_missions', ['mission', 'max_missions'], 0))
        self.mission_timeout_sec = get_param('mission_timeout', ['mission', 'mission_timeout'], 120)
        
        self.debug_config = self.config.get('debug', {}) 
        
        self.safety_config = self.config.get('safety', {})
        self.enable_altitude_limits_safety = get_param('enable_altitude_limits', ['safety', 'enable_altitude_limits'], True)
        self.enable_velocity_limits_safety = get_param('enable_velocity_limits', ['safety', 'enable_velocity_limits'], True)
        # max_emergency_velocity used to be safety_config.get, direct mapping now
        self.max_emergency_vel = get_param('max_emergency_velocity', ['safety', 'max_emergency_velocity'], 2.0)

        # INTEGRATED: Exploration & Data Collection Specific Parameters
        self.exploration_config = self.config.get('exploration_data_collection',{})
        self.enable_agitation = get_param('enable_agitation', ['exploration_data_collection', 'enable_agitation'], True)
        self.agitation_probability = get_param('agitation_probability', ['exploration_data_collection', 'agitation_probability'], 0.05)
        self.agitation_strength = get_param('agitation_strength', ['exploration_data_collection', 'agitation_strength'], 0.25)
        self.agitation_vertical_scale = get_param('agitation_vertical_scale', ['exploration_data_collection', 'agitation_vertical_scale'], 0.2)


    def log_configuration(self):
        rospy.loginfo("[APF] ---- Current Effective Configuration ----")
        rospy.loginfo(f"  Data Collection Mode: {self.data_collection_mode}")
        rospy.loginfo(f"  Force Field: k_att={self.k_att:.2f}, k_rep={self.k_rep:.2f}, history_size={self.force_history_size}")
        rospy.loginfo(f"  Velocity: vel_cap={self.vel_cap:.2f}m/s, max_emergency_vel={self.max_emergency_vel:.2f}m/s")
        rospy.loginfo(f"  Altitude: takeoff_h={self.takeoff_height:.2f}m, safe_min={self.min_safe_altitude:.2f}m, safe_max={self.max_altitude:.2f}m")
        rospy.loginfo(f"  Goal: hover_dur={self.hover_duration.to_sec():.2f}s (active), reach_thresh={self.goal_reach_threshold:.2f}m, speed_thresh={self.goal_speed_threshold:.2f}m/s")
        rospy.loginfo(f"  Takeoff: stable_count_req={self.stable_count_req} (active), gains=[{self.takeoff_gains.get('p_gain_large',0):.2f},{self.takeoff_gains.get('p_gain_small',0):.2f}]")
        rospy.loginfo(f"  Stuck Escape: dur={self.escape_duration:.2f}, vel={self.escape_velocity_mag:.2f}, boost_dur={self.post_escape_boost_duration:.2f}, k_att_mult={self.post_escape_k_att_multiplier:.2f}")
        rospy.loginfo(f"  Exploration: agitation_enabled={self.enable_agitation}, prob={self.agitation_probability:.2f}, strength={self.agitation_strength:.2f}, vert_scale={self.agitation_vertical_scale:.2f}")
        rospy.loginfo(f"  Spawn: fixed_pos_xyz=[{self.fixed_spawn_pos[0]:.2f}, {self.fixed_spawn_pos[1]:.2f}, {self.fixed_spawn_pos[2]:.2f}]")
        rospy.loginfo(f"-------------------------------------------")

    def sample_new_goal(self):
        x = random.uniform(self.goal_area['x_min'], self.goal_area['x_max'])
        y = random.uniform(self.goal_area['y_min'], self.goal_area['y_max'])
        base_z_for_goal_calc = self.fixed_spawn_pos[2] 
        if self.data_collection_mode :
             base_z_for_goal_calc += random.uniform(-0.2,0.2) # Add slight variation for base Z too
        z = base_z_for_goal_calc + random.uniform(-self.altitude_variation, self.altitude_variation)
        z = np.clip(z, self.min_safe_altitude, self.max_altitude)
        goal = np.array([x, y, z])
        rospy.loginfo(f"[APF] New goal sampled: [{goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}] (Ref Z for goal calc from spawn: {base_z_for_goal_calc:.2f})")
        return goal

    def generate_waypoints(self, start_pos_for_wp, goal_pos_for_wp):
        waypoints = []
        if self.num_waypoints == 0: 
            waypoints.append(goal_pos_for_wp)
            rospy.loginfo(f"[APF] No intermediate waypoints. Target: {goal_pos_for_wp}")
            return waypoints
        for i in range(1, self.num_waypoints + 1): 
            alpha = i / (self.num_waypoints + 1.0) 
            offset = np.random.uniform(-1.0, 1.0, 3) * self.waypoint_offset_range
            offset[2] *= self.waypoint_vertical_scale 
            waypoint = start_pos_for_wp + alpha * (goal_pos_for_wp - start_pos_for_wp) + offset
            if i == 1 and self.data_collection_mode:
                 # Target mission's takeoff_height for the first waypoint after start_pos_for_wp
                 waypoint[2] = np.clip(waypoint[2], self.takeoff_height - 0.3, self.takeoff_height + 0.3) 
            waypoint[2] = np.clip(waypoint[2], self.min_safe_altitude, self.max_altitude)
            waypoints.append(waypoint)
        waypoints.append(goal_pos_for_wp) 
        rospy.loginfo(f"[APF] Generated {len(waypoints)} waypoints from {start_pos_for_wp} to {goal_pos_for_wp}. First intermediate: {waypoints[0] if waypoints else 'N/A'}")
        return waypoints
    
    def respawn_at_start(self):
        rospy.loginfo("[APF] Attempting to respawn drone to fixed start position...")
        self.navigating = False 
        for _ in range(5): self.send_velocity_command(np.zeros(3)); self.rate.sleep()
        state_msg = ModelState(); state_msg.model_name = 'iris' 
        target_spawn_pos_vec = self.fixed_spawn_pos
        state_msg.pose.position.x = target_spawn_pos_vec[0]; state_msg.pose.position.y = target_spawn_pos_vec[1]; state_msg.pose.position.z = target_spawn_pos_vec[2]
        state_msg.pose.orientation.x = 0.0; state_msg.pose.orientation.y = 0.0; state_msg.pose.orientation.z = 0.0; state_msg.pose.orientation.w = 1.0
        state_msg.twist.linear.x = 0.0; state_msg.twist.linear.y = 0.0; state_msg.twist.linear.z = 0.0
        state_msg.twist.angular.x = 0.0; state_msg.twist.angular.y = 0.0; state_msg.twist.angular.z = 0.0
        try:
            response = self.set_model_state_service(state_msg) 
            if response.success:
                rospy.loginfo(f"[APF] Drone respawn service call to {target_spawn_pos_vec} successful.")
                self.curr_pos = target_spawn_pos_vec.copy(); self.curr_vel = np.zeros(3)
                self.takeoff_complete = False; self.escape_mode = False; self.post_escape_boost_active = False 
                self.force_history.clear(); self.current_waypoint_idx = 0
                wait_start_time = rospy.Time.now()
                settle_duration_sec = max(1.0, self.respawn_delay - 0.1) 
                settle_duration = rospy.Duration(settle_duration_sec)
                rospy.loginfo(f"[APF] Respawn: Waiting up to {settle_duration.to_sec():.1f}s for MAVROS pose to update...")
                initial_settle_loops = 0
                for _ in range(int(self.control_frequency * settle_duration_sec)): 
                    if rospy.is_shutdown(): return False
                    if (rospy.Time.now() - wait_start_time) > settle_duration: break 
                    self.send_velocity_command(np.zeros(3)) 
                    dist_to_spawn = np.linalg.norm(self.curr_pos - target_spawn_pos_vec)
                    if initial_settle_loops < self.control_frequency : 
                         rospy.loginfo(f"[APF]  Settle check {initial_settle_loops}: Dist to spawn: {dist_to_spawn:.3f}m. Curr pose: [{self.curr_pos[0]:.3f}, {self.curr_pos[1]:.3f}, {self.curr_pos[2]:.3f}]")
                    elif initial_settle_loops % (self.control_frequency // 2) == 0 : 
                         rospy.loginfo(f"[APF]  Settle check {initial_settle_loops}: Dist to spawn: {dist_to_spawn:.3f}m.")
                    if dist_to_spawn < 0.3: 
                        rospy.loginfo(f"[APF] MAVROS Pose settled near spawn point: {self.curr_pos}")
                        return True
                    self.rate.sleep(); initial_settle_loops += 1
                rospy.logwarn(f"[APF] MAVROS Pose not settled close. Last: {self.curr_pos}, Dist: {np.linalg.norm(self.curr_pos - target_spawn_pos_vec):.2f}m. Proceeding.")
                return True 
            else: rospy.logerr(f"[APF] Respawn service call failed: {response.status_message}"); return False
        except rospy.ServiceException as e: rospy.logerr(f"[APF] Respawn service exception: {e}"); return False

    def state_callback(self, msg):
        self.current_state = msg

    def pose_callback(self, msg):
        new_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        if self.start_pos_of_session is None and not np.allclose(new_pos, np.zeros(3), atol=0.1):
            self.start_pos_of_session = new_pos.copy()
            rospy.loginfo(f"[APF] Overall session's initial drone position recorded: {self.start_pos_of_session}")
        self.curr_pos = new_pos

    def velocity_callback(self, msg):
        self.curr_vel = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

    def depth_callback(self, msg):
        try:
            cv_image_original = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_img_cv = cv_image_original.copy() 
            if depth_img_cv.dtype != np.float32: 
                depth_img_cv = depth_img_cv.astype(np.float32)
                if msg.encoding == "16UC1": depth_img_cv /= 1000.0
            depth_img_cv[np.isnan(depth_img_cv)] = np.inf 
            depth_img_cv[depth_img_cv <= 0.05] = np.inf 
            self.depth_image = median_filter(depth_img_cv, size=3)
        except CvBridgeError as e: rospy.logerr_throttle(5.0, f"[APF] Depth CvBridge error: {e}")
        except Exception as e_gen: rospy.logerr_throttle(5.0, f"[APF] Depth processing error: {e_gen}")

    def detect_obstacles(self):
        # (Aligned with your original YAML logic for this node's rep_force calculation)
        if self.depth_image is None:
            self.last_repulsive_force_raw = np.zeros(3)
            return False, np.zeros(3)
        h, w = self.depth_image.shape
        roi_h_start, roi_h_end = h // 6, 5 * h // 6 # Wider vertical ROI from original
        roi_w_start, roi_w_end = w // 6, 5 * w // 6 
        roi = self.depth_image[roi_h_start:roi_h_end, roi_w_start:roi_w_end]
        close_obstacle_mask = roi < self.depth_threshold
        
        if not np.any(close_obstacle_mask):
            self.last_repulsive_force_raw = np.zeros(3)
            return False, np.zeros(3)
        
        base_rep_force_sum = np.zeros(3); count = 0
        # This uses the 'step' method from your original code
        step = max(1, min(h,w) // 20) # Original stepping logic
        for v_img in range(0, h, step): # Iterate over full image, not just ROI (original logic)
            for u_img in range(0, w, step):
                depth = self.depth_image[v_img, u_img]
                if np.isfinite(depth) and 0.1 < depth < self.depth_threshold:
                    x_cam = (u_img - self.cx) * depth / self.fx
                    y_cam = (v_img - self.cy) * depth / self.fy
                    obstacle_body = np.array([depth, -x_cam, -y_cam]) # X-fwd, Y-left, Z-up
                    # Transform to world - This needs proper rotation from drone orientation for accuracy
                    # The original script simplified this. Sticking to that:
                    obstacle_world = self.curr_pos + obstacle_body 
                    vec_to_drone_world = self.curr_pos - obstacle_world
                    dist = np.linalg.norm(vec_to_drone_world)
                    if 0.1 < dist < self.obs_radius:
                        strength = (self.obs_radius - dist) / self.obs_radius
                        base_rep_force_sum += strength * (vec_to_drone_world / (dist + 0.1)) # Original had 0.1 denominator
                        count += 1
        if count > 0:
            avg_base_rep_force = base_rep_force_sum / count
            # k_rep scaling done in compute_velocity_command to match original script
            self.last_repulsive_force_raw = avg_base_rep_force.copy()
            return True, avg_base_rep_force
        self.last_repulsive_force_raw = np.zeros(3)
        return False, np.zeros(3)

    def compute_attractive_force(self):
        # (Uses post_escape_k_att_multiplier)
        if not self.waypoints or self.current_waypoint_idx >= len(self.waypoints):
            if self.goal_pos is None: return np.zeros(3)
            target = self.goal_pos
            is_final_goal_target = True
        else:
            target = self.waypoints[self.current_waypoint_idx]
            is_final_goal_target = (self.current_waypoint_idx == len(self.waypoints) - 1)
        direction = target - self.curr_pos
        distance = np.linalg.norm(direction)
        if distance < 0.1: return np.zeros(3) # Your original was 0.1
        unit_direction = direction / (distance + 1e-6) 
        current_k_att_val = self.k_att
        if self.post_escape_boost_active: 
            current_k_att_val *= self.post_escape_k_att_multiplier
            rospy.logdebug_throttle(0.5, f"[APF] Attractive boost. k_att used: {current_k_att_val:.2f}")
        
        magnitude = current_k_att_val
        # Using your original scaling when close to goal
        if is_final_goal_target and distance < 2.0: 
            magnitude *= (distance / 2.0) 
        # No min clamping or special intermediate WP scaling in original
        return magnitude * unit_direction

    def update_waypoint(self):
        # (Matches your original logic, plus log_once fix)
        if self.waypoints and self.current_waypoint_idx < len(self.waypoints):
            current_target_wp = self.waypoints[self.current_waypoint_idx]
            distance_to_wp = np.linalg.norm(current_target_wp - self.curr_pos)
            is_final_wp_in_list = (self.current_waypoint_idx == len(self.waypoints) - 1)
            threshold = self.goal_reach_threshold if is_final_wp_in_list else self.waypoint_reach_threshold

            if distance_to_wp < threshold:
                if is_final_wp_in_list:
                    if not self.logged_approaching_final_wp_for_current_target:
                        rospy.loginfo(f"[APF] Approaching final target WP {self.current_waypoint_idx + 1}. Dist: {distance_to_wp:.2f}m. Goal logic in navigate_to_goal takes over.")
                        self.logged_approaching_final_wp_for_current_target = True 
                else: 
                    rospy.loginfo(f"[APF] Reached waypoint {self.current_waypoint_idx +1}, moving to next waypoint") # Matched original log
                    self.current_waypoint_idx += 1 
                    self.force_history.clear(); self.last_progress_dist_to_target = float('inf')
                    self.logged_approaching_final_wp_for_current_target = False 
        else: self.logged_approaching_final_wp_for_current_target = False


    def check_stuck_condition(self):
        # (Using original simpler logic, then add the progress check after if needed)
        current_speed = np.linalg.norm(self.curr_vel)
        current_time = rospy.Time.now()
        
        if current_speed > self.stuck_speed_threshold: # Your original primary check
            self.last_movement_time = current_time
            self.stuck_count = 0 # Your original used stuck_count
            if self.escape_mode:
                if self.debug_config.get('log_stuck_detection', True):
                    rospy.loginfo("[APF] Movement detected, exiting escape mode.")
                self.escape_mode = False
                # Trigger post-escape boost upon exiting due to movement
                self.post_escape_boost_active = True 
                self.post_escape_boost_end_time = rospy.Time.now() + rospy.Duration(self.post_escape_boost_duration)
                rospy.loginfo(f"[APF] Post-escape boost started for {self.post_escape_boost_duration}s after movement.")
            return False 
        
        time_no_movement_sec = (current_time - self.last_movement_time).to_sec()
        
        if not self.escape_mode and time_no_movement_sec > self.stuck_time_threshold:
            self.escape_mode = True
            self.post_escape_boost_active = False 
            self.escape_start_time = current_time
            self.escape_direction = np.random.uniform(-1, 1, 3) # Your original random escape
            self.escape_direction[2] = self.escape_upward_bias 
            if np.linalg.norm(self.escape_direction) > 1e-6: # Avoid division by zero
                 self.escape_direction = self.escape_direction / np.linalg.norm(self.escape_direction)
            else: # Fallback if random vector is zero
                 self.escape_direction = np.array([0.5, 0.5, self.escape_upward_bias]) 
                 self.escape_direction = self.escape_direction / np.linalg.norm(self.escape_direction)
            
            if self.debug_config.get('log_stuck_detection', True):
                rospy.logwarn(f"[APF] Drone stuck! (Speed: {current_speed:.2f} for {time_no_movement_sec:.1f}s). Activating escape. Dir: {self.escape_direction}")
            self.force_history.clear() 
            return True
        
        return self.escape_mode 

    def compute_velocity_command(self):
        # (Aligned with your original structure more, agitation added)
        if not self.takeoff_complete or not self.navigating: 
            return np.zeros(3)

        self.update_waypoint() 
        
        # Manage post-escape boost from successful escape (movement detected)
        if self.post_escape_boost_active and rospy.Time.now() > self.post_escape_boost_end_time:
            self.post_escape_boost_active = False
            rospy.loginfo("[APF] Post-escape boost ended.")

        is_stuck_and_escaping = self.check_stuck_condition()
        if is_stuck_and_escaping: 
            escape_elapsed_time = (rospy.Time.now() - self.escape_start_time).to_sec()
            if escape_elapsed_time < self.escape_duration:
                return self.escape_velocity_mag * self.escape_direction
            else: 
                self.escape_mode = False 
                self.last_movement_time = rospy.Time.now() # Prevent immediate re-stuck
                rospy.loginfo("[APF] Escape maneuver duration ended.")
                # Note: Post-escape boost is triggered when movement is detected by check_stuck_condition
        
        attractive_f = self.compute_attractive_force() # Considers post_escape_boost_active
        obstacle_present, base_repulsive_f = self.detect_obstacles()
        self.obstacle_detected = obstacle_present
        
        # Apply k_rep scaling as in your original script
        scaled_repulsive_f = base_repulsive_f * self.k_rep if obstacle_present else np.zeros(3)
        # Ensure scaled_repulsive_f is world frame if base_repulsive_f is world frame
        
        total_force = attractive_f + scaled_repulsive_f
        
        # Agitation (from your YAML)
        if self.enable_agitation and self.data_collection_mode and random.random() < self.agitation_probability:
            nudge = np.random.uniform(-1, 1, 3) * self.agitation_strength
            nudge[2] *= self.agitation_vertical_scale
            total_force += nudge
            rospy.logdebug(f"[APF] Applied agitation nudge: {nudge}")
        
        # Altitude management (original simple logic)
        # This does not actively try to reach waypoint Z, but rather stays within safe band
        if self.curr_pos[2] < self.min_safe_altitude:
            z_error = self.min_safe_altitude - self.curr_pos[2]
            total_force[2] += self.low_altitude_gain * z_error
            rospy.logwarn_throttle(1.0, f"[APF] Low altitude! Boosting up. CurrZ: {self.curr_pos[2]:.2f}, MinSafe: {self.min_safe_altitude:.2f}")
        elif self.curr_pos[2] > self.max_altitude:
            z_error = self.curr_pos[2] - self.max_altitude
            total_force[2] -= self.high_altitude_gain * z_error
            rospy.logwarn_throttle(1.0, f"[APF] High altitude! Reducing. CurrZ: {self.curr_pos[2]:.2f}, Max: {self.max_altitude:.2f}")


        self.force_history.append(total_force.copy())
        if len(self.force_history) > self.force_history_size: self.force_history.pop(0)
        smooth_force = np.mean(self.force_history, axis=0) if len(self.force_history) > 1 else total_force
        
        velocity_magnitude = np.linalg.norm(smooth_force) 
        
        # Using your original scripts safety check location
        if self.safety_config.get('enable_velocity_limits', True): # Check from config
            # Max emergency velocity from safety config in YAML (e.g., 2.0 or 2.5)
            current_max_emergency_vel = self.max_emergency_vel 
            if velocity_magnitude > current_max_emergency_vel :
                rospy.logwarn_throttle(1.0, f"[APF] Emergency velocity limit on smoothed_force Mag: {velocity_magnitude:.2f} -> {current_max_emergency_vel:.2f}")
                smooth_force = smooth_force / velocity_magnitude * current_max_emergency_vel
                velocity_magnitude = current_max_emergency_vel # Update after cap

        # Apply vel_cap (main operational speed limit)
        if velocity_magnitude > self.vel_cap:
            smooth_force = smooth_force / velocity_magnitude * self.vel_cap
        
        return smooth_force

    def send_velocity_command(self, velocity_cmd):
        msg = TwistStamped(); msg.header.stamp = rospy.Time.now(); msg.header.frame_id = "map" 
        msg.twist.linear.x = velocity_cmd[0]; msg.twist.linear.y = velocity_cmd[1]; msg.twist.linear.z = velocity_cmd[2]
        msg.twist.angular.x = 0.0; msg.twist.angular.y = 0.0; msg.twist.angular.z = 0.0
        self.vel_pub.publish(msg)

    def arm_and_set_offboard(self):
        rospy.loginfo("[APF] Sending initial setpoints before arming/offboard...");
        for _ in range(self.initial_setpoints_count): 
            if rospy.is_shutdown(): return False
            self.send_velocity_command(np.zeros(3)); self.rate.sleep()
        rospy.loginfo("[APF] Attempting to arm and set OFFBOARD mode...")
        last_request_time = rospy.Time.now() - rospy.Duration(5.0) 
        request_interval = rospy.Duration(1.0) 
        arm_offboard_timeout = rospy.Duration(15.0); start_arm_offboard_time = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - start_arm_offboard_time < arm_offboard_timeout):
            current_ros_time = rospy.Time.now()
            if (current_ros_time - last_request_time) >= request_interval :
                if not self.current_state.armed:
                    rospy.loginfo("[APF] Sending arm command...")
                    try:
                        arm_response = self.arm_service(True)
                        if arm_response.success: rospy.loginfo("[APF] Arm command sent, success from service.")
                        else: rospy.logwarn(f"[APF] Arm command service response indicates failure: {arm_response}")
                    except rospy.ServiceException as e: rospy.logerr(f"[APF] Arm service call failed: {e}")
                if self.current_state.armed and self.current_state.mode != "OFFBOARD": 
                    rospy.loginfo("[APF] Sending set_mode OFFBOARD command...")
                    try:
                        mode_response = self.set_mode_service(custom_mode='OFFBOARD')
                        if mode_response.mode_sent: rospy.loginfo("[APF] Set_mode OFFBOARD command sent, success from service.")
                        else: rospy.logwarn(f"[APF] Set_mode OFFBOARD command service response indicates failure: {mode_response}")
                    except rospy.ServiceException as e: rospy.logerr(f"[APF] Set_mode service call failed: {e}")
                last_request_time = current_ros_time 
            if self.current_state.armed and self.current_state.mode == "OFFBOARD":
                rospy.loginfo("[APF] Drone is ARMED and in OFFBOARD mode."); return True
            self.send_velocity_command(np.zeros(3)); self.rate.sleep()
        rospy.logerr(f"[APF] Timeout or shutdown while attempting to arm and set OFFBOARD mode. Current state: Armed={self.current_state.armed}, Mode='{self.current_state.mode}'")
        return False 

    def takeoff_to_altitude(self):
        # Using the logic from your original script
        target_abs_z = self.takeoff_height 
        rospy.loginfo(f"[APF] Starting takeoff. Current Z: {self.curr_pos[2]:.2f}m, Target Abs Z: {target_abs_z:.2f}m...")
        stable_counter = 0; timeout_cycle_counter = 0
        p_gain_l = self.takeoff_gains.get('p_gain_large',0.8)
        p_gain_s = self.takeoff_gains.get('p_gain_small',0.5)
        max_takeoff_v = self.max_takeoff_vel # From YAML via load_parameters
        normal_takeoff_v = self.normal_takeoff_vel # From YAML
        
        while not rospy.is_shutdown():
            altitude_err = target_abs_z - self.curr_pos[2]
            # Your original gain switching based on 1.0m error
            gain = p_gain_l if abs(altitude_err) > 1.0 else p_gain_s 
            vz_cmd = np.clip(altitude_err * gain, -max_takeoff_v, max_takeoff_v)
            # Your original further clipping if under small gain logic region but command is high
            if abs(altitude_err) <= 1.0 and abs(vz_cmd) > normal_takeoff_v:
                 vz_cmd = np.sign(vz_cmd) * normal_takeoff_v

            self.send_velocity_command([0.0, 0.0, vz_cmd])
            if timeout_cycle_counter % (self.control_frequency) == 0: 
                rospy.loginfo(f"[APF] Takeoff: CurrZ={self.curr_pos[2]:.2f}m, TargetZ={target_abs_z:.2f}m, Err={altitude_err:.2f}m, VzCmd={vz_cmd:.2f}m/s")
            
            is_alt_stable = abs(altitude_err) < self.takeoff_alt_err_thresh # YAML stability.altitude_error_threshold
            is_vel_stable = abs(self.curr_vel[2]) < self.takeoff_vel_thresh # YAML stability.velocity_threshold
            
            if is_alt_stable and is_vel_stable: stable_counter += 1
            else: stable_counter = 0 
            
            if stable_counter >= self.stable_count_req: # self.stable_count_req already respects data_collection_mode
                rospy.loginfo(f"[APF] Takeoff completed and stable at {self.curr_pos[2]:.2f}m.")
                # Post-takeoff hover respects data_collection_mode via self.hover_duration (if its dc part is very short)
                # For clarity, using specific very short post-takeoff hover.
                post_takeoff_hover_s = 0.1 if self.data_collection_mode else 0.5
                post_takeoff_hover_cycles = int(post_takeoff_hover_s * self.control_frequency)
                for _ in range(post_takeoff_hover_cycles):
                    if rospy.is_shutdown(): return False
                    self.send_velocity_command(np.zeros(3)); self.rate.sleep()
                self.last_movement_time = rospy.Time.now(); self.last_progress_time = rospy.Time.now(); self.last_progress_dist_to_target = float('inf')
                self.takeoff_complete = True; return True
            
            timeout_cycle_counter += 1
            if timeout_cycle_counter >= self.takeoff_max_timeout_cycles: # YAML takeoff.max_timeout_cycles
                if self.curr_pos[2] > self.takeoff_min_alt: # YAML takeoff.minimum_acceptable_altitude
                    rospy.logwarn(f"[APF] Takeoff timeout, but alt ({self.curr_pos[2]:.2f}m) acceptable. Proceeding.")
                    self.last_movement_time = rospy.Time.now(); self.last_progress_time = rospy.Time.now(); self.last_progress_dist_to_target = float('inf')
                    self.takeoff_complete = True; return True
                else: rospy.logerr(f"[APF] Takeoff failed - alt too low ({self.curr_pos[2]:.2f}m) after timeout."); return False
            self.rate.sleep()
        return False 

    def navigate_to_goal(self):
        # (Uses refined is_last_waypoint_the_target)
        rospy.loginfo(f"[APF] Starting navigation. Current pos: {self.curr_pos}, Goal: {self.goal_pos}")
        self.waypoints = self.generate_waypoints(self.curr_pos, self.goal_pos)
        self.current_waypoint_idx = 0; self.force_history.clear() 
        self.last_progress_dist_to_target = float('inf') ; self.last_progress_time = rospy.Time.now()
        self.logged_approaching_final_wp_for_current_target = False
        near_goal_stabilize_time = None; mission_phase_start_time = rospy.Time.now()
        
        while not rospy.is_shutdown():
            if self.mission_timeout_sec > 0 and \
               (rospy.Time.now() - mission_phase_start_time).to_sec() > self.mission_timeout_sec:
                rospy.logwarn(f"[APF] Navigation phase timeout after {(rospy.Time.now() - mission_phase_start_time).to_sec():.1f}s."); return False 
            if not self.current_state.armed or self.current_state.mode != "OFFBOARD":
                rospy.logwarn_throttle(3.0, "[APF] Nav: Disarmed/not OFFBOARD! Re-engaging.")
                self.navigating = False; self.send_velocity_command(np.zeros(3))
                if not self.arm_and_set_offboard(): rospy.logerr("[APF] Nav: Failed to re-engage. Abort."); return False 
                rospy.loginfo("[APF] Nav: Re-engaged. Resuming."); self.navigating = True 
                self.last_movement_time = rospy.Time.now(); self.last_progress_dist_to_target = float('inf'); self.last_progress_time = rospy.Time.now(); self.force_history.clear() 
            
            velocity_command = self.compute_velocity_command(); self.send_velocity_command(velocity_command)
            dist_to_final_goal = np.linalg.norm(self.goal_pos - self.curr_pos); current_speed = np.linalg.norm(self.curr_vel)
            
            # Correct check if the current target is the actual final goal
            is_last_waypoint_the_target = (not self.waypoints) or \
                                        (self.waypoints and self.current_waypoint_idx == len(self.waypoints) - 1)

            if is_last_waypoint_the_target and \
               dist_to_final_goal < self.goal_reach_threshold and \
               current_speed < self.goal_speed_threshold:
                if near_goal_stabilize_time is None:
                    near_goal_stabilize_time = rospy.Time.now()
                    rospy.loginfo(f"[APF] Near final goal (Dist: {dist_to_final_goal:.2f}m, Speed: {current_speed:.2f}m/s). Stabilizing for {self.hover_duration.to_sec():.1f}s...")
                elif (rospy.Time.now() - near_goal_stabilize_time) > self.hover_duration: 
                    rospy.loginfo("[APF] Final goal reached and stabilized! Mission phase completed."); return True 
            else: near_goal_stabilize_time = None 
            
            if hasattr(self, '_last_nav_log_time') and (rospy.Time.now() - self._last_nav_log_time).to_sec() > self.log_interval:
                wp_info = "Final Goal"
                if self.waypoints and self.current_waypoint_idx < len(self.waypoints): wp_info = f"WP {self.current_waypoint_idx + 1}/{len(self.waypoints)}"
                rospy.loginfo(f"[APF] Nav: D_Goal={dist_to_final_goal:.1f}m, Speed={current_speed:.2f}m/s, Alt={self.curr_pos[2]:.1f}m, Obs={self.obstacle_detected}, Tgt: {wp_info}")
                self._last_nav_log_time = rospy.Time.now()
            elif not hasattr(self, '_last_nav_log_time'): self._last_nav_log_time = rospy.Time.now()
            self.rate.sleep()
        return False 

    def run(self):
        # (Logic to always respawn at fixed_spawn_pos after mission_count > 1)
        rospy.loginfo("[APF] ======== APF Navigation System Starting ========")
        rospy.loginfo(f"[APF] Using Config: Data Collection Mode = {'ACTIVE' if self.data_collection_mode else 'INACTIVE'}")
        while not rospy.is_shutdown() and not self.current_state.connected:
            rospy.logwarn_throttle(5.0,"[APF] Waiting for FCU connection..."); self.rate.sleep()
        if rospy.is_shutdown(): rospy.loginfo("[APF] Shutdown during FCU wait."); return
        
        rospy.loginfo("[APF] Waiting for initial position data from MAVROS...")
        while not rospy.is_shutdown() and (self.start_pos_of_session is None or np.allclose(self.start_pos_of_session, np.zeros(3), atol=0.1)):
            rospy.logwarn_throttle(2.0, f"[APF] Waiting for valid initial drone position (current_pos: {self.curr_pos})..."); self.rate.sleep()
        if rospy.is_shutdown(): rospy.loginfo("[APF] Shutdown during initial pose wait."); return
        rospy.loginfo(f"[APF] Initial session drone position recorded: {self.start_pos_of_session}")
        
        # For the very first mission, start from where the drone actually is.
        # For subsequent missions, respawn_at_start() will reset self.curr_pos to self.fixed_spawn_pos
        self.curr_pos = self.start_pos_of_session.copy() 

        while not rospy.is_shutdown():
            self.mission_count += 1; rospy.loginfo(f"\n{'-'*20} Preparing Mission {self.mission_count} {'-'*20}")
            if self.max_missions > 0 and self.mission_count > self.max_missions:
                rospy.loginfo(f"[APF] Reached max missions ({self.max_missions}). Exiting."); break
            
            if self.mission_count > 1: # For all missions AFTER the first one
                rospy.loginfo(f"[APF] Mission {self.mission_count}: Attempting respawn to fixed start position.")
                if not self.respawn_at_start(): # This sets self.curr_pos to self.fixed_spawn_pos
                    rospy.logerr("[APF] Respawn failed before starting mission. Exiting."); break
            # self.curr_pos is now either initial_session_pos (for mission 1) or fixed_spawn_pos (for missions > 1)
            
            self.navigating = False; self.send_velocity_command(np.zeros(3))
            self.takeoff_complete = False; self.escape_mode = False; self.post_escape_boost_active = False
            self.logged_approaching_final_wp_for_current_target = False 

            self.goal_pos = self.sample_new_goal() # Goal sampling logic uses fixed_spawn_pos Z for reference
            rospy.loginfo(f"[APF] Mission {self.mission_count}: Starting Pos for this Mission (after any respawn/before takeoff): {self.curr_pos}, New Goal: {self.goal_pos}")
            
            if not self.arm_and_set_offboard():
                rospy.logerr(f"[APF] Mission {self.mission_count}: Failed to arm/set_offboard. Retrying...");
                if self.continuous_mode:
                    if self.mission_count > 1: # No need to respawn if first mission arm fails from initial spot
                        if not self.respawn_at_start(): rospy.logerr("[APF] Respawn fail after arm fail. Exit."); break
                else: break 
                rospy.sleep(1.0); continue 
            
            if not self.takeoff_to_altitude(): # Takeoff will be from self.curr_pos
                rospy.logerr(f"[APF] Mission {self.mission_count}: Takeoff failed.")
                if self.continuous_mode:
                    rospy.logwarn("[APF] Respawn after takeoff failure...")
                    if self.respawn_at_start(): rospy.loginfo("[APF] Respawn successful.")
                    else: rospy.logerr("[APF] Respawn failed. Exit."); break 
                    continue 
                else: rospy.loginfo("[APF] Non-continuous & takeoff fail. Exit."); break 
            
            rospy.loginfo(f"[APF] Mission {self.mission_count}: Post-takeoff pos: {self.curr_pos}. Nav to goal: {self.goal_pos}")
            self.navigating = True 
            mission_success = self.navigate_to_goal()
            self.navigating = False; self.send_velocity_command(np.zeros(3)) 

            if mission_success:
                rospy.loginfo(f"[APF] Mission {self.mission_count} COMPLETED SUCCESSFULLY!")
                post_mission_hover_s = self.goal_config.get('hover_duration_data_collection', 0.1) if self.data_collection_mode else self.goal_config.get('hover_duration', 1.0)
                post_mission_hover_cycles = int(post_mission_hover_s * self.control_frequency)
                rospy.loginfo(f"[APF] Post-mission hover for {post_mission_hover_s:.1f}s...")
                for _ in range(post_mission_hover_cycles):
                    if rospy.is_shutdown(): break
                    self.send_velocity_command(np.zeros(3)); self.rate.sleep()
                if rospy.is_shutdown(): break
                if not self.continuous_mode: rospy.loginfo("[APF] Continuous mode disabled. Exiting."); break
                
                rospy.loginfo("[APF] Respawning to fixed start for next mission.")
                if not self.respawn_at_start(): rospy.logerr("[APF] Respawn failed. Exiting."); break
            else: 
                rospy.logwarn(f"[APF] Mission {self.mission_count} FAILED or was interrupted.")
                if not self.continuous_mode : rospy.loginfo("[APF] Non-continuous & mission fail. Exiting."); break
                
                rospy.logwarn("[APF] Respawning to fixed start after mission failure.")
                if not self.respawn_at_start(): rospy.logerr("[APF] Respawn failed. Exiting."); break
        
        rospy.loginfo("[APF] ======== APF Navigation System Shutting Down ========")

def main():
    config_file_path_arg = None
    cleaned_args = rospy.myargv(argv=sys.argv) 
    if len(cleaned_args) > 1:
        path_candidate = cleaned_args[1]
        if not (path_candidate.startswith('__') or ':=' in path_candidate):
             expanded_path = os.path.expanduser(path_candidate)
             if os.path.isfile(expanded_path):
                config_file_path_arg = expanded_path
                rospy.loginfo(f"[APF Main] Using config file from argument: {config_file_path_arg}")
             else:
                rospy.logwarn(f"[APF Main] Argument '{path_candidate}' provided but not a valid file path. Will try default config locations.")
        else:
            rospy.loginfo(f"[APF Main] Argument '{path_candidate}' appears to be a ROS parameter. Will load config via default paths or internal defaults.")
    try:
        navigator = APFDepthNavigator(config_file_path_arg)
        if hasattr(navigator, 'rate') and navigator.rate is not None: 
            navigator.run()
        else:
            rospy.logerr("[APF Main] Navigator initialization did not complete successfully (e.g., services not found). Cannot run.")
    except rospy.ROSInterruptException:
        rospy.loginfo("[APF Main] Mission interrupted by user (ROSInterruptException).")
    except Exception as e:
        rospy.logfatal(f"[APF Main] Unhandled critical exception in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rospy.loginfo("[APF Main] Final shutdown procedures.")

if __name__ == '__main__':
    main()