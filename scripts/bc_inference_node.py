#!/usr/bin/env python3

import rospy
import torch
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os
import time # For robust waiting

# ROS Messages
from sensor_msgs.msg import Image as RosImage
from nav_msgs.msg import Odometry # Preferred for getting body-frame velocities
from geometry_msgs.msg import TwistStamped, Point
from visualization_msgs.msg import Marker
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

# Model and Preprocessing (Assume these are accessible in PYTHONPATH or same directory)
# If not, you'd copy the necessary functions/classes here.
from bc_model import create_model # Assumes your bc_model.py is findable

# --- Centralized Preprocessing Function (Mirror of il_dataloader.py) ---
def preprocess_depth_for_model_input_inference(depth_image_meters, target_size=(60, 90), max_depth_value=10.0):
    """
    Preprocesses a depth image (already in meters) for model input.
    - Clips and Normalizes by max_depth_value to [0,1]
    - Resizes to target_size
    Args:
        depth_image_meters (np.ndarray): Depth image in float32, units of meters.
        target_size (tuple): (height, width) for resizing.
        max_depth_value (float): Maximum depth value for clipping and normalization.
    Returns:
        np.ndarray: Processed depth image, float32, normalized and resized [0,1].
    """
    # 1. Handle NaNs/Infs if necessary (simulator might give clean data)
    img_processed = np.nan_to_num(depth_image_meters, nan=0.0, posinf=max_depth_value, neginf=0.0)
    
    # 2. Clip to reasonable depth range and normalize to [0,1]
    img_normalized = np.clip(img_processed, 0, max_depth_value) / max_depth_value
    
    # 3. Resize to target size
    img_resized = cv2.resize(img_normalized, (target_size[1], target_size[0]))
    
    return img_resized
# --- End Preprocessing ---


class BCInferenceNodeClean:
    def __init__(self):
        rospy.init_node('bc_inference_node_clean')

        # --- Essential Parameters (Load from ROS params) ---
        self.model_path = rospy.get_param('~model_path', '/catkin_ws/src/apf_depth_nav_node/scripts/bc_training/full_lr0.0001_seed42_20250606_032242/final_model.pt') # Critical
        self.image_topic = rospy.get_param('~image_topic', '/camera/depth/image_raw')
        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom') # For current velocity state
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/mavros/setpoint_velocity/cmd_vel')

        self.takeoff_altitude = rospy.get_param('~takeoff_altitude', 1.5) # meters
        self.control_frequency = rospy.get_param('~control_frequency', 20) # Hz
        self.max_pred_velocity = rospy.get_param('~max_pred_velocity', 1.5) # m/s, for clipping model output

        # Visualization and Safety (Optional)
        self.enable_visualization = rospy.get_param('~enable_visualization', True)
        self.min_safety_distance = rospy.get_param('~min_safety_distance', 0.75) # meters, for basic safety stop

        # --- Model & Preprocessing Config (DEFAULTS - will be overridden by checkpoint config if found) ---
        self.model_input_image_size = (60, 90)
        self.model_input_max_depth = 10.0
        self.model_velocity_dim = 4 # Default (vx,vy,vz,yaw_rate)
        # Default model creation parameters (if not found in config)
        self.default_model_type = 'full'
        self.default_dropout_rate = 0.2
        self.default_image_channels = 1 # For create_model if needed
        self.default_output_dim = 3     # For create_model if needed

        # --- Initialization ---
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rate = rospy.Rate(self.control_frequency)
        self.model = None
        self.model_loaded_successfully = False

        # --- State Variables ---
        self.latest_depth_image_raw = None # Raw depth image from sensor (meters)
        self.current_odom = None
        self.mavros_state = State()
        self.is_ready_for_control = False
        self.emergency_stop_active = False

        # --- Load Model ---
        if not self._load_model_and_config():
            rospy.signal_shutdown("Failed to load model or critical configuration.")
            return

        # --- ROS Subscribers ---
        rospy.Subscriber(self.image_topic, RosImage, self._image_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_callback, queue_size=1)
        rospy.Subscriber('/mavros/state', State, self._mavros_state_callback, queue_size=1)

        # --- ROS Publishers ---
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, TwistStamped, queue_size=1)
        if self.enable_visualization:
            self.debug_img_pub = rospy.Publisher('~debug_processed_image', RosImage, queue_size=1)
            self.velocity_marker_pub = rospy.Publisher('~predicted_velocity_marker', Marker, queue_size=1)

        # --- ROS Service Clients ---
        self._connect_mavros_services()

        rospy.on_shutdown(self._shutdown_hook)
        rospy.loginfo(f"BC Inference Node initialized. Device: {self.device}.")
        # This log will now reflect actual loaded values if config was found
        rospy.loginfo(f"Effective model input image size: {self.model_input_image_size}, Max depth: {self.model_input_max_depth}m, Vel_dim: {self.model_velocity_dim}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ MODIFIED SECTION START: _load_model_and_config +++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _load_model_and_config(self):
        if not os.path.exists(self.model_path):
            rospy.logerr(f"Model file not found: {self.model_path}")
            return False
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            rospy.loginfo(f"Checkpoint loaded from {self.model_path}")

            train_config = None
            config_key_found = None

            if 'config' in checkpoint and checkpoint['config'] is not None:
                train_config = checkpoint['config']
                config_key_found = 'config'
            elif 'config_snapshot' in checkpoint and checkpoint['config_snapshot'] is not None:
                train_config = checkpoint['config_snapshot']
                config_key_found = 'config_snapshot'

            # Parameters for creating the model instance
            model_type_to_use = self.default_model_type
            dropout_to_use = self.default_dropout_rate # Assuming create_model takes 'dropout'
            image_channels_to_use = self.default_image_channels # For create_model if it uses it
            output_dim_to_use = self.default_output_dim # For create_model if it uses it

            if train_config:
                rospy.loginfo(f"Loaded training config from checkpoint using key '{config_key_found}'.")
                # Log the loaded config for debugging
                # import json
                # rospy.loginfo(f"Full train_config: {json.dumps(train_config, indent=2)}")


                # Override inference node's defaults with values from the loaded training configuration
                # Preprocessing parameters
                self.model_input_image_size = tuple(train_config.get('image_size', self.model_input_image_size))
                self.model_input_max_depth = train_config.get('max_depth', self.model_input_max_depth)
                
                # Model architecture parameters
                # self.model_velocity_dim is crucial for _get_current_velocity_input_tensor AND create_model
                self.model_velocity_dim = train_config.get('velocity_dim', self.model_velocity_dim)
                if self.model_velocity_dim != 4:
                     rospy.logwarn(f"Model configured with velocity_dim={self.model_velocity_dim} from checkpoint. Ensure this is intended for model input and creation.")

                model_type_to_use = train_config.get('model_type', self.default_model_type)
                # Training script uses 'dropout_rate', create_model in inference script likely expects 'dropout'
                dropout_to_use = train_config.get('dropout_rate', self.default_dropout_rate)
                image_channels_to_use = train_config.get('image_channels', self.default_image_channels)
                output_dim_to_use = train_config.get('output_dim', self.default_output_dim)

            else:
                rospy.logwarn(f"No 'config' or 'config_snapshot' dictionary in checkpoint (or it's None). Using default parameters defined in inference node.")
                # model_type_to_use and dropout_to_use will remain their defaults set above

            rospy.loginfo(f"Creating model with: type='{model_type_to_use}', vel_dim={self.model_velocity_dim}, "
                          f"img_h={self.model_input_image_size[0]}, img_w={self.model_input_image_size[1]}, "
                          f"dropout={dropout_to_use}, img_channels={image_channels_to_use}, output_dim={output_dim_to_use}")

            self.model = create_model(
                model_type=model_type_to_use,
                image_channels=image_channels_to_use, # Pass if your create_model uses it
                velocity_dim=self.model_velocity_dim,
                output_dim=output_dim_to_use,         # Pass if your create_model uses it
                image_height=self.model_input_image_size[0],
                image_width=self.model_input_image_size[1],
                dropout_rate=dropout_to_use # IMPORTANT: Ensure create_model expects 'dropout_rate'
                                             # OR if it expects 'dropout', change this to 'dropout=dropout_to_use'
                                             # Based on your train_bc.py, create_model takes 'dropout_rate'
            )

            state_dict_key = None
            if 'model_state_dict' in checkpoint:
                state_dict_key = 'model_state_dict'
            # Add more checks if the state_dict might be under other keys or directly at the root
            # For instance, if the checkpoint IS the state_dict:
            elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()) and \
                 any(k.endswith(('.weight', '.bias', '.running_mean', '.running_var', '.num_batches_tracked')) for k in checkpoint.keys()):
                # This heuristic tries to guess if the checkpoint itself is a state_dict
                rospy.loginfo("Checkpoint appears to be a raw state_dict.")
                # In this case, no specific key, the checkpoint *is* the state_dict.
            else:
                rospy.logwarn("Could not find 'model_state_dict' key. Attempting to load checkpoint directly (assuming it's the state_dict).")


            if state_dict_key:
                self.model.load_state_dict(checkpoint[state_dict_key])
            else:
                # If no specific key, assume the checkpoint itself is the state_dict
                # This is a common practice if only the model weights are saved.
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded_successfully = True
            rospy.loginfo("Model loaded and set to evaluation mode.")
            return True
        except Exception as e:
            rospy.logerr(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ MODIFIED SECTION END +++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _connect_mavros_services(self):
        rospy.loginfo("Waiting for MAVROS services...")
        try:
            rospy.wait_for_service('/mavros/cmd/arming', timeout=15.0)
            rospy.wait_for_service('/mavros/set_mode', timeout=15.0)
            self.arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
            self.set_mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)
            rospy.loginfo("MAVROS services connected.")
        except rospy.ServiceException as e:
            rospy.logerr(f"MAVROS Service connection failed: {e}")
            rospy.signal_shutdown("MAVROS services not available.")
        except rospy.ROSException as e:
            rospy.logerr(f"Timeout waiting for MAVROS services: {e}")
            rospy.signal_shutdown("MAVROS services not available (timeout).")


    def _image_callback(self, msg):
        try:
            if msg.encoding == "32FC1":
                self.latest_depth_image_raw = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            elif msg.encoding == "16UC1":
                depth_mm = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                self.latest_depth_image_raw = depth_mm.astype(np.float32) / 1000.0
            else:
                rospy.logwarn_throttle(5.0, f"Unsupported depth encoding: {msg.encoding}. Expecting '32FC1' or '16UC1'.")
                self.latest_depth_image_raw = None
                return
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error in image_callback: {e}")
            self.latest_depth_image_raw = None

    def _odom_callback(self, msg):
        self.current_odom = msg

    def _mavros_state_callback(self, msg):
        self.mavros_state = msg

    def _preprocess_current_image(self):
        if self.latest_depth_image_raw is None:
            return None
        
        processed_np = preprocess_depth_for_model_input_inference(
            self.latest_depth_image_raw.copy(),
            target_size=self.model_input_image_size, # This will now use the value from config if loaded
            max_depth_value=self.model_input_max_depth # This will now use the value from config if loaded
        )
        
        image_tensor = torch.from_numpy(processed_np).unsqueeze(0).float()
        
        if self.enable_visualization:
            self._publish_debug_image(processed_np)
            
        return image_tensor

    def _get_current_velocity_input_tensor(self):
        if self.current_odom is None:
            rospy.logwarn_throttle(5.0, "Odometry not available, using zero velocity input for model.")
            # self.model_velocity_dim will be set from config if available
            return torch.zeros(1, self.model_velocity_dim, dtype=torch.float32).to(self.device)

        vx = self.current_odom.twist.twist.linear.x
        vy = self.current_odom.twist.twist.linear.y
        vz = self.current_odom.twist.twist.linear.z
        yaw_rate = self.current_odom.twist.twist.angular.z
        
        # self.model_velocity_dim is now set based on loaded config (or default if no config)
        if self.model_velocity_dim == 4:
            vel_np = np.array([vx, vy, vz, yaw_rate], dtype=np.float32)
        elif self.model_velocity_dim == 3:
            # IMPORTANT: Decide which 3 velocities the model was TRAINED on.
            # Example: vx, vy, vz (common for position control, less so for direct vel cmd)
            vel_np = np.array([vx, vy, vz], dtype=np.float32)
            rospy.logwarn_throttle(10.0, f"Using 3D velocity input (vx,vy,vz) as model_velocity_dim is {self.model_velocity_dim}. Ensure this matches training.")
            # Or, if it was vx, vy, yaw_rate:
            # vel_np = np.array([vx, vy, yaw_rate], dtype=np.float32)
            # rospy.logwarn_throttle(10.0, f"Using 3D velocity input (vx,vy,yaw_rate) as model_velocity_dim is {self.model_velocity_dim}. Ensure this matches training.")
        else:
            rospy.logerr_throttle(10.0, f"Unsupported model_velocity_dim: {self.model_velocity_dim} (from config or default). Using zeros for velocity input.")
            vel_np = np.zeros(self.model_velocity_dim, dtype=np.float32)

        return torch.from_numpy(vel_np).unsqueeze(0).to(self.device)

    def _perform_safety_check(self):
        if self.latest_depth_image_raw is None:
            if self.emergency_stop_active:
                 rospy.logwarn_throttle(5.0, "SAFETY: No depth image, maintaining emergency stop.")
            return self.emergency_stop_active 

        h, w = self.latest_depth_image_raw.shape
        roi_h_start = int(h * 0.3)
        roi_h_end = int(h * 0.7)
        roi_w_start = int(w * 0.3)
        roi_w_end = int(w * 0.7)
        center_roi = self.latest_depth_image_raw[roi_h_start:roi_h_end, roi_w_start:roi_w_end]
        
        valid_depths_in_roi = center_roi[center_roi > 0.01] 

        obstacle_too_close = False
        min_dist_in_roi_for_log = -1.0

        if valid_depths_in_roi.size > 0:
            min_dist_in_roi_val = np.min(valid_depths_in_roi)
            min_dist_in_roi_for_log = min_dist_in_roi_val 
            if min_dist_in_roi_val < self.min_safety_distance:
                obstacle_too_close = True
        
        if obstacle_too_close:
            if not self.emergency_stop_active: 
                rospy.logwarn(f"SAFETY: Obstacle too close ({min_dist_in_roi_for_log:.2f}m < {self.min_safety_distance:.2f}m). Activating emergency stop.")
            self.emergency_stop_active = True
        else: 
            if self.emergency_stop_active: 
                rospy.loginfo("SAFETY: Path clear or no immediate obstacle. Deactivating emergency stop.")
            self.emergency_stop_active = False
            
        return self.emergency_stop_active


    def _prepare_for_flight(self):
        rospy.loginfo("Preparing for flight...")
        while not rospy.is_shutdown() and not self.mavros_state.connected:
            rospy.logwarn_throttle(5.0,"Waiting for FCU connection...")
            self.rate.sleep()
        if rospy.is_shutdown(): return False

        rospy.loginfo("Sending initial setpoints...")
        for _ in range(100):
            if rospy.is_shutdown(): return False
            self.cmd_vel_pub.publish(TwistStamped()) # Publish empty TwistStamped
            self.rate.sleep()

        rospy.loginfo("Setting OFFBOARD mode...")
        try:
            # Ensure set_mode service is called with base_mode=0 for custom_mode
            if not self.set_mode_service(base_mode=0, custom_mode="OFFBOARD").mode_sent:
                rospy.logerr("Failed to set OFFBOARD mode (mode_sent is False).")
                return False
            mode_wait_start = rospy.Time.now()
            while self.mavros_state.mode != "OFFBOARD":
                if rospy.is_shutdown() or (rospy.Time.now() - mode_wait_start).to_sec() > 10.0:
                    rospy.logerr(f"Timeout or shutdown while waiting for OFFBOARD mode confirmation (current: {self.mavros_state.mode}).")
                    return False
                rospy.logwarn_throttle(1.0, f"Waiting for OFFBOARD mode confirmation (current: {self.mavros_state.mode})")
                # It's good practice to keep sending setpoints while waiting for mode change
                self.cmd_vel_pub.publish(TwistStamped())
                self.rate.sleep()
            rospy.loginfo("OFFBOARD mode confirmed.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call to set OFFBOARD mode failed: {e}")
            return False

        rospy.loginfo("Arming drone...")
        if not self.mavros_state.armed:
            try:
                if not self.arm_service(True).success:
                    rospy.logerr("Failed to arm drone (service call returned success=False).")
                    return False
                arm_wait_start = rospy.Time.now()
                while not self.mavros_state.armed:
                    if rospy.is_shutdown() or (rospy.Time.now() - arm_wait_start).to_sec() > 10.0:
                        rospy.logerr("Timeout or shutdown while waiting for arming confirmation.")
                        return False
                    rospy.logwarn_throttle(1.0, "Waiting for arming confirmation...")
                    self.rate.sleep()
                rospy.loginfo("Drone armed.")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call to arm drone failed: {e}")
                return False
        else:
            rospy.loginfo("Drone already armed.")

        rospy.loginfo(f"Taking off to {self.takeoff_altitude}m...")
        takeoff_start_time = rospy.Time.now()
        initial_z = self.current_odom.pose.pose.position.z if self.current_odom else 0.0

        while not rospy.is_shutdown():
            current_z = self.current_odom.pose.pose.position.z if self.current_odom else initial_z
            
            if current_z >= self.takeoff_altitude * 0.95: # 95% of target altitude
                rospy.loginfo("Reached target takeoff altitude.")
                break
            
            if (rospy.Time.now() - takeoff_start_time).to_sec() > 30.0: # 30s timeout
                rospy.logerr("Takeoff timeout!")
                # Consider disarming or landing here as a safety measure
                return False

            vz_command = 0.3 # Default takeoff speed
            if self.takeoff_altitude - current_z < 0.5 : # Slow down when close
                vz_command = 0.15

            cmd = TwistStamped()
            cmd.header.stamp = rospy.Time.now()
            cmd.header.frame_id = "base_link" # Or "map", "odom" depending on PX4 settings for setpoint_velocity
            cmd.twist.linear.z = vz_command
            self.cmd_vel_pub.publish(cmd)
            self.rate.sleep()
        
        rospy.loginfo("Hovering to stabilize after takeoff...")
        hover_start_time = rospy.Time.now()
        hover_duration = 2.0 # seconds
        while not rospy.is_shutdown() and (rospy.Time.now() - hover_start_time).to_sec() < hover_duration:
            # Send zero velocity to hover
            self.cmd_vel_pub.publish(TwistStamped())
            self.rate.sleep()
        
        self.is_ready_for_control = True
        rospy.loginfo("Preparation for flight complete. Ready for BC control.")
        return True

    def run_control_loop(self):
        if not self.model_loaded_successfully:
            rospy.logerr("Model not loaded. Cannot start control loop.")
            return

        rospy.loginfo("Waiting for initial sensor data (depth, odom)...")
        while not rospy.is_shutdown() and (self.latest_depth_image_raw is None or self.current_odom is None):
            rospy.logwarn_throttle(2.0, "Waiting for sensor data...")
            self.rate.sleep()
        if rospy.is_shutdown(): return

        if not self._prepare_for_flight():
            rospy.logerr("Flight preparation failed. Shutting down.")
            rospy.signal_shutdown("Flight preparation failed.")
            return

        rospy.loginfo("Starting BC model control loop...")
        log_counter = 0
        while not rospy.is_shutdown():
            if not self.is_ready_for_control or not self.mavros_state.armed or self.mavros_state.mode != "OFFBOARD":
                rospy.logwarn_throttle(2.0, f"Not ready for control. Armed: {self.mavros_state.armed}, Mode: {self.mavros_state.mode}. Publishing zero velocity.")
                self.cmd_vel_pub.publish(TwistStamped())
                self.rate.sleep()
                continue

            self._perform_safety_check() 

            if self.emergency_stop_active:
                self.cmd_vel_pub.publish(TwistStamped()) # Publish empty TwistStamped
                # Log is already handled inside _perform_safety_check for activation
                self.rate.sleep()
                continue

            processed_image_tensor = self._preprocess_current_image()
            current_velocity_tensor = self._get_current_velocity_input_tensor()

            if processed_image_tensor is None: # Check if preprocessing failed
                rospy.logwarn_throttle(1.0, "Failed to preprocess image. Sending zero velocity.")
                self.cmd_vel_pub.publish(TwistStamped()) # Publish empty TwistStamped
                self.rate.sleep()
                continue
            
            # Log current velocity input to the model
            if log_counter % (self.control_frequency * 1) == 0: # Log every 1 second (adjust as needed)
                rospy.loginfo(f"Input Vel to Model: {current_velocity_tensor.cpu().numpy().squeeze()}")

            # Add channel dimension for image: [Batch, Channels, Height, Width]
            # Model expects [B, C, H, W] but _preprocess_current_image returns [B, H, W]
            image_input_for_model = processed_image_tensor.unsqueeze(1).to(self.device) # Add channel dim

            try:
                with torch.no_grad():
                    predicted_vel_tensor = self.model(image_input_for_model, current_velocity_tensor)
                
                predicted_vel_np = predicted_vel_tensor.squeeze().cpu().numpy()

                if log_counter % (self.control_frequency * 5) == 0: # Log every 5 seconds
                    rospy.loginfo(f"Raw model output tensor shape: {predicted_vel_tensor.shape}, Squeezed numpy shape: {predicted_vel_np.shape}")

                # Ensure predicted_vel_np is suitable for command (e.g., at least 3 elements for vx,vy,vz)
                if predicted_vel_np.ndim == 0 or predicted_vel_np.size < 3:
                    rospy.logwarn_throttle(1.0, f"Unexpected model output shape: {predicted_vel_np.shape}. Expected at least 3 elements. Using zeros.")
                    predicted_vel_np_3d = np.zeros(3)
                else:
                    predicted_vel_np_3d = predicted_vel_np[:3] # Take first 3 (vx, vy, vz)

                # Clip predicted velocity to a maximum magnitude
                vel_magnitude = np.linalg.norm(predicted_vel_np_3d)
                if vel_magnitude > self.max_pred_velocity:
                    predicted_vel_np_3d = (predicted_vel_np_3d / vel_magnitude) * self.max_pred_velocity
                    if log_counter % (self.control_frequency * 2) == 0: # Log clipping less frequently
                        rospy.loginfo(f"Clipped vel from {vel_magnitude:.2f} to {self.max_pred_velocity:.2f}")

                cmd = TwistStamped()
                cmd.header.stamp = rospy.Time.now()
                cmd.header.frame_id = "base_link" # Ensure this is the correct frame for your setup
                cmd.twist.linear.x = float(predicted_vel_np_3d[0])
                cmd.twist.linear.y = float(predicted_vel_np_3d[1])
                cmd.twist.linear.z = float(predicted_vel_np_3d[2])
                # Yaw rate control: If your model predicts yaw rate as the 4th output and output_dim was >=4
                # And if self.default_output_dim or train_config output_dim was 4.
                # if predicted_vel_np.size >= 4:
                #    cmd.twist.angular.z = float(predicted_vel_np[3])
                # else:
                cmd.twist.angular.z = 0.0 # Default to no yaw rate command
                
                self.cmd_vel_pub.publish(cmd)

                if self.enable_visualization:
                    self._publish_velocity_marker(predicted_vel_np_3d)

                if log_counter % (self.control_frequency * 2) == 0: # Log predicted command every 2 secs
                     rospy.loginfo(f"Pred Vel Cmd: vx={cmd.twist.linear.x:.2f}, vy={cmd.twist.linear.y:.2f}, vz={cmd.twist.linear.z:.2f}, az={cmd.twist.angular.z:.2f}")
                log_counter += 1

            except Exception as e:
                rospy.logerr(f"Error during model inference or publishing: {e}")
                import traceback
                traceback.print_exc()
                self.cmd_vel_pub.publish(TwistStamped()) # Safety: send zero velocity on error

            self.rate.sleep()

    def _publish_debug_image(self, processed_np_image_0_1):
        try:
            # Convert [0,1] float to [0,255] uint8
            debug_img_uint8 = (processed_np_image_0_1 * 255).astype(np.uint8)
            # If it's single channel, cvtColor to BGR for easier visualization in tools like RViz
            if debug_img_uint8.ndim == 2 or debug_img_uint8.shape[-1] == 1:
                 debug_img_colored = cv2.cvtColor(debug_img_uint8, cv2.COLOR_GRAY2BGR)
            else: # Assuming it's already BGR or similar
                 debug_img_colored = debug_img_uint8

            ros_image_msg = self.bridge.cv2_to_imgmsg(debug_img_colored, "bgr8")
            ros_image_msg.header.stamp = rospy.Time.now()
            # ros_image_msg.header.frame_id = "camera_depth_optical_frame" # Or appropriate camera frame
            self.debug_img_pub.publish(ros_image_msg)
        except Exception as e:
            rospy.logwarn(f"Error publishing debug image: {e}")

    def _publish_velocity_marker(self, velocity_np_3d):
        marker = Marker()
        # marker.header.frame_id = "base_link" # Or more specific frame like drone's body frame from odom
        if self.current_odom and self.current_odom.child_frame_id:
             marker.header.frame_id = self.current_odom.child_frame_id # e.g., "base_link" or "iris/base_link"
        else:
             marker.header.frame_id = "base_link" # Fallback
        marker.header.stamp = rospy.Time.now()
        marker.ns = "predicted_velocity"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Arrow start point (origin of the frame_id)
        marker.points.append(Point(0,0,0))
        # Arrow end point (scaled velocity vector)
        # Scale for visualization if velocities are too small/large to see clearly
        arrow_scale = 1.0 # Adjust if needed
        marker.points.append(Point(x=velocity_np_3d[0]*arrow_scale, y=velocity_np_3d[1]*arrow_scale, z=velocity_np_3d[2]*arrow_scale))

        marker.scale.x = 0.05 # Shaft diameter
        marker.scale.y = 0.1  # Head diameter
        marker.scale.z = 0.1  # Head length (if not using ARROW type that defines it)

        marker.color.a = 0.9 # Alpha
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.lifetime = rospy.Duration(0.5) # Marker will disappear after 0.5s if not republished
        self.velocity_marker_pub.publish(marker)


    def _shutdown_hook(self):
        rospy.loginfo("BC Inference Node shutting down. Sending zero velocities.")
        if hasattr(self, 'cmd_vel_pub') and self.cmd_vel_pub.get_num_connections() > 0 :
            # Send zero velocity multiple times to ensure it's received
            for _ in range(int(self.control_frequency / 2)): # Send for 0.5 seconds
                try:
                    self.cmd_vel_pub.publish(TwistStamped())
                    rospy.sleep(1.0 / self.control_frequency) 
                except rospy.ROSException: # E.g. if master is already down or publisher is invalid
                    rospy.logwarn("ROSException during shutdown zero velocity publish.")
                    break
                except Exception as e:
                    rospy.logwarn(f"Exception during shutdown zero velocity publish: {e}")
                    break
        rospy.loginfo("Shutdown zero velocities sent (or attempted).")


if __name__ == '__main__':
    try:
        node = BCInferenceNodeClean()
        # Check model_loaded_successfully before running control loop
        if hasattr(node, 'model_loaded_successfully') and node.model_loaded_successfully:
            node.run_control_loop()
        else:
            rospy.logerr("Failed to initialize or load model. Node will not run control loop. Check logs above for errors.")
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received. Shutting down gracefully.")
    except Exception as e:
        rospy.logfatal(f"Unhandled critical exception in BCInferenceNodeClean __main__: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # The shutdown_hook registered with rospy.on_shutdown handles zero velocity.
        # No need for extra logic here unless the node didn't initialize fully.
        rospy.loginfo("BCInferenceNodeClean main block finished.")