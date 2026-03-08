#!/usr/bin/env python3

import rospy
import torch
# import torch.nn.functional as F # Not used
import numpy as np
import cv2
import os
import json
import pickle
from datetime import datetime
from collections import deque
import threading
# import time # Not used

from geometry_msgs.msg import TwistStamped, PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from mavros_msgs.msg import State

from bc_model import create_model
# --- MODIFIED --- Import the centralized preprocessing function
from il_dataloader import preprocess_depth_for_model_input

class DAggerDataCollector:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        rospy.init_node('dagger_data_collector', anonymous=True)
        self.bridge = CvBridge()
        
        self.load_bc_model()
        
        self.data_buffer = []
        # self.expert_labels = deque(maxlen=1000) # Not directly used for saving, expert_action is saved per datapoint
        self.episode_data = [] # To store lists of trajectories
        self.current_trajectory = [] # A list of datapoints for the current episode
        
        self.current_depth_meters = None # --- MODIFIED --- Store as float32 meters
        self.current_velocity_input = np.zeros(4) # vx, vy, vz, yaw_rate
        self.current_position = np.zeros(3)
        self.expert_action_control = None # 3D: vx, vy, vz from APF
        self.model_action_control = None # 3D: vx, vy, vz from BC model
        self.collecting = False
        self.drone_armed = False # Initialize
        self.drone_mode = "" # Initialize
        
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Topics for expert vs policy commands
        self.expert_cmd_topic = self.config.get('expert_cmd_topic', '/apf/expert_cmd')
        self.policy_cmd_topic = self.config.get('policy_cmd_topic', '/mavros/setpoint_velocity/cmd_vel')
        if self.expert_cmd_topic == self.policy_cmd_topic:
            rospy.logwarn("[DAgger] expert_cmd_topic equals policy_cmd_topic; expert labels will mirror policy outputs. Set a separate expert topic to get real supervision.")

        with open(os.path.join(self.output_dir, 'dagger_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        self.setup_subscribers()
        
        self.collection_thread = None
        self.stop_collection_event = threading.Event() # --- MODIFIED --- Renamed for clarity
        
        rospy.loginfo(f"[DAgger] Data collector initialized. Output: {self.output_dir}")
        rospy.loginfo(f"[DAgger] Using device: {self.device}")

    def load_bc_model(self):
        model_path = self.config['bc_model_path']
        if not os.path.exists(model_path):
            rospy.logerr(f"[DAgger] BC model not found: {model_path}") # --- MODIFIED --- logerr
            raise FileNotFoundError(f"BC model not found: {model_path}")
        
        rospy.loginfo(f"[DAgger] Loading BC model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint.get('config', {}) # If config not in checkpoint, use defaults in create_model
        
        self.bc_model = create_model(
            model_type=model_config.get('model_type', self.config.get('model_type', 'full')),
            velocity_dim=4, # Standard for this project
            image_height=model_config.get('image_size', self.config.get('image_size', [60, 90]))[0],
            image_width=model_config.get('image_size', self.config.get('image_size', [60, 90]))[1],
            dropout=0.0 
        ).to(self.device)
        
        self.bc_model.load_state_dict(checkpoint['model_state_dict'])
        self.bc_model.eval()
        rospy.loginfo("[DAgger] BC model loaded successfully")

    def setup_subscribers(self):
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback, queue_size=1)
        rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, self.velocity_callback, queue_size=1)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_callback, queue_size=1)
        # Expert supervision (should come from APF or human teleop)
        rospy.Subscriber(self.expert_cmd_topic, TwistStamped, self.expert_action_callback, queue_size=1)
        # Optional: tap the policy command being flown for logging/debug
        rospy.Subscriber(self.policy_cmd_topic, TwistStamped, self.policy_action_callback, queue_size=1)
        rospy.Subscriber('/mavros/state', State, self.state_callback, queue_size=1)

    def depth_callback(self, msg):
        try:
            # --- MODIFIED --- Process to float32 meters and store
            if msg.encoding == "32FC1":
                depth_img_meters = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            elif msg.encoding == "16UC1":
                depth_img_uint16 = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                depth_img_meters = depth_img_uint16.astype(np.float32) / 1000.0
            else:
                rospy.logwarn_throttle(5.0, f"[DAgger] Unsupported depth encoding: {msg.encoding}")
                return
            
            depth_img_meters = np.array(depth_img_meters, dtype=np.float32)
            
            max_depth_val = self.config.get('max_depth', 10.0)
            depth_img_meters[np.isnan(depth_img_meters)] = max_depth_val # Replace NaN with max_depth
            depth_img_meters[depth_img_meters <= 0] = max_depth_val      # Replace non-positive with max_depth
            depth_img_meters = np.clip(depth_img_meters, 0, max_depth_val) # Clip to max_depth
            
            self.current_depth_meters = depth_img_meters
            
        except Exception as e:
            rospy.logerr(f"[DAgger] Depth processing error: {e}", exc_info=True)

    def velocity_callback(self, msg):
        self.current_velocity_input = np.array([
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z, msg.twist.angular.z
        ], dtype=np.float32)

    def pose_callback(self, msg):
        self.current_position = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        ], dtype=np.float32)

    def expert_action_callback(self, msg): # APF commands
        self.expert_action_control = np.array([
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
        ], dtype=np.float32)

    def policy_action_callback(self, msg):
        # Command actually being flown (for reference/debug)
        self.policy_action_control = np.array([
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
        ], dtype=np.float32)

    def state_callback(self, msg):
        self.drone_armed = msg.armed
        self.drone_mode = msg.mode

    def predict_action(self, depth_img_meters_input, current_velocity_input_val): # --- MODIFIED --- parameter names
        if depth_img_meters_input is None:
            return np.zeros(3, dtype=np.float32) # Model outputs 3D
        
        try:
            # --- MODIFIED --- Use centralized preprocessing for model input
            # depth_img_meters_input is already float32, in meters, and basic cleaned (NaN, clip)
            processed_depth_np = preprocess_depth_for_model_input(
                depth_img_meters_input, 
                target_size=tuple(self.config.get('image_size', [60, 90])),
                max_depth_value=self.config.get('max_depth', 10.0)
            )
            
            depth_tensor = torch.from_numpy(processed_depth_np).unsqueeze(0).unsqueeze(0).to(self.device).float()
            vel_tensor = torch.from_numpy(current_velocity_input_val).unsqueeze(0).to(self.device).float()
            
            with torch.no_grad():
                prediction = self.bc_model(depth_tensor, vel_tensor) # Should be 3D output
                action = prediction.cpu().numpy().flatten() # Now 3D
            
            # Clamp to reasonable range (for 3D action)
            # If max_velocity is scalar, it applies to all. If it's a list/array of 3, it's per component.
            max_vel_val = self.config.get('max_velocity', 3.0)
            action = np.clip(action, -max_vel_val, max_vel_val)
            
            return action # 3D action
            
        except Exception as e:
            rospy.logerr(f"[DAgger] Model prediction error: {e}", exc_info=True)
            return np.zeros(3, dtype=np.float32) # Model outputs 3D

    def collect_datapoint(self):
        if self.current_depth_meters is None or self.expert_action_control is None:
            # rospy.logwarn_throttle(2.0, "[DAgger] Waiting for depth or expert action to collect datapoint.")
            return False
        
        self.model_action_control = self.predict_action(self.current_depth_meters, self.current_velocity_input)
        
        action_diff_val = np.linalg.norm(self.expert_action_control - self.model_action_control)
        
        datapoint = {
            'timestamp': rospy.Time.now().to_sec(),
            'depth_image_meters': self.current_depth_meters.copy(), # Store the float32 meters image
            'velocity_input': self.current_velocity_input.copy(),
            'position': self.current_position.copy(),
            'expert_action': self.expert_action_control.copy(),
            'model_action': self.model_action_control.copy(),
            'policy_action': self.policy_action_control.copy() if hasattr(self, 'policy_action_control') and self.policy_action_control is not None else None,
            'action_difference': action_diff_val
        }
        
        self.current_trajectory.append(datapoint)
        # self.data_buffer.append(datapoint) # data_buffer will be populated from trajectories later if needed
        
        # Log progress periodically
        if len(self.current_trajectory) % 50 == 0:
             rospy.loginfo(f"[DAgger] Collected {len(self.current_trajectory)} points in current trajectory. Last action_diff: {action_diff_val:.3f}")
        return True

    def start_collection(self):
        if self.collecting:
            rospy.logwarn("[DAgger] Collection already running")
            return
        
        rospy.loginfo("[DAgger] Starting data collection...")
        self.collecting = True
        self.stop_collection_event.clear()
        
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True # Allow main program to exit even if thread is running
        self.collection_thread.start()

    def stop_collection_process(self): # Renamed from stop_collection
        if not self.collecting:
            rospy.logwarn("[DAgger] Collection not running")
            return

        rospy.loginfo("[DAgger] Stopping data collection...")
        self.collecting = False # Signal to loop
        self.stop_collection_event.set() # Signal to thread
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0) # Wait for thread to finish
            if self.collection_thread.is_alive():
                rospy.logwarn("[DAgger] Collection thread did not terminate gracefully.")
        rospy.loginfo("[DAgger] Collection stopped.")


    def _collection_loop(self):
        rate = rospy.Rate(self.config.get('collection_frequency', 10)) # Hz
        
        while not self.stop_collection_event.is_set() and not rospy.is_shutdown():
            if self.collecting and self.drone_armed and "OFFBOARD" in self.drone_mode.upper(): # Check mode case-insensitively
                if not self.collect_datapoint():
                    pass # Optional: log if datapoint collection failed
            elif self.collecting: # If collecting is true but not armed/offboard
                if len(self.current_trajectory) > 0: # Finalize trajectory if drone disarmed/mode changed
                    rospy.loginfo("[DAgger] Drone disarmed or mode changed. Finalizing current trajectory.")
                    self.finalize_trajectory() # Finalize before potentially starting a new one
            
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo("[DAgger] Collection loop interrupted.")
                break
        rospy.loginfo("[DAgger] Collection loop finished.")


    def finalize_trajectory(self):
        if not self.current_trajectory:
            return
            
        trajectory_len = len(self.current_trajectory)
        if trajectory_len > 0: # Min length for a trajectory
            action_diffs = [dp['action_difference'] for dp in self.current_trajectory if 'action_difference' in dp]
            mean_action_diff = np.mean(action_diffs) if action_diffs else 0.0
            max_action_diff = np.max(action_diffs) if action_diffs else 0.0

            trajectory_info = {
                'length': trajectory_len,
                'start_time': self.current_trajectory[0]['timestamp'],
                'end_time': self.current_trajectory[-1]['timestamp'],
                'mean_action_diff': mean_action_diff,
                'max_action_diff': max_action_diff
            }
            
            self.episode_data.append({
                'trajectory_data': self.current_trajectory.copy(), # Store the actual data
                'info': trajectory_info
            })
            
            rospy.loginfo(f"[DAgger] Trajectory finalized: {trajectory_info['length']} points, "
                         f"avg diff: {trajectory_info['mean_action_diff']:.4f}")
            
        self.current_trajectory.clear()

    def save_collected_data(self, iteration_num=None):
        # Consolidate all data from trajectories into a flat list for "raw_datapoints"
        # This is for backward compatibility with the original structure of the pickle file
        # if it expects a flat list of all datapoints.
        all_collected_datapoints = []
        for episode in self.episode_data:
            all_collected_datapoints.extend(episode['trajectory_data'])

        if not all_collected_datapoints:
            rospy.logwarn("[DAgger] No data to save (all_collected_datapoints is empty).")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = "dagger_data"
        if iteration_num is not None:
            filename = f"{base_filename}_iter_{iteration_num}_{timestamp}.pkl"
        else:
            filename = f"{base_filename}_{timestamp}.pkl"
        
        filepath = os.path.join(self.output_dir, filename)
        
        save_data = {
            'config': self.config,
            'collection_info': {
                'timestamp': timestamp,
                'total_datapoints': len(all_collected_datapoints),
                'num_trajectories': len(self.episode_data),
                'iteration': iteration_num
            },
            'trajectories_info': [ep['info'] for ep in self.episode_data], # Summary of trajectories
            'raw_datapoints': all_collected_datapoints # Flat list of all datapoints
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            rospy.loginfo(f"[DAgger] Saved {len(all_collected_datapoints)} datapoints ({len(self.episode_data)} trajectories) to {filepath}")
            self._save_collection_summary(filepath.replace('.pkl', '_summary.json'), all_collected_datapoints)
            return filepath
        except Exception as e:
            rospy.logerr(f"[DAgger] Error saving data to {filepath}: {e}", exc_info=True)
            return None


    def _save_collection_summary(self, filepath, datapoints_list): # --- MODIFIED --- takes list of dps
        if not datapoints_list:
            return
        
        action_diffs = [dp['action_difference'] for dp in datapoints_list]
        expert_actions = np.array([dp['expert_action'] for dp in datapoints_list])
        model_actions = np.array([dp['model_action'] for dp in datapoints_list])
        
        summary = {
            'total_datapoints': len(datapoints_list),
            'num_trajectories': len(self.episode_data), # From self.episode_data
            'action_difference_stats': {
                'mean': float(np.mean(action_diffs)) if action_diffs else 0.0,
                'std': float(np.std(action_diffs)) if action_diffs else 0.0,
                'min': float(np.min(action_diffs)) if action_diffs else 0.0,
                'max': float(np.max(action_diffs)) if action_diffs else 0.0,
                'median': float(np.median(action_diffs)) if action_diffs else 0.0
            },
            'expert_action_stats': {
                'mean': expert_actions.mean(axis=0).tolist() if expert_actions.size > 0 else [0,0,0],
                'std': expert_actions.std(axis=0).tolist() if expert_actions.size > 0 else [0,0,0],
            },
            'model_action_stats': {
                'mean': model_actions.mean(axis=0).tolist() if model_actions.size > 0 else [0,0,0],
                'std': model_actions.std(axis=0).tolist() if model_actions.size > 0 else [0,0,0],
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

    def get_collection_stats(self): # --- MODIFIED --- to use all_collected_datapoints logic
        all_collected_datapoints = []
        for episode in self.episode_data:
            all_collected_datapoints.extend(episode['trajectory_data'])

        if not all_collected_datapoints:
            return {
                'total_points': 0, 'trajectories': 0, 'current_trajectory_length': len(self.current_trajectory),
                'mean_action_diff': 0, 'max_action_diff': 0, 'collecting': self.collecting
            }
        
        action_diffs = [dp['action_difference'] for dp in all_collected_datapoints]
        
        return {
            'total_points': len(all_collected_datapoints),
            'trajectories': len(self.episode_data),
            'current_trajectory_length': len(self.current_trajectory),
            'mean_action_diff': np.mean(action_diffs) if action_diffs else 0.0,
            'max_action_diff': np.max(action_diffs) if action_diffs else 0.0,
            'collecting': self.collecting
        }

    def clear_data(self):
        # self.data_buffer.clear() # Not used directly for primary storage anymore
        self.episode_data.clear()
        self.current_trajectory.clear()
        rospy.loginfo("[DAgger] Data buffer and trajectories cleared")

def main_dagger_collector(): # Renamed to avoid conflict
    # --- MODIFIED --- Example config
    config = {
        'bc_model_path': os.path.expanduser('~/catkin_ws/src/apf_depth_nav_node/scripts/bc_training/full_lr0.0001_seed42_20250603_220558/final_model.pt'), # Example path
        'output_dir': os.path.expanduser('~/catkin_ws/src/apf_depth_nav_node/dagger_data'), # Example path
        'collection_frequency': 10,  # Hz
        'image_size': [60, 90], # Must match model and dataloader
        'max_depth': 10.0,      # Must match dataloader
        'max_velocity': 2.0     # For clamping model output
    }
    
    # Ensure model path exists for testing
    if not os.path.exists(config['bc_model_path']):
        rospy.logerr(f"BC Model for DAgger collector not found at {config['bc_model_path']}. Please train a BC model first or update path.")
        # You might want to create a dummy model file for basic script testing if a real one isn't available
        # For example:
        # dummy_model_dir = os.path.dirname(config['bc_model_path'])
        # os.makedirs(dummy_model_dir, exist_ok=True)
        # dummy_checkpoint = {'model_state_dict': {}, 'config': {'image_size': [60,90], 'model_type': 'lightweight'}}
        # torch.save(dummy_checkpoint, config['bc_model_path'])
        # rospy.loginfo(f"Created a dummy model file at {config['bc_model_path']} for DAgger collector testing.")
        # return # Or raise an error to stop

    collector = DAggerDataCollector(config)
    
    # Cleanup hook
    def shutdown_hook():
        rospy.loginfo("[DAgger] Shutdown hook called.")
        if collector.collecting:
            collector.stop_collection_process()
        collector.finalize_trajectory() # Finalize any pending trajectory
        collector.save_collected_data()
        rospy.loginfo("[DAgger] Data saved on shutdown.")

    rospy.on_shutdown(shutdown_hook)

    try:
        collector.start_collection()
        rospy.loginfo("[DAgger] Collection running. Press Ctrl+C to stop and save.")
        
        # Keep main thread alive, checking status
        while not rospy.is_shutdown():
            stats = collector.get_collection_stats()
            if stats and stats['collecting']:
                rospy.loginfo_throttle(10.0, f"[DAgger] Stats: Points={stats['total_points']}, Trajs={stats['trajectories']}, "
                                           f"CurTrajLen={stats['current_trajectory_length']}, AvgDiff={stats['mean_action_diff']:.3f}")
            elif not stats['collecting'] and collector.collection_thread and collector.collection_thread.is_alive():
                # This case should ideally not happen if stop_collection_process is used correctly
                rospy.logwarn_throttle(5.0, "[DAgger] Collector not 'collecting' but thread is alive.")

            rospy.sleep(1.0) # Loop to keep main thread responsive and for status checks
            
    except rospy.ROSInterruptException:
        rospy.loginfo("[DAgger] Collection interrupted by ROSInterruptException.")
    except Exception as e:
        rospy.logerr(f"[DAgger] Main loop error: {e}", exc_info=True)
    finally:
        # This will be called before the on_shutdown hook if an exception occurs here
        # The on_shutdown hook handles saving, so just ensure collection is stopped.
        if hasattr(collector, 'collecting') and collector.collecting: # Check if collector was fully initialized
             collector.stop_collection_process()
        rospy.loginfo("[DAgger] Main DAgger collector function finished.")


if __name__ == '__main__':
    main_dagger_collector()