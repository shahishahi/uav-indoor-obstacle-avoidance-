#!/usr/bin/env python3

import rospy
import os
import csv # For simplified CSV metadata if needed in future, not main focus now
import rosbag
import math
# import numpy as np # Not strictly needed if not doing complex analysis here
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped, PoseStamped 
from mavros_msgs.msg import State # Only if you decide to record MAVROS state
import message_filters
from datetime import datetime
# from cv_bridge import CvBridge # Not needed if not converting/analyzing images here

class APFBagRecorder:
    def __init__(self):
        rospy.init_node('apf_bag_recorder_node', anonymous=True)
        
        # Core Parameters for the Recorder
        self.save_dir = rospy.get_param('~save_dir', '/home/shahi/apf_logs') # Base directory for all logs
        base_bag_name_param = rospy.get_param('~bag_name', 'apf_depth_vel.bag') # More descriptive default
        self.sync_slop = rospy.get_param('~sync_slop', 0.1) # Synchronization slop in seconds
        self.min_expert_speed_to_record = rospy.get_param('~min_expert_speed_to_record', 0.1) # m/s, slightly lower default to capture more nuanced movements initially

        # Topic names from APF node and drone state
        self.depth_topic_in = '/camera/depth/image_raw'
        self.expert_vel_topic_in = '/mavros/setpoint_velocity/cmd_vel' # APF output
        self.current_vel_topic_in = '/mavros/local_position/velocity_local' # Drone's actual velocity
        self.current_pose_topic_in = '/mavros/local_position/pose' # Drone's actual pose

        # Topic names to be written in the bag (for BC consistency)
        self.depth_topic_out = '/depth_image'
        self.expert_vel_topic_out = '/expert_velocity'
        self.current_vel_topic_out = '/input_velocity' # Often used as state for BC model
        self.current_pose_topic_out = '/pose_data' # Optional: if pose is part of BC model state

        # Optional: Record MAVROS state? (Can make bags large)
        self.record_mavros_state = rospy.get_param('~record_mavros_state', False)
        if self.record_mavros_state:
            self.mavros_state_topic_in = '/mavros/state'
            self.mavros_state_topic_out = '/mavros_state_info'


        os.makedirs(self.save_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_id = f"session_{timestamp_str}"
        
        base_name = base_bag_name_param.replace('.bag', '')
        self.bag_name = f"{base_name}_{timestamp_str}.bag"
        
        self.session_dir = os.path.join(self.save_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        self.bag_path = os.path.join(self.session_dir, self.bag_name)
        
        try:
            self.bag = rosbag.Bag(self.bag_path, 'w', chunk_threshold=768*1024*10) # Write in 5MB chunks
            rospy.loginfo(f"[Recorder] Recording to: {self.bag_path}")
            rospy.loginfo(f"[Recorder] Session ID: {self.session_id}")
            rospy.loginfo(f"[Recorder] Min expert speed to record: {self.min_expert_speed_to_record} m/s")
            rospy.loginfo(f"[Recorder] Sync Slop: {self.sync_slop} s")
        except Exception as e:
            rospy.logfatal(f"[Recorder] Failed to open bag file: {e}")
            rospy.signal_shutdown(f"Failed to open bag: {e}")
            return # Important to stop further execution
        
        self.session_start_time_obj = datetime.now()
        self.ros_start_time = rospy.Time.now() 
        
        self.frame_count_recorded = 0 
        self.frame_count_received_sync = 0
        
        self.create_session_info_file() # Create basic info file
        
        # Setup synchronized subscribers
        depth_sub = message_filters.Subscriber(self.depth_topic_in, Image)
        expert_vel_sub = message_filters.Subscriber(self.expert_vel_topic_in, TwistStamped)
        current_vel_sub = message_filters.Subscriber(self.current_vel_topic_in, TwistStamped)
        current_pose_sub = message_filters.Subscriber(self.current_pose_topic_in, PoseStamped) # Add pose
        
        subscribers_list = [depth_sub, expert_vel_sub, current_vel_sub, current_pose_sub]
        
        # Optional MAVROS state subscriber (not part of the main sync)
        if self.record_mavros_state:
            rospy.Subscriber(self.mavros_state_topic_in, State, self.mavros_state_callback)
            self.last_mavros_state_msg = None # To store the latest state msg

        # Synchronizer for main data recording
        # ApproximateTimeSynchronizer is good for real-world topics that might not be perfectly aligned
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subscribers_list,
            queue_size=50,  # Larger queue if messages can arrive with some delay variance
            slop=self.sync_slop 
        )
        self.sync.registerCallback(self.synced_callback)
        
        rospy.on_shutdown(self.cleanup)
        rospy.loginfo("[Recorder] Node initialized. Recording synchronized navigation data...")
    
    def create_session_info_file(self):
        """Create a basic session info file for this recording session"""
        info_path = os.path.join(self.session_dir, 'session_info.txt')
        try:
            with open(info_path, 'w') as f:
                f.write(f"APF Data Recording Session\n")
                f.write(f"===========================\n")
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Start Time: {self.session_start_time_obj.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Bag File: {self.bag_name}\n")
                f.write(f"Save Directory: {self.session_dir}\n")
                f.write(f"Configured Parameters:\n")
                f.write(f"  - Sync Slop: {self.sync_slop}s\n")
                f.write(f"  - Min Expert Speed to Record: {self.min_expert_speed_to_record} m/s\n")
                f.write(f"Input Topics:\n")
                f.write(f"  - Depth: {self.depth_topic_in}\n")
                f.write(f"  - Expert Velocity (APF cmd): {self.expert_vel_topic_in}\n")
                f.write(f"  - Current Velocity (Drone state): {self.current_vel_topic_in}\n")
                f.write(f"  - Current Pose (Drone state): {self.current_pose_topic_in}\n")
                if self.record_mavros_state:
                    f.write(f"  - MAVROS State: {self.mavros_state_topic_in}\n")
                f.write(f"Output Topics in Bag:\n")
                f.write(f"  - Depth: {self.depth_topic_out}\n")
                f.write(f"  - Expert Velocity: {self.expert_vel_topic_out}\n")
                f.write(f"  - Input (Current) Velocity: {self.current_vel_topic_out}\n")
                f.write(f"  - Pose Data: {self.current_pose_topic_out}\n")
                if self.record_mavros_state:
                    f.write(f"  - MAVROS State Info: {self.mavros_state_topic_out}\n")

        except Exception as e:
            rospy.logwarn(f"[Recorder] Could not create session info file: {e}")
    
    def mavros_state_callback(self, msg):
        """Store the latest MAVROS state if recording it."""
        self.last_mavros_state_msg = msg

    def synced_callback(self, depth_msg, expert_vel_msg, current_vel_msg, current_pose_msg):
        self.frame_count_received_sync += 1
        try:
            # Use a common timestamp, e.g., from depth or the latest of the synced messages
            # For ApproximateTimeSynchronizer, msgs might have slightly different stamps within 'slop'
            # Using depth_msg.header.stamp is common. Or, you can average them or use rospy.Time.now()
            # Let's use the header stamp of the expert_vel_msg as the reference time for the bundle
            common_timestamp = expert_vel_msg.header.stamp 

            # Filter based on expert speed
            ex_vx = expert_vel_msg.twist.linear.x
            ex_vy = expert_vel_msg.twist.linear.y
            ex_vz = expert_vel_msg.twist.linear.z
            expert_speed = math.sqrt(ex_vx**2 + ex_vy**2 + ex_vz**2)

            if expert_speed < self.min_expert_speed_to_record:
                if self.frame_count_received_sync % 500 == 0: # Log less frequently when skipping
                     rospy.logdebug(f"[Recorder] Skipping frame, expert speed {expert_speed:.2f} < {self.min_expert_speed_to_record:.2f} m/s")
                return 

            # Write synchronized data to bag
            self.bag.write(self.depth_topic_out, depth_msg, common_timestamp) 
            self.bag.write(self.expert_vel_topic_out, expert_vel_msg, common_timestamp) 
            self.bag.write(self.current_vel_topic_out, current_vel_msg, common_timestamp) 
            self.bag.write(self.current_pose_topic_out, current_pose_msg, common_timestamp)

            # If recording MAVROS state, write the last received one with the common_timestamp
            # This is not perfectly synced, but closest available without adding it to the main TimeSynchronizer
            if self.record_mavros_state and self.last_mavros_state_msg:
                # Create a new State message with the synced timestamp
                state_msg_to_write = State()
                state_msg_to_write.header.stamp = common_timestamp # Overwrite stamp
                state_msg_to_write.header.frame_id = self.last_mavros_state_msg.header.frame_id
                state_msg_to_write.connected = self.last_mavros_state_msg.connected
                state_msg_to_write.armed = self.last_mavros_state_msg.armed
                state_msg_to_write.guided = self.last_mavros_state_msg.guided
                state_msg_to_write.manual_input = self.last_mavros_state_msg.manual_input
                state_msg_to_write.mode = self.last_mavros_state_msg.mode
                state_msg_to_write.system_status = self.last_mavros_state_msg.system_status
                self.bag.write(self.mavros_state_topic_out, state_msg_to_write, common_timestamp)


            self.frame_count_recorded += 1
            
            if self.frame_count_recorded % 100 == 0 and self.frame_count_recorded > 0: # Log every 100 *recorded* frames
                rospy.loginfo(f"[Recorder] Recorded {self.frame_count_recorded} frames (syncs received: {self.frame_count_received_sync}).")
                
        except Exception as e: # Catch any other errors during callback
            rospy.logerr(f"[Recorder] Error in synced_callback: {e}")
            import traceback
            traceback.print_exc()
    
    def cleanup(self):
        rospy.loginfo("[Recorder] Starting cleanup and saving bag...")
        final_ros_time = rospy.Time.now()
        duration_sec = (final_ros_time - self.ros_start_time).to_sec()
        
        try:
            if hasattr(self, 'bag') and self.bag is not None: # Ensure bag was opened
                # Force buffer flush before closing, might take a moment
                rospy.loginfo("[Recorder] Flushing bag buffer...")
                self.bag.flush()
                self.bag.close()
                rospy.loginfo(f"[Recorder] Bag file saved and closed: {self.bag_path}")
            else:
                rospy.logwarn("[Recorder] Bag object not found or already closed during cleanup.")
        except Exception as e:
            rospy.logerr(f"[Recorder] Error closing bag: {e}")
        
        # Create a simple summary file
        summary_path = os.path.join(self.session_dir, 'recording_summary.txt')
        try:
            with open(summary_path, 'w') as f:
                f.write(f"Recording Session Summary\n")
                f.write(f"=========================\n")
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Bag File: {self.bag_name}\n")
                f.write(f"Start Time (Obj): {self.session_start_time_obj.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Start Time (ROS): {self.ros_start_time.to_sec():.3f}\n")
                f.write(f"End Time (ROS): {final_ros_time.to_sec():.3f}\n")
                f.write(f"Duration: {duration_sec:.2f} seconds\n")
                f.write(f"Synchronized Sets Received: {self.frame_count_received_sync}\n")
                f.write(f"Frames Recorded to Bag: {self.frame_count_recorded}\n")
                f.write(f"Filter - Min Expert Speed: {self.min_expert_speed_to_record} m/s\n")
            rospy.loginfo(f"[Recorder] Recording summary saved to: {summary_path}")
        except Exception as e:
            rospy.logerr(f"[Recorder] Error writing summary file: {e}")

        rospy.loginfo(f"[Recorder] Session {self.session_id} ended. Data saved to: {self.session_dir}")

if __name__ == '__main__':
    try:
        recorder = APFBagRecorder()
        rospy.spin() # Keep node alive until shutdown
    except rospy.ROSInterruptException:
        rospy.loginfo("[Recorder] Shutdown requested.")
    except Exception as e: # Catch any other unexpected errors during node setup
        rospy.logfatal(f"[Recorder] Critical error during initialization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rospy.loginfo("[Recorder] Node is terminating.")