#!/usr/bin/env python3

import rosbag
import rospy # For rospy.Time, rospy.Duration
import os
import cv2
import csv
import numpy as np
import argparse
# import shutil # Not actively used in this version, but can be added if needed
import gc
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from datetime import datetime
import psutil
import time


class MemoryOptimizedBagConverter:
    def __init__(self, bag_file, output_dir, sync_slop=0.25, preserve_existing=True, chunk_size=100):
        self.bag_file = bag_file
        self.base_output_dir = output_dir
        self.sync_slop = sync_slop  # in seconds
        self.preserve_existing = preserve_existing
        self.chunk_size = chunk_size
        
        bag_basename = os.path.basename(bag_file).replace('.bag', '')
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if preserve_existing:
            self.output_dir = os.path.join(output_dir, f"{bag_basename}")
        else:
            # If not preserving, use the output_dir directly.
            # Consider if self.output_dir should be cleaned if it exists.
            # For now, it will write into it, potentially overwriting.
            self.output_dir = output_dir
            
        self.image_topic = '/depth_image'
        self.input_vel_topic = '/input_velocity'
        self.expert_vel_topic = '/expert_velocity'
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.img_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(self.img_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, 'controls.csv')
        self.info_path = os.path.join(self.output_dir, 'dataset_info.txt')
        self.master_csv_path = os.path.join(self.base_output_dir, 'all_datasets.csv')
        
        self.bridge = CvBridge()
        self.process = psutil.Process(os.getpid())
        
        # Search indices for optimized velocity lookup, initialized per conversion
        self.input_vel_search_idx = 0
        self.expert_vel_search_idx = 0
        
        self.create_dataset_info() # Create info file early
        
    def monitor_memory(self, context=""):
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        print(f"[Memory] {context}: {memory_mb:.1f} MB")
        return memory_mb
        
    def create_dataset_info(self):
        try:
            with open(self.info_path, 'w') as f:
                f.write(f"APF Dataset Conversion Info\n")
                f.write(f"===========================\n")
                f.write(f"Source Bag: {self.bag_file}\n")
                f.write(f"Conversion Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n") 
                f.write(f"Output Directory: {self.output_dir}\n")
                f.write(f"Sync Slop: {self.sync_slop}s\n")
                f.write(f"Chunk Size: {self.chunk_size}\n")
                f.write(f"Topics:\n")
                f.write(f"  - Images: {self.image_topic}\n")
                f.write(f"  - Input Velocity: {self.input_vel_topic}\n")
                f.write(f"  - Expert Velocity: {self.expert_vel_topic}\n")
        except Exception as e:
            print(f"[Converter] Warning: Could not create dataset info: {e}")
    
    def get_bag_metadata(self):
        try:
            with rosbag.Bag(self.bag_file, 'r') as bag:
                rosbag_info_dict = bag.get_type_and_topic_info()
                start_time_ros = rospy.Time.from_sec(bag.get_start_time())
                end_time_ros = rospy.Time.from_sec(bag.get_end_time())
                duration_ros = end_time_ros - start_time_ros
                
                topic_info_data = {}
                if rosbag_info_dict and rosbag_info_dict.topics:
                    for topic, topic_data in rosbag_info_dict.topics.items():
                        topic_info_data[topic] = {
                            'message_count': topic_data.message_count,
                            'message_type': topic_data.msg_type
                        }
                
                return {
                    'start_time': start_time_ros.to_sec(),
                    'end_time': end_time_ros.to_sec(),
                    'duration': duration_ros.to_sec(),
                    'topics': topic_info_data
                }
        except Exception as e:
            print(f"[Converter] Error reading bag metadata: {e}")
            return None
            
    def extract_velocity_messages(self):
        input_vel_msgs = []
        expert_vel_msgs = []
        print(f"[Converter] Extracting velocity messages...")
        try:
            with rosbag.Bag(self.bag_file, 'r') as bag:
                vel_topics = [self.input_vel_topic, self.expert_vel_topic]
                for topic, msg, t in bag.read_messages(topics=vel_topics):
                    # msg.header.stamp is a rospy.Time object
                    if topic == self.input_vel_topic:
                        input_vel_msgs.append({
                            'timestamp': msg.header.stamp, 
                            'linear': (msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z),
                            'angular': msg.twist.angular.z
                        })
                    elif topic == self.expert_vel_topic:
                        expert_vel_msgs.append({
                            'timestamp': msg.header.stamp,
                            'linear': (msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z)
                        })
        except Exception as e:
            print(f"[Converter] Error reading velocity messages: {e}")
            return None, None
            
        print(f"[Converter] Loaded {len(input_vel_msgs)} input velocity messages")
        print(f"[Converter] Loaded {len(expert_vel_msgs)} expert velocity messages")
        self.monitor_memory("After loading velocities")
        return input_vel_msgs, expert_vel_msgs
    
    def _find_closest_velocity_advancing(self, target_time_ros, sorted_vel_msgs, slop_sec, vel_type_for_idx):
        """
        Finds the closest velocity message to target_time_ros within slop_sec.
        Uses and updates an internal index (self.input_vel_search_idx or self.expert_vel_search_idx)
        to optimize search for monotonically increasing target_time_ros.
        `vel_type_for_idx` must be 'input' or 'expert'.
        """
        current_search_idx = 0
        if vel_type_for_idx == 'input':
            current_search_idx = self.input_vel_search_idx
        elif vel_type_for_idx == 'expert':
            current_search_idx = self.expert_vel_search_idx
        else:
            # Should not happen if called correctly
            raise ValueError("Invalid vel_type_for_idx for advancing index.")

        n = len(sorted_vel_msgs)
        if n == 0: # No messages to search in
            return None
        
        # If current_search_idx is already past the end, no more matches possible for this list
        if current_search_idx >= n:
            return None

        best_match_msg = None
        min_abs_diff_sec = float('inf')

        # 1. Advance current_search_idx:
        # Skip messages that are "too old" for the current target_time_ros.
        # A message is "too old" if its timestamp is less than (target_time_ros - slop_sec).
        # So, (target_time_ros - msg_time).to_sec() > slop_sec
        idx = current_search_idx
        while idx < n and \
              (target_time_ros - sorted_vel_msgs[idx]['timestamp']).to_sec() > slop_sec:
            idx += 1
        
        # Update the class member index for the next call for this specific velocity type
        if vel_type_for_idx == 'input':
            self.input_vel_search_idx = idx
        else: # expert
            self.expert_vel_search_idx = idx
        
        # 2. Search forward from the updated `idx`:
        # Look for a message within the [-slop_sec, +slop_sec] window around target_time_ros.
        # Stop if messages are "too new" (timestamp > target_time_ros + slop_sec).
        iter_idx = idx
        while iter_idx < n:
            msg_data = sorted_vel_msgs[iter_idx]
            msg_time_ros = msg_data['timestamp']
            
            # Calculate difference: msg_time_ros - target_time_ros
            # Positive if msg is after target, negative if before.
            diff_sec = (msg_time_ros - target_time_ros).to_sec()

            if diff_sec > slop_sec:
                # This message and subsequent ones are too late (beyond target_time_ros + slop_sec).
                # Since messages are sorted, no need to check further for *this* target_time_ros.
                break 
            
            # If we are here, msg_time_ros is <= target_time_ros + slop_sec.
            # The first loop advanced `idx` so that sorted_vel_msgs[idx]['timestamp'] is approximately
            # >= target_time_ros - slop_sec.
            # So, any message checked here is potentially within the overall slop window.
            
            abs_diff_sec = abs(diff_sec)
            # We must explicitly check if it's within slop, as the first loop is an optimization
            # and doesn't guarantee the message at `idx` is strictly within `target - slop`.
            if abs_diff_sec <= slop_sec: 
                if abs_diff_sec < min_abs_diff_sec:
                    min_abs_diff_sec = abs_diff_sec
                    best_match_msg = msg_data
            
            iter_idx += 1
            
        return best_match_msg

    def process_depth_image(self, image_msg, timestamp_sec):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
            
            if cv_image.dtype == np.float32:
                cv_image = np.nan_to_num(cv_image, nan=0.0, posinf=10.0, neginf=0.0) # Replace NaN/inf
                cv_image = np.clip(cv_image, 0.0, 10.0) # Clip to 0-10 meters
                cv_image = (cv_image * 1000).astype(np.uint16) # Convert to mm, uint16
            elif cv_image.dtype == np.uint16:
                cv_image = np.clip(cv_image, 0, 10000) # Assuming 10000 is 10m in mm
            else:
                print(f"[Converter] Warning: Unexpected image dtype: {cv_image.dtype}. Attempting conversion and clipping.")
                cv_image = cv_image.astype(np.uint16) 
                cv_image = np.clip(cv_image, 0, 10000)
            
            filename = f"{timestamp_sec:.6f}.png" # Timestamp as filename
            filepath = os.path.join(self.img_dir, filename)
            
            # Use lower PNG compression for faster writing.
            # Range is 0 (no/fastest) to 9 (max compression/slowest). OpenCV default is often 3.
            # Value 1 should provide a good speedup.
            success = cv2.imwrite(filepath, cv_image, [cv2.IMWRITE_PNG_COMPRESSION, 1]) 
            
            del cv_image # Explicitly delete to assist GC
            
            if not success:
                print(f"[Converter] Failed to save image: {filename}")
                return None
            return filename
            
        except Exception as e:
            # Catch CvBridgeError specifically if needed, or general Exception
            print(f"[Converter] Error processing image at {timestamp_sec:.2f}: {e}")
            return None

    def _process_buffered_image_chunk(self, image_msg_list, input_vel_msgs, expert_vel_msgs, csv_writer):
        """Processes a list of image messages already loaded into memory."""
        matched_count = 0
        unmatched_count = 0

        for i, image_msg in enumerate(image_msg_list):
            img_header_stamp = image_msg.header.stamp # This is a rospy.Time object
            timestamp_sec = img_header_stamp.to_sec()

            # Find matching velocity messages using the optimized advancing pointer method
            input_vel = self._find_closest_velocity_advancing(
                img_header_stamp, input_vel_msgs, self.sync_slop, 'input'
            )
            expert_vel = self._find_closest_velocity_advancing(
                img_header_stamp, expert_vel_msgs, self.sync_slop, 'expert'
            )

            if input_vel and expert_vel:
                # Process depth image (includes saving)
                filename = self.process_depth_image(image_msg, timestamp_sec)
                
                if filename:
                    try:
                        csv_writer.writerow([
                            timestamp_sec,
                            input_vel['linear'][0], input_vel['linear'][1], input_vel['linear'][2], 
                            input_vel['angular'],
                            expert_vel['linear'][0], expert_vel['linear'][1], expert_vel['linear'][2],
                            filename
                        ])
                        matched_count += 1
                    except Exception as e:
                        print(f"[Converter] Error writing CSV data for image at {timestamp_sec:.2f}: {e}")
                        unmatched_count += 1 
                else:
                    # process_depth_image returned None (failed to save/process)
                    unmatched_count += 1
            else:
                # Velocities not found
                unmatched_count += 1
            
            if (i + 1) % 20 == 0: # Progress update within chunk processing, less frequent than every 10
                print(f"[Converter]  Sub-progress (chunk): {i + 1}/{len(image_msg_list)} processed.")
                
        return matched_count, unmatched_count

    def get_image_count(self):
        """Get total number of images without loading them all."""
        try:
            with rosbag.Bag(self.bag_file, 'r') as bag:
                info = bag.get_type_and_topic_info()
                # Ensure info and info.topics are not None before accessing
                if info and info.topics and self.image_topic in info.topics:
                    return info.topics[self.image_topic].message_count
                else:
                    print(f"[Converter] Warning: Topic {self.image_topic} not found or bag info empty.")
                    return 0
        except Exception as e:
            print(f"[Converter] Error getting image count: {e}")
            return 0
            
    def update_master_dataset_list(self, dataset_stats):
        """Update master list of all converted datasets."""
        try:
            file_exists = os.path.isfile(self.master_csv_path)
            with open(self.master_csv_path, 'a', newline='') as csvfile:
                fieldnames = [
                    'dataset_name', 'bag_file', 'conversion_time', 'output_dir',
                    'matched_frames', 'unmatched_frames', 'total_frames',
                    'bag_duration_sec', 'success_rate', 'dataset_size_mb'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(dataset_stats)
        except Exception as e:
            print(f"[Converter] Warning: Could not update master dataset list: {e}")

    def get_directory_size(self, path):
        """Calculate directory size in MB."""
        total_size = 0
        try:
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath): # Avoids issues with broken symlinks
                        total_size += os.path.getsize(filepath)
            return round(total_size / (1024 * 1024), 2) # Convert bytes to MB
        except Exception as e:
            print(f"[Converter] Error calculating directory size for {path}: {e}")
            return 0.0

    def convert(self):
        """Main conversion function with memory and speed optimizations."""
        print(f"[Converter] Starting optimized conversion for: {self.bag_file}")
        print(f"[Converter] Output to: {self.output_dir}, Chunk size: {self.chunk_size}, Sync slop: {self.sync_slop}s")
        self.monitor_memory("Initial")

        overall_start_time = time.time()

        bag_metadata = self.get_bag_metadata()
        total_images_in_bag = self.get_image_count()
        
        if total_images_in_bag == 0:
            print("[Converter] No images found in the specified image topic. Aborting.")
            # Update info file to reflect no images found
            try:
                with open(self.info_path, 'a') as f:
                    f.write(f"\nConversion Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Error: No images found in topic '{self.image_topic}'. Conversion aborted.\n")
            except Exception as e:
                print(f"[Converter] Warning: Could not update dataset info with failure: {e}")
            return False

        input_vel_msgs, expert_vel_msgs = self.extract_velocity_messages()
        if input_vel_msgs is None or expert_vel_msgs is None:
            # Error already printed by extract_velocity_messages
            return False

        # Sort velocity messages by timestamp for efficient lookup
        input_vel_msgs.sort(key=lambda v: v['timestamp'])
        expert_vel_msgs.sort(key=lambda v: v['timestamp'])
        self.monitor_memory("Velocities extracted and sorted")

        # Reset search indices for velocity lookup for this conversion run
        self.input_vel_search_idx = 0
        self.expert_vel_search_idx = 0

        total_matched_frames = 0
        total_unmatched_frames = 0
        
        with open(self.csv_path, 'w', newline='') as csvfile:
            csv_data_writer = csv.writer(csvfile)
            # Header consistent with dataloader expectations
            csv_data_writer.writerow(['timestamp', 'vx_input', 'vy_input', 'vz_input', 'yaw_rate', 
                                      'vx', 'vy', 'vz', 'image_file'])

            image_message_buffer = [] # Buffer to hold a chunk of image messages
            images_read_from_bag_count = 0 # Counter for images read from bag

            # Single pass over the bag for image messages
            with rosbag.Bag(self.bag_file, 'r') as bag:
                print(f"[Converter] Reading and processing {total_images_in_bag} image messages from bag...")
                # Iterate through all image messages ONCE
                for topic, msg, ros_bag_time in bag.read_messages(topics=[self.image_topic]):
                    if topic == self.image_topic:
                        image_message_buffer.append(msg)
                        images_read_from_bag_count += 1

                        # Process the buffer when it's full or if it's the last image
                        if len(image_message_buffer) >= self.chunk_size or images_read_from_bag_count == total_images_in_bag:
                            chunk_start_msg_num = images_read_from_bag_count - len(image_message_buffer) + 1
                            print(f"\n[Converter] Processing chunk: images {chunk_start_msg_num}-{images_read_from_bag_count} of {total_images_in_bag}")
                            self.monitor_memory(f"Buffer full (images up to {images_read_from_bag_count})")
                            
                            matched_in_chunk, unmatched_in_chunk = self._process_buffered_image_chunk(
                                image_message_buffer,
                                input_vel_msgs,
                                expert_vel_msgs,
                                csv_data_writer
                            )
                            
                            total_matched_frames += matched_in_chunk
                            total_unmatched_frames += unmatched_in_chunk
                            
                            image_message_buffer.clear() # Important: clear buffer for next chunk
                            gc.collect() # Force garbage collection after processing a chunk and clearing buffer
                            self.monitor_memory(f"After processing chunk (images up to {images_read_from_bag_count})")
                        
                        # Optional: More frequent global progress update based on images read from bag
                        if images_read_from_bag_count % (self.chunk_size * 2) == 0 and images_read_from_bag_count < total_images_in_bag : 
                             print(f"[Converter] Overall progress: {images_read_from_bag_count}/{total_images_in_bag} images read from bag.")
            
            # End of with rosbag.Bag...
        # End of with open CSV...
        
        overall_duration = time.time() - overall_start_time
        print(f"[Converter] Bag processing and data extraction finished in {overall_duration:.2f} seconds.")

        # Calculate final statistics
        total_frames_considered = total_matched_frames + total_unmatched_frames # Should equal total_images_in_bag if all processed
        success_rate = (total_matched_frames / total_frames_considered * 100) if total_frames_considered > 0 else 0
        final_dataset_size_mb = self.get_directory_size(self.output_dir)

        # Update dataset_info.txt with final results
        try:
            with open(self.info_path, 'a') as f: # Append to existing info file
                f.write(f"\nConversion Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Conversion Duration: {overall_duration:.2f} seconds\n")
                f.write(f"\nConversion Results:\n")
                f.write(f"==================\n")
                f.write(f"Total images in bag topic ('{self.image_topic}'): {total_images_in_bag}\n")
                f.write(f"Total frames considered for matching: {total_frames_considered}\n")
                f.write(f"Successfully matched and saved: {total_matched_frames}\n")
                f.write(f"Unmatched/Failed frames: {total_unmatched_frames}\n")
                f.write(f"Success rate: {success_rate:.1f}%\n")
                f.write(f"Final Dataset size: {final_dataset_size_mb} MB\n")
                f.write(f"CSV file: {os.path.basename(self.csv_path)}\n")
                f.write(f"Images directory: {os.path.basename(self.img_dir)}\n")
                if bag_metadata:
                    f.write(f"Bag duration (from metadata): {bag_metadata['duration']:.1f} seconds\n")
        except Exception as e:
            print(f"[Converter] Warning: Could not update dataset info with final results: {e}")
        
        # Update master dataset list
        dataset_name = os.path.basename(self.output_dir)
        dataset_stats = {
            'dataset_name': dataset_name,
            'bag_file': os.path.basename(self.bag_file),
            'conversion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # End time of conversion
            'output_dir': self.output_dir,
            'matched_frames': total_matched_frames,
            'unmatched_frames': total_unmatched_frames,
            'total_frames': total_frames_considered, # total images processed
            'bag_duration_sec': bag_metadata['duration'] if bag_metadata and bag_metadata['duration'] is not None else 0,
            'success_rate': round(success_rate, 1),
            'dataset_size_mb': final_dataset_size_mb
        }
        self.update_master_dataset_list(dataset_stats)
        
        print(f"\n[Converter] Conversion Summary for Dataset: {dataset_name}")
        print(f"  Successfully matched frames: {total_matched_frames}")
        print(f"  Unmatched/Failed frames: {total_unmatched_frames}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Final dataset size: {final_dataset_size_mb} MB")
        print(f"  Total conversion time: {overall_duration:.2f} seconds")
        print(f"  Dataset saved at: {self.output_dir}")
        print(f"  CSV file: {self.csv_path}")
        self.monitor_memory("Final")
        
        return True

class BatchConverter:
    """Utility class for batch processing multiple bag files"""
    
    def __init__(self, input_dir, output_dir, sync_slop=0.25, chunk_size=300):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sync_slop = sync_slop
        self.chunk_size = chunk_size
        
    def find_bag_files(self):
        """Find all bag files in input directory"""
        bag_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.bag'):
                    bag_files.append(os.path.join(root, file))
        return bag_files
    
    def convert_all(self):
        """Convert all bag files found in input directory"""
        bag_files = self.find_bag_files()
        
        if not bag_files:
            print(f"[BatchConverter] No bag files found in: {self.input_dir}")
            return False
        
        print(f"[BatchConverter] Found {len(bag_files)} bag files to process.")
        
        successful_conversions = 0
        failed_conversions = 0
        
        for i, bag_file_path in enumerate(bag_files):
            print(f"\n[BatchConverter] Processing file {i+1}/{len(bag_files)}: {os.path.basename(bag_file_path)}")
            
            try:
                # For batch, always preserve existing by creating unique subdirectories
                # The preserve_existing=True is the default for MemoryOptimizedBagConverter
                converter_instance = MemoryOptimizedBagConverter(
                    bag_file_path, 
                    self.output_dir, # Base output directory for all datasets
                    sync_slop=self.sync_slop, 
                    preserve_existing=True, # Ensures unique output subfolder for each bag
                    chunk_size=self.chunk_size
                )
                conversion_success = converter_instance.convert()
                
                if conversion_success:
                    successful_conversions += 1
                else:
                    failed_conversions += 1
                    
            except Exception as e:
                print(f"[BatchConverter] Critical error during conversion of {bag_file_path}: {e}")
                failed_conversions += 1
            
            # Force garbage collection between processing large bag files
            gc.collect()
            print(f"[BatchConverter] Memory after GC post-conversion: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")

        print(f"\n[BatchConverter] Batch conversion finished.")
        print(f"  Total successful conversions: {successful_conversions}")
        print(f"  Total failed conversions: {failed_conversions}")
        print(f"  All datasets (if any) saved under base directory: {self.output_dir}")
        
        return successful_conversions > 0

def main():
    parser = argparse.ArgumentParser(description='Convert ROS bag(s) to CSV dataset(s) with memory optimization')
    parser.add_argument('--bag_file', help='Path to a single input ROS bag file')
    parser.add_argument('--input_dir', help='Directory containing multiple ROS bag files (for batch processing)')
    parser.add_argument('--output_dir', required=True, help='Base output directory for dataset(s)')
    parser.add_argument('--sync_slop', type=float, default=0.25, help='Time synchronization tolerance in seconds (default: 0.25s)')
    parser.add_argument('--chunk_size', type=int, default=50, help='Number of images to buffer and process at once (default: 50, lower for less memory)')
    parser.add_argument('--overwrite', action='store_true', help='If processing a single bag, allow overwriting the output_dir if it is used directly (no unique subfolder). Not recommended for batch.')
    parser.add_argument('--batch', action='store_true', help='Enable batch processing of all .bag files in input_dir')
    
    args = parser.parse_args()
    
    # Argument validation
    if args.batch:
        if not args.input_dir:
            print("Error: --input_dir is required for batch processing (--batch).")
            parser.print_help()
            return
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory for batch processing not found: {args.input_dir}")
            return
        if args.bag_file:
            print("Warning: --bag_file is ignored when --batch is specified.")
    elif not args.bag_file:
        print("Error: --bag_file is required for single file processing (if --batch is not used).")
        parser.print_help()
        return
    elif not os.path.isfile(args.bag_file):
        print(f"Error: Single bag file not found: {args.bag_file}")
        return

    overall_success = False
    if args.batch:
        print(f"Starting batch conversion from input directory: {args.input_dir}")
        batch_processor = BatchConverter(
            args.input_dir, 
            args.output_dir, 
            sync_slop=args.sync_slop, 
            chunk_size=args.chunk_size
        )
        overall_success = batch_processor.convert_all()
    else:
        # Single file processing
        print(f"Starting single file conversion for: {args.bag_file}")
        # preserve_existing=True by default, creates unique subfolder.
        # If --overwrite is true, preserve_existing becomes False, potentially overwriting output_dir.
        preserve = not args.overwrite 
        if not preserve:
             print(f"Warning: --overwrite specified. Output will be written directly to '{args.output_dir}'. Existing files may be overwritten if not using unique subfolders elsewhere.")

        converter_instance = MemoryOptimizedBagConverter(
            args.bag_file, 
            args.output_dir, 
            sync_slop=args.sync_slop, 
            preserve_existing=preserve, # For single bag, user can control this via --overwrite
            chunk_size=args.chunk_size
        )
        overall_success = converter_instance.convert()
    
    if overall_success:
        print("\nConversion process completed successfully for one or more files.")
    else:
        print("\nConversion process failed or no files were successfully converted.")
   
if __name__ == '__main__':
    # Support direct execution with hardcoded paths for backward compatibility / easy testing
    if len(os.sys.argv) == 1:  # No command-line arguments provided
        print("Memory-Optimized APF Bag to CSV Converter - No CLI args provided.")
        print("================================================================")
        print("This script converts ROS bag files containing depth images and velocity topics")
        print("into a dataset format with images and a controls.csv file.")
        print("\nUsage examples via command line:")
        print("  Single file: python script_name.py --bag_file /path/to/your.bag --output_dir /path/to/output_datasets")
        print("  Batch mode:  python script_name.py --batch --input_dir /path/to/bags --output_dir /path/to/output_datasets")
        print("  Low memory:  Add --chunk_size 25 (or lower) for very large bags or limited RAM.")
        print("\nAttempting to run with default hardcoded paths for testing (if bag exists)...")
        
        # Default paths for backward compatibility or quick testing
        # IMPORTANT: Modify these paths if you want to use this default execution block
        default_bag_file_path = '/home/shahi/apf_logs/session_20250606_031122/apf_depth_vel.bag' # Example path, change as needed
        default_output_base_dir = '/home/shahi/apf_logs/il_dataset' # Use a different output for testing
        
        if os.path.exists(default_bag_file_path):
            print(f"Found default bag file for testing: {default_bag_file_path}")
            print(f"Output will be in a subfolder of: {default_output_base_dir}")
            print("Using optimized processing with chunk_size=25 for this default run.")
            
            # Ensure the base output directory for datasets exists
            os.makedirs(default_output_base_dir, exist_ok=True)
            
            converter = MemoryOptimizedBagConverter(
                default_bag_file_path, 
                default_output_base_dir, 
                chunk_size=25, # Small chunk size for testing memory efficiency
                preserve_existing=True # Creates unique subfolder, good for testing
            )
            converter.convert()
        else:
            print(f"Default bag file not found: {default_bag_file_path}")
            print("Please specify bag file path and output directory using command line arguments as shown above.")
    else:
        # Normal execution with command-line arguments
        main()