import numpy as np
from enum import Enum

class APF_Core:
    """
    A pure computational engine for Artificial Potential Field calculations.
    This class is framework-agnostic (no ROS code) and focuses on:
    1. Processing depth data to find key obstacle points.
    2. Calculating attractive, repulsive, and tangential (escape) forces.
    3. Providing a final, combined force vector.
    """
    def __init__(self, config):
        self.config = config
        self.apf_params = config['apf']
        self.cam_params = config['obstacle_detection']['camera']
        
        # Internal state for the calculator
        self.force_history = []
        
    def calculate_total_force(self, current_pos, goal_pos, depth_image, is_stuck=False):
        """
        Main entry point to calculate the final velocity command vector.

        Args:
            current_pos (np.array): Drone's current [x, y, z] position in the world frame.
            goal_pos (np.array): The target [x, y, z] position in the world frame.
            depth_image (np.array): The raw depth image.
            is_stuck (bool): Flag indicating if the mission manager has detected a stuck condition.

        Returns:
            np.array: The combined 3D force vector.
        """
        # 1. Calculate the primary forces
        attractive_f = self._calculate_attractive_force(current_pos, goal_pos)
        obstacle_vectors = self._process_depth_image(depth_image)
        repulsive_f = self._calculate_repulsive_force(current_pos, obstacle_vectors)

        # 2. If stuck, calculate an escape force tangential to the main repulsive force
        escape_f = np.zeros(3)
        if is_stuck and np.linalg.norm(repulsive_f) > 0.1:
            escape_f = self._calculate_tangential_escape_force(repulsive_f)

        # 3. Combine all forces
        total_force = attractive_f + repulsive_f + escape_f
        
        # 4. Smooth the final force using a moving average
        self.force_history.append(total_force)
        if len(self.force_history) > self.apf_params['force_history_size']:
            self.force_history.pop(0)
        
        smoothed_force = np.mean(self.force_history, axis=0) if self.force_history else np.zeros(3)

        return smoothed_force, obstacle_vectors # Return vectors for visualization

    def _calculate_attractive_force(self, current_pos, goal_pos):
        """Calculates a conic attractive force (constant magnitude)."""
        direction = goal_pos - current_pos
        distance = np.linalg.norm(direction)
        if distance < 0.1:
            return np.zeros(3)
            
        # Scale down force when very close to the goal to allow for a smooth stop
        magnitude = self.apf_params['k_att']
        if distance < 2.0:
            magnitude *= (distance / 2.0)

        return (direction / distance) * magnitude

    def _process_depth_image(self, depth_image):
        """
        Efficiently processes a depth image to find the most relevant obstacle points.
        Instead of checking every pixel, it divides the image into sectors and finds
        the closest point in each, reducing noise and computational cost.
        
        Returns:
            list of np.array: A list of 3D obstacle points in the CAMERA's frame.
        """
        if depth_image is None:
            return []

        h, w = depth_image.shape
        num_sectors = 5
        sector_width = w // num_sectors
        obstacle_vectors = []

        for i in range(num_sectors):
            # Define a central region of interest for each sector
            sector_roi = depth_image[h//4 : 3*h//4, i*sector_width : (i+1)*sector_width]
            
            # Find the minimum depth, ignoring non-finite values (inf, nan)
            min_depth = np.min(sector_roi[np.isfinite(sector_roi)]) if np.any(np.isfinite(sector_roi)) else float('inf')

            if min_depth < self.config['obstacle_detection']['depth_threshold']:
                # Find the coordinates of this closest point
                coords = np.where(sector_roi == min_depth)
                v_roi, u_roi = coords[0][0], coords[1][0]
                
                # Convert back to full image coordinates
                v_img = v_roi + h // 4
                u_img = u_roi + i * sector_width
                
                # Deproject 2D pixel to 3D point in camera frame
                # Z-forward, X-right, Y-down (standard camera frame)
                z_cam = min_depth
                x_cam = (u_img - self.cam_params['cx']) * z_cam / self.cam_params['fx']
                y_cam = (v_img - self.cam_params['cy']) * z_cam / self.cam_params['fy']
                
                # This point is in the CAMERA's coordinate frame
                obstacle_vectors.append(np.array([z_cam, -x_cam, -y_cam])) # Convert to Drone Body Frame (X-fwd, Y-left, Z-up)

        return obstacle_vectors

    def _calculate_repulsive_force(self, current_pos, obstacle_vectors_world):
        """
        Calculates the total repulsive force from a list of obstacle points.
        These points are assumed to be already transformed into the world frame.
        """
        total_repulsive_force = np.zeros(3)
        d0 = self.apf_params['repulsive_dist_influence']
        Kr = self.apf_params['k_rep']

        for obs_pos_world in obstacle_vectors_world:
            vec_to_obs = obs_pos_world - current_pos
            dist_to_obs = np.linalg.norm(vec_to_obs)

            if 0.1 < dist_to_obs < d0:
                # Stable repulsive force formula
                force_magnitude = Kr * ((1.0 / dist_to_obs) - (1.0 / d0)) * (1.0 / dist_to_obs**2)
                direction = -vec_to_obs / dist_to_obs # Force pushes away from obstacle
                total_repulsive_force += force_magnitude * direction
        
        return total_repulsive_force

    def _calculate_tangential_escape_force(self, repulsive_force):
        """
        Calculates a force 90 degrees to the main repulsive force to "slide"
        along obstacles and escape local minima. This provides a "wall following" behavior.
        """
        # We only care about escaping in the XY plane
        repulsive_xy = repulsive_force[:2]
        
        # Rotate the 2D vector by 90 degrees
        # A better method might choose the direction that is more aligned with the goal
        tangential_xy = np.array([-repulsive_xy[1], repulsive_xy[0]])
        
        # Normalize and apply gain
        norm = np.linalg.norm(tangential_xy)
        if norm < 1e-6:
            return np.zeros(3)
        
        escape_force = np.zeros(3)
        escape_force[:2] = (tangential_xy / norm) * self.apf_params['escape_force_gain']
        return escape_force
        
    def reset(self):
        """Resets the internal state of the calculator."""
        self.force_history.clear()