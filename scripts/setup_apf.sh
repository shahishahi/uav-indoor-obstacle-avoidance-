#!/bin/bash

# APF Navigation Setup Script
# This script helps you set up the APF navigation system with configuration files

echo "========================================="
echo "APF Navigation Configuration Setup"
echo "========================================="

# Updated paths for your workspace
WORKSPACE_DIR="$HOME/catkin_ws/src/apf_depth_nav_node"
CONFIG_DIR="$WORKSPACE_DIR/config"
CONFIG_FILE="$CONFIG_DIR/apf_config.yaml"
BACKUP_DIR="$CONFIG_DIR/backups"
SCRIPTS_DIR="$WORKSPACE_DIR/scripts"

# Create necessary directories
mkdir -p "$CONFIG_DIR"
mkdir -p "$BACKUP_DIR"
mkdir -p "$SCRIPTS_DIR"

# Function to backup existing config
backup_config() {
    if [ -f "$CONFIG_FILE" ]; then
        timestamp=$(date +"%Y%m%d_%H%M%S")
        backup_file="$BACKUP_DIR/apf_config_backup_$timestamp.yaml"
        cp "$CONFIG_FILE" "$backup_file"
        echo "Existing config backed up to: $backup_file"
    fi
}

# Function to create default config
create_default_config() {
    echo "Creating default configuration file at: $CONFIG_FILE"
    cat > "$CONFIG_FILE" << 'EOF'
# APF Navigation Configuration File
# Artificial Potential Field based navigation for depth camera equipped drones

# Force field parameters
force_field:
  k_att: 0.8          # Attractive force gain (towards goal)
  k_rep: 2.0          # Repulsive force gain (from obstacles)
  force_history_size: 5    # Number of previous forces to consider for smoothing
  min_force_threshold: 0.1 # Minimum force magnitude to consider
  max_force_magnitude: 5.0 # Maximum allowed force magnitude

# Velocity control
velocity:
  max_linear_vel: 1.0      # Maximum linear velocity (m/s)
  max_angular_vel: 1.0     # Maximum angular velocity (rad/s)
  vel_smoothing_factor: 0.3 # Velocity smoothing (0.0 = no smoothing, 1.0 = max smoothing)
  emergency_stop_distance: 0.3 # Distance to trigger emergency stop

# Obstacle detection and avoidance
obstacles:
  depth_threshold: 2.0     # Maximum depth to consider for obstacles (meters)
  min_obstacle_distance: 0.5 # Minimum safe distance from obstacles
  obstacle_expansion_radius: 0.3 # Expand obstacle boundaries
  vertical_fov_limit: 0.6  # Vertical field of view limit (0.0 to 1.0)
  horizontal_fov_limit: 0.8 # Horizontal field of view limit (0.0 to 1.0)

# Camera parameters (Intel RealSense D435 typical values)
camera:
  fx: 462.1    # Focal length x
  fy: 462.1    # Focal length y  
  cx: 320.0    # Principal point x
  cy: 240.0    # Principal point y
  width: 640   # Image width
  height: 480  # Image height

# Goal and waypoint management
navigation:
  goal_tolerance: 0.5      # Distance tolerance to consider goal reached
  waypoint_tolerance: 0.3  # Distance tolerance for waypoint navigation
  max_goal_distance: 10.0  # Maximum goal distance to consider
  goal_timeout: 30.0       # Timeout for reaching goal (seconds)

# Safety parameters
safety:
  takeoff_height: 1.5      # Default takeoff height
  min_altitude: 0.5        # Minimum flight altitude
  max_altitude: 3.0        # Maximum flight altitude
  battery_low_threshold: 20.0 # Battery percentage for low warning
  
# Behavior modes
modes:
  continuous_navigation: true  # Continuous navigation mode
  hover_at_goal: true         # Hover when goal is reached
  return_to_launch: false     # Return to launch point when done
  obstacle_avoidance_only: false # Only avoid obstacles, no goal seeking

# ROS topic names
topics:
  depth_image: '/camera/depth/image_rect_raw'
  depth_info: '/camera/depth/camera_info'
  cmd_vel: '/cmd_vel'
  pose: '/mavros/local_position/pose'
  goal: '/apf_nav/goal'
  status: '/apf_nav/status'

# Logging and debugging
debug:
  verbose_output: false
  publish_debug_markers: true
  log_forces: false
  save_trajectories: false
EOF
    echo "Default configuration created successfully!"
}

# Function to validate config file
validate_config() {
    if command -v python3 &> /dev/null; then
        python3 -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print('✓ Configuration file is valid YAML')
    
    # Basic validation of required sections
    required_sections = ['force_field', 'velocity', 'obstacles', 'camera']
    for section in required_sections:
        if section not in config:
            print(f'⚠ Warning: Missing section [{section}]')
        else:
            print(f'✓ Section [{section}] found')
            
except Exception as e:
    print('✗ Configuration file has errors:', e)
    exit(1)
"
    else
        echo "Python3 not available for validation. Please check YAML syntax manually."
    fi
}

# Function to show usage examples
show_usage() {
    echo ""
    echo "========================================="
    echo "Usage Examples:"
    echo "========================================="
    echo ""
    echo "1. Run with default config:"
    echo "   rosrun apf_depth_nav_node apf_depth_nav_node.py"
    echo ""
    echo "2. Run with specific config file:"
    echo "   rosrun apf_depth_nav_node apf_depth_nav_node.py /path/to/config.yaml"
    echo ""
    echo "3. Run with launch file:"
    echo "   roslaunch apf_depth_nav_node apf_navigation.launch"
    echo ""
    echo "4. Run with launch file and custom config:"
    echo "   roslaunch apf_depth_nav_node apf_navigation.launch config_file:=/path/to/config.yaml"
    echo ""
    echo "5. Override specific parameters:"
    echo "   roslaunch apf_depth_nav_node apf_navigation.launch k_att:=1.2 vel_cap:=1.0"
    echo ""
    echo "Configuration file location: $CONFIG_FILE"
    echo "Edit this file to customize your drone's behavior!"
    echo ""
    echo "Package location: $WORKSPACE_DIR"
}

# Function to check workspace structure
check_workspace() {
    echo "Checking workspace structure..."
    
    if [ ! -d "$WORKSPACE_DIR" ]; then
        echo "⚠ Warning: Package directory not found: $WORKSPACE_DIR"
        echo "Creating package structure..."
        mkdir -p "$WORKSPACE_DIR"/{config,scripts,launch}
    fi
    
    if [ ! -f "$SCRIPTS_DIR/apf_depth_nav_node.py" ]; then
        echo "⚠ Warning: Main script not found: $SCRIPTS_DIR/apf_depth_nav_node.py"
    else
        echo "✓ Main script found"
    fi
    
    echo "✓ Workspace structure checked"
}

# Main setup process
case "$1" in
    "backup")
        backup_config
        echo "Configuration backed up successfully!"
        ;;
    "create")
        check_workspace
        backup_config
        create_default_config
        echo "Please edit $CONFIG_FILE to customize settings."
        ;;
    "validate")
        if [ -f "$CONFIG_FILE" ]; then
            validate_config
        else
            echo "No configuration file found at $CONFIG_FILE"
            echo "Run: $0 create"
        fi
        ;;
    "edit")
        if [ -f "$CONFIG_FILE" ]; then
            ${EDITOR:-nano} "$CONFIG_FILE"
        else
            echo "No configuration file found. Creating default..."
            create_default_config
            ${EDITOR:-nano} "$CONFIG_FILE"
        fi
        ;;
    "show")
        if [ -f "$CONFIG_FILE" ]; then
            echo "Current configuration:"
            echo "====================="
            cat "$CONFIG_FILE"
        else
            echo "No configuration file found at $CONFIG_FILE"
        fi
        ;;
    "workspace")
        check_workspace
        echo "Workspace structure:"
        echo "==================="
        tree "$WORKSPACE_DIR" 2>/dev/null || find "$WORKSPACE_DIR" -type d
        ;;
    *)
        echo "APF Navigation Setup Script"
        echo ""
        echo "Commands:"
        echo "  create    - Create default configuration file"
        echo "  backup    - Backup existing configuration"
        echo "  validate  - Validate configuration file syntax"
        echo "  edit      - Edit configuration file"
        echo "  show      - Display current configuration"
        echo "  workspace - Check and display workspace structure"
        echo ""
        echo "Examples:"
        echo "  $0 create     # Create new config"
        echo "  $0 edit       # Edit existing config"
        echo "  $0 validate   # Check config syntax"
        echo "  $0 workspace  # Check workspace"
        echo ""
        show_usage
        ;;
esac