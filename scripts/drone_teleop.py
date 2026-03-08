#!/usr/bin/env python3

import rospy
import pygame
import numpy as np
from geometry_msgs.msg import TwistStamped, PoseStamped, Pose
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import time

# --- Main Configuration ---
TAKEOFF_ALTITUDE = 1.5                # Altitude for both initial and respawn takeoffs
INITIAL_MANUAL_FLIGHT_ALTITUDE = 1.5  # Manual control will start by holding this altitude
TAKEOFF_VZ = 0.8                      # Vertical speed during takeoff ascent (m/s)

# Manual Control Speed Settings
MAX_SPEED_X = 1.0; MAX_SPEED_Y = 1.0; MAX_SPEED_YAW = 0.8
VELOCITY_RAMP_FACTOR = 0.2

# Altitude Adjustment Settings
ALTITUDE_ADJUSTMENT_RATE = 0.3; ALTITUDE_HOLD_KP = 0.8; MAX_ALTITUDE_CORRECTION_VZ = 0.3

# System & Simulation
CONTROL_FREQUENCY = 20; MODEL_NAME_FOR_GAZEBO_RESPAWN = "iris"

# --- FIXED RESPAWN POSITION ---
FIXED_RESPAWN_POSE = Pose(); FIXED_RESPAWN_POSE.position.x = -9.0; FIXED_RESPAWN_POSE.position.y = 0.0
FIXED_RESPAWN_POSE.position.z = 0.1; FIXED_RESPAWN_POSE.orientation.w = 1.0

# --- Key mapping ---
KEY_FORWARD=pygame.K_w; KEY_BACKWARD=pygame.K_s; KEY_LEFT=pygame.K_a; KEY_RIGHT=pygame.K_d
KEY_INCREASE_ALTITUDE_TARGET=pygame.K_SPACE; KEY_DECREASE_ALTITUDE_TARGET=pygame.K_LSHIFT
KEY_YAW_LEFT=pygame.K_q; KEY_YAW_RIGHT=pygame.K_e; KEY_PANIC_HOVER=pygame.K_x
KEY_INITIAL_TAKEOFF=pygame.K_RETURN; KEY_LAND_DISARM=pygame.K_l; KEY_RESPAWN_AND_TAKEOFF=pygame.K_r

# --- Global State Variables ---
is_armed = False; current_mode = ""; current_altitude_map_frame = 0.0
current_commanded_vx, current_commanded_vy, current_commanded_vz, current_commanded_wz = 0.0, 0.0, 0.0, 0.0
initial_ground_altitude_map_frame = 0.0; initial_altitude_captured_for_takeoff = False
first_takeoff_complete = False

# State Machine
STATE_IDLE=0; STATE_ARM_REQUESTED=1; STATE_OFFBOARD_REQUESTED=2; STATE_TAKING_OFF=3
STATE_MANUAL_CONTROL=4; STATE_LANDING_REQUESTED=5; STATE_SEQUENCE_IN_PROGRESS=6
flight_state = STATE_IDLE

# Proxies
vel_pub = None; arming_client = None; set_mode_client = None; set_model_state_client = None
pygame_initialized = False; screen = None; font = None; running = True


def handle_manual_control_inputs():
    global current_commanded_vx, current_commanded_vy, current_commanded_wz
    keys = pygame.key.get_pressed()
    if keys[KEY_PANIC_HOVER]:
        current_commanded_vx, current_commanded_vy, current_commanded_wz = 0.0, 0.0, 0.0; return
    desired = {'vx': 0.0, 'vy': 0.0, 'wz': 0.0}
    if keys[KEY_FORWARD]: desired['vx'] = MAX_SPEED_X
    elif keys[KEY_BACKWARD]: desired['vx'] = -MAX_SPEED_X
    if keys[KEY_RIGHT]: desired['vy'] = -MAX_SPEED_Y
    elif keys[KEY_LEFT]: desired['vy'] = MAX_SPEED_Y
    if keys[KEY_YAW_RIGHT]: desired['wz'] = -MAX_SPEED_YAW
    elif keys[KEY_YAW_LEFT]: desired['wz'] = MAX_SPEED_YAW
    current_commanded_vx += VELOCITY_RAMP_FACTOR * (desired['vx'] - current_commanded_vx)
    current_commanded_vy += VELOCITY_RAMP_FACTOR * (desired['vy'] - current_commanded_vy)
    current_commanded_wz += VELOCITY_RAMP_FACTOR * (desired['wz'] - current_commanded_wz)

def handle_altitude_target_adjustment():
    global INITIAL_MANUAL_FLIGHT_ALTITUDE
    keys = pygame.key.get_pressed()
    adjustment = ALTITUDE_ADJUSTMENT_RATE / CONTROL_FREQUENCY
    if keys[KEY_INCREASE_ALTITUDE_TARGET]: INITIAL_MANUAL_FLIGHT_ALTITUDE += adjustment
    elif keys[KEY_DECREASE_ALTITUDE_TARGET]: INITIAL_MANUAL_FLIGHT_ALTITUDE -= adjustment
    INITIAL_MANUAL_FLIGHT_ALTITUDE = np.clip(INITIAL_MANUAL_FLIGHT_ALTITUDE, 0.5, 5.0)

def state_cb(msg):
    global is_armed, current_mode
    is_armed = msg.armed; current_mode = msg.mode

def local_pose_cb(msg):
    global current_altitude_map_frame, initial_ground_altitude_map_frame, initial_altitude_captured_for_takeoff
    current_altitude_map_frame = msg.pose.position.z
    if flight_state == STATE_ARM_REQUESTED and not initial_altitude_captured_for_takeoff:
        initial_ground_altitude_map_frame = current_altitude_map_frame
        initial_altitude_captured_for_takeoff = True
        rospy.loginfo(f"[PoseCB] Initial takeoff ground altitude captured: {initial_ground_altitude_map_frame:.2f}m")

def call_service(proxy, *args, **kwargs):
    try:
        rospy.wait_for_service(proxy.resolved_name, timeout=1.0)
        return proxy(*args, **kwargs)
    except (rospy.ServiceException, rospy.ROSException, AttributeError): return None

def publish_velocity_command(vx, vy, vz, avz):
    global current_commanded_vx, current_commanded_vy, current_commanded_vz, current_commanded_wz
    current_commanded_vx, current_commanded_vy, current_commanded_vz, current_commanded_wz = vx, vy, vz, avz
    vel_cmd = TwistStamped()
    vel_cmd.header.stamp = rospy.Time.now()
    vel_cmd.twist.linear.x, vel_cmd.twist.linear.y, vel_cmd.twist.linear.z, vel_cmd.twist.angular.z = float(vx), float(vy), float(vz), float(avz)
    vel_pub.publish(vel_cmd)

def reset_to_idle_state(reason=""):
    global flight_state, initial_altitude_captured_for_takeoff
    rospy.logwarn(f"Resetting to IDLE state. Reason: {reason}")
    if is_armed: call_service(arming_client, False)
    flight_state = STATE_IDLE
    publish_velocity_command(0,0,0,0)
    initial_altitude_captured_for_takeoff = False

def update_pygame_display():
    if not pygame_initialized: return
    screen.fill((30, 30, 30))
    state_names = ["IDLE", "ARM_REQ", "OFFBOARD_REQ", "TAKING_OFF", "MANUAL_CONTROL", "LAND_REQ", "AUTO_SEQUENCE"]
    target_alt = f"(Tgt: {INITIAL_MANUAL_FLIGHT_ALTITUDE:.2f}m)" if flight_state == STATE_MANUAL_CONTROL else ""
    status = f"State: {state_names[flight_state]} | Armed: {is_armed} | Mode: {current_mode} | Alt: {current_altitude_map_frame:.2f}m {target_alt}"
    vel = f"CmdVel: Vx:{current_commanded_vx: .2f} Vy:{current_commanded_vy: .2f} Vz:{current_commanded_vz: .2f} Yaw:{current_commanded_wz: .2f}"
    status_sfc = font.render(status, True, (255, 255, 255)); vel_sfc = font.render(vel, True, (200, 200, 200))
    screen.blit(status_sfc, (10, 10)); screen.blit(vel_sfc, (10, 35))
    pygame.display.flip()

def _arm_and_set_offboard(timeout=10.0):
    start_time = rospy.Time.now(); rate = rospy.Rate(10)
    while (rospy.Time.now() - start_time) < rospy.Duration(timeout):
        if not is_armed: call_service(arming_client, True)
        elif current_mode != "OFFBOARD": call_service(set_mode_client, custom_mode="OFFBOARD")
        if is_armed and current_mode == "OFFBOARD": return True
        publish_velocity_command(0,0,0,0); rate.sleep()
    return False

def _execute_takeoff(start_alt, timeout=15.0):
    start_time = rospy.Time.now(); rate = rospy.Rate(CONTROL_FREQUENCY)
    target_altitude = start_alt + TAKEOFF_ALTITUDE
    while (rospy.Time.now() - start_time) < rospy.Duration(timeout):
        error = target_altitude - current_altitude_map_frame
        if abs(error) < 0.15: return True
        vz = np.clip(error * 1.5, -TAKEOFF_VZ, TAKEOFF_VZ)
        publish_velocity_command(0, 0, vz, 0); rate.sleep()
    return False

def handle_respawn_and_takeoff_sequence():
    global flight_state
    flight_state = STATE_SEQUENCE_IN_PROGRESS
    rospy.loginfo("--- STARTING ONE-TOUCH RESPAWN & TAKEOFF ---")

    if set_model_state_client:
        rospy.loginfo("Step 1: Teleporting...")
        call_service(set_model_state_client, ModelState(model_name=MODEL_NAME_FOR_GAZEBO_RESPAWN, pose=FIXED_RESPAWN_POSE))
        rospy.sleep(0.5)
    
    rospy.loginfo("Step 2: Arming...")
    if not _arm_and_set_offboard():
        reset_to_idle_state("Arm/Offboard failure in auto-sequence."); return
    
    rospy.loginfo("Step 3: Taking off...")
    if not _execute_takeoff(start_alt=current_altitude_map_frame):
        reset_to_idle_state("Takeoff failure in auto-sequence."); return
    
    rospy.loginfo("--- SEQUENCE COMPLETE --- Ready for manual control.");
    publish_velocity_command(0,0,0,0);
    flight_state = STATE_MANUAL_CONTROL

def print_instructions():
    print("\n--- Drone Teleop: Dual-Mode Control ---")
    print(f"  '{pygame.key.name(KEY_INITIAL_TAKEOFF).upper()}':  Manually Arm & Take off (FIRST FLIGHT ONLY).")
    print(f"  '{pygame.key.name(KEY_RESPAWN_AND_TAKEOFF).upper()}':   One-Touch Respawn & Auto-Takeoff (after first flight).")
    print(f"  '{pygame.key.name(KEY_LAND_DISARM).upper()}':   Land at current position and disarm.")
    print("------------------------------------------")

def main():
    global flight_state, running, pygame_initialized, first_takeoff_complete
    global vel_pub, arming_client, set_mode_client, set_model_state_client, screen, font

    rospy.init_node('drone_teleop_dual_mode', anonymous=True)
    vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=1)
    rospy.Subscriber('/mavros/state', State, state_cb); rospy.Subscriber('/mavros/local_position/pose', PoseStamped, local_pose_cb, queue_size=1)
    arming_client=rospy.ServiceProxy('/mavros/cmd/arming', CommandBool); set_mode_client=rospy.ServiceProxy('/mavros/set_mode', SetMode)
    set_model_state_client = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    pygame.init(); screen = pygame.display.set_mode((750, 60)); font = pygame.font.Font(None, 20); pygame_initialized = True
    print_instructions()

    rate = rospy.Rate(CONTROL_FREQUENCY)
    flight_state = STATE_IDLE
    first_takeoff_complete = False

    while running and not rospy.is_shutdown():
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                # *** THE CRITICAL FIX IS HERE ***
                # Respawn key ('R') is now checked first and works from manual control.
                if event.key == KEY_RESPAWN_AND_TAKEOFF and first_takeoff_complete:
                    if flight_state in [STATE_IDLE, STATE_MANUAL_CONTROL]:
                        handle_respawn_and_takeoff_sequence()

                # Land key ('L') only works when flying.
                elif event.key == KEY_LAND_DISARM and flight_state == STATE_MANUAL_CONTROL:
                    flight_state = STATE_LANDING_REQUESTED
                
                # Initial takeoff key ('Enter') only works from IDLE, before the first flight.
                elif event.key == KEY_INITIAL_TAKEOFF and not first_takeoff_complete:
                    if flight_state == STATE_IDLE:
                        flight_state = STATE_ARM_REQUESTED
        if not running: break

        # --- State Machine Logic ---
        if flight_state == STATE_IDLE:
            publish_velocity_command(0,0,0,0)

        elif flight_state == STATE_ARM_REQUESTED:
            publish_velocity_command(0,0,0,0)
            if not is_armed: call_service(arming_client, True)
            elif initial_altitude_captured_for_takeoff: flight_state = STATE_OFFBOARD_REQUESTED

        elif flight_state == STATE_OFFBOARD_REQUESTED:
            publish_velocity_command(0,0,0,0)
            if current_mode != "OFFBOARD": call_service(set_mode_client, custom_mode="OFFBOARD")
            else: flight_state = STATE_TAKING_OFF

        elif flight_state == STATE_TAKING_OFF:
            target_alt = initial_ground_altitude_map_frame + TAKEOFF_ALTITUDE
            error = target_alt - current_altitude_map_frame
            if abs(error) < 0.15:
                rospy.loginfo("Initial takeoff complete. Future restarts via 'R' key.")
                flight_state = STATE_MANUAL_CONTROL
                first_takeoff_complete = True
                publish_velocity_command(0,0,0,0)
            else:
                vz = np.clip(error * 1.5, -TAKEOFF_VZ, TAKEOFF_VZ)
                publish_velocity_command(0, 0, vz, 0)
            if not is_armed or current_mode != "OFFBOARD": reset_to_idle_state("Lost Arm/Offboard during takeoff")
        
        elif flight_state == STATE_MANUAL_CONTROL:
            handle_manual_control_inputs()
            handle_altitude_target_adjustment()
            alt_error = INITIAL_MANUAL_FLIGHT_ALTITUDE - current_altitude_map_frame
            vz = np.clip(alt_error * ALTITUDE_HOLD_KP, -MAX_ALTITUDE_CORRECTION_VZ, MAX_ALTITUDE_CORRECTION_VZ)
            if abs(alt_error) < 0.05: vz = 0.0
            publish_velocity_command(current_commanded_vx, current_commanded_vy, vz, current_commanded_wz)
            if not is_armed or current_mode != "OFFBOARD": reset_to_idle_state("Lost Arm/Offboard during flight")
        
        elif flight_state == STATE_LANDING_REQUESTED:
            publish_velocity_command(0,0,-0.4,0)
            if current_altitude_map_frame < 0.2: reset_to_idle_state("Landed")
        
        if flight_state != STATE_SEQUENCE_IN_PROGRESS: update_pygame_display()
        try: rate.sleep()
        except rospy.ROSInterruptException: running = False

    rospy.loginfo("Exiting...")
    reset_to_idle_state("Shutdown initiated.")
    if pygame_initialized: pygame.quit()

if __name__ == '__main__':
    try: main()
    except Exception as e:
        rospy.logfatal(f"Unhandled critical exception in main(): {e}"); import traceback; traceback.print_exc()
    finally:
        if 'pygame_initialized' in globals() and pygame_initialized: pygame.quit()
        rospy.loginfo("Script fully terminated.")