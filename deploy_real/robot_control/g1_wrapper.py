import numpy as np
import time
import unitree_interface
from rich import print

ContollerMapping ={
    "A": 0x0100,
    "B": 0x0200,
    "X": 0x0400,
    "Y": 0x0800,
    "R1": 0x0001,
    "L1": 0x0002,
    "start": 0x0004,
    "select": 0x0008,
    "R2": 0x0010,
    "L2": 0x0020,
    "F1": 0x0040,
    "F2": 0x0080,
    "up": 0x1000,
    "right": 0x2000,
    "down": 0x4000,
    "left": 0x8000
    }

INIT_SEQUENCE = [
    [
        0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.,
        0., 0., 0.,
        0., 0.4, 0., 1.2, 0., 0., 0.,
        0., -0.4, 0., 1.2, 0., 0., 0.
    ],
    [
        0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.,
        0., 0., 0.,
        0., 0.985, 0., 1.2, 0., 0., 0.,
        0., -0.985, 0., 1.2, 0., 0., 0.
    ],
    [
        0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.,
        0., 0., 0.,
        0., 1.57, 0., 1.2, 0., 0., 0.,
        0., -1.57, 0., 1.2, 0., 0., 0.
    ],
    [
        0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.,
        0., 0., 0.,
        0., 1.57, 0., 0.0, 0., 0., 0.,
        0., -1.57, 0., 0.0, 0., 0., 0.
    ],
    [
        0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.,
        0., 0., 0.,
        0., 0., 0., -0.3, 0., 0., 0.,
        0., 0., 0., -0.3, 0., 0., 0.
    ]
]

class G1RealWorldEnv:
    def __init__(self, net, config):
        
        self.config = config
        self.robot = unitree_interface.create_robot(net, unitree_interface.RobotType.G1, unitree_interface.MessageType.HG)
        self.running = True
        
        self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25,
                25, 25, 25, 25,
            ])
        
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.tau_est = np.zeros(config.num_actions, dtype=np.float32)
        self.temperature = np.zeros((config.num_actions, 2), dtype=np.float32)
        self.voltage = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.counter = 0
        
        # Get robot configuration
        self.robot_config = self.robot.get_config()
        self.num_motors = self.robot.get_num_motors()
        print(f"Robot: {self.robot_config.name}")
        print(f"Motors: {self.num_motors}")
        print(f"Message type: {self.robot_config.message_type}")

        # Set control mode to PR (Pitch/Roll)
        self.robot.set_control_mode(unitree_interface.ControlMode.PR)
        control_mode = self.robot.get_control_mode()
        print(f"Control mode set to: {'PR' if control_mode == unitree_interface.ControlMode.PR else 'AB'}")

        # read the low state
        self.low_state = self.robot.read_low_state()
        self.low_cmd = self.robot.create_zero_command()
        print(f"Current robot state: {self.low_state}")
        print("[green]Successfully connected to the robot[/green]")

        self.controller_mapping = ContollerMapping

        # Arm SDK mode (optional — set via set_arm_sdk after construction)
        self.arm_sdk = None
        self.arm_sdk_joint_indices = None

    def set_arm_sdk(self, arm_sdk, joint_indices):
        """Enable arm SDK dispatching in send_robot_action."""
        print("[green]Arm SDK enabled. send_robot_action will dispatch to Arm SDK for specified joints.[/green]")
        self.arm_sdk = arm_sdk
        self.arm_sdk_joint_indices = joint_indices

    def read_robot_state(self) -> unitree_interface.LowState:
        """Read current robot state"""
        return self.robot.read_low_state()

    def read_controller_input(self) -> unitree_interface.WirelessController:
        """Read wireless controller input"""
        controller = self.robot.read_wireless_controller()
        return controller

    def send_cmd(self, cmd: unitree_interface.MotorCommand):
        self.robot.write_low_command(cmd)

    def move_to_default_pos(self):
        print("Waiting for the start signal to move to default pos...")
        while self.read_controller_input().keys != self.controller_mapping["start"]:
            time.sleep(self.config.control_dt)
        
        print("Moving to default pos.")
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        for seq in INIT_SEQUENCE:
            tgt_qpos = np.array(seq, dtype=np.float32)
            qpos = self.get_robot_state()[0]
            for i in range(num_step):
                alpha = i / num_step
                interp_qpos = qpos * (1 - alpha) + tgt_qpos * alpha
                self.send_robot_action(interp_qpos)
                time.sleep(self.config.control_dt)
    
    def move_to_exit_pos(self):
        print("Moving to exit pos.")
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        for seq in reversed(INIT_SEQUENCE):
            tgt_qpos = np.array(seq, dtype=np.float32)
            qpos = self.get_robot_state()[0]
            for i in range(num_step):
                alpha = i / num_step
                interp_qpos = qpos * (1 - alpha) + tgt_qpos * alpha
                self.send_robot_action(interp_qpos)
                time.sleep(self.config.control_dt)

    def default_pos_state(self):
        logger.info("Enter default pos state. Waiting for Button A signal...")
        while self.read_controller_input().keys != self.controller_mapping["A"]:
            # keep the default pos
            default_pos = np.array(INIT_SEQUENCE[-1], dtype=np.float32)
            self.send_robot_action(default_pos)
            time.sleep(self.config.control_dt)
    
    def get_robot_state(self):
        self.counter += 1
        low_state = self.read_robot_state()
        # Get the current joint position and velocity
        for i in range(len(self.config.joint2motor_idx)):
            self.qj[i] = low_state.motor.q[self.config.joint2motor_idx[i]]
            self.dqj[i] = low_state.motor.dq[self.config.joint2motor_idx[i]]
            self.tau_est[i] = low_state.motor.tau_est[self.config.joint2motor_idx[i]]
            self.voltage[i] = low_state.motor.voltage[self.config.joint2motor_idx[i]]
            self.temperature[i] = low_state.motor.temperature[self.config.joint2motor_idx[i]]
            
        # imu_state quaternion: w, x, y, z
        quat = low_state.imu.quat.copy()
        ang_vel = np.array(low_state.imu.omega, dtype=np.float32)
        accel = np.array(low_state.imu.accel, dtype=np.float32) # [m/s^2] 
        dof_pos = self.qj.copy()
        dof_vel = self.dqj.copy()
        dof_temp = self.temperature.copy()
        dof_tau = self.tau_est.copy()
        dof_vol = self.voltage.copy()

        # print(f"Robot state at step {self.counter}:")
        # print(f"  DOF positions: {dof_pos}")
        # print(f"  DOF velocities: {dof_vel}")
        # print(f"  IMU quaternion: {quat}")
        # print(f"  IMU angular velocity: {ang_vel}")
        # print(f"  DOF temperatures: {dof_temp}")
        # print(f"  DOF torques: {dof_tau}")
        # print(f"  DOF voltages: {dof_vol}")
        return (dof_pos, dof_vel, quat, ang_vel, dof_temp, dof_tau, dof_vol)
    
    def send_robot_action(self, target_dof_pos, kp_scale=1.0, kd_scale=1.0):
        if self.arm_sdk is not None:
            arm_cmd = self.arm_sdk.create_zero_command()
            arm_cmd.q_target = [float(target_dof_pos[idx]) for idx in self.arm_sdk_joint_indices]
            arm_cmd.dq_target = np.zeros_like(target_dof_pos)
            arm_cmd.kp = [self.config.kps[idx] * kp_scale for idx in self.arm_sdk_joint_indices]
            arm_cmd.kd = [self.config.kds[idx] * kd_scale for idx in self.arm_sdk_joint_indices]
            arm_cmd.tau_ff = np.zeros_like(target_dof_pos)
            arm_cmd.weight = 1.0
            self.arm_sdk.write_arm_command(arm_cmd)
        else:
            cmd = self.robot.create_zero_command()
            cmd.q_target = target_dof_pos.copy()
            cmd.dq_target = np.zeros_like(target_dof_pos)
            kps = [self.config.kps[i] * kp_scale for i in range(len(self.config.kps))]
            kds = [self.config.kds[i] * kd_scale for i in range(len(self.config.kds))]
            cmd.kp = kps
            cmd.kd = kds
            cmd.tau_ff = np.zeros_like(target_dof_pos)
            self.send_cmd(cmd)
        
    def close(self):
        exit()

if __name__ == "__main__":
    # example usage
    from config import Config
    config = Config("configs/g1.yaml")
    net="eno1"
    env = G1RealWorldEnv(net=net, config=config)
    test_arm_sdk = True
    if test_arm_sdk:
        arm_sdk = unitree_interface.ArmSdkInterface.create_g1_7dof(net, re_init=False)
        arm_sdk_joint_indices = arm_sdk.get_joint_indices()
        env.set_arm_sdk(arm_sdk, arm_sdk_joint_indices)

    while True:
        try:
            state = env.get_robot_state()
            controller = env.read_controller_input()
            env.move_to_default_pos()
        except KeyboardInterrupt:
            env.move_to_exit_pos()
            env.close()
            break