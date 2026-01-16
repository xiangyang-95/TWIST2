import os
import sys
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import unitree_interface as ui
except ImportError:
    logger.error("unitree_interface module not found. Build unitree_sdk2 first before running this script.")
    sys.exit(1)

Inspire_Num_Motors = 12


class InspireController:
    def __init__(self, interface: str, domain_id=0):
        if not interface:
            raise RuntimeError(f"Interface is required")

        self.interface = interface
        self.domain_id = domain_id

        logger.info(f"Initialize InspireController on {interface}...")
        self.inspire_hand_interface = ui.InspireHandInterface.create_inspire_hand(
            self.interface, True
        )

        # Internal command state
        # Initialize with default command (all zeros)
        self.cmd = self.inspire_hand_interface.create_zero_command()
        self.cmd.kp = self.inspire_hand_interface.get_default_kp()
        self.cmd.kd = self.inspire_hand_interface.get_default_kd()

        self.joint_mapping = {
            0: "r_pinky",
            1: "r_ring",
            2: "r_middle",
            3: "r_index",
            4: "r_thumb_bend",
            5: "r_thumb_rotation",
            6: "l_pinky",
            7: "l_ring",
            8: "l_middle",
            9: "l_index",
            10: "l_thumb_bend",
            11: "l_thumb_rotation",
        }

    def _ctrl_joint(self, id: int, value: float):
        if value < 0 or value > 1:
            logger.warning(f"Joint value {value} out of range [0, 1]")

        # clip value to [0, 1]
        value = max(0, min(1, value))

        # Update the command array
        if 0 <= id < Inspire_Num_Motors:
            self.cmd.q_target[id] = value

            # Write the updated command to DDS
            self.inspire_hand_interface.write_hand_command(self.cmd)
        else:
            logger.error(f"Joint id {id} out of range")

    @property
    def hand_state(self):
        """
        Return the current hand state using the new API structure.
        Note: The structure differs from the old unitree_sdk2py LowState_.
        Access motor positions via hand_state.motor.q
        """
        return self.inspire_hand_interface.read_hand_state()

    def open_hand(self):
        for joint_id in range(0, 12):
            self.cmd.q_target[joint_id] = 0.0
        self.inspire_hand_interface.write_hand_command(self.cmd)

    def close_hand(self):
        for joint_id in range(0, 12):
            self.cmd.q_target[joint_id] = 1.0
        self.inspire_hand_interface.write_hand_command(self.cmd)

    def ctrl_dual_hand(self, left_q_target: list, right_q_target: list):
        """set current left, right hand motor state target q"""
        if len(left_q_target) != 6 or len(right_q_target) != 6:
            raise ValueError(
                "left_q_target and right_q_target must have length 6 each")

        # Update command for left hand motors (indices 6-11)
        for i in range(6):
            self.cmd.q_target[6 + i] = left_q_target[i]

        # Update command for right hand motors (indices 0-5)
        for i in range(6):
            self.cmd.q_target[i] = right_q_target[i]

        # Send commands
        self.inspire_hand_interface.write_hand_command(self.cmd)


if __name__ == "__main__":
    # Simplet test for ctrl_dual_hand
    interface = ""
    if interface == "":
        logger.error("Please set the interface variable to the correct network interface.")
        sys.exit(1)

    inspire_controller = InspireController(
        interface=interface
    )

    logger.info("Running test sequence ...")
    for i in range(10):
        left_targets = [i / 10.0] * 6
        right_targets = [1.0 - (i / 10.0)] * 6
        logger.info(f"Step {i}: Left targets: {left_targets}, Right targets: {right_targets}")
        inspire_controller.ctrl_dual_hand(left_targets, right_targets)
        time.sleep(0.5)

    logger.info("Test sequence completed.")
