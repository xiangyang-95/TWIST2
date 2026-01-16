import logging
import numpy as np
import sys
import os

# Add unitree_sdk2 python binding path
# Helper logic to find the module if path is not set up
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# Search in likely build locations relative to this file
possible_paths = [
    os.path.abspath(os.path.join(current_file_dir, '../../../thirdparty/unitree_sdk2/build/lib')),
    os.path.abspath(os.path.join(current_file_dir, '../../../../thirdparty/unitree_sdk2/build/lib')),
]

module_found = False
for path in possible_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)
        try:
            import unitree_interface as ui
            module_found = True
            break
        except ImportError:
            continue

if not module_found:
    print("Warning: unitree_interface module not found in likely locations. Import might fail.")
    try:
        import unitree_interface as ui
    except ImportError:
        pass # Will crash later if not found

logger = logging.getLogger(__name__)

# Joint Indices
Inspire_Num_Motors = 12

class InspireController:
    def __init__(self, interface: str, domain_id=0):
        if not interface:
            raise RuntimeError(f"Interface is required")
        
        self.interface = interface
        self.domain_id = domain_id
        
        print(f"Initialize InspireController on {interface}...")
        
        # Initialize the C++ binding interface
        # We use create_left_hand as a handle for the single topic interface
        # The topic is shared (rt/inspire/cmd) efficiently managing all 12 motors
        self.inspire_hand_interface = ui.InspireHandInterface.create_left_hand(self.interface, True)
        
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

        

