import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

current_state = [
 -1.54092741e-03, -1.20889808e-04, -9.67440831e-03, -1.58237624e-02,
 -1.22022871e-02,  1.14009428e-01,  2.72871674e-02, -1.22633882e-02,
  4.45357318e-02, -1.33464139e-01, -2.37165396e-02,  5.32012340e-02,
  2.39592529e-02,  2.27730880e-02,  3.02783216e-02, -6.90317133e-02,
  2.19804173e-02, -4.23689649e-02,  2.51007717e-02, -9.66834234e-03,
  1.03255099e-01,  3.12982779e-01, -1.61312846e-01,  3.34745145e-01,
 -3.91319770e-01, -1.88255569e-01,  1.23823272e-01, -1.22579548e-01,
 -2.71717697e-01,  4.17533338e-01,  3.98258794e-01,  2.11671731e-01,
 -1.79829648e-01, -1.78381979e-01
]

dataset_id = "lerobot/twist-dataset"
dataset_root = "/mnt/2eb9e109-0bb6-41db-a49a-483d3806fe10/xy-ws/unitree-g1-ws/TWIST2/lerobot/lerobot_twist_dataset"
temp_dataset = LeRobotDataset(
    dataset_id,
    root=dataset_root,
)
first_frame = temp_dataset[0]
initial_state = first_frame['observation.state'].cpu().numpy()
initial_action = first_frame['action'].cpu().numpy()

print(f"Initial state from dataset: {initial_state}")

# Get first 35 and compare
print("Comparing first 35 elements of current_state and initial_state:")
for i in range(35):
    print(f"Index {i}: current_state = {current_state[i]}, initial_state = {initial_state[i]}")
    np.testing.assert_almost_equal(current_state[i], initial_state[i], decimal=5)
    