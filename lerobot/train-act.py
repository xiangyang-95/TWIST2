import torch
from pathlib import Path
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]

    return [i / fps for i in delta_indices]


def main():
    print("Starting act training script ...")
    output_directory = Path("outputs/act")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dataset_id = "lerobot/twist-dataset"
    dataset_root = "/mnt/2eb9e109-0bb6-41db-a49a-483d3806fe10/xy-ws/unitree-g1-ws/TWIST2/lerobot/lerobot_twist_dataset"

    dataset_metadata = LeRobotDatasetMetadata(
        dataset_id,
        root=dataset_root
    )
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    cfg.use_amp = True
    # cfg.optimizer_lr = 1e-5
    print(f"ACT Config: {cfg}")

    policy = ACTPolicy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    print(f"Loading policy to device: {device} ...")
    policy.to(device)

    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }
    dataset = LeRobotDataset(
        dataset_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps
    )

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    training_steps = 10000
    log_freq = 1
    step = 0
    done = False
    loss_list = []
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
                loss_list.append(loss.item())

            step += 1
            if step % 100 == 0:
                print(f"Saving checkpoint at step {step} ...")
                checkpoint_dir = output_directory / f"checkpoint-{step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)

            if step >= training_steps:
                done = True
                break
        break
    
    # Plot the loss curve
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid()
    plt.savefig(output_directory / "final/training_loss_curve.png")

    # Save the policy checkpoint, alongside the pre/post processors
    save_dir = output_directory / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)

if __name__ == "__main__":
    main()
