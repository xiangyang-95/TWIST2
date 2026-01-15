import torch
from pathlib import Path
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]

    return [i / fps for i in delta_indices]


def main():
    print("Starting diffusion training script ...")
    output_directory = Path("outputs/diffusion")
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
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
    cfg.use_amp = True
    cfg.optimizer_lr = 1e-5
    print(f"Diffusion Config: {cfg}")

    policy = DiffusionPolicy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()

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
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            print(f"DEBUG: batch['observation.images.head_image'] shape: {batch['observation.images.head_image'].shape}")
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")

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

    # Save the policy checkpoint, alongside the pre/post processors
    save_dir = output_directory / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
