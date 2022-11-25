import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config/multi_modality_config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
