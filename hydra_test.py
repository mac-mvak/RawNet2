import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def my_app(cfg : DictConfig):
    print(cfg)
    return cfg




cfg = my_app()

print(cfg['user'])


