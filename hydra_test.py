import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(version_base=None, config_path='hydra_config', config_name='config')
def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    #print(cfg['n_gpu'])
    #u = cfg['data']['train']
    #w = cfg.get('data')
    #print(w)


if __name__ == "__main__":
    my_app()
