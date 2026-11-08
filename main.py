import hydra

from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig):
    config_data = config.data
    print(f"Data path: ", config_data.path) 

if __name__ == "__main__":
    main()
