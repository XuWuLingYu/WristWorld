import argparse
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="VGGT Training Launch Script")
    parser.add_argument("--config_file", type=str, default="default.yaml", 
                       help="Path to config file (relative to training/config/)")
    args = parser.parse_args()
    
    # Extract config name from path
    config_name = args.config_file.replace("training/config/", "").replace(".yaml", "")
    
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=config_name)
        
    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
