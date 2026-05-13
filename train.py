import os
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = '0'  # Disable weights_only
import torch

# Save the original function
_original_torch_load = torch.load

# Create a wrapper that defaults weights_only=False
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Replace torch.load with your patched version
torch.load = patched_torch_load

import dotenv
import hydra
import torch
from omegaconf import DictConfig

from gotennet.utils.utils import find_config_directory  # Import the utility function

# Load environment variables from `.env` file if it exists
# Recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

# Find configs directory using the utility function
config_dir = find_config_directory()

# Enable TF32 precision for CUDA operations to leverage Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = True


@hydra.main(version_base="1.3", config_path=config_dir, config_name="train.yaml")
def main(cfg: DictConfig) -> float:
    """
    Main training function called by Hydra.

    This function serves as the entry point for the training process. It imports
    necessary modules, applies optional utilities, trains the model, and returns
    the optimized metric value.

    Args:
        cfg (DictConfig): Configuration composed by Hydra from command line arguments
                         and config files. Contains all parameters for training.

    Returns:
        float: Value of the optimized metric for hyperparameter optimization.
    """
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from gotennet import utils
    from gotennet.training_pipeline import train

    # Applies optional utilities
    utils.extras(cfg)

    # Train model
    metric_dict, _ = train(cfg)

    # Safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict,
        metric_name=cfg.get("optimized_metric"),
    )

    # Return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
