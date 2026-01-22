import yaml

def load_safe_yaml(config_path):
    """
    Load contents of a yaml

    Args:
        config_path: path to the yaml file
    
    Returns:
        contents of yaml as dict
    """
    with open(config_path, "r") as file:
            return yaml.safe_load(file)