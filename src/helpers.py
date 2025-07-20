import yaml

def load_config(name):
    """
    Loads a specific value from config.yaml based on the provided key name.
    
    Args:
        name (str): The key to retrieve from the config.yaml file.
        
    Returns:
        The value associated with the key.
        
    Raises:
        FileNotFoundError: If config.yaml is not found.
        KeyError: If the specified key is not in the config file.
    """
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            return config[name]
    except FileNotFoundError:
        raise FileNotFoundError("config.yaml file not found")
    except KeyError:
        raise KeyError(f"Key '{name}' not found in config.yaml")
