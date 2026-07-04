import os
import yaml
import sys

def main():
    print("Configuring Hermes Agent to use local Chinese Router Proxy...")
    
    # Load port from config.yaml
    port = 8000
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            port = config.get("routing", {}).get("port", 8000)
    except Exception:
        pass

    # Locate ~/.hermes directory
    home_dir = os.path.expanduser("~")
    hermes_dir = os.path.join(home_dir, ".hermes")
    config_path = os.path.join(hermes_dir, "config.yaml")
    
    # Create directory if it doesn't exist
    if not os.path.exists(hermes_dir):
        try:
            os.makedirs(hermes_dir)
            print(f"Created Hermes config directory at: {hermes_dir}")
        except Exception as e:
            print(f"Error creating directory {hermes_dir}: {e}")
            sys.exit(1)
            
    # Load existing config or initialize new one
    config_data = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    config_data = loaded
            print(f"Loaded existing Hermes config from {config_path}")
        except Exception as e:
            print(f"Warning: Failed to load existing config.yaml ({e}). A new config will be generated.")
            
    # Modify config to point to our local proxy
    config_data["model"] = {
        "default": "custom-router",
        "provider": "local-proxy"
    }
    
    if "providers" not in config_data or not isinstance(config_data["providers"], dict):
        config_data["providers"] = {}
        
    config_data["providers"]["local-proxy"] = {
        "base_url": f"http://localhost:{port}/v1",
        "api_key": "dummy-key-handled-by-proxy"
    }
    
    # Save config
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False)
        print(f"Successfully configured Hermes! Config written to: {config_path}")
        print("\nHermes will now default to your local proxy:")
        print(f"  - Base URL: http://localhost:{port}/v1")
        print("  - Default Model: custom-router")
        print("  - Dynamic task-based routing & fallback chains are enabled!")
    except Exception as e:
        print(f"Error writing configuration to {config_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
