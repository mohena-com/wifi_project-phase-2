import os
import sys
import subprocess
from pathlib import Path
from config_reader import ConfigReader

def install_requirements(config_path=None):
    print(f'0. install_requirements:')

    """
    Install requirements from requirements.txt file.
    
    Args:
        config_path (str, optional): Path to the configuration file. If None, uses default path.
    
    Returns:
        bool: True if installation was successful, False otherwise.
    """
    try:
        print(f'0. try start:')

        # Load configuration
        if config_path is None:
            config_path = Path('/content/drive/MyDrive/Intellipaat-sem3/wifi_project/FINAL_PROJECT/wifi_project/config/har_config.properties')
            print(f'0. config_path: {config_path}')
        config = ConfigReader(config_path)
        print(f'0. config_path: {config_path}')
        print(f'0. config: {config}')
        # Get requirements path
        if os.path.exists('/content/drive'):
            # Running in Colab
            requirements_path = Path('/content/drive/MyDrive/Intellipaat-sem3/wifi_project/FINAL_PROJECT/wifi_project/requirements.txt')
            print(f'1. requirements_path: {requirements_path}')
        else:
            # Running locally
            requirements_path = Path(__file__).parent.parent / 'requirements.txt'
            print(f'2. requirements_path: {requirements_path}')

        print(f"Looking for requirements file at: {requirements_path}")
        
        # Check if running in Google Colab
        try:
            from google.colab import drive
            IN_COLAB = True
            print("Running in Google Colab")
            
            # Mount Google Drive if not already mounted
            if not os.path.exists('/content/drive'):
                drive.mount('/content/drive')
                print("Google Drive mounted successfully")
        except ImportError:
            IN_COLAB = False
            print("Running locally")
        
        # Verify requirements file exists
        if not requirements_path.exists():
            print(f"Error: Requirements file not found at {requirements_path}")
            print("Please ensure the requirements.txt file exists at the correct location.")
            return False
        
        print(f"Installing requirements from {requirements_path}")
        
        # Install requirements using pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
        print("Requirements installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    # Allow passing config path as command line argument
    import argparse
    parser = argparse.ArgumentParser(description='Install project requirements')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    install_requirements(args.config) 