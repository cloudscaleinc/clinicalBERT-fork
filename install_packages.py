import subprocess
import sys

requirements_file = 'requirements.txt'
failed_packages = []

with open(requirements_file, 'r') as f:
    for line in f:
        package = line.strip() 
        if package and not package.startswith("#"): 
            print(f"Installing {package}...")
            try:
                # Use sys.executable to ensure using the correct pip in the current environment
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}. Error: {e}")
                failed_packages.append(package)

if failed_packages:
    print("\nFailed to install the following packages:")
    for pkg in failed_packages:
        print(pkg)
else:
    print("\nAll packages installed successfully!")
