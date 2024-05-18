import pkg_resources
import subprocess

def get_installed_packages():
    """Returns a dictionary of installed packages and their versions."""
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    return installed_packages

def update_requirements_file(requirements_file='requirements.txt'):
    """Reads the requirements.txt file, gets the installed package versions,
    and writes the packages with their versions back to the file."""
    try:
        # Read the packages from the requirements.txt file
        with open(requirements_file, 'r') as file:
            packages = file.readlines()

        # Get installed packages and their versions
        installed_packages = get_installed_packages()

        # Prepare the updated package list with versions
        updated_packages = []
        for package in packages:
            package = package.strip()
            if package in installed_packages:
                version = installed_packages[package]
                updated_packages.append(f"{package}=={version}\n")
            else:
                # If package is not found in installed packages, keep it as is
                updated_packages.append(f"{package}\n")

        # Write the updated packages with versions back to requirements.txt
        with open(requirements_file, 'w') as file:
            file.writelines(updated_packages)

        print(f"Updated {requirements_file} with package versions.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    update_requirements_file()