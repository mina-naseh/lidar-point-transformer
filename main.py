import subprocess

def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        # Run the script and wait for it to complete
        subprocess.run(["python", script_name], check=True)
        print(f"{script_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}")
        exit(1)  # Exit the program if a script fails
    except FileNotFoundError:
        print(f"Python interpreter or {script_name} not found.")
        exit(1)  # Exit the program if the script is not found

if __name__ == "__main__":
    # List of scripts to run in order
    scripts_to_run = ["main_pre.py", "main_lmf.py", "main_pyg.py"]

    for script in scripts_to_run:
        run_script(script)
