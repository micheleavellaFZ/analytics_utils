from pathlib import Path
import subprocess


def run_python_script(
    script_path: str | Path, script_args: None | list[str] = None
) -> None:
    command = ["python3", str(script_path)]
    if script_args:
        command += script_args
    try:
        result = subprocess.run(
            command,
            check=True,  # This will raise an exception if the command fails
            capture_output=True,
            text=True,
        )

        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")

    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise
