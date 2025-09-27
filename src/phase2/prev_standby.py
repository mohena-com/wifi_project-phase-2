import subprocess
import time

def prevent_standby(duration_seconds):
    """
    Prevent macOS from sleeping for duration_seconds seconds using caffeinate.
    """
    # Start caffeinate subprocess
    proc = subprocess.Popen(['caffeinate'])

    try:
        print(f"Standby prevention started for {duration_seconds} seconds...")
        for remaining in range(duration_seconds, 0, -1):
            print(f"Time left: {remaining} seconds", end="\r", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user, stopping standby prevention...")
    finally:
        # Terminate caffeinate subprocess
        proc.terminate()
        proc.wait()
        print("Standby prevention stopped.")

if __name__ == '__main__':
    # For example, prevent standby for 1 hour = 3600 seconds
    prevent_standby(3600*10)
