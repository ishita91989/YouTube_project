import schedule
import time
import subprocess

def run_client_side():
    print("Running client-side.py")
    subprocess.run(["python", "client-side-back.py"], check=True)

def run_aggregation():
    print("Running aggregation.py")
    subprocess.run(["python", "aggregation-back.py"], check=True)

# Schedule the functions
schedule.every().minute.do(run_client_side)
schedule.every().minute.do(run_aggregation)

# Run the scheduler in an infinite loop
while True:
    schedule.run_pending()
    time.sleep(1)  # Sleep for a short time to prevent high CPU usage
