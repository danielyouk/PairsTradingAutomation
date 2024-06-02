import subprocess
import logging

# Set up logging
logging.basicConfig(filename='/Users/songyouk/PairsTradingAutomation/lecture03/twsstartmacos_python.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)

def run_shell_script():
    try:
        logging.info("Starting the shell script.")
        
        # Run the shell script
        result = subprocess.run(['/bin/bash', '/opt/ibc/twsstartmacos.sh'], capture_output=True, text=True, timeout = 30)
        
        # Log the output and error
        logging.info("Shell script output: %s", result.stdout)
        if result.stderr:
            logging.error("Shell script error: %s", result.stderr)
        
        logging.info("Shell script finished with return code: %d", result.returncode)
        
    except Exception as e:
        logging.error("An error occurred: %s", str(e))

if __name__ == "__main__":
    run_shell_script()

