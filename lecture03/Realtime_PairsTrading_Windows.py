import logging # need to be installed with pip install logging
import subprocess # default package
from ib_insync import IB # need to be installed with pip install ib_insync
import time # default package

logging.basicConfig(filename=r'lecture03\log\tws_check.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
def check_tws_connection():
    ib = IB()
    try:
        ib.connect()
        if ib.isConnected():
            logging.info("TWS is already logged in, skipping IBC start.")
            return True
        else:
            return False
    except Exception as e:
        # add pause for 5 seconds to allow TWS to start
        time.sleep(5)
        logging.error(f"Error connecting to TWS: {e}")
        return False

if not check_tws_connection():
    logging.info("Starting IBC as TWS is not logged in.")
    
    result = subprocess.run(['C:\\IBC\\StartTWS.bat'], capture_output=True, text=True, timeout=30)
