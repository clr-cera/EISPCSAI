import sys
from controller import Controller
import logging
import time

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("log.txt"), logging.StreamHandler()],
    )
    controller = Controller()
    start_time = time.time()
    controller.start()
    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time} seconds")
