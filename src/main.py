import sys
from controller import Controller
import logging

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("log.txt"), logging.StreamHandler()],
    )
    controller = Controller()
    controller.start()
