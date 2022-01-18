import sys

from loguru import logger

logger.add(sys.stderr, level="DEBUG")
