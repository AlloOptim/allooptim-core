"""Trading Allocation Package"""

import logging

from common.config.logger import setup_service_logger

setup_service_logger("allocation")

test_logger = logging.getLogger(__name__)
test_logger.info("Allocation package initialized successfully")
