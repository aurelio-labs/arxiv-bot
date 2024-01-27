import logging

import colorlog


class CustomFormatter(colorlog.ColoredFormatter):
    def __init__(self):
        super().__init__(
            "%(log_color)s%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
            reset=True,
            style="%",
        )


def setup_custom_logger(name):
    formatter = CustomFormatter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


logger = setup_custom_logger("__name__")