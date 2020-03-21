import logging.config

__version__ = "0.1.0"

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": "%(asctime)-15s %(message)s"}},
        "handlers": {
            "console": {
                "level": "DEBUG",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "entiretydotai": {"handlers": ["console"], "level": "DEBUG", "propagate": False}
        },
    }
)

logger = logging.getLogger("entiretydotai")

from .data.nlp_dataset import FlairDataset