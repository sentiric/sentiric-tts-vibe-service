# sentiric-tts-vibe-service/app/core/logging.py
import logging
import sys
import structlog

def setup_logging(log_level: str, env: str):
    log_level = log_level.upper()

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if env == "development":
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        processors = shared_processors + [structlog.processors.JSONRenderer()]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers = [] 
        uvicorn_logger.propagate = True

logger = structlog.get_logger()