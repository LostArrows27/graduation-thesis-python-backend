import logging


info_log = logging.getLogger('uvicorn.info')
error_log = logging.getLogger('uvicorn.error')


def log_info(message):
    info_log.info(message)


def log_error(message):
    error_log.error(message)
