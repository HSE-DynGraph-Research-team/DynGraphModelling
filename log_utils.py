import logging

def get_logger(model_id):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(model_id)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'./logs/{model_id}.log')
    # fh = logging.FileHandler(f'log/{model_id}_{str(time.time())}.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
