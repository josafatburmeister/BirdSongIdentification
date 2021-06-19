import logging, verboselogs
from tqdm.notebook import tqdm
from collections.abc import Iterable


class Logger:
    logger = verboselogs.VerboseLogger("Birdsong-Logger")
    logger.propagate = False
    # print logs to console
    console = logging.StreamHandler()
    logger.addHandler(console)

logger = Logger.logger


class ProgressBar:
    def iterable(self):
        if self.is_pipeline_run:
            return self.sequence
        else:
            return self.tqdm

    def __init__(self, sequence: Iterable, desc: str = "", position: int = 0, is_pipeline_run: bool = False):
        self.sequence = sequence
        self.is_pipeline_run = is_pipeline_run
        if self.is_pipeline_run:
            logger.info(desc)
        else:
            self.tqdm = tqdm(sequence, desc=desc, position=position)

    def write(self, text: str):
        if self.is_pipeline_run:
            logger.info(text)
        else:
            self.tqdm.write(text)
