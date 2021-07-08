import logging
import verboselogs
from collections.abc import Iterable
from typing import Optional

from tqdm.notebook import tqdm


class Logger:
    logger = verboselogs.VerboseLogger("Birdsong-Logger")
    logger.propagate = False
    # print logs to console
    console = logging.StreamHandler()
    logger.addHandler(console)


logger = Logger.logger


class ProgressBar:
    def iterable(self):
        if not self.sequence:
            raise NameError("No sequence provided")
        if self.is_pipeline_run:
            return self.sequence
        else:
            return self.tqdm

    def __init__(self, total: int = 0, sequence: Optional[Iterable] = None, desc: str = "", position: int = 0,
                 is_pipeline_run: bool = False):
        self.is_pipeline_run = is_pipeline_run
        self.sequence = sequence
        if self.is_pipeline_run:
            logger.info(desc)
        else:
            if self.sequence:
                self.tqdm = tqdm(sequence, desc=desc, position=position)
            elif total:
                self.tqdm = tqdm(total=total, desc=desc, position=position)
            else:
                raise NameError("Either total or sequence parameter must be set")

    def update(self, n: int = 1):
        if not self.is_pipeline_run:
            self.tqdm.update(n)

    def write(self, text: str):
        if self.is_pipeline_run:
            logger.info(text)
        else:
            self.tqdm.write(text)
