import logging
from collections.abc import Iterable
from typing import Optional, Union, Type

import verboselogs
from tqdm.notebook import tqdm, tqdm_notebook


class Logger:
    """
    Class that holds a global logger instance.
    """

    logger = verboselogs.VerboseLogger("Birdsong-Logger")
    logger.propagate = False
    # print logs to console
    console = logging.StreamHandler()
    logger.addHandler(console)


logger = Logger.logger


class ProgressBar:
    """
    Class for logging progress in both notebook and non-notebook environments. In notebook environments a tqdm progress
    bar is used for progress visualization. In non-notebook environments plain text lines are output for progress logging.
    """

    def __init__(self,
                 total: int = 0,
                 sequence: Optional[Iterable] = None,
                 desc: str = "",
                 position: int = 0,
                 is_pipeline_run: bool = False) -> None:
        """

        Args:
            total: Number of steps until the progress bar is finished.
            sequence: Itarable for which the progress bar is to be incremented at each iteration.
            desc: Descriptive text to be displayed next to the progress bar.
            position: Start position of the progress bar.
            is_pipeline_run: Whether the progress bar is used in a non-notebook execution environment.
        """
        self.is_pipeline_run = is_pipeline_run
        self.sequence = sequence
        if self.is_pipeline_run:
            logger.info(desc)
        else:
            if self.sequence is not None:
                self.tqdm = tqdm(sequence, desc=desc, position=position)
            elif total:
                self.tqdm = tqdm(total=total, desc=desc, position=position)
            else:
                raise NameError("Either total or sequence parameter must be set")

    def iterable(self) -> Union[Optional[Iterable], Type[tqdm_notebook]]:
        """
        Creates an Iterable from the progress bar.

        Returns:
            Iterable wrapper of the progress bar that iterates over the underlying sequence.
        """
        if self.sequence is None:
            raise NameError("No sequence provided")
        if self.is_pipeline_run:
            return self.sequence
        else:
            return self.tqdm

    def update(self, n: int = 1) -> None:
        """
        Updates the progress bar.

        Args:
            n: Number of steps by which the progress bar should be increased.

        Returns:
            None
        """
        if not self.is_pipeline_run:
            self.tqdm.update(n)

    def write(self, text: str) -> None:
        """
        Logs text below the progress bar.

        Args:
            text: Text to be logged.

        Returns:
            None
        """
        if self.is_pipeline_run:
            logger.info(text)
        else:
            self.tqdm.write(text)
