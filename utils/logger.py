"""
Logging utilities for training and evaluation.
"""

import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Optional


def setup_logger(log_file: Optional[Path] = None, level=logging.INFO):
    """
    Setup logging configuration.

    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Only add console handler if not already present
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


class TensorBoardLogger:
    """
    Wrapper for TensorBoard SummaryWriter.
    """

    def __init__(self, log_dir: Path):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        """Log histogram."""
        self.writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        self.writer.add_text(tag, text, step)

    def close(self):
        """Close writer."""
        self.writer.close()
