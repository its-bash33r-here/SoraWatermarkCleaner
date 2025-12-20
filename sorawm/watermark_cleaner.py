from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np

from sorawm.cleaner.e2fgvi_hq_cleaner import E2FGVIHDCleaner, E2FGVIHDConfig
from sorawm.cleaner.lama_cleaner import LamaCleaner
from sorawm.schemas import CleanerType


class WaterMarkCleaner:
    def __new__(cls, cleaner_type: CleanerType, enable_torch_compile: bool):
        """
        Selects and instantiates a watermark cleaner implementation based on `cleaner_type`.
        
        Parameters:
            cleaner_type (CleanerType): The cleaner implementation to create.
            enable_torch_compile (bool): When creating an E2FGVI_HQ cleaner, enable or disable `torch.compile` in its configuration.
        
        Returns:
            Cleaner: An instance of the corresponding cleaner class (e.g., `LamaCleaner` or `E2FGVIHDCleaner`).
        
        Raises:
            ValueError: If `cleaner_type` is not a recognized CleanerType.
        """
        if cleaner_type == CleanerType.LAMA:
            return LamaCleaner()
        elif cleaner_type == CleanerType.E2FGVI_HQ:
            e2fgvi_hq_config = E2FGVIHDConfig(enable_torch_compile=enable_torch_compile)
            return E2FGVIHDCleaner(config=e2fgvi_hq_config)
        else:
            raise ValueError(f"Invalid cleaner type: {cleaner_type}")