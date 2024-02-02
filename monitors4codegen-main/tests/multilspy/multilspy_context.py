"""
Provides the MultilspyContext class, which stores the context for a Multilspy test.
"""

import dataclasses
from monitors4codegen.multilspy.multilspy_config import MultilspyConfig
from monitors4codegen.multilspy.multilspy_logger import MultilspyLogger

@dataclasses.dataclass
class MultilspyContext:
    """
    Stores the context for a Multilspy test.
    """
    config: MultilspyConfig
    logger: MultilspyLogger
    source_directory: str