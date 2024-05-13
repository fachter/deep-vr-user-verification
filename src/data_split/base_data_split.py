import pathlib
from typing import Dict, List, Optional


class BaseDataSplit:
    split: Optional[Dict[str, List[pathlib.Path]]] = None
