from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class TrainingResult:
    models: Union[List[object], object]
    evaluation: Dict
    hyperparameters: Dict
