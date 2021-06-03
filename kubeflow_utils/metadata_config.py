from dataclasses import dataclass
from typing import List, Union


@dataclass
class MetadataConfig:
    model_names: Union[str, List[str]]
    model_description: str
    model_type: str
    model_version: str
    dataset_name: str
    dataset_description: str
    dataset_path: str
    dataset_version: str
    maturity_state: str
    training_framework_name: str
    training_framework_version: str
    owner: str
