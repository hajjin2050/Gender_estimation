from typing import Optional, Dict, Any
import hashlib

from mlflow.entities import Dataset

from data.dataset_source import DatasetSource

class CustomDataset(Dataset):
    def __init__(self, source: DatasetSource, name: Optional[str] = None, dataset_type: Optional[str] = None):
        """
        CustomDataset 초기화 메서드.

        [input]
        source : DatasetSource
            데이터셋 소스 정보.
        name : str, optional
            데이터셋 이름.
        dataset_type : str, optional
            데이터셋 타입 ('train' 또는 'val').

        [output]
        None
        -------
        """
        super().__init__(source, name, self._compute_digest())
        self.dataset_type = dataset_type

    def _compute_digest(self) -> str:
        """
        데이터셋 소스의  계산.

        [input]
        None

        [output]
        str
            .
        -------
        """
        hasher = hashlib.md5()
        hasher.update(self.source.to_json().encode('utf-8'))
        return hasher.hexdigest()[:8]

    def to_dict(self) -> Dict[str, str]:
        """
        데이터셋 정보를 딕셔너리로 반환

        [input]
        None

        [output]
        dict
            데이터셋 정보 딕셔너리
        -------
        """
        return {
            "name": self.name,
            "digest": self.digest,
            "source": self.source.to_json(),
            "source_type": self.source._get_source_type(),
            "dataset_type": self.dataset_type
        }

    @property
    def profile(self) -> Optional[Any]:
        return None

    @property
    def schema(self) -> Optional[Dict[str, Any]]:
        """
        데이터셋 스키마 반환

        [input]
        None

        [output]
        dict
            데이터셋 스키마
        -------
        """
        schema = {
            "columns": [
                {"name": "image_id", "type": "str"},
                {"name": "gender", "type": "str"}
            ],
            "dataset_type": self.dataset_type
        }
        return schema