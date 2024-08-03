import json

# 데이터 소스 코드
class DatasetSource:
    """
    데이터셋 소스 관리 클래스.
    
    Parameters
    ----------
    source : str
        데이터셋 소스 경로 또는 위치.
    """
    def __init__(self, source: str):
        """
        DatasetSource 초기화:
        [input]
        source : str
            데이터셋 소스 경로 또는 위치
        [output]
        None
        -------
        """
        self._source = source

    def to_json(self) -> str:
        """
        데이터셋 소스를 JSON 문자열로 변환.

        [input]
        None

        [output]
        str
            JSON 형식의 데이터셋 소스 문자열.
        -------
        """
        return json.dumps({"source": self._source})

    def _get_source_type(self) -> str:
        """
        데이터셋 소스 유형 반환.
        [input]
        None
        [output]
        str
            데이터셋 소스 유형.
        -------
        """
        return "file_path"