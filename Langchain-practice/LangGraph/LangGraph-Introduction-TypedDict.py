# 1. TypedDict와 Dict의 차이점 예시
from typing import Dict, TypedDict

# 일반적인 파이썬 딕셔너리 사용
sample_dict: Dict[str, str] = {
    "name": "수린",
    "age": "27",
    "job": "개발자"
}

# TypedDict 사용
class Person(TypedDict):
    name: str
    age: int
    job: str

typed_dict: Person = {"name": "surin", "age": 27, "job": "student"}

# dict의 경우
sample_dict["age"] = 35    # 문자열에서 정수로 변경되어도 오류 없음
sample_dict["new_field"] = "추가 정보"    # 새로운 필드 추가 가능
print(sample_dict)

# TypedDict의 경우
typed_dict["age"] = 35    # 정수형으로 올바르게 사용해야 함.
typed_dict["age"] = "35"    # 타입 체커가 오류를 감지
typed_dict["new_field"] = (
    "추가 정보"    # 타입 체커가 정의되지 않은 키이기 때문에 오류를 발생시킨다.
)
print(typed_dict)