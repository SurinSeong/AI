from langchain.tools import tool

# 데코레이터를 사용해 함수를 도구로 변환한다.
@tool
def add_numbers(a: int, b:int) -> int:
    """Add two numbers"""
    return a + b


@tool
def multiply_numbers(a: int, b:int) -> int:
    """Nultiply two numbers"""
    return a * b

# 도구 실행
print(add_numbers.invoke({"a": 3, "b": 4}))
print(multiply_numbers.invoke({"a": 3, "b": 4}))