import re
import os
import sys

def format_execution_result(execution_result_str: str) -> str:
    """CodeExecutor 결과를 사용자 친화적 메시지로 포맷"""
    if not execution_result_str:
        return "실행 결과가 없습니다."

    # Ensure it's a string
    execution_result_str = str(execution_result_str).strip()

    if execution_result_str.startswith("ModuleNotFoundError: No module named"):
        try:
            # Extract module name, handling potential variations like quotes
            match = re.search(r"No module named \'(.+?)\'", execution_result_str)
            if match:
                missing_module = match.group(1)
                return f"[오류] 코드를 실행하려면 '{missing_module}' 패키지가 필요합니다.\n터미널에서 '{sys.executable} -m pip install {missing_module}' 명령어로 설치해주세요."
        except Exception:
            pass # Fallback to generic message if parsing fails
        return f"[오류] 필요한 파이썬 패키지를 찾을 수 없습니다: {execution_result_str}"

    elif execution_result_str.startswith("FileNotFoundError: Required command "):
        try:
            missing_command = re.search(r"\'(.+?)\'", execution_result_str).group(1)
            return f"[오류] 코드 실행에 필요한 '{missing_command}' 명령어를 찾을 수 없습니다.\n관련 언어/도구를 설치하고 PATH 환경 변수를 확인해주세요."
        except Exception:
            pass
        return f"[오류] 실행에 필요한 명령어를 찾을 수 없습니다: {execution_result_str}"
    elif "SyntaxError:" in execution_result_str:
        return f"[오류] 코드 문법 오류가 있습니다:\n{execution_result_str}"
    elif "NameError:" in execution_result_str:
        return f"[오류] 정의되지 않은 이름(변수/함수)을 사용했습니다:\n{execution_result_str}"
    elif "TypeError:" in execution_result_str:
        return f"[오류] 잘못된 타입의 값을 사용했습니다:\n{execution_result_str}"
    elif "IndexError:" in execution_result_str:
        return f"[오류] 잘못된 인덱스를 사용했습니다:\n{execution_result_str}"
    elif "KeyError:" in execution_result_str:
        return f"[오류] 존재하지 않는 키를 사용했습니다:\n{execution_result_str}"
    elif "AttributeError:" in execution_result_str:
        return f"[오류] 객체에 존재하지 않는 속성이나 메서드를 사용했습니다:\n{execution_result_str}"
    elif "ImportError:" in execution_result_str:
        return f"[오류] 모듈 가져오기(import)에 실패했습니다:\n{execution_result_str}"
    # Add more specific error checks as needed

    else:
        # Generic error formatting if not specifically caught
        if "error" in execution_result_str.lower() or "exception" in execution_result_str.lower():
             # Try to keep it concise
             lines = execution_result_str.splitlines()
             if len(lines) > 5:
                 return f"[오류] 실행 중 오류 발생:\n" + "\n".join(lines[:2] + ["..."] + lines[-2:])
             else:
                 return f"[오류] 실행 중 오류 발생:\n{execution_result_str}"

    # If no specific error pattern is matched, return the original string
    return execution_result_str

def is_fixable_code_error(error_message: str) -> bool:
    """실행 결과 오류 메시지가 코드 자체의 문제로 수정 가능한지 판단"""
    if not error_message:
        return False

    # 설치/환경 오류 키워드 제외
    if any(kw in error_message for kw in [
        "ModuleNotFoundError",
        "FileNotFoundError: Required command",
        "cannot find file", # Common compiler error
        "No such file or directory", # Common system error
        "not recognized as an internal or external command", # Windows command error
        "command not found" # Linux/macOS command error
        ]):
        return False

    # 일반적인 코드 오류 키워드 포함 (더 포괄적으로)
    if any(kw in error_message for kw in [
        "SyntaxError", "NameError", "TypeError", "ValueError", "IndexError",
        "AttributeError", "KeyError", "ImportError", # Python specific
        "error:", "Exception", "Traceback", # General error indicators
        "undeclared identifier", "expected ';'", # C/C++ common errors
        "NullReferenceException", "InvalidOperationException", # C# common errors
        "panic:", # Rust panic
        "Uncaught ReferenceError", "Uncaught TypeError" # JavaScript common errors
        ]):
        # Double check it's not an import error that looks like a ModuleNotFoundError
        if "ImportError: cannot import name" in error_message:
            return True # This is often a code structure/circular dependency issue
        if "ImportError" in error_message and "No module named" not in error_message:
            return True # Other import errors might be code-related

        return True

    return False # 위 조건에 해당하지 않으면 수정 불가능으로 판단 