# Agentic-AI-for-codes

## 프로젝트 구조 및 파일 설명

이 프로젝트는 사용자의 요청에 따라 웹 검색, 코드 생성, 파일 실행 등 다양한 작업을 수행하는 에이전트 AI입니다.

- **`main.py`**: 
    - 에이전트 AI (`AgentAI`)의 메인 실행 파일입니다.
    - 사용자 입력을 받아 작업을 계획하고 실행하며, 대화형 인터페이스를 제공합니다.
    - 각 기능 모듈(`TaskPlanner`, `CodeGeneratorAgent`, `WebHandler`, `FileManager`, `CodeExecutor`, `ModelManager`, `ResultFormatter`, `constants`)을 통합하여 전체 워크플로우를 관리합니다.
    - 주요 클래스/함수: `AgentAI`, `run_interactive`, `run_task`, `_execute_*_step`

- **`model_manager.py`**: 
    - OpenAI API 클라이언트 초기화 및 LLM 호출을 중앙에서 관리합니다.
    - 작업 유형(planning, code_gen, summarization 등)에 따라 사용할 모델을 선택합니다.
    - LLM API 호출 및 기본 오류 처리를 담당합니다.
    - 주요 클래스/함수: `ModelManager`, `get_model_for_task`, `call_llm`

- **`constants.py`**: 
    - 프로젝트 전체에서 사용되는 상수(주로 작업 유형)를 정의합니다.
    - 코드의 일관성을 유지하고 오타를 방지합니다.

- **`task_planner.py`**: 
    - 사용자의 자연어 요청을 분석하여 수행할 작업 단계를 계획합니다.
    - 명시적인 키워드 패턴을 우선 감지하고, 해당하지 않으면 LLM을 사용하여 계획을 생성합니다.
    - 주요 클래스/함수: `TaskPlanner`, `plan_task`, `_detect_explicit_patterns`, `_plan_with_llm`

- **`code_generator.py`**: 
    - LLM을 사용하여 코드를 생성하거나 수정합니다.
    - 작업 요청에서 언어를 감지하고, 코드 생성/수정/실행 요청 여부를 판단합니다.
    - LLM 응답에서 코드 블록을 추출하고, 유해 코드를 검사하며, 필요한 패키지를 감지합니다.
    - 생성된 코드를 파일에 저장합니다.
    - 주요 클래스/함수: `CodeGeneratorAgent`, `run`, `_detect_language_and_request`, `_clean_llm_code_output`, `_find_python_imports`, `_check_required_packages`

- **`web_handler.py`**: 
    - 주어진 쿼리로 웹 검색(Google)을 수행하고, 검색 결과 페이지의 내용을 가져옵니다.
    - 수집된 텍스트 내용을 LLM을 사용하여 요약합니다.
    - 주요 클래스/함수: `WebHandler`, `perform_web_search_and_summarize`, `_fetch_web_content`, `_summarize_text`

- **`file_manager.py`**: 
    - 파일 시스템 관련 작업(파일/디렉토리 생성, 삭제, 이동, 읽기, 쓰기, 탐색)을 처리합니다.
    - 파일 확장자를 기반으로 언어를 감지하는 유틸리티 함수를 포함합니다.
    - 주요 클래스/함수: `FileManager`, `manage_files`, `explore_directory`, `analyze_file`, `LANGUAGE_MAP`

- **`code_executor.py`**: 
    - 주어진 코드 문자열 또는 파일을 실행합니다.
    - 다양한 프로그래밍 언어(Python, C++, Java, C#, JavaScript 등)의 실행 및 컴파일을 지원합니다.
    - 임시 디렉토리를 사용하여 실행 환경을 관리합니다.
    - 주요 클래스/함수: `CodeExecutor`, `execute_code`, `execute_file`, `_execute_with_popen`, `_compile_code`

- **`result_formatter.py`**: 
    - 여러 단계의 작업 결과를 사용자 친화적인 형식으로 조합합니다.
    - 각 단계의 성공/실패 여부와 결과를 명확하게 표시합니다.
    - 주요 클래스/함수: `ResultFormatter`, `combine_step_results`

- **`utils.py`**: 
    - 프로젝트 전반에서 사용되는 유틸리티 함수를 포함합니다.
    - 코드 실행 결과 문자열을 포맷팅하고, 수정 가능한 오류인지 판단하는 함수 등을 제공합니다.
    - 주요 함수: `format_execution_result`, `is_fixable_code_error`

- **`.env`**: 
    - OpenAI API 키와 같은 민감한 환경 변수를 저장합니다.

- **`output/`**: 
    - `CodeGeneratorAgent`가 생성한 코드 파일이 저장되는 디렉토리입니다.

- **`agent.log`**: 
    - 에이전트 실행 중 발생하는 로그 메시지가 기록되는 파일입니다.

- **`requirements.txt`**: 
    - 프로젝트 실행에 필요한 Python 패키지 목록입니다.