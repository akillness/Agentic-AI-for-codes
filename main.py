from typing import List, Dict, Any, Callable, Tuple, Union
import os
import re
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from googlesearch import search as google_search
import requests
from bs4 import BeautifulSoup
from file_manager import FileManager
from code_executor import CodeExecutor
from code_generator import CodeGeneratorAgent
import sys
import shutil

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent.log', encoding='utf-8')
    ]
)

class AgentAI:
    # 작업 타입 상수 정의
    TASK_SEARCH = "search"
    TASK_CODE_GENERATION = "code_generation"
    TASK_FILE_EXECUTION = "file_execution"
    TASK_CODE_BLOCK_EXECUTION = "code_block_execution"
    TASK_COMPILATION = "compilation"
    TASK_COMPILED_RUN = "compiled_run"
    TASK_DIRECTORY_EXPLORATION = "directory_exploration"
    TASK_FILE_MANAGEMENT = "file_management"
    
    def __init__(self, name: str, description: str, memory_limit: int = 10):
        """초기화 함수
        
        Args:
            name (str): 에이전트의 이름
            description (str): 에이전트의 설명
            memory_limit (int, optional): 메모리에 저장할 최대 대화 수. Defaults to 10.
        """
        load_dotenv()  # .env 파일에서 환경 변수 로드
        
        self.name = name
        self.description = description
        self.memory: List[Dict[str, Any]] = []
        self.memory_limit = memory_limit
        
        # OpenAI API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI()
        logging.info(f"{self.name} 에이전트가 성공적으로 초기화되었습니다.")
        
        # 에이전트 및 도구 초기화
        self.code_generator = CodeGeneratorAgent(client=self.client) # CodeGeneratorAgent 인스턴스 생성
        
        # 시스템 메시지 설정
        self.system_message = f"""당신은 {name}이라는 이름의 AI 에이전트입니다.
{description}

당신은 다음과 같은 작업들을 수행할 수 있습니다:
1. 파일 관리 (생성, 삭제, 이동)
2. 코드 실행 (파일 또는 코드 블록)
3. 인터넷 검색
4. 코드 생성
5. 디렉토리 탐색
6. 컴파일 및 컴파일된 파일 실행

각 작업을 필요에 따라 순차적으로 수행할 수 있습니다."""

    def run_interactive(self):
        """대화형 모드로 실행"""
        print(f"\n=== {self.name} 시작 ===")
        print(f"{self.description}")

        while True:
            try:
                # 사용자 입력 받기
                user_input = input("\n명령어를 입력하세요 (도움말: help): ").strip()
                
                # 종료 명령 처리
                if user_input.lower() in ['exit', '종료', 'quit', 'q']:
                    print("프로그램을 종료합니다.")
                    break                
                
                # 작업 실행
                if user_input:
                    print("\n=== 작업 실행 ===")
                    result_message = self.run_task(user_input)
                    print(f"\n결과:\n{result_message}")
                    print("=" * 50)
            
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                break

    def _manage_memory(self):
        """메모리 관리 함수"""
        if len(self.memory) > self.memory_limit:
            removed = self.memory[:-self.memory_limit]
            self.memory = self.memory[-self.memory_limit:]
            logging.info(f"{len(removed)}개의 오래된 대화가 메모리에서 제거되었습니다.")
    
    def _fetch_web_content(self, query: str) -> List[str]:
        """인터넷 검색을 수행하고 웹 콘텐츠를 반환"""
        logging.info(f"웹 검색 시도: {query}")
        
        search_results_text = []
        urls_processed = []
        logging.info("구글 검색 시작...")
        
        try:
            # 검색 결과 가져오기 (최대 3개 URL)
            for url in google_search(query, num=3):
                urls_processed.append(url) # 어떤 URL을 처리했는지 기록
                try:
                    logging.info(f"URL 처리 중: {url}")
                    response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'}) # User-Agent 추가
                    response.raise_for_status() # HTTP 오류 확인
                    response.encoding = response.apparent_encoding # 인코딩 자동 감지 시도
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
                        script_or_style.decompose()
                    
                    text = soup.get_text(separator=' ', strip=True)
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    cleaned_text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    if cleaned_text: # 내용이 있는 경우에만 추가
                        search_results_text.append(cleaned_text[:1500]) # 요약을 위해 조금 더 긴 내용 사용
                    logging.info(f"URL {url} 처리 완료 (내용 길이: {len(cleaned_text)})")
                    
                except requests.exceptions.RequestException as e:
                    logging.warning(f"URL {url} 요청 오류: {str(e)}")
                except Exception as e:
                    logging.warning(f"URL {url} 처리 중 오류: {str(e)}")
                
                if len(search_results_text) >= 2: # 최대 2개의 성공적인 결과만 사용 (요약 부담 줄이기)
                    break
                    
        except Exception as e:
            logging.error(f"구글 검색 API 호출 중 오류: {e}")
            return []
        
        return search_results_text
    
    def _summarize_text(self, query: str, text_content: List[str]) -> str:
        """LLM을 사용하여 텍스트 내용을 요약"""
        if not text_content:
            return "관련 정보를 찾을 수 없습니다."

        # 검색 결과 텍스트들을 하나로 합침
        context = "\n\n---\n\n".join(text_content)
        context_for_llm = context[:4000] # LLM 토큰 제한 고려하여 컨텍스트 길이 제한
        logging.info(f"텍스트 요약 시도 (컨텍스트 길이: {len(context_for_llm)})...")
        
        try:
            # LLM을 이용한 요약
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Summarize the following text context to directly answer the user's original question. Provide a concise and relevant answer based *only* on the provided text."},
                    {"role": "user", "content": f"Original Question: {query}\n\nContext:\n{context_for_llm}"} # 원본 질문과 컨텍스트 전달
                ],
                temperature=0.3, # 좀 더 사실 기반 요약을 위해 temperature 낮춤
                max_tokens=150 # 요약 길이 제한
            )
            summary = response.choices[0].message.content.strip()
            
            if not summary:
                logging.warning("LLM 요약 결과가 비어 있습니다.")
                # 요약 실패 시, 간단한 결과라도 보여주기 (예: 첫번째 결과 일부)
                return f"검색 결과를 요약하는 데 실패했습니다. 첫 번째 검색 결과 일부: \n{text_content[0][:300]}..."

            logging.info("텍스트 요약 성공")
            return summary
            
        except Exception as e:
            logging.error(f"LLM 요약 API 호출 중 오류: {e}")
            # 요약 실패 시, 간단한 결과라도 보여주기
            return f"텍스트를 요약하는 중 오류가 발생했습니다. 첫 번째 검색 결과 일부: \n{text_content[0][:300]}..."
    
    def _perform_web_search_and_summarize(self, query: str) -> Dict[str, Any]:
        """웹 검색 수행 및 결과 요약"""
        # 웹 검색 수행
        search_results = self._fetch_web_content(query)
        
        if not search_results:
            return {
                "success": False,
                "result": "웹 검색 중 오류가 발생했거나 관련 정보를 찾을 수 없습니다.",
                "raw_content": []
            }
        
        # 검색 결과 요약
        summary = self._summarize_text(query, search_results)
        
        return {
            "success": True,
            "result": summary,
            "raw_content": search_results
        }

    def _plan_task(self, task: str) -> List[Dict[str, Any]]:
        """작업 계획 생성
        
        사용자 입력을 분석하여 필요한 작업 단계들을 계획합니다.
        """
        logging.info(f"작업 계획 생성 시작: {task}")
        
        # 특정 패턴 감지를 위한 간단한 키워드 체크
        task_lower = task.lower()
        
        # 코드 생성과 검색이 모두 언급된 경우 -> 검색 후 코드 생성 패턴
        is_search_code_pattern = (
            any(kw in task_lower for kw in ["검색", "search"]) and 
            any(kw in task_lower for kw in ["코드", "프로그램", "작성", "만들", "짜줘", "generate", "create", "code"])
        )
        
        # 코드 생성 후 실행 패턴
        is_code_execute_pattern = (
            any(kw in task_lower for kw in ["코드", "프로그램", "작성", "만들", "짜줘", "generate", "create", "code"]) and
            any(kw in task_lower for kw in ["실행", "돌려", "run", "execute"])
        )
        
        # 검색 + 코드 생성 + 실행 패턴
        is_search_code_execute_pattern = is_search_code_pattern and is_code_execute_pattern
        
        # 특정 프로그래밍 언어 감지
        languages = ["python", "파이썬", "java", "자바", "c++", "c#", "javascript", "js", "go", "rust"]
        detected_language = next((lang for lang in languages if lang in task_lower), None)
        
        # 명시적인 패턴 감지를 통한 작업 계획 바로 반환
        if is_search_code_execute_pattern:
            logging.info("명시적 패턴 감지: 검색 + 코드 생성 + 실행")
            search_query = task  # 전체 쿼리를 검색에 사용
            
            return [
                {
                    "task_type": self.TASK_SEARCH,
                    "description": f"웹에서 '{search_query}' 관련 정보 검색",
                    "parameters": {"query": search_query}
                },
                {
                    "task_type": self.TASK_CODE_GENERATION,
                    "description": f"검색 결과를 활용하여 요청된 코드 생성",
                    "parameters": {"task": task, "use_search_context": True}
                },
                {
                    "task_type": self.TASK_FILE_EXECUTION,
                    "description": "생성된 코드 파일 실행",
                    "parameters": {}  # 파일 경로는 code_generation 단계에서 context에 저장
                }
            ]
        elif is_search_code_pattern:
            logging.info("명시적 패턴 감지: 검색 + 코드 생성")
            search_query = task  # 전체 쿼리를 검색에 사용
            
            return [
                {
                    "task_type": self.TASK_SEARCH,
                    "description": f"웹에서 '{search_query}' 관련 정보 검색",
                    "parameters": {"query": search_query}
                },
                {
                    "task_type": self.TASK_CODE_GENERATION,
                    "description": f"검색 결과를 활용하여 요청된 코드 생성",
                    "parameters": {"task": task, "use_search_context": True}
                }
            ]
        elif is_code_execute_pattern:
            logging.info("명시적 패턴 감지: 코드 생성 + 실행")
            return [
                {
                    "task_type": self.TASK_CODE_GENERATION,
                    "description": f"요청된 코드 생성",
                    "parameters": {"task": task, "use_search_context": False}
                },
                {
                    "task_type": self.TASK_FILE_EXECUTION,
                    "description": "생성된 코드 파일 실행",
                    "parameters": {}  # 파일 경로는 code_generation 단계에서 context에 저장
                }
            ]
            
        # 위의 명시적 패턴들에 맞지 않는 경우 LLM을 통한 작업 계획 생성
        try:
            plan_prompt = f"""
사용자의 입력을 분석하여 어떤 작업들을 수행해야 하는지 계획을 세워주세요.
다음과 같은 작업 유형이 있습니다:
- search: 웹 검색 수행
- code_generation: 코드 생성
- file_execution: 파일 실행
- code_block_execution: 코드 블록 실행
- compilation: 코드 컴파일
- compiled_run: 컴파일된 파일 실행
- directory_exploration: 디렉토리 탐색
- file_management: 파일 관리 (생성, 삭제, 이동)

응답은 다음 형식의 JSON으로 작성해주세요:
[
    {{
        "task_type": "작업_유형",
        "description": "이 단계에서 수행할 작업에 대한 설명",
        "parameters": {{
            "param1": "값1",
            "param2": "값2",
            ...
        }}
    }},
    ...
]

각 작업 유형별로 필요한 매개변수:
- search: query (검색어)
- code_generation: task (수행할 작업 설명), use_search_context (검색 결과 활용 여부, true/false)
- file_execution: file_path (실행할 파일 경로)
- code_block_execution: code (실행할 코드), language (코드 언어)
- compilation: file_path (컴파일할 파일 경로)
- compiled_run: file_path (원본 파일 경로)
- directory_exploration: dir_path (탐색할 디렉토리 경로)
- file_management: action (생성/삭제/이동), path (대상 경로), new_path (이동 시 새 경로)

사용자 입력: {task}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that analyzes user requests and creates a plan of action. Respond only with the required JSON format, no explanations or markdown."},
                    {"role": "user", "content": plan_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            plan_json = response.choices[0].message.content.strip()
            logging.info(f"Raw plan JSON: {plan_json}")
            
            try:
                plan = json.loads(plan_json)
                
                # 응답이 배열이 아니라 객체인 경우 처리
                if isinstance(plan, dict):
                    if "tasks" in plan:
                        plan = plan["tasks"]
                    elif "task_type" in plan:
                        # 단일 작업으로 받은 경우 리스트로 변환
                        plan = [plan]
                    else:
                        # 알 수 없는 형식의 경우 기본 검색 작업으로 변환
                        plan = [{"task_type": "search", "description": "기본 웹 검색 수행", "parameters": {"query": task}}]
                
                # 계획이 문자열일 경우 (잘못된 JSON 파싱)
                if isinstance(plan, str):
                    # 기본 검색 작업으로 변환
                    plan = [{"task_type": "search", "description": "기본 웹 검색 수행", "parameters": {"query": task}}]
                
                # 배열이 아닌 경우 배열로 변환
                if not isinstance(plan, list):
                    plan = [{"task_type": "search", "description": "기본 웹 검색 수행", "parameters": {"query": task}}]
                
                # 각 작업에 필수 필드가 있는지 확인
                for i, step in enumerate(plan):
                    if not isinstance(step, dict):
                        # 작업이 딕셔너리가 아닌 경우, 적절한 형식으로 변환
                        if isinstance(step, str):
                            task_type = "search" if "검색" in step else "code_generation"
                            plan[i] = {
                                "task_type": task_type,
                                "description": step,
                                "parameters": {"query": task} if task_type == "search" else {"task": task, "use_search_context": True}
                            }
                        else:
                            # 알 수 없는 타입인 경우 검색 작업으로 대체
                            plan[i] = {"task_type": "search", "description": "기본 웹 검색 수행", "parameters": {"query": task}}
                    else:
                        # 필수 필드가 없는 경우 추가
                        if "task_type" not in step:
                            step["task_type"] = "search"
                        if "description" not in step:
                            step["description"] = f"{step['task_type']} 작업 수행"
                        if "parameters" not in step:
                            if step["task_type"] == "search":
                                step["parameters"] = {"query": task}
                            elif step["task_type"] == "code_generation":
                                step["parameters"] = {"task": task, "use_search_context": True}
                            else:
                                step["parameters"] = {}
                
                logging.info(f"생성된 작업 계획: {plan}")
                return plan
                
            except json.JSONDecodeError as e:
                logging.error(f"JSON 파싱 오류: {e}, 원본: {plan_json}")
                # JSON 파싱 오류 시 기본 작업 반환
                if is_search_code_pattern:
                    # 검색 + 코드 생성 패턴으로 추정
                    return [
                        {"task_type": "search", "description": "웹 검색 수행", "parameters": {"query": task}},
                        {"task_type": "code_generation", "description": "코드 생성 작업 수행", "parameters": {"task": task, "use_search_context": True}}
                    ]
                else:
                    return [{"task_type": "code_generation", "description": "코드 생성 작업 수행", "parameters": {"task": task, "use_search_context": False}}]
            
        except Exception as e:
            logging.error(f"작업 계획 생성 중 오류: {e}", exc_info=True)
            # 오류 발생 시 기본 작업 (코드 생성) 계획 반환
            return [{"task_type": "code_generation", "description": "코드 생성 작업 수행", "parameters": {"task": task, "use_search_context": False}}]
        
    def _format_execution_result(self, execution_result_str: str) -> str:
        """CodeExecutor 결과를 사용자 친화적 메시지로 포맷"""
        if execution_result_str.startswith("ModuleNotFoundError: "):
            missing_module = execution_result_str.split(": ")[1].strip()            
            return f"[오류] 코드를 실행하려면 '{missing_module}' 패키지가 필요합니다.\n터미널에서 'pip install {missing_module}' 명령어로 설치해주세요."
        elif execution_result_str.startswith("FileNotFoundError: Required command "):
            missing_command = re.search(r"'(.+?)'", execution_result_str).group(1)
            return f"[오류] 코드 실행에 필요한 '{missing_command}' 명령어를 찾을 수 없습니다.\n관련 언어/도구를 설치하고 PATH 환경 변수를 확인해주세요."
        else:
            return execution_result_str # 오류 없으면 그대로 반환

    def _is_fixable_code_error(self, error_message: str) -> bool:
        """실행 결과 오류 메시지가 코드 자체의 문제로 수정 가능한지 판단"""
        if not error_message:
            return False
            
        # 설치/환경 오류 키워드 제외
        if any(kw in error_message for kw in ["ModuleNotFoundError", "FileNotFoundError: Required command"]):
            return False
            
        # 일반적인 코드 오류 키워드 포함
        if any(kw in error_message for kw in ["SyntaxError", "NameError", "TypeError", "ValueError", "IndexError", "AttributeError", "KeyError", "ImportError"]):
            return True
            
        return False # 위 조건에 해당하지 않으면 수정 불가능으로 판단

    def _execute_search_step(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """검색 단계 실행"""
        query = parameters.get("query", "")
        if not query:
            return {"success": False, "result": "검색어가 제공되지 않았습니다."}
        
        search_result = self._perform_web_search_and_summarize(query)
        context["search_result"] = search_result
        return search_result

    def _execute_code_generation_step(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """코드 생성 단계 실행"""
        task = parameters.get("task", "")
        use_search_context = parameters.get("use_search_context", False)
        
        search_context = None
        if use_search_context and "search_result" in context and context["search_result"].get("success", False):
            search_context = context["search_result"].get("result")
        
        # CodeGeneratorAgent 호출
        agent_result = self.code_generator.run(task, search_context=search_context)
        
        # 결과 처리
        result_type = agent_result.get('result_type')
        if result_type == 'code_generation':
            generated_code = agent_result.get('generated_code', '코드 내용을 가져올 수 없습니다.')
            save_msg = agent_result.get('saved_path_message', '저장 경로 정보 없음')
            saved_file_path = agent_result.get('saved_file_path')
            execute_request = agent_result.get('execute_request', False)
            language_name = agent_result.get('language', 'unknown')
            required_packages = agent_result.get('required_packages', [])
            
            final_message = f"--- 생성된 코드 ({language_name}) ---\n{generated_code}\n-------------------\n{save_msg}"
            
            # 필요한 패키지 설치
            installation_successful = True
            if required_packages:
                logging.info(f"필요 패키지 감지됨: {required_packages}. 자동 설치 시도...")
                final_message += f"\n\n[알림] 다음 패키지 자동 설치 시도: {', '.join(required_packages)}"
                
                install_command_list = [sys.executable, "-m", "pip", "install"] + required_packages
                install_ret, _, install_stderr = CodeExecutor._execute_with_popen(install_command_list, timeout=120)
                
                if install_ret == 0:
                    logging.info(f"패키지 설치 성공: {', '.join(required_packages)}")
                    final_message += "\n설치 성공."
                else:
                    installation_successful = False
                    logging.error(f"패키지 설치 실패: {install_stderr}")
                    final_message += f"\n설치 실패:\n{install_stderr[:500]}..."
            
            # 자동 실행 요청이 있을 경우
            if execute_request and installation_successful and saved_file_path:
                context["pending_execution"] = {
                    "file_path": saved_file_path,
                    "type": "file"
                }
            
            return {
                "success": True,
                "result": final_message,
                "file_path": agent_result.get('saved_file_path'),
                "language": language_name,
                "execute_request": execute_request and installation_successful
            }
        else:
            error_msg = agent_result.get('result', 'LLM 처리 중 알 수 없는 오류 발생')
            return {"success": False, "result": error_msg}

    def _execute_file_execution_step(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """파일 실행 단계 (오류 발생 시 자동 수정 시도 포함)"""
        file_path = parameters.get("file_path")
        if not file_path and context.get("pending_execution") and context["pending_execution"].get("type") == "file":
            file_path = context["pending_execution"].get("file_path")
            language = context["pending_execution"].get('language') # 언어 정보도 가져옴
        else:
            # file_path가 제공된 경우 언어 분석
            if file_path:
                _, language = FileManager.analyze_file(file_path)
            else:
                return {"success": False, "result": "실행할 파일 경로가 제공되지 않았습니다."}
        
        # 첫 실행 시도
        logging.info(f"파일 실행 시도 (1차): {file_path}")
        execution_result_str = CodeExecutor.execute_file(file_path)
        formatted_result = self._format_execution_result(execution_result_str)
        
        # 실행 성공 여부 판단 (단순 오류 문자열 체크 개선)
        is_successful = not self._is_fixable_code_error(execution_result_str) # Use original error string for fixable check
        
        # 수정 시도 횟수 context 관리
        correction_attempt = context.get("correction_attempts", {}).get(file_path, 0) # 파일 경로별 시도 횟수 관리
        
        # 오류가 있고, 수정 가능하며, 아직 수정 시도 안 한 경우
        if not is_successful and self._is_fixable_code_error(execution_result_str) and correction_attempt == 0:
            logging.warning(f"코드 실행 오류 감지 ({file_path}), 자동 수정 시도...")
            # 수정 시도 기록 (파일 경로 기준)
            context.setdefault("correction_attempts", {})[file_path] = 1
            
            original_task = context.get("original_task")
            if not original_task:
                logging.error("코드 수정을 위한 원본 작업 설명을 찾을 수 없습니다.")
                return {"success": False, "result": formatted_result + "\\n(자동 수정 실패: 원본 작업 설명 없음)"}
            
            # 오류난 코드 읽기
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    previous_code = f.read()
            except Exception as e:
                logging.error(f"수정 위해 파일을 읽는 중 오류: {e}")
                return {"success": False, "result": formatted_result + f"\\n(자동 수정 실패: 파일 읽기 오류 {e})"}
            
            # CodeGeneratorAgent에게 수정 요청
            logging.info("CodeGeneratorAgent에게 코드 수정 요청 전달...")
            correction_result = self.code_generator.run(
                task=original_task, 
                previous_code=previous_code, 
                error_message=execution_result_str # 원본 오류 메시지 전달
            )
            
            if correction_result.get('result_type') == 'code_generation' and correction_result.get('saved_file_path'):
                corrected_file_path = correction_result['saved_file_path']
                # 생성된 수정 코드를 원래 파일에 덮어쓰기 (다른 파일이면 경로 변경)
                if corrected_file_path != file_path:
                    try:
                        # shutil.move 대신 shutil.copy2 사용 (메타데이터 보존)
                        # 파일을 덮어쓰기 전에 원본 파일이 존재하는지 확인하고 삭제
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        shutil.copy2(corrected_file_path, file_path)
                        # 임시 수정 파일 삭제
                        if os.path.exists(corrected_file_path):
                             os.remove(corrected_file_path)
                        logging.info(f"수정된 코드를 원본 파일에 덮어씀: {file_path}")
                        corrected_file_path = file_path # 경로 업데이트
                    except Exception as e:
                         logging.error(f"수정된 코드 덮어쓰기 실패: {e}")
                         # 실패 시 새 파일 경로 유지, 하지만 실행은 원본 경로로 시도해야 함.
                         # 이 경우, 수정이 실패했다고 보고하는 것이 나을 수 있음.
                         return {
                            "success": False,
                            "result": formatted_result + f"\\n(자동 수정 실패: 수정 코드 저장 오류 {e})"
                        }
                
                # 수정된 코드로 재실행 시도
                logging.info(f"수정된 코드 파일 실행 시도 (2차): {corrected_file_path}")
                second_execution_result_str = CodeExecutor.execute_file(corrected_file_path)
                second_formatted_result = self._format_execution_result(second_execution_result_str)
                
                second_is_successful = not self._is_fixable_code_error(second_execution_result_str)
                
                final_result_message = f"초기 실행 오류:\n{formatted_result}\\n\\n자동 수정 후 재실행 결과:\n{second_formatted_result}"
                
                return {
                    "success": second_is_successful,
                    "result": final_result_message
                }
            else:
                # 수정 코드 생성 실패
                logging.error("코드 수정 실패: CodeGeneratorAgent가 코드를 반환하지 않음")
                return {
                    "success": False,
                    "result": formatted_result + "\\n(자동 수정 실패: 수정된 코드를 생성하지 못함)"
                }
        else:
            # 첫 실행 성공 또는 수정 불가능한 오류 또는 이미 수정 시도함
            return {
                "success": is_successful,
                "result": formatted_result
            }

    def _execute_code_block_execution_step(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """코드 블록 실행 단계"""
        code = parameters.get("code", "")
        language = parameters.get("language", "python")
        
        if not code:
            return {"success": False, "result": "실행할 코드가 제공되지 않았습니다."}
        
        execution_result_str = CodeExecutor.execute_code(code, language)
        formatted_result = self._format_execution_result(execution_result_str)
        
        return {
            "success": "오류" not in formatted_result.lower(),
            "result": formatted_result
        }

    def _execute_compilation_step(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """컴파일 단계"""
        file_path = parameters.get("file_path", "")
        if not file_path:
            return {"success": False, "result": "컴파일할 파일 경로가 제공되지 않았습니다."}
        
        _, language = FileManager.analyze_file(file_path)
        
        if language not in ['c++', 'c', 'rust', 'c#']:
            return {"success": False, "result": f"컴파일이 필요하지 않은 언어입니다: {language}"}
        
        temp_dir = CodeExecutor.get_temp_dir()
        output_file = os.path.join(temp_dir, 'temp_out')
        if language == 'c#': 
            output_file += ".exe"
        
        cmd = list(CodeExecutor.COMMAND_MAP[language])
        compile_cmd = list(cmd)  # 명령어 리스트 복사
        compile_cmd.append(file_path)
        
        if '{output}' in cmd:
            idx = cmd.index('{output}')
            if language == 'c#': 
                compile_cmd[idx] = f'/out:{output_file}'
            else: 
                compile_cmd[idx] = output_file
        
        compile_ret, _, compile_stderr = CodeExecutor._execute_with_popen(compile_cmd, timeout=30)
        
        if compile_ret == 0:
            logging.info("컴파일 성공")
            context["compiled_file"] = {
                "original_path": file_path,
                "output_path": output_file,
                "language": language
            }
            return {"success": True, "result": "컴파일이 완료되었습니다."}
        else:
            logging.error(f"컴파일 실패: {compile_stderr}")
            return {"success": False, "result": f"컴파일 중 오류 발생:\n{compile_stderr}"}

    def _execute_compiled_run_step(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """컴파일된 파일 실행 단계"""
        file_path = parameters.get("file_path", "")
        output_file = None
        language = None
        
        # 이전 컴파일 단계에서의 정보 사용
        if "compiled_file" in context:
            compiled_info = context["compiled_file"]
            if not file_path or file_path == compiled_info.get("original_path"):
                output_file = compiled_info.get("output_path")
                language = compiled_info.get("language")
        
        # 컴파일 정보가 없는 경우 파일 분석
        if not output_file:
            if not file_path:
                return {"success": False, "result": "실행할 컴파일된 파일의 원본 경로가 제공되지 않았습니다."}
                
        _, language = FileManager.analyze_file(file_path)
        temp_dir = CodeExecutor.get_temp_dir()
        output_file = os.path.join(temp_dir, 'temp_out')
        if language == 'c#':
            output_file += ".exe"
        
        if not os.path.exists(output_file):
            return {"success": False, "result": f"{file_path}에 대한 컴파일된 실행 파일({output_file})이 없습니다. 먼저 컴파일해주세요."}
        
        cmd_to_run = [output_file]
        if language == 'c#' and os.name != 'nt':
            cmd_to_run.insert(0, 'mono')
        
        run_ret, run_stdout, run_stderr = CodeExecutor._execute_with_popen(cmd_to_run, timeout=10)
        
        # 오류가 있으면 포맷팅하여 반환
        if run_stderr:
            formatted_result = self._format_execution_result(run_stderr)
            if formatted_result != run_stderr:  # 특별히 포맷팅된 오류
                return {"success": False, "result": formatted_result}
        
        # 정상 실행 또는 일반 오류
        output_text = ""
        if run_stdout:
            output_text += f"실행 결과:\n{run_stdout}"
        if run_stderr and not output_text:
            output_text += f"실행 중 오류:\n{run_stderr}"
            return {"success": False, "result": output_text}
        elif run_stderr:
            output_text += f"\nStandard Error:\n{run_stderr}"
        
        return {"success": True, "result": output_text if output_text else "실행 결과가 없습니다."}

    def _execute_directory_exploration_step(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """디렉토리 탐색 단계"""
        dir_path = parameters.get("dir_path", "")
        if not dir_path:
            return {"success": False, "result": "탐색할 디렉토리 경로가 제공되지 않았습니다."}
        
        explore_result = FileManager.explore_directory(dir_path)
        
        if explore_result:
            message = f"디렉토리: {explore_result['path']}\n"
            message += f"총 크기: {explore_result['total_size']:,} bytes\n"
            message += f"항목 수: {len(explore_result['items'])}\n\n"
            message += "파일 목록:\n"
            for item in explore_result['items']:
                if item['type'] == 'file':
                    message += f"📄 {item['name']} ({item['size']:,} bytes) - {item['file_type']} [{item['language']}]\n"
                else:
                    message += f"📁 {item['name']} ({item['size']:,} bytes)\n"
            return {"success": True, "result": message}
        else:
            return {"success": False, "result": "디렉토리 탐색 중 오류가 발생했습니다."}

    def _execute_file_management_step(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """파일 관리 단계"""
        action = parameters.get("action", "")
        path = parameters.get("path", "")
        new_path = parameters.get("new_path", None)
        
        if not action or not path:
            return {"success": False, "result": "파일 관리 작업에 필요한 파라미터가 부족합니다."}
        
        if action == "이동" and not new_path:
            return {"success": False, "result": "파일 이동 작업에는 새 경로가 필요합니다."}
        
        action_map = {'생성': 'create', '삭제': 'delete', '이동': 'move'}
        if action in action_map:
            action = action_map[action]
        
        result = FileManager.manage_files(action, path, new_path)
        return {"success": "완료" in result, "result": result}

    def run_task(self, task: str) -> str:
        """주어진 작업을 계획하고 실행"""
        logging.info(f"작업 시작: {task}")
        
        # 1. 작업 계획 생성
        plan = self._plan_task(task)
        
        # 2. 각 단계별 실행 및 결과 수집
        step_results = []
        context = {"original_task": task}
        
        for step in plan:
            task_type = step.get("task_type", "")
            parameters = step.get("parameters", {})
            description = step.get("description", f"{task_type} 작업")
            
            logging.info(f"단계 실행: {description} (유형: {task_type})")
            
            try:
                # 작업 유형에 따라 적절한 실행 함수 호출
                if task_type == self.TASK_SEARCH:
                    result = self._execute_search_step(parameters, context)
                elif task_type == self.TASK_CODE_GENERATION:
                    result = self._execute_code_generation_step(parameters, context)
                elif task_type == self.TASK_FILE_EXECUTION:
                    result = self._execute_file_execution_step(parameters, context)
                elif task_type == self.TASK_CODE_BLOCK_EXECUTION:
                    result = self._execute_code_block_execution_step(parameters, context)
                elif task_type == self.TASK_COMPILATION:
                    result = self._execute_compilation_step(parameters, context)
                elif task_type == self.TASK_COMPILED_RUN:
                    result = self._execute_compiled_run_step(parameters, context)
                elif task_type == self.TASK_DIRECTORY_EXPLORATION:
                    result = self._execute_directory_exploration_step(parameters, context)
                elif task_type == self.TASK_FILE_MANAGEMENT:
                    result = self._execute_file_management_step(parameters, context)
                else:
                    result = {"success": False, "result": f"알 수 없는 작업 유형: {task_type}"}
                
                step_results.append({
                    "task_type": task_type,
                    "description": description,
                    "result": result
                })
                
                # 실패한 경우 로깅 (실패해도 다음 단계 진행)
                if not result.get("success", False):
                    logging.warning(f"단계 실행 실패: {description} - 결과: {result.get('result')}")
                    # 특정 작업 유형의 실패는 이후 단계의 실행을 막을 수 있음 (예: 컴파일 실패)
                    if task_type in [self.TASK_COMPILATION]:
                        break
                
            except Exception as e:
                logging.error(f"단계 {description} 실행 중 오류: {e}", exc_info=True)
                step_results.append({
                    "task_type": task_type,
                    "description": description,
                    "result": {"success": False, "result": f"실행 중 예외 발생: {str(e)}"}
                })
                # 오류 발생 시 다음 단계 진행
        
        # 3. 결과 조합
        final_result_message = self._combine_step_results(step_results)
        
        # 4. 메모리에 저장
        self.memory.append({
            "task": task,
            "plan": plan,
            "results": step_results,
            "final_result": final_result_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # 5. 메모리 관리
        self._manage_memory()
        
        logging.info("작업 완료")
        return final_result_message
    
    def _combine_step_results(self, step_results: List[Dict[str, Any]]) -> str:
        """여러 단계의 결과를 하나의 메시지로 조합"""
        if not step_results:
            return "작업을 수행할 수 없습니다."
        
        # 한 단계만 있는 경우, 해당 결과를 그대로 반환
        if len(step_results) == 1:
            return step_results[0]["result"].get("result", "알 수 없는 결과")
        
        # 여러 단계가 있는 경우, 각 단계의 결과를 조합
        combined_message = ""
        
        for i, step in enumerate(step_results, 1):
            task_type = step["task_type"]
            description = step["description"]
            result_data = step["result"]
            result_text = result_data.get("result", "결과 없음")
            success = result_data.get("success", False)
            
            # 실패한 단계는 중요하게 표시
            if not success:
                combined_message += f"\n== 단계 {i}: {description} (실패) ==\n{result_text}\n"
            else:
                combined_message += f"\n== 단계 {i}: {description} ==\n{result_text}\n"
        
        return combined_message.strip()

if __name__ == "__main__":
    # 에이전트 생성
    agent = AgentAI(
        name="코더",
        description="검색과 코드 생성을 도와주는 AI 에이전트입니다.",
        memory_limit=5
    )
    
    # 대화형 모드로 실행
    agent.run_interactive() 