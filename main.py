from typing import List, Dict, Any, Callable, Tuple, Union
import os
import re
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import sys
import shutil

# Local imports
from file_manager import FileManager
from code_executor import CodeExecutor
from code_generator import CodeGeneratorAgent
from utils import is_fixable_code_error, format_execution_result # Import helpers from utils
from web_handler import WebHandler # Import WebHandler
from task_planner import TaskPlanner # Import TaskPlanner
from result_formatter import ResultFormatter # Import ResultFormatter

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
        self.web_handler = WebHandler(client=self.client)
        self.task_planner = TaskPlanner(client=self.client)
        self.result_formatter = ResultFormatter()
        
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
    
    def _execute_search_step(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """검색 단계 실행 - WebHandler 사용"""
        query = parameters.get("query", "")
        if not query:
            return {"success": False, "result": "검색어가 제공되지 않았습니다."}
        
        search_result = self.web_handler.perform_web_search_and_summarize(query) # Delegate to WebHandler
        context["search_result"] = search_result
        return search_result

    def _execute_code_generation_step(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """코드 생성 단계 실행 - CodeGeneratorAgent 사용"""
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
                # Explicitly set this in the context for the next step
                context["pending_execution"] = {
                    "file_path": saved_file_path,
                    "type": "file",
                    "language": language_name
                }
                logging.info(f"Pending execution set for: {saved_file_path}")
                
                # Now check if the next task in plan is a file execution
                # If it's not, we need to add an execution step
                plan = context.get("plan", [])
                current_step_index = context.get("current_step_index", 0)
                
                if current_step_index < len(plan) - 1:
                    next_step = plan[current_step_index + 1]
                    if next_step.get("task_type") != self.TASK_FILE_EXECUTION:
                        logging.info("Auto-adding file execution step for generated code")
                        # We can't modify the plan directly, but we'll set a flag to handle the execution
                        context["execute_after_generation"] = True
            
            return {
                "success": True,
                "result": final_message,
                "file_path": saved_file_path,
                "language": language_name,
                "execute_request": execute_request and installation_successful
            }
        else:
            error_msg = agent_result.get('result', 'LLM 처리 중 알 수 없는 오류 발생')
            return {"success": False, "result": error_msg}

    def _execute_file_execution_step(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """파일 실행 단계 (오류 발생 시 자동 수정 시도 포함) - utils 함수 사용"""
        file_path = parameters.get("file_path")
        language = None # Initialize language

        # Check pending execution context first
        if not file_path and context.get("pending_execution") and context["pending_execution"].get("type") == "file":
            pending_info = context["pending_execution"]
            file_path = pending_info.get("file_path")
            language = pending_info.get('language') # Get language from context
            logging.info(f"Using pending execution context with file: {file_path}, language: {language}")
            # Don't clear pending execution yet - it might be needed for auto-execution at end of plan
        elif file_path:
            # Analyze language if file_path is provided directly
            _, language = FileManager.analyze_file(file_path)
            logging.info(f"Using direct file path: {file_path}, detected language: {language}")
        else:
            logging.error("No file path provided and no pending execution found")
            return {"success": False, "result": "실행할 파일 경로가 제공되지 않았습니다."}

        # Ensure file_path is valid before proceeding
        if not file_path or not os.path.exists(file_path):
             logging.error(f"File not found: {file_path}")
             return {"success": False, "result": f"실행할 파일을 찾을 수 없습니다: {file_path}"}

        # 첫 실행 시도
        logging.info(f"파일 실행 시도 (1차): {file_path}")
        execution_result_str = CodeExecutor.execute_file(file_path)
        formatted_result = format_execution_result(execution_result_str) # Use helper from utils

        # 실행 성공 여부 판단
        is_successful = not is_fixable_code_error(execution_result_str) # Use helper from utils
        
        # Set execution performed flag
        context["execution_performed"] = True

        # 수정 시도 횟수 context 관리
        correction_attempt = context.get("correction_attempts", {}).get(file_path, 0)

        # 오류가 있고, 수정 가능하며, 아직 수정 시도 안 한 경우
        if not is_successful and is_fixable_code_error(execution_result_str) and correction_attempt == 0:
            logging.warning(f"코드 실행 오류 감지 ({file_path}), 자동 수정 시도...")
            context.setdefault("correction_attempts", {})[file_path] = 1

            original_task = context.get("original_task")
            if not original_task:
                logging.error("코드 수정을 위한 원본 작업 설명을 찾을 수 없습니다.")
                return {"success": False, "result": formatted_result + "\n(자동 수정 실패: 원본 작업 설명 없음)"}

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    previous_code = f.read()
            except Exception as e:
                logging.error(f"수정 위해 파일을 읽는 중 오류: {e}")
                return {"success": False, "result": formatted_result + f"\n(자동 수정 실패: 파일 읽기 오류 {e})"}

            logging.info("CodeGeneratorAgent에게 코드 수정 요청 전달...")
            correction_result = self.code_generator.run(
                task=original_task,
                previous_code=previous_code,
                error_message=execution_result_str
            )

            if correction_result.get('result_type') == 'code_generation' and correction_result.get('saved_file_path'):
                corrected_file_path = correction_result['saved_file_path']
                corrected_language = correction_result.get('language', language) # Update language if provided

                # 수정된 코드를 원래 파일에 덮어쓰기
                try:
                    # Ensure the corrected file exists before attempting to move/copy
                    if not os.path.exists(corrected_file_path):
                         raise FileNotFoundError(f"Generated correction file not found: {corrected_file_path}")

                    # Remove original before moving to avoid potential issues on some systems
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    shutil.move(corrected_file_path, file_path) # Move corrected code to original path
                    logging.info(f"수정된 코드를 원본 파일 위치로 이동: {file_path}")
                    # Update context if file path changed (though it shouldn't with move)
                    # If pending_execution still exists and points to the old path, update it? Or rely on the next execution using file_path directly.

                except Exception as e:
                     logging.error(f"수정된 코드 덮어쓰기/이동 실패: {e}")
                     return {
                        "success": False,
                        "result": formatted_result + f"\n(자동 수정 실패: 수정 코드 저장 오류 {e})"
                    }

                # 수정된 코드로 재실행 시도
                logging.info(f"수정된 코드 파일 실행 시도 (2차): {file_path}")
                second_execution_result_str = CodeExecutor.execute_file(file_path)
                second_formatted_result = format_execution_result(second_execution_result_str) # Use helper

                second_is_successful = not is_fixable_code_error(second_execution_result_str) # Use helper

                final_result_message = f"초기 실행 오류:\n{formatted_result}\n\n자동 수정 후 재실행 결과:\n{second_formatted_result}"

                return {
                    "success": second_is_successful,
                    "result": final_result_message
                }
            else:
                logging.error("코드 수정 실패: CodeGeneratorAgent가 코드를 반환하지 않음")
                return {
                    "success": False,
                    "result": formatted_result + "\n(자동 수정 실패: 수정된 코드를 생성하지 못함)"
                }
        else:
            return {
                "success": is_successful,
                "result": formatted_result
            }

    def _execute_code_block_execution_step(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """코드 블록 실행 단계 - utils 함수 사용"""
        code = parameters.get("code", "")
        language = parameters.get("language", "python")
        
        if not code:
            return {"success": False, "result": "실행할 코드가 제공되지 않았습니다."}
        
        execution_result_str = CodeExecutor.execute_code(code, language)
        formatted_result = format_execution_result(execution_result_str) # Use helper from utils
        
        # Check if the formatted result indicates an error more robustly
        is_successful = not formatted_result.startswith("[오류]")
        
        return {
            "success": is_successful,
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
        # Ensure temp_dir exists
        os.makedirs(temp_dir, exist_ok=True)

        # Use filename without extension for output base
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(temp_dir, base_name)
        if os.name == 'nt': # Add .exe on Windows
             output_file += ".exe"

        # Handle different compilation commands and output flags
        cmd_base = list(CodeExecutor.COMMAND_MAP.get(language, []))
        if not cmd_base:
            return {"success": False, "result": f"컴파일 명령어를 찾을 수 없습니다: {language}"}

        compile_cmd = list(cmd_base) # Copy base command

        # Replace placeholders correctly
        try:
            if '{input}' in compile_cmd:
                 compile_cmd[compile_cmd.index('{input}')] = file_path
            else:
                 compile_cmd.append(file_path) # Default append input if no placeholder

            if '{output}' in compile_cmd:
                 idx = compile_cmd.index('{output}')
                 # Handle C# specific output flag
                 if language == 'c#':
                     compile_cmd[idx] = f'/out:{output_file}'
                 else:
                     compile_cmd[idx] = output_file
            # If no output placeholder, C/C++/Rust compilers often use -o
            elif language in ['c', 'c++', 'rust'] and '-o' not in compile_cmd:
                 compile_cmd.extend(['-o', output_file])
            # Add default output logic if needed, otherwise assume compiler handles it

        except ValueError:
             logging.error(f"Error processing compile command template for {language}")
             return {"success": False, "result": "컴파일 명령어 설정 오류"}

        logging.info(f"Executing compile command: {' '.join(compile_cmd)}")
        compile_ret, _, compile_stderr = CodeExecutor._execute_with_popen(compile_cmd, timeout=60) # Increased timeout

        if compile_ret == 0:
            logging.info(f"컴파일 성공: {output_file}")
            context["compiled_file"] = {
                "original_path": file_path,
                "output_path": output_file,
                "language": language
            }
            return {"success": True, "result": f"컴파일이 완료되었습니다. 실행 파일: {output_file}"}
        else:
            logging.error(f"컴파일 실패: {compile_stderr}")
            # Use format_execution_result for compiler errors too?
            formatted_error = format_execution_result(compile_stderr)
            return {"success": False, "result": f"컴파일 중 오류 발생:\n{formatted_error}"}

    def _execute_compiled_run_step(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """컴파일된 파일 실행 단계 - utils 함수 사용"""
        file_path = parameters.get("file_path", "") # This is the *original* source file path
        output_file = None
        language = None

        # Try to get info from the previous compilation step
        if "compiled_file" in context:
            compiled_info = context["compiled_file"]
            # Check if the provided file_path matches the one that was compiled
            if not file_path or file_path == compiled_info.get("original_path"):
                output_file = compiled_info.get("output_path")
                language = compiled_info.get("language")
            else:
                # file_path provided, but doesn't match context - use the provided one
                logging.warning(f"Provided file path '{file_path}' differs from compiled context '{compiled_info.get('original_path')}'. Attempting execution based on provided path.")
                # Re-determine output path based on provided file_path
                _, language = FileManager.analyze_file(file_path)
                temp_dir = CodeExecutor.get_temp_dir()
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_file = os.path.join(temp_dir, base_name)
                if os.name == 'nt': output_file += ".exe"

        elif file_path:
            # No compile context, determine output path from file_path parameter
            logging.info(f"No compile context found, determining executable path for: {file_path}")
            _, language = FileManager.analyze_file(file_path)
            temp_dir = CodeExecutor.get_temp_dir()
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file = os.path.join(temp_dir, base_name)
            if os.name == 'nt': output_file += ".exe"
        else:
            return {"success": False, "result": "실행할 컴파일된 파일의 원본 경로가 제공되지 않았습니다."}

        # Check if the determined output file exists
        if not output_file or not os.path.exists(output_file):
            return {"success": False, "result": f"컴파일된 실행 파일({output_file or 'N/A'})이 없습니다. 먼저 컴파일해주세요."}

        # Determine how to run the executable
        cmd_to_run = [output_file]
        if language == 'c#' and os.name != 'nt': # Need mono for C# on non-Windows
            # Check if mono exists
            mono_path = shutil.which('mono')
            if not mono_path:
                 return {"success": False, "result": "C# 실행을 위해 'mono'를 찾을 수 없습니다. 설치해주세요."}
            cmd_to_run.insert(0, mono_path)
        elif language == 'java': # Java needs 'java' command
             # Compiled Java usually means .class files, direct execution needs setup.
             # Assuming the compilation step produced an executable JAR or similar is complex.
             # Let's stick to C/C++/Rust/C# for compiled execution for now.
             return {"success": False, "result": "컴파일된 Java 파일 실행은 현재 지원되지 않습니다."}

        logging.info(f"Executing compiled file: {' '.join(cmd_to_run)}")
        run_ret, run_stdout, run_stderr = CodeExecutor._execute_with_popen(cmd_to_run, timeout=30) # Increased timeout

        # Format output/error
        output_text = ""
        if run_stdout:
            output_text += f"실행 결과:\n{run_stdout.strip()}"

        formatted_error = ""
        if run_stderr:
            formatted_error = format_execution_result(run_stderr.strip()) # Use helper
            # Append stderr only if it's significant or no stdout
            if formatted_error.startswith("[오류]") or not run_stdout:
                 if output_text: output_text += "\n\n"
                 output_text += f"실행 중 오류:\n{formatted_error}"
            elif run_stdout: # Append non-critical stderr if there was also stdout
                 output_text += f"\n\nStandard Error:\n{run_stderr.strip()}"

        final_success = not formatted_error.startswith("[오류]")

        return {"success": final_success, "result": output_text if output_text else "실행 결과가 없습니다."}

    def _execute_directory_exploration_step(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """디렉토리 탐색 단계 - FileManager 사용"""
        dir_path = parameters.get("dir_path", ".") # Default to current directory

        explore_result = FileManager.explore_directory(dir_path)

        if explore_result and explore_result.get('success'):
            data = explore_result['data']
            message = f"디렉토리: {data['path']}\n"
            message += f"총 크기: {data['total_size']:,} bytes\n"
            message += f"항목 수: {len(data['items'])}\n\n"
            message += "파일 및 디렉토리 목록:\n"
            for item in data['items']:
                item_type_icon = "📁" if item['type'] == 'directory' else "📄"
                size_str = f"({item['size']:,} bytes)"
                extra_info = f" - {item.get('file_type', '')} [{item.get('language', 'N/A')}]" if item['type'] == 'file' else ""
                message += f"{item_type_icon} {item['name']} {size_str}{extra_info}\n"
            return {"success": True, "result": message.strip()}
        else:
            error_msg = explore_result.get('message', '디렉토리 탐색 중 알 수 없는 오류가 발생했습니다.')
            logging.error(f"디렉토리 탐색 실패: {dir_path} - {error_msg}")
            return {"success": False, "result": f"디렉토리 탐색 오류: {error_msg}"}

    def _execute_file_management_step(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """파일 관리 단계 - FileManager 사용"""
        action = parameters.get("action", "")
        path = parameters.get("path", "")
        new_path = parameters.get("new_path", None)
        content = parameters.get("content", None) # Added for create

        if not action or not path:
            return {"success": False, "result": "파일 관리 작업에 필요한 파라미터가 부족합니다 (action, path)."}

        if action.lower() == "move" and not new_path:
            return {"success": False, "result": "파일 이동 작업에는 새 경로(new_path)가 필요합니다."}
        if action.lower() == "create" and content is None:
             # Allow creating empty files/dirs, but log it
             logging.info(f"파일/디렉토리 생성 요청: {path} (내용 없음)")
             # FileManager.manage_files handles content=None

        # Normalize action name if needed (e.g., Korean to English)
        action_map = {'생성': 'create', '삭제': 'delete', '이동': 'move', '읽기': 'read', '쓰기': 'write'}
        action_lower = action.lower()
        if action_lower in action_map:
            action = action_map[action_lower]
        elif action in ['create', 'delete', 'move', 'read', 'write']:
             action = action # Already in correct format
        else:
            return {"success": False, "result": f"알 수 없는 파일 관리 작업: {action}"}

        # Call FileManager
        result_dict = FileManager.manage_files(action, path, new_path=new_path, content=content)

        return {"success": result_dict.get('success', False), "result": result_dict.get('message', '알 수 없는 결과')}

    def run_task(self, task: str) -> str:
        """주어진 작업을 계획하고 실행 - TaskPlanner 및 ResultFormatter 사용"""
        logging.info(f"작업 시작: {task}")
        
        # 1. 작업 계획 생성 - Delegate to TaskPlanner
        plan = self.task_planner.plan_task(task)
        
        # 2. 각 단계별 실행 및 결과 수집
        step_results = []
        context = {
            "original_task": task,
            "correction_attempts": {}, # Initialize correction attempts context
            "plan": plan  # Add the plan to the context for reference
        }
        
        i = 0
        while i < len(plan):
            step = plan[i]
            task_type = step.get("task_type", "")
            parameters = step.get("parameters", {})
            description = step.get("description", f"{task_type} 작업")
            
            # Store current step index in context
            context["current_step_index"] = i
            
            logging.info(f"단계 실행: {description} (유형: {task_type})")
            step_result_data = {"success": False, "result": "알 수 없는 오류"} # Default result

            try:
                # 작업 유형에 따라 적절한 실행 함수 호출
                if task_type == self.TASK_SEARCH:
                    step_result_data = self._execute_search_step(parameters, context)
                elif task_type == self.TASK_CODE_GENERATION:
                    step_result_data = self._execute_code_generation_step(parameters, context)
                    
                    # Check if we need to execute the generated code even if not in the plan
                    if context.get("execute_after_generation") and step_result_data.get("success"):
                        logging.info("Auto-executing generated code (not in original plan)")
                        exec_result = self._execute_file_execution_step({}, context)
                        # Add this result to the step results as a synthetic step
                        auto_exec_step = {
                            "task_type": self.TASK_FILE_EXECUTION, 
                            "description": "생성된 코드 자동 실행",
                            "result": exec_result
                        }
                        step_results.append(auto_exec_step)
                        # Remove the flag to avoid duplicate execution
                        context.pop("execute_after_generation", None)

                elif task_type == self.TASK_FILE_EXECUTION:
                    step_result_data = self._execute_file_execution_step(parameters, context)
                elif task_type == self.TASK_CODE_BLOCK_EXECUTION:
                    step_result_data = self._execute_code_block_execution_step(parameters, context)
                elif task_type == self.TASK_COMPILATION:
                    step_result_data = self._execute_compilation_step(parameters, context)
                    # If compilation fails, stop the plan execution for compile/run sequences
                    if not step_result_data.get("success", False):
                         logging.warning(f"컴파일 단계 실패 ({description}), 후속 실행 단계 중단.")
                         step_results.append({
                            "task_type": task_type,
                            "description": description,
                            "result": step_result_data
                         })
                         break # Stop processing further steps
                elif task_type == self.TASK_COMPILED_RUN:
                    step_result_data = self._execute_compiled_run_step(parameters, context)
                elif task_type == self.TASK_DIRECTORY_EXPLORATION:
                    step_result_data = self._execute_directory_exploration_step(parameters, context)
                elif task_type == self.TASK_FILE_MANAGEMENT:
                    step_result_data = self._execute_file_management_step(parameters, context)
                else:
                    step_result_data = {"success": False, "result": f"알 수 없는 작업 유형: {task_type}"}
                
                step_results.append({
                    "task_type": task_type,
                    "description": description,
                    "result": step_result_data
                })

                # 실패한 경우 로깅 (실패해도 다음 단계 진행, 컴파일 제외)
                if not step_result_data.get("success", False):
                    logging.warning(f"단계 실행 실패: {description} - 결과: {step_result_data.get('result')}")

            except Exception as e:
                logging.error(f"단계 {description} 실행 중 오류: {e}", exc_info=True)
                step_result_data = {"success": False, "result": f"실행 중 예외 발생: {str(e)}"}
                step_results.append({
                    "task_type": task_type,
                    "description": description,
                    "result": step_result_data
                })
                # Decide if we should break on general exceptions? Maybe not.
            
            # Move to the next step
            i += 1
            
            # If we have a pending execution that wasn't part of the plan,
            # check if we should execute it now
            if i == len(plan) and context.get("pending_execution") and not context.get("execution_performed"):
                logging.info("Detected pending execution at the end of plan, performing execution")
                exec_result = self._execute_file_execution_step({}, context)
                step_results.append({
                    "task_type": self.TASK_FILE_EXECUTION, 
                    "description": "추가 파일 실행 단계",
                    "result": exec_result
                })
                context["execution_performed"] = True

        # 3. 결과 조합 - Delegate to ResultFormatter
        final_result_message = self.result_formatter.combine_step_results(step_results)

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

if __name__ == "__main__":
    # 에이전트 생성
    agent = AgentAI(
        name="코더",
        description="검색과 코드 생성을 도와주는 AI 에이전트입니다.",
        memory_limit=5
    )
    
    # 대화형 모드로 실행
    agent.run_interactive() 