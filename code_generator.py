from typing import List, Callable, Any, Dict, Tuple
from functools import wraps
import re
import os
import logging
from datetime import datetime
from openai import OpenAI
from file_manager import FileManager

# tool 데코레이터는 ToolCallingAgent에서 직접 사용하지 않으므로 주석 처리 또는 삭제 가능
# def tool(func: Callable) -> Callable:
#     """데코레이터: 함수를 도구로 등록"""
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         return func(*args, **kwargs)
#     wrapper.is_tool = True
#     return wrapper

class CodeGeneratorAgent:
    def __init__(self, client: OpenAI):
        """초기화 함수
        
        Args:
            client (OpenAI): OpenAI API 클라이언트
        """
        self.client = client
        self.output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 언어 감지 키워드 (소문자)
        self.language_keywords = {
            "python": ["python", "파이썬"],
            "javascript": ["javascript", "js", "자바스크립트"],
            "java": ["java", "자바"],
            "c++": ["c++", "cpp", "씨쁠쁠"],
            "c": ["c", "씨", "씨언어"],
            "c#": ["c#", "cs", "csharp", "씨샵"],
            "go": ["go", "golang", "고"],
            "rust": ["rust", "러스트"],
            "ruby": ["ruby", "루비"],
            "php": ["php"],
            "typescript": ["typescript", "ts", "타입스크립트"],
            "swift": ["swift", "스위프트"],
            "kotlin": ["kotlin", "코틀린"],
            "r": ["r", "알"]
            # 필요에 따라 언어 추가
        }
        # 코드 생성 감지 키워드
        self.codegen_keywords = ["코드", "프로그램", "작성", "만들", "짜줘", "generate", "create", "write", "code", "program"]
        # 코드 실행 감지 키워드
        self.execution_keywords = ["실행", "돌려", "run", "execute"]

        # 언어 이름 -> 확장자 매핑 (FileManager 활용)
        # FileManager.LANGUAGE_MAP의 key(확장자)와 value(언어이름)를 뒤집음
        self.lang_to_ext = {v: k for k, v in FileManager.LANGUAGE_MAP.items()}
        # FileManager에 없는 언어 수동 추가 (예시)
        self.lang_to_ext.setdefault('javascript', '.js') 
        self.lang_to_ext.setdefault('typescript', '.ts')
        self.lang_to_ext.setdefault('swift', '.swift')
        self.lang_to_ext.setdefault('kotlin', '.kt')
        # C# 확장자 추가 (.cs는 FileManager에 이미 있을 수 있지만 확인)
        self.lang_to_ext.setdefault('c#', '.cs')

        # pip 설치가 필요할 수 있는 일반적인 패키지 목록 (표준 라이브러리 제외)
        self.common_pip_packages = {
            'requests', 'numpy', 'pandas', 'matplotlib', 'scipy', 'pygame', 
            'beautifulsoup4', 'bs4', 'selenium', 'pillow', 'PIL', 'flask', 
            'django', 'sqlalchemy', 'fastapi', 'tensorflow', 'keras', 
            'torch', 'scikit-learn', 'sklearn' 
            # tkinter는 시스템 설치 필요하므로 제외
        }

    def _detect_language_and_request(self, task: str) -> Tuple[str | None, bool, bool]:
        """작업 문자열에서 언어, 코드 생성, 실행 요청 여부 감지"""
        task_lower = task.lower()
        detected_language = None
        is_codegen_request = any(keyword in task_lower for keyword in self.codegen_keywords)
        is_execution_request = any(keyword in task_lower for keyword in self.execution_keywords)
        
        # 코드 생성 요청이 없으면 언어 감지 불필요
        if not is_codegen_request:
            return None, False, False

        for lang_name, keywords in self.language_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                detected_language = lang_name
                break 
                
        # 언어가 특정되지 않았지만 코드 생성 요청은 있을 수 있음 (기본 Python으로 처리할 수도 있음)
        # 여기서는 언어가 명시된 경우만 코드 생성 처리
        if detected_language is None:
             is_codegen_request = False # 언어 없으면 코드 생성 불가 처리

        return detected_language, is_codegen_request, is_execution_request
        
    def _clean_llm_code_output(self, code_content: str, language: str) -> str:
        """LLM 응답에서 코드 블록 마크다운 제거"""
        # 정규표현식으로 ```language ... ``` 또는 ``` ... ``` 제거 시도
        pattern = rf"^```(?:{language}|\w*)?\s*\n(.*?)\n```$"
        match = re.match(pattern, code_content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # 간단히 앞뒤 ``` 제거 시도 (마크다운이 불완전할 경우 대비)
        if code_content.startswith("```") and code_content.endswith("```"):
             lines = code_content.splitlines()
             if len(lines) > 1:
                 # 첫 줄(```language)과 마지막 줄(```) 제거
                 return "\n".join(lines[1:-1]).strip()
        return code_content # 제거할 패턴 없으면 원본 반환

    def _find_python_imports(self, code: str) -> List[str]:
        """Python 코드에서 import 문을 찾아 모듈 이름 리스트 반환 (간단한 정규식 사용)"""
        # import module / from module import ... / import module as alias
        import_pattern = r'^\s*(?:import|from)\s+([\w\.]+)'
        matches = re.findall(import_pattern, code, re.MULTILINE)
        # 점(.)으로 시작하는 상대 임포트 등은 제외하고, 첫 부분만 추출
        modules = {m.split('.')[0] for m in matches if m and not m.startswith('.')}
        return list(modules)

    def _check_required_packages(self, modules: List[str]) -> List[str]:
        """임포트된 모듈 중 설치가 필요할 수 있는 패키지 목록 반환"""
        required = []
        for module in modules:
            # bs4는 beautifulsoup4로 설치해야 함
            package_name = 'beautifulsoup4' if module == 'bs4' else module
            # scikit-learn은 sklearn으로 설치해야 함
            package_name = 'scikit-learn' if module == 'sklearn' else package_name
            # PIL은 Pillow로 설치해야 함
            package_name = 'Pillow' if module == 'PIL' else package_name
            
            if package_name in self.common_pip_packages:
                required.append(package_name)
        return sorted(list(set(required))) # 중복 제거 및 정렬

    def _is_refusal_message(self, text: str) -> bool:
        """Checks if the text likely contains a refusal message from the LLM."""
        text_lower = text.lower()
        refusal_keywords_ko = ["죄송합니다", "할 수 없습니다", "안전하지 않은", "악의적인", "해로운", "대신"]
        refusal_keywords_en = ["sorry", "cannot generate", "unable to", "unsafe", "malicious", "harmful", "destructive", "instead", "as an ai", "i cannot"]

        # Check for keywords
        if any(keyword in text_lower for keyword in refusal_keywords_ko + refusal_keywords_en):
            return True

        # Check for phrases indicating refusal or ethical concerns
        refusal_phrases = [
            "i cannot fulfill this request",
            "i cannot create code that",
            "my purpose is to help",
            "potentially harmful",
            "violates safety policies",
            "요청을 수행할 수 없습니다",
            "안전 정책에 위배",
            "해로운 코드는 생성할 수 없습니다",
        ]
        if any(phrase in text_lower for phrase in refusal_phrases):
             return True

        # Check if the response doesn't look like typical code (e.g., lacks common code symbols)
        # This is a heuristic and might need refinement
        code_symbols = ['{', '}', '(', ')', '=', ';', ':', 'import', 'def', 'class', 'function', 'var', 'let', 'const']
        if len(text) < 500 and not any(symbol in text_lower for symbol in code_symbols): # Short response without code symbols might be refusal
             # Exception: Very short, simple code might not have many symbols
             if 'print(' not in text_lower and 'console.log(' not in text_lower:
                 logging.warning(f"Response might be a refusal (short, lacks common code symbols): {text[:100]}...")
                 return True

        return False

    def _is_potentially_harmful(self, code: str) -> bool:
        """Performs a basic check for potentially harmful patterns in the code."""
        code_lower = code.lower()
        harmful_patterns = [
            r'rm\s+-rf',       # File deletion
            r'deltree',        # File deletion (Windows)
            r'format\s+[a-z]:', # Formatting drives (Windows)
            r':(){:|:&};:',    # Fork bomb (Bash)
            r'os\.fork\(\)',    # Forking (check context if used repeatedly in loops)
            # Add more specific patterns as needed, e.g., accessing common sensitive file paths
            r' shutil\.rmtree', # Recursive directory removal
            # Be careful not to block legitimate uses, e.g., os.remove for temp files
        ]
        
        # Check for os.system or subprocess calls with potentially dangerous commands
        suspicious_cmd_pattern = r'(?:os\.system|subprocess\.(?:run|call|check_call|check_output|Popen))\s*\(\s*[\'"](.*?)[\'"]'
        suspicious_matches = re.findall(suspicious_cmd_pattern, code, re.IGNORECASE)
        
        suspicious_commands = ['rm', 'del', 'format', 'mkfs', 'dd ', 'shutdown', 'reboot', 'wget ', 'curl '] # Add more

        for match in suspicious_matches:
            command_part = match.split(' ')[0] # Get the command itself
            if any(cmd in command_part for cmd in suspicious_commands):
                 # Check for very common safe uses, like removing a specific temp file
                 if 'rm ' in match and '/tmp/' in match and ' -rf ' not in match :
                     continue # Allow removing specific files in /tmp
                 logging.warning(f"Suspicious command found via os.system/subprocess: {match}")
                 return True

        for pattern in harmful_patterns:
            if re.search(pattern, code_lower):
                logging.warning(f"Potentially harmful pattern matched: {pattern}")
                return True
        return False

    def run(self, \
            task: str, \
            search_context: str | None = None, \
            previous_code: str | None = None, \
            error_message: str | None = None, \
            print_results: bool = False\
            ) -> Dict[str, Any]:
        """작업 실행 (다국어 코드 생성, 패키지 감지, 실행 요청 감지, 코드 수정)
        
        Args:
            task (str): 사용자의 원본 작업 요청
            search_context (str | None, optional): 웹 검색 결과 (요약). Defaults to None.
            previous_code (str | None, optional): 이전에 생성되었던 코드 (수정 요청 시). Defaults to None.
            error_message (str | None, optional): 이전 코드 실행 시 발생한 오류 메시지 (수정 요청 시). Defaults to None.
            print_results (bool, optional): 결과를 콘솔에 출력할지 여부. Defaults to False.
            
        Returns:
            Dict[str, Any]: 작업 처리 결과 딕셔너리
        """
        
        detected_language, is_codegen_request, is_execution_request = self._detect_language_and_request(task)
        is_correction_request = bool(previous_code and error_message)

        if is_correction_request or (is_codegen_request and detected_language):
            # 수정 요청인 경우 언어 재감지 (혹시나 해서)
            if is_correction_request and not detected_language:
                # 간단하게 python 코드로 가정 (추후 개선 가능)
                detected_language = 'python' 
                logging.info("수정 요청에서 언어가 불분명하여 Python으로 가정합니다.")
            
            if not detected_language: # 코드 생성인데 언어가 없는 이상한 경우
                 return {
                    "task": task,
                    "result_type": "error",
                    "result": "코드 생성을 요청했으나, 작업할 프로그래밍 언어를 특정할 수 없습니다.",
                    "status": "failed"
                }

            if is_correction_request:
                logging.info(f"LLM 코드 수정 요청 ({detected_language}): {task}")
            else:
                logging.info(f"LLM 코드 생성 요청 ({detected_language}): {task}")
            
            # 특정 키워드를 감지하여 프롬프트 강화
            is_gui_request = any(keyword in task.lower() for keyword in ['gui', '인터페이스', 'ui', '화면', 'window', 'tkinter', 'pygame', 'pyqt', 'qt'])
            is_particle_request = any(keyword in task.lower() for keyword in ['파티클', '폭죽', '입자', 'particle', 'firework', '애니메이션', 'animation'])
            
            # --- 시스템 프롬프트 구성 --- 
            if is_correction_request:
                system_prompt = f"""You are a code correction assistant. Fix the provided {detected_language} code based on the user's original request and the error message. 
Output *only* the corrected, complete, raw code in {detected_language}. 
Do not include explanations, comments, markdown formatting (like ```{detected_language}), or introductory phrases. Just output the corrected code itself.

EXTREMELY IMPORTANT: Ensure the corrected code is safe and does not contain harmful, destructive, or malicious operations (like rm -rf, file deletion, fork bombs, etc.). Focus *only* on fixing the error according to the error message."""
            else: # 코드 생성 요청
                system_prompt = f"""You are a code generation assistant. Generate *only* the raw code in the requested language ({detected_language}) based on the user's request. Do not include any explanations, comments, markdown formatting (like ```{detected_language}), or introductory phrases. Just output the code itself.

EXTREMELY IMPORTANT: Do NOT generate any harmful, destructive, malicious, or dangerous code. Never include commands like:
- rm -rf, deltree, format, or any destructive file system operations
- Fork bombs or infinite loops that consume system resources
- Network attacks or scanning tools
- Code that creates, modifies, or deletes system files
- Code that attempts to access sensitive user data
- Code that disables security features

Only generate educational, useful, and safe code that demonstrates the requested functionality."""
            
            # 특정 요청에 대한 프롬프트 강화 (생성 시에만)
            if not is_correction_request:
                if is_gui_request:
                    system_prompt += f"\n\nInclude necessary imports and full implementation for a GUI application in {detected_language}. Ensure the code is complete, runnable, and properly handles window creation, user interface components, and interactions."
                
                if is_particle_request:
                    system_prompt += f"\n\nCreate code for particle or fireworks simulation. Include animations, particle physics, color effects, and visual rendering. Ensure the code is complete, self-contained, and demonstrates particle movement, colors, and dynamic effects."
                    
                    if detected_language == 'python':
                        system_prompt += "\n\nRecommended libraries: pygame for visual effects, tkinter for simple GUIs, or numpy for calculations. Use appropriate timing, loops, and event handling to create smooth animations."
            
            # 검색 컨텍스트 추가 (생성 시에만)
            if search_context and not is_correction_request:
                system_prompt += f"\n\nUse the following search results as context if relevant:\n--- SEARCH CONTEXT ---\n{search_context}\n--- END SEARCH CONTEXT ---"
                logging.info("코드 생성 시 검색 컨텍스트 사용")

            # --- 사용자 프롬프트 구성 --- 
            if is_correction_request:
                user_prompt = f"Original Request: {task}\n\nPrevious {detected_language} Code (with error):\n```\n{previous_code}\n```\n\nError Message:\n```\n{error_message}\n```\n\nPlease fix the code based on the error message and the original request. Provide only the corrected code."
            else: # 코드 생성 요청
                user_prompt = f"Generate {detected_language} code for the following request: {task}"
                if is_gui_request and is_particle_request:
                    user_prompt += "\n\nCreate a self-contained program that shows animated particles or fireworks with a proper GUI interface. The program should:\n1. Open a window with controls (buttons, sliders, etc.)\n2. Display animated particles or fireworks effects\n3. Allow users to interact with or control the particle effects\n4. Include proper animation loops and timing\n5. Handle window events correctly"

            # --- LLM 호출 ---
            messages = []
            messages.append({"role": "system", "content": system_prompt})
            
            user_content_parts = [f"Task: {task}"]
            if is_correction_request:
                user_content_parts.append(f"\\n\\nPrevious Code:\\n{previous_code}")
                user_content_parts.append(f"\\n\\nError Message:\\n{error_message}")
            
            messages.append({"role": "user", "content": "\\n".join(user_content_parts)})

            generated_code = None
            required_packages = []
            status = "failed" # Default status
            llm_response_text = "" # Store full response

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini", # Or your preferred model
                    messages=messages,
                    temperature=0.1, # Low temp for deterministic code
                    max_tokens=1500, # Adjust as needed
                    n=1,
                    stop=None,
                )
                llm_response_text = response.choices[0].message.content.strip()
                
                # --- Refined Code Extraction ---
                # 1. Search for the first code block using regex
                code_block_pattern = r"```(?:python|\\w*)?\\s*\\n(.*?)\\n```"
                match = re.search(code_block_pattern, llm_response_text, re.DOTALL | re.IGNORECASE)

                if match:
                    # 2. Extract code if block found
                    extracted_code = match.group(1).strip()
                    logging.info(f"Code block extracted successfully. Length: {len(extracted_code)}")
                    
                    # 3. Basic safety check on extracted code (can be enhanced)
                    if self._is_potentially_harmful(extracted_code):
                         logging.warning(f"Potentially harmful code pattern detected in extracted block.")
                         result_message = f"LLM generated potentially harmful code patterns."
                         status = "failed_safety"
                    else:
                         generated_code = extracted_code
                         status = "success"
                         logging.info(f"Code generation successful ({detected_language}).")
                else:
                    # 4. If no code block found, check if it's a refusal
                    logging.warning("No code block found in LLM response.")
                    if self._is_refusal_message(llm_response_text):
                        logging.warning(f"LLM refused to generate/correct code. Response: {llm_response_text}")
                        result_message = f"LLM이 코드 생성/수정을 거부했습니다:\\n---\\n{llm_response_text}\\n---"
                        status = "refused"
                    else:
                        # Not a refusal, but no code block - maybe just text?
                        logging.warning("LLM response did not contain a code block and wasn't detected as refusal.")
                        result_message = f"LLM이 코드를 생성하지 못했습니다. 응답 내용:\\n---\\n{llm_response_text}\\n---"
                        status = "failed_no_code"
                
                # --- Package Check (only if code generation was successful) ---
                if status == "success" and detected_language == 'python':
                    imports = self._find_python_imports(generated_code)
                    required_packages = self._check_required_packages(imports)
                    if required_packages:
                        logging.info(f"Detected required packages: {required_packages}")

            except Exception as e:
                logging.error(f"LLM API 호출 중 오류 발생: {e}", exc_info=True)
                result_message = f"LLM API 호출 중 오류가 발생했습니다: {e}"
                status = "failed_api"

            # --- 결과 처리 및 파일 저장 ---
            if status == "success":
                # Save the generated code
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                extension = self.lang_to_ext.get(detected_language, '.txt')
                filename = f"{detected_language}_code_{timestamp}{extension}"
                saved_file_path = os.path.join(self.output_dir, filename)
                
                try:
                    with open(saved_file_path, 'w', encoding='utf-8') as f:
                        f.write(generated_code)
                    logging.info(f"Generated code saved to: {saved_file_path}")
                    save_msg = f"코드가 성공적으로 생성되어 '{saved_file_path}'에 저장되었습니다."
                    
                    final_result = {
                        "task": task,
                        "result_type": "code_generation",
                        "result": f"Code generation successful for language: {detected_language}",
                        "generated_code": generated_code,
                        "saved_file_path": saved_file_path,
                        "saved_path_message": save_msg,
                        "language": detected_language,
                        "execute_request": is_execution_request,
                        "required_packages": required_packages,
                        "status": status
                    }
                    
                except IOError as e:
                    logging.error(f"파일 저장 실패: {saved_file_path}, 오류: {e}")
                    final_result = {
                        "task": task,
                        "result_type": "error",
                        "result": f"생성된 코드를 파일에 저장하는 중 오류 발생: {e}",
                        "status": "failed_save"
                    }
            else: # Handle various failure cases
                final_result = {
                    "task": task,
                    "result_type": "error",
                    "result": result_message, # Use the message set in the try/except block
                    "full_llm_response": llm_response_text, # Include full response for debugging
                    "status": status
                }

            if print_results:
                print(f"--- Code Generation Result ---\nTask: {task}\nStatus: {final_result.get('status')}\nResult: {final_result.get('result')}")
                if final_result.get('generated_code'):
                    print(f"Language: {final_result.get('language')}")
                    print(f"Saved to: {final_result.get('saved_file_path')}")
                    if final_result.get('required_packages'):
                        print(f"Required Packages: {final_result.get('required_packages')}")
                    print(f"Code:\\n{final_result.get('generated_code')[:500]}...") # Print first 500 chars
                elif final_result.get('full_llm_response'):
                     print(f"LLM Response:\\n{final_result.get('full_llm_response')}")
                print("-" * 30)
                
            return final_result
            
        else: # 작업 유형이 코드 생성/수정이 아닌 경우
             # This part might not be strictly necessary if AgentAI routes tasks correctly
             logging.info(f"Task '{task}' is not a code generation or correction request.")
             return {
                "task": task,
                "result_type": "info",
                "result": "작업 유형이 코드 생성 또는 수정이 아닙니다.",
                "status": "not_applicable"
            }

        # --- 코드 생성이 아니거나 언어를 특정할 수 없는 경우 ---
        result = {
            "task": task,
            "result_type": "unknown",
            "result": "요청을 처리할 수 없습니다. 명확한 명령어를 사용하거나 지원되는 언어(Python, C++, Java 등)와 함께 코드 생성을 요청해주세요.",
            "status": "failed"
        }
            
        return result 