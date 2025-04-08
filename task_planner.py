from typing import List, Dict, Any
import logging
import json
import re  # Move import to the top
from model_manager import ModelManager
import constants # Import constants

class TaskPlanner:
    def __init__(self, model_manager: ModelManager):
        """Initializes the TaskPlanner."""
        self.model_manager = model_manager
        logging.info("TaskPlanner initialized.")

    def _detect_explicit_patterns(self, task: str) -> List[Dict[str, Any]] | None:
        """Detects explicit, common task patterns based on keywords."""
        task_lower = task.lower()

        # Keywords for different actions
        search_kws = ["검색", "찾아줘", "알아봐", "search", "find", "look up"]
        code_gen_kws = ["코드", "프로그램", "작성", "만들", "짜줘", "generate", "create", "code", "write", "develop"]
        execute_kws = ["실행", "돌려", "run", "execute", "start"]
        compile_kws = ["컴파일", "빌드", "compile", "build"]
        dir_kws = ["디렉토리", "폴더", "directory", "folder", "ls", "list files"]
        file_manage_kws = ["파일 관리", "file manage", "생성", "삭제", "이동", "복사", "create", "delete", "move", "copy", "read", "write"]

        # Detect presence of keywords
        has_search = any(kw in task_lower for kw in search_kws)
        has_code_gen = any(kw in task_lower for kw in code_gen_kws)
        has_execute = any(kw in task_lower for kw in execute_kws)
        has_compile = any(kw in task_lower for kw in compile_kws)
        has_dir = any(kw in task_lower for kw in dir_kws)
        has_file_manage = any(kw in task_lower for kw in file_manage_kws)

        # --- Define explicit patterns --- #

        # Pattern: Compile then Run
        if has_compile and has_execute and not has_code_gen and not has_search:
            # Extract potential file path
            # This is a simple heuristic, might need refinement
            file_path_match = re.search(r'([\w\.\-\/]+\.(?:cpp|c|rs|cs))\b', task)
            file_path = file_path_match.group(1) if file_path_match else "unknown_file_to_compile"
            logging.info(f"Explicit pattern detected: Compile + Run ({file_path})")
            return [
                {
                    "task_type": constants.TASK_COMPILATION,
                    "description": f"'{file_path}' 파일 컴파일",
                    "parameters": {"file_path": file_path}
                },
                {
                    "task_type": constants.TASK_COMPILED_RUN,
                    "description": "컴파일된 파일 실행",
                    "parameters": {"file_path": file_path} # Pass original path
                }
            ]

        # Pattern: Search -> Code Gen -> Execute
        if has_search and has_code_gen and has_execute:
            logging.info("Explicit pattern detected: Search + Code Gen + Execute")
            return [
                {
                    "task_type": constants.TASK_SEARCH,
                    "description": f"'{task}' 관련 정보 웹 검색",
                    "parameters": {"query": task}
                },
                {
                    "task_type": constants.TASK_CODE_GENERATION,
                    "description": f"검색 결과를 활용하여 '{task}' 요청 코드 생성",
                    "parameters": {"task": task, "use_search_context": True}
                },
                {
                    "task_type": constants.TASK_FILE_EXECUTION,
                    "description": "생성된 코드 파일 실행",
                    "parameters": {} # File path comes from code_generation context
                }
            ]

        # Pattern: Search -> Code Gen
        if has_search and has_code_gen and not has_execute:
            logging.info("Explicit pattern detected: Search + Code Gen")
            return [
                {
                    "task_type": constants.TASK_SEARCH,
                    "description": f"'{task}' 관련 정보 웹 검색",
                    "parameters": {"query": task}
                },
                {
                    "task_type": constants.TASK_CODE_GENERATION,
                    "description": f"검색 결과를 활용하여 '{task}' 요청 코드 생성",
                    "parameters": {"task": task, "use_search_context": True}
                }
            ]

        # Pattern: Code Gen -> Execute
        if not has_search and has_code_gen and has_execute:
            logging.info("Explicit pattern detected: Code Gen + Execute")
            return [
                {
                    "task_type": constants.TASK_CODE_GENERATION,
                    "description": f"'{task}' 요청 코드 생성",
                    "parameters": {"task": task, "use_search_context": False}
                },
                {
                    "task_type": constants.TASK_FILE_EXECUTION,
                    "description": "생성된 코드 파일 실행",
                    "parameters": {} # File path comes from code_generation context
                }
            ]

        # Pattern: Directory Exploration
        if has_dir and not (has_code_gen or has_search or has_execute or has_file_manage):
            # Try to extract path, default to current dir
            dir_path_match = re.search(r'(?:디렉토리|폴더|directory|folder)\s+([\w\.\-\/\~]+)', task_lower)
            dir_path = dir_path_match.group(1) if dir_path_match else "."
            logging.info(f"Explicit pattern detected: Directory Exploration ({dir_path})")
            return [
                 {
                     "task_type": constants.TASK_DIRECTORY_EXPLORATION,
                     "description": f"'{dir_path}' 디렉토리 탐색",
                     "parameters": {"dir_path": dir_path}
                 }
             ]

        # Pattern: Simple File Management (Create/Delete)
        if has_file_manage and not (has_code_gen or has_search or has_execute):
            action = None
            path = None
            # Simple extraction, needs improvement for robustness
            if any(kw in task_lower for kw in ["생성", "create"]):
                 action = "create"
                 match = re.search(r'(?:생성|create)\s+([\w\.\-\/\~]+)', task_lower)
                 path = match.group(1) if match else None
            elif any(kw in task_lower for kw in ["삭제", "delete"]):
                 action = "delete"
                 match = re.search(r'(?:삭제|delete)\s+([\w\.\-\/\~]+)', task_lower)
                 path = match.group(1) if match else None
            # Add more actions like move, copy, read, write if needed

            if action and path:
                logging.info(f"Explicit pattern detected: File Management ({action} {path})")
                return [
                     {
                         "task_type": constants.TASK_FILE_MANAGEMENT,
                         "description": f"파일 관리: {path} {action}",
                         "parameters": {"action": action, "path": path}
                     }
                 ]

        # Add more explicit patterns here as needed

        return None # No explicit pattern matched

    def _plan_with_llm(self, task: str) -> List[Dict[str, Any]]:
        """Uses LLM to generate a task plan when no explicit pattern matches."""
        logging.info("No explicit pattern matched, using LLM for planning.")
        plan_prompt = f"""
사용자의 요청을 분석하여 수행해야 할 작업 계획을 JSON 배열 형식으로 생성해주세요.

사용 가능한 작업 유형:
- {constants.TASK_SEARCH}: 웹 검색 (파라미터: query)
- {constants.TASK_CODE_GENERATION}: 코드 생성 (파라미터: task, use_search_context)
- {constants.TASK_FILE_EXECUTION}: 생성된 코드 파일 실행 (파라미터: file_path - 이전 단계에서 전달됨, 또는 명시적 지정)
- {constants.TASK_CODE_BLOCK_EXECUTION}: 코드 블록 직접 실행 (파라미터: code, language)
- {constants.TASK_COMPILATION}: 코드 컴파일 (파라미터: file_path)
- {constants.TASK_COMPILED_RUN}: 컴파일된 파일 실행 (파라미터: file_path - 원본 소스 파일 경로)
- {constants.TASK_DIRECTORY_EXPLORATION}: 디렉토리 내용 확인 (파라미터: dir_path)
- {constants.TASK_FILE_MANAGEMENT}: 파일 생성/삭제/이동/읽기/쓰기 (파라미터: action[create|delete|move|read|write], path, [new_path], [content])

규칙:
1. 응답은 반드시 JSON 배열이어야 합니다 (예: `[{{"task_type": ...}}, ...]`). 다른 텍스트는 포함하지 마세요.
2. 각 단계는 `task_type`, `description`, `parameters` 키를 포함해야 합니다.
3. `description`은 해당 단계에서 수행할 작업을 간결하게 설명합니다.
4. `parameters`는 각 작업 유형에 필요한 정보를 포함합니다.
5. `code_generation` 후 `file_execution`이 필요하면, `file_execution`의 `parameters`는 비워두세요 (경로는 자동으로 전달됩니다).
6. `compilation` 후 `compiled_run`이 필요하면, `compiled_run`의 `parameters`에 원본 소스 `file_path`를 지정하세요.
7. 여러 단계가 필요할 수 있습니다. 예를 들어, 정보 검색 후 코드 생성, 또는 코드 생성 후 컴파일 및 실행.
8. 사용자의 요청을 최대한 반영하여 필요한 모든 단계를 포함하세요.

사용자 요청:
{task}

JSON 계획:
"""

        try:
            llm_result = self.model_manager.call_llm(
                task_type='planning',
                messages=[
                    {"role": "system", "content": "You are a planning assistant. Generate a JSON array representing the steps needed to fulfill the user request, following the provided instructions and schema. Respond ONLY with the JSON array."},
                    {"role": "user", "content": plan_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            if not llm_result["success"]:
                raise Exception(f"LLM planning call failed: {llm_result['error']}")

            raw_plan_json = llm_result["content"]
            logging.info(f"Raw plan JSON from LLM: {raw_plan_json}")

            # Attempt to parse the JSON
            plan = self._parse_and_validate_plan(raw_plan_json, task)
            return plan

        except Exception as e:
            logging.error(f"LLM-based planning failed: {e}", exc_info=True)
            # Fallback plan: Simple code generation or search based on keywords
            if any(kw in task.lower() for kw in self._get_keywords("code_gen_kws")):
                return [{ "task_type": constants.TASK_CODE_GENERATION, "description": "코드 생성 시도", "parameters": {"task": task, "use_search_context": False}}]
            else:
                return [{ "task_type": constants.TASK_SEARCH, "description": "웹 검색 시도", "parameters": {"query": task}}]

    def _parse_and_validate_plan(self, plan_json: str, original_task: str) -> List[Dict[str, Any]]:
        """Parses the JSON plan and validates its structure, providing fallbacks."""
        try:
            # Clean potential markdown fences
            if plan_json.startswith("```json"):
                plan_json = plan_json[7:]
            if plan_json.endswith("```"):
                plan_json = plan_json[:-3]
            plan_json = plan_json.strip()

            data = json.loads(plan_json)

            # LLM might return a dictionary with a key like "plan" or "tasks"
            if isinstance(data, dict):
                possible_keys = ["plan", "tasks", "steps"]
                for key in possible_keys:
                    if key in data and isinstance(data[key], list):
                        plan = data[key]
                        break
                else:
                    # If it's a dict but looks like a single task, wrap it
                    if "task_type" in data:
                         plan = [data]
                    else:
                         raise ValueError("LLM returned a dictionary without a recognized plan list.")
            elif isinstance(data, list):
                plan = data
            else:
                raise ValueError("LLM response is not a JSON list or a recognized dictionary.")

            # Validate individual steps
            validated_plan = []
            for i, step in enumerate(plan):
                if not isinstance(step, dict):
                    logging.warning(f"Plan step {i} is not a dictionary: {step}. Skipping.")
                    continue

                # Ensure required keys exist
                if "task_type" not in step:
                    logging.warning(f"Plan step {i} missing 'task_type': {step}. Skipping.")
                    continue
                if "description" not in step:
                    step["description"] = f"{step['task_type']} 작업 수행" # Add default description
                if "parameters" not in step or not isinstance(step["parameters"], dict):
                    logging.warning(f"Plan step {i} missing or invalid 'parameters': {step}. Setting to empty dict.")
                    step["parameters"] = {} # Add default empty params

                validated_plan.append(step)

            if not validated_plan:
                 raise ValueError("LLM plan parsing resulted in an empty plan.")

            logging.info(f"Successfully parsed and validated LLM plan: {validated_plan}")
            return validated_plan

        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse or validate LLM plan JSON: {e}. Raw JSON: \n{plan_json}")
            # Fallback plan if parsing/validation fails
            if any(kw in original_task.lower() for kw in self._get_keywords("code_gen_kws")):
                return [{ "task_type": constants.TASK_CODE_GENERATION, "description": "LLM 계획 실패 후 코드 생성 시도", "parameters": {"task": original_task, "use_search_context": False}}]
            else:
                return [{ "task_type": constants.TASK_SEARCH, "description": "LLM 계획 실패 후 웹 검색 시도", "parameters": {"query": original_task}}]

    def _get_keywords(self, kw_type: str) -> List[str]:
         """Helper to get keyword lists, prevents repeating them."""
         # This could be refactored slightly from _detect_explicit_patterns
         # For simplicity here, we just return them directly
         if kw_type == "code_gen_kws":
             return ["코드", "프로그램", "작성", "만들", "짜줘", "generate", "create", "code", "write", "develop"]
         # Add other types if needed for fallbacks
         return []

    def plan_task(self, task: str) -> List[Dict[str, Any]]:
        """Generates a task plan, first checking explicit patterns, then using LLM."""
        logging.info(f"Generating task plan for: {task}")

        # 1. Check for explicit patterns
        explicit_plan = self._detect_explicit_patterns(task)
        if explicit_plan:
            logging.info(f"Using explicit plan: {explicit_plan}")
            return explicit_plan

        # 2. If no explicit pattern, use LLM
        llm_plan = self._plan_with_llm(task)
        logging.info(f"Using LLM generated plan: {llm_plan}")
        return llm_plan 