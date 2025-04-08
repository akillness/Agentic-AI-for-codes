from typing import List, Dict, Any

class ResultFormatter:
    def __init__(self):
        pass # No initialization needed for now

    def combine_step_results(self, step_results: List[Dict[str, Any]]) -> str:
        """Combines results from multiple steps into a single message."""
        if not step_results:
            return "수행된 작업 단계가 없습니다."

        # If only one step, return its result directly
        if len(step_results) == 1:
            result_data = step_results[0].get("result", {})
            return result_data.get("result", "알 수 없는 결과")

        # Combine results from multiple steps
        combined_message = ""
        for i, step in enumerate(step_results, 1):
            description = step.get("description", f"단계 {i}")
            result_data = step.get("result", {})
            result_text = result_data.get("result", "결과 없음")
            success = result_data.get("success", False)

            status = "성공" if success else "실패"
            combined_message += f"\n== 단계 {i}: {description} ({status}) ==\n"
            combined_message += f"{result_text}\n"
            # Add extra newline for separation, except for the last step
            if i < len(step_results):
                 combined_message += "\n" + "-" * 30 + "\n"

        return combined_message.strip() 