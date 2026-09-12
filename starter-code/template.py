"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

load_dotenv()

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>

Quy tắc quan trọng:
- Nếu câu hỏi chỉ cần 1 tool, hãy gọi Action rồi NGAY SAU KHI nhận Observation, đưa ra Final Answer trong cùng lượt.
- Nếu câu hỏi cần nhiều tool, hãy gọi từng tool một, mỗi lượt 1 Action.
- Nếu câu hỏi KHÔNG cần tool (FAQ, chính sách chung), hãy đưa ra Final Answer ngay mà KHÔNG cần Action.
- LUÔN kết thúc bằng "Final Answer:" khi đã có đủ thông tin.
"""


# ============================================================
# Milestone 1: Chatbot Baseline — gọi LLM 1 lượt, không dùng tool
# ============================================================
class ChatbotBaseline:
    """Baseline LLM Chatbot without ReAct Loop or Tools"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    # TODO: Trả về câu trả lời không dùng tool
    def query(self, user_input: str) -> Dict[str, Any]:
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel('gemini-3.6-flash')
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn du lịch. Hãy trả lời "
                    f"KHÔNG dùng tool hay internet: {user_input}"
                )
                return {
                    "status": "success",
                    "tool_calls": [],
                    "answer": response.text
                }
            except Exception as e:
                return {
                    "status": "error",
                    "tool_calls": [],
                    "answer": f"Lỗi gọi Gemini API: {e}"
                }
        # Fallback khi không có API key
        return {
            "status": "success",
            "tool_calls": [],
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}"
        }


# ============================================================
# Milestone 3 + 4: ReAct Agent — Thought-Action-Observation Loop
# ============================================================
class ReActAgent:
    """Production-grade ReAct Agent with Tool Registry and Safeguards"""

    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    # Helper: chuẩn hoá mã sân bay từ text tự nhiên
    def parse_city_code(self, text: str) -> str:
        text_upper = text.upper()
        for code in ["SGN", "HAN", "DAD"]:
            if code in text_upper:
                return code
        if "HÀ NỘI" in text_upper:
            return "HAN"
        if "HỒ CHÍ MINH" in text_upper or "SÀI GÒN" in text_upper:
            return "SGN"
        if "ĐÀ NẴNG" in text_upper:
            return "DAD"
        return "SGN"

    # Helper: gọi Gemini API trả về text
    def _call_llm(self, prompt: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel('gemini-3.6-flash')
        response = model.generate_content(prompt)
        return response.text

    # Helper: tách chuỗi JSON ra khỏi markdown nếu LLM bọc ```json ... ```
    def _clean_json(self, raw: str) -> str:
        s = raw.strip()
        if s.startswith("```json"):
            s = s[7:]
        elif s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        return s.strip()

    # Milestone 3: thực thi 1 bước Thought → Action → Observation
    def plan_and_execute_step(self, user_input: str, iteration: int) -> Tuple[str, bool]:
        """
        Trả về (result_text, is_final).
        - is_final = True  → result_text là Final Answer, dừng vòng lặp.
        - is_final = False → result_text chứa llm_response + Observation, tiếp tục lặp.
        """
        if not self.api_key:
            return "Lỗi: Không có GEMINI_API_KEY.", True

        try:
            llm_response = self._call_llm(user_input)
        except Exception as e:
            return f"Lỗi gọi API: {e}", True

        step_data: Dict[str, Any] = {"step": iteration, "llm_response": llm_response}

        # --- TODO 3: Phân tích Thought / Action từ Agent ---
        # Nếu LLM đã đưa ra Final Answer → dừng
        if "Final Answer:" in llm_response:
            final_answer = llm_response.split("Final Answer:")[-1].strip()
            self.trace.append(step_data)
            return final_answer, True

        # Nếu LLM đưa ra Action → parse JSON, gọi tool
        if "Action:" in llm_response:
            try:
                action_str = llm_response.split("Action:")[-1].split("Observation:")[0].strip()
                action_str = self._clean_json(action_str)
                action_dict = json.loads(action_str)

                # Trap 1: .strip().lower() để tránh KeyError
                tool_name = action_dict.get("name", "").strip().lower()
                tool_args = action_dict.get("args", {})

                # Chuẩn hoá city_code nếu cần
                if tool_name == "get_weather_forecast" and "city_code" in tool_args:
                    tool_args["city_code"] = self.parse_city_code(tool_args["city_code"])

                # --- TODO 4: Thực thi Tool trong TOOL_MAP nếu có Action ---
                if tool_name in TOOL_MAP:
                    observation = TOOL_MAP[tool_name](**tool_args)
                    obs_str = f"Observation: {json.dumps(observation, ensure_ascii=False)}"
                else:
                    obs_str = f"Observation: Tool '{tool_name}' not found."

            except json.JSONDecodeError:
                # Trap 2: Format Drift → Invalid JSON
                obs_str = "Observation: Invalid JSON format"
            except Exception as e:
                obs_str = f"Observation: Error — {e}"

            # --- TODO 5: Ghi lại Observation ---
            step_data["observation"] = obs_str
            self.trace.append(step_data)
            return f"{llm_response}\n{obs_str}", False

        # Không có Action cũng không có Final Answer → yêu cầu lại
        step_data["observation"] = "Observation: Hãy đưa ra Action hoặc Final Answer."
        self.trace.append(step_data)
        return f"{llm_response}\nObservation: Hãy đưa ra Action hoặc Final Answer.", False

    def run(self, user_input: str) -> dict:
        # --- TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces ---
        self.trace = []
        tools_str = json.dumps(TOOL_DEFINITIONS, ensure_ascii=False, indent=2)
        prompt_context = SYSTEM_PROMPT.format(tools=tools_str) + f"\nUser: {user_input}\n"

        # --- TODO 2: Thiết lập vòng lặp while iteration < self.max_iterations ---
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1

            result, is_final = self.plan_and_execute_step(prompt_context, iteration)

            if is_final:
                return {
                    "status": "completed",
                    "iterations": iteration,
                    "answer": result,
                    "trace": self.trace
                }

            # Nối kết quả (llm_response + Observation) vào prompt cho vòng lặp tiếp theo
            prompt_context += f"{result}\n"

        # --- Milestone 4: Safeguard — giới hạn max_iterations ---
        return {
            "status": "max_iterations_reached",
            "answer": "Không thể hoàn thành trong số bước tối đa.",
            "trace": self.trace
        }


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result.get("answer"))
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()