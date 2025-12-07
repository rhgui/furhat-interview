# demo_llm_cv_questions.py
# ------------------------------------------------------------
# 测试流程：
#   1) 从 PDF 读取 CV 文本
#   2) 用 CVParser + Gemini 解析出结构化 CV 信息
#   3) 用 QuestionGenerator + Gemini 基于 CV 生成 Q1–Q5
#      - Q1, Q2: 固定暖场问题
#      - Q3, Q4, Q5: LLM 从 CV 生成
#
# 运行示例：
#   python demo_llm_cv_questions.py --cv my_cv.pdf --position "Data Scientist Intern"
#
# 前提：
#   - 当前文件夹里有：
#       cv_parser.py
#       question_generator.py
#   - .env 里设置了 GEMINI_API_KEY
#   - 已安装依赖：PyPDF2, python-dotenv, google-genai 等
# ------------------------------------------------------------

import argparse
import json

from cv_parser import CVParser
from question_generator import QuestionGenerator


def main():
    # ---------------- 参数解析 ----------------
    parser = argparse.ArgumentParser(
        description="Demo: use LLM to parse CV and generate interview questions."
    )
    parser.add_argument(
        "--cv",
        type=str,
        default="dummy_cv.pdf",
        help="Path to the CV PDF file (default: dummy_cv.pdf)",
    )
    parser.add_argument(
        "--position",
        type=str,
        default="Software Engineer Intern",
        help="Target position / role for the interview.",
    )
    args = parser.parse_args()

    cv_path = args.cv
    position = args.position

    print("========================================")
    print(f"[1] 读取并解析简历: {cv_path}")
    print("========================================")

    # ---------------- 1. 解析 CV ----------------
    cv_parser = CVParser()
    cv_data = cv_parser.parse_cv(cv_path)

    # 打印解析结果，方便调试
    print("\n--- Parsed CV JSON ---")
    print(json.dumps(cv_data, ensure_ascii=False, indent=2))

    # 简单检查一下是否有错误字段
    if isinstance(cv_data, dict) and cv_data.get("error"):
        print("\n[警告] CV 解析返回 error 字段，后续生成问题可能不太可靠。")
        print("error =", cv_data["error"])

    # ---------------- 2. 基于 CV 生成问题 ----------------
    print("\n========================================")
    print(f"[2] 基于 CV + 职位 \"{position}\" 生成问题")
    print("========================================")

    qgen = QuestionGenerator()
    questions = qgen.generate_full_question_set(cv_data=cv_data, position=position)

    # ---------------- 3. 以可读格式打印问题 ----------------
    print("\n--- Generated Questions (Q1–Q5) ---")
    for q in questions:
        qid = q.get("id", "?")
        text = q.get("text", "")
        guided = q.get("guided_main", "")
        expected_time = q.get("expected_time_s", 0)
        source = q.get("source", "")
        category = q.get("category", "")

        print("\n----------------------------------------")
        print(f"ID: {qid}")
        print(f"Source: {source} | Category: {category}")
        print(f"Main question: {text}")
        if guided:
            print(f"Guided version: {guided}")
        print(f"Expected answer time: {expected_time:.1f} s")

    # ---------------- 4. 如需要，也可以导出为 JSON 文件 ----------------
    output_path = "generated_questions_from_cv.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print("\n========================================")
    print(f"所有问题已保存到: {output_path}")
    print("Demo 运行结束。")
    print("========================================")


if __name__ == "__main__":
    main()
