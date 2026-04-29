# 基于多Agent协作的虚假新闻识别与溯源系统
# 毕业设计 | GitHub开源项目
import pandas as pd
from agents import (
    MultimodalParseAgent,
    CredibilityEvalAgent,
    TraceAnalysisAgent,
    InteractiveAgent
)

def main():
    print("="*60)
    print("🚀 虚假新闻识别多Agent系统启动中...")
    print("🎯 核心功能：多模态解析 | 可信度评估 | 传播溯源 | 闭环优化")
    print("="*60)

    # 1. 加载数据集（10w+条数据集模拟，此处用示例数据）
    df = pd.read_csv("data/sample_news.csv")
    news_list = df.to_dict("records")

    # 2. 初始化4大Agent
    multimodal_agent = MultimodalParseAgent()
    credibility_agent = CredibilityEvalAgent()
    trace_agent = TraceAnalysisAgent()
    interactive_agent = InteractiveAgent()

    results = []
    total_token = 0

    # 3. 多Agent协同处理新闻数据
    for news in news_list:
        print(f"\n📰 正在处理新闻ID：{news['news_id']}")

        # Step1：多模态解析
        parsed = multimodal_agent.parse_news(news)
        print(f"✅ 多模态解析完成 | 提取实体：{parsed['entities']}")

        # Step2：可信度评估（LSTM+大模型推理）
        eval_result = credibility_agent.predict_credibility(parsed)
        status = "❌ 虚假新闻" if eval_result["is_fake"] else "✅ 真实新闻"
        print(f"📊 可信度评估完成 | 检测结果：{status} | 评分：{eval_result['credibility_score']}")

        # Step3：溯源分析（生成传播图谱）
        trace_result = trace_agent.generate_trace_graph(eval_result)
        print(f"🔍 溯源分析完成 | 传播链路：{trace_result['trace_nodes']}")

        # Token统计
        token = multimodal_agent.token_count + credibility_agent.token_count
        total_token += token

        results.append({
            "parsed_data": parsed,
            "eval_result": eval_result,
            "trace_result": trace_result,
            "token": token
        })

    # 4. 生成最终报告（人工交互闭环）
    report = interactive_agent.generate_report(results)
    print("\n" + "="*60)
    print("📋 系统运行报告（量化指标）")
    print("="*60)
    for key, value in report.items():
        print(f"{key}: {value}")
    print("="*60)
    print("✅ 项目运行完成！图谱已保存，可上传GitHub作为演示证明")

if __name__ == "__main__":
    main()
