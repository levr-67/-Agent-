import networkx as nx
import matplotlib.pyplot as plt

class TraceAnalysisAgent:
    """
    溯源分析Agent：生成传播链路图谱、统计传播路径
    对应系统核心模块3
    """
    def __init__(self):
        self.token_count = 0
        self.graph = nx.DiGraph()

    def generate_trace_graph(self, eval_result: dict) -> dict:
        """生成虚假新闻传播溯源图谱"""
        news_id = eval_result["news_id"]
        is_fake = eval_result["is_fake"]

        if is_fake:
            # 模拟传播链路
            nodes = [f"源头{news_id}", "社交媒体", "自媒体", "用户群"]
            edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
            self.graph.add_edges_from(edges)
            self.token_count += 500

            # 保存图谱图片（GitHub可视化展示）
            plt.figure(figsize=(8, 5))
            nx.draw(self.graph, with_labels=True, node_color="red", font_size=10)
            plt.title(f"虚假新闻{news_id}传播链路图谱")
            plt.savefig(f"trace_graph_{news_id}.png")
            plt.close()

        return {
            "news_id": news_id,
            "is_fake": is_fake,
            "trace_nodes": list(self.graph.nodes) if is_fake else [],
            "trace_status": "completed"
        }
