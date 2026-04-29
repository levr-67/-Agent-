import torch
import torch.nn as nn

class LSTMFakeNewsModel(nn.Module):
    """LSTM虚假新闻检测模型（模拟训练好的模型）"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=100, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)

class CredibilityEvalAgent:
    """
    可信度评估Agent：LSTM+大模型推理，交叉验证新闻真实性
    对应系统核心模块2
    """
    def __init__(self):
        self.model = LSTMFakeNewsModel()
        self.token_count = 0
        # 模拟权威知识库
        self.trust_sources = ["新华社", "人民日报", "央视新闻"]

    def verify_source(self, source: str) -> bool:
        """验证信源权威性"""
        return source in self.trust_sources

    def predict_credibility(self, parsed_data: dict) -> dict:
        """新闻可信度评分（0-1）"""
        # 模拟模型推理
        dummy_input = torch.randn(1, 1, 100)
        score = self.model(dummy_input).item()
        self.token_count += 3000  # 单条Token消耗：3000（匹配版本一描述）

        # 信源交叉验证
        source_trust = self.verify_source(parsed_data["source"])
        final_score = score * 0.7 + 0.3 if source_trust else score * 0.7

        return {
            "news_id": parsed_data["news_id"],
            "credibility_score": round(final_score, 2),
            "is_fake": final_score < 0.5,
            "verify_source": source_trust,
            "status": "evaluated"
        }
