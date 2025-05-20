import unittest
from unittest.mock import Mock, patch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import LLMInference


class TestLLMInference(unittest.TestCase):
    def setUp(self):
        """测试开始前的设置"""
        self.model = "test-model"
        self.api_base = "http://test-api-base"
        self.api_key = "test-api-key"
        self.llm = LLMInference(self.model, self.api_base, self.api_key)

    @patch("openai.OpenAI")
    def test_inference(self, mock_openai):
        """测试inference方法"""
        # 设置模拟响应
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="测试回复"))]
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # 执行测试
        result = self.llm.inference("测试输入")

        # 验证结果
        self.assertEqual(result, "测试回复")
        mock_client.chat.completions.create.assert_called_once_with(
            model=self.model, messages=[{"role": "user", "content": "测试输入"}]
        )

    @patch("openai.OpenAI")
    def test_analyze(self, mock_openai):
        """测试analyze方法"""
        # 设置模拟响应
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="分析结果"))]
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # 执行测试
        result = self.llm.analyze(
            summary="测试摘要",
            target="测试目标",
            favor_rate="60%",
            neutral_rate="20%",
            against_rate="20%",
            top_words=["词1", "词2"],
        )

        # 验证结果
        self.assertEqual(result, "分析结果")
        self.assertTrue(mock_client.chat.completions.create.called)

    def test_call_method(self):
        """测试__call__方法"""
        self.llm.inference = Mock(return_value="测试回复")
        result = self.llm("测试输入")
        self.assertEqual(result, "测试回复")
        self.llm.inference.assert_called_once_with("测试输入")


if __name__ == "__main__":
    unittest.main()
