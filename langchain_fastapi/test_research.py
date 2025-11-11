import asyncio
import requests
import json

BASE_URL = "http://localhost:8000"

def test_capability_analysis():
    """测试能力分析"""
    response = requests.post(
        f"{BASE_URL}/research/capability-analysis/invoke",
        json={"input": {"question": "请解释深度学习中的注意力机制"}}
    )
    return response.json()

def test_comparison():
    """测试模型对比"""
    response = requests.post(
        f"{BASE_URL}/research/comparison",
        json={
            "question": "什么是大语言模型？",
            "compare_with_openai": True
        }
    )
    return response.json()

def test_performance():
    """测试性能"""
    response = requests.post(
        f"{BASE_URL}/research/performance-test",
        json={
            "test_type": "technical",
            "content": "解释神经网络的反向传播算法",
            "iterations": 2
        }
    )
    return response.json()

def test_technical_specs():
    """测试技术规格调研"""
    response = requests.post(
        f"{BASE_URL}/research/technical-specs/invoke",
        json={"input": {"focus_area": "architecture"}}
    )
    return response.json()

def run_all_tests():
    """运行所有测试"""
    print("=== DeepSeek 调研平台测试 ===\n")

    print("1. 能力分析测试:")
    result = test_capability_analysis()
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n2. 模型对比测试:")
    result = test_comparison()
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n3. 性能测试:")
    result = test_performance()
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n4. 技术规格调研:")
    result = test_technical_specs()
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    run_all_tests()