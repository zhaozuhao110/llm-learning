# -*- coding: utf-8 -*-
"""
tools.py - 工具库模块

本文件定义了智能体可以调用的具体工具函数。
工具就是智能体的“手”和“眼”，让它能与外部世界交互（查天气、搜索网络等）。
"""

import os
import requests
from tavily import TavilyClient

def get_weather(city: str) -> str:
    """
    工具函数 1：获取天气
    
    功能：通过调用 wttr.in 免费 API 查询指定城市的实时天气。
    参数：city (str) - 城市名称（如 "Beijing"）
    返回：str - 格式化后的天气描述字符串
    """
    
    # 1. 构造请求 URL
    # format=j1 表示请求 JSON 格式的数据，方便代码解析
    # f-string 是 Python 中拼接字符串的便捷方式
    url = f"https://wttr.in/{city}?format=j1"
    
    try:
        # 2. 发起 HTTP 网络请求
        # requests.get 就像你在浏览器地址栏输入网址一样
        response = requests.get(url)
        
        # 3. 检查请求是否成功
        # 如果状态码不是 200（成功），这里会抛出异常
        response.raise_for_status()
        
        # 4. 将返回的文本解析为 JSON 字典对象
        data = response.json()
        
        # 5. 提取我们需要的数据
        # JSON 结构层级较深，需要一层层剥洋葱
        # current_condition[0] 是当前时刻的天气状况
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value'] # 天气描述（如 Clear）
        temp_c = current_condition['temp_C']                        # 摄氏温度
        
        # 6. 将提取的数据拼装成人类可读的句子返回
        return f"{city} 当前天气：{weather_desc}，气温 {temp_c} 摄氏度"
        
    except requests.exceptions.RequestException as e:
        # 捕捉网络类错误（如断网、网址错误）
        return f"错误：查询天气遇到网络问题 - {e}"    
    except (KeyError, IndexError) as e:
        # 捕捉解析类错误（如城市名写错导致返回的数据格式不对）
        return f"错误：解析天气数据失败，可能是城市名称无效 - {e}"

def get_attraction(city: str, weather: str) -> str:
    """
    工具函数 2：推荐景点
    
    功能：根据城市和天气，利用 Tavily 搜索引擎查找适合的旅游景点。
    参数：
        city (str) - 城市名称
        weather (str) - 天气状况（从上一个工具获取）
    返回：str - 搜索到的景点建议
    """
    
    # 1. 从环境变量中读取 API 密钥
    # 就像从保险柜里拿钥匙一样，避免把密码直接写在代码里
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误：未配置 TAVILY_API_KEY 环境变量，请在 .env 文件中配置。" 
    
    # 2. 初始化 Tavily 搜索客户端
    tavily = TavilyClient(api_key=api_key)
    
    # 3. 构造搜索关键词（Prompt Engineering 的一部分）
    # 我们把城市和天气拼在一起，让搜索结果更精准
    query = f"'{city}' 在 '{weather}' 天气下值得去的旅游景点推荐及理由"
    
    try:
        # 4. 调用搜索 API
        # include_answer=True 表示让 AI 直接生成一个总结性的答案，而不只是给一堆链接
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        
        # 5. 处理搜索结果
        # 优先返回 AI 总结好的答案
        if response.get("answer"):
            return response["answer"]
        
        # 如果没有总结答案，就自己拼凑搜索到的标题和内容
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        
        if not formatted_results:
             return "抱歉，没有找到相关的旅游景点推荐。"

        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误：执行 Tavily 搜索时出现问题 - {e}"

# --- 工具注册 ---
# 将函数名（字符串）映射到真实的函数对象
# 这样主程序就可以通过字符串 "get_weather" 找到并执行 get_weather 函数
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}
