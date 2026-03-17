# -*- coding: utf-8 -*-
"""
agent.py - 智能体主程序

本文件是智能体的核心控制逻辑，负责：
1. 配置大模型客户端
2. 维护对话历史（Memory）
3. 运行 "思考-行动-观察" (Thought-Action-Observation) 循环
"""

import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# 导入我们在 tools.py 和 prompts.py 中定义的工具和提示词
from tools import available_tools
from prompts import AGENT_SYSTEM_PROMPT

# 加载 .env 文件中的环境变量（比如 API Key）
# 这行代码会让 os.environ.get() 能读到 .env 里的内容
load_dotenv()

class OpenAICompatibleClient:
    """
    一个简单的 LLM 客户端封装类。
    
    作用：统一调用不同服务商（OpenAI, 阿里云, DeepSeek 等）的接口。
    只要服务商兼容 OpenAI 的 SDK 格式，都可以用这个类。
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        # 初始化 OpenAI 官方 SDK 客户端
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """
        核心方法：发送提示词给大模型，获取回复。
        
        参数：
            prompt (str): 当前的用户问题或对话历史
            system_prompt (str): 系统提示词（人设和规则）
        返回：
            str: 大模型的回答内容
        """
        print("🤖 [LLM] 正在思考...")
        try:
            # 构造消息列表，包含系统人设和用户输入
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            # 调用 API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False # 不使用流式输出，一次性获取完整结果
            )
            # 提取回答文本
            answer = response.choices[0].message.content
            print("✅ [LLM] 思考完成。")
            return answer
        except Exception as e:
            print(f"❌ [Error] 调用 LLM API 失败: {e}")
            return "错误：调用语言模型服务时出错。"

def run_agent():
    """
    智能体的主循环逻辑 (ReAct 模式)
    """
    
    # --- 1. 配置阶段 ---
    # 从环境变量读取配置，如果没读到则使用默认值
    API_KEY = os.environ.get("LLM_API_KEY", "your_api_key_here")
    BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    MODEL_ID = os.environ.get("LLM_MODEL_ID", "gpt-3.5-turbo")
    
    # 初始化我们封装的客户端
    llm = OpenAICompatibleClient(
        model=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # --- 2. 初始化任务 ---
    user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
    
    # prompt_history 用于记录整个对话过程（用户输入 + 模型思考 + 工具结果）
    # 这样模型才能“记住”之前发生了什么
    prompt_history = [f"用户请求: {user_prompt}"]

    print(f"👤 用户输入: {user_prompt}\n" + "="*40)

    # --- 3. 开始 ReAct 循环 ---
    # 我们设置最大循环 5 次，防止模型陷入死循环
    for i in range(5): 
        print(f"--- 第 {i+1} 轮循环 ---\n")
        
        # 3.1. 拼接历史记录，形成完整的 Prompt
        full_prompt = "\n".join(prompt_history)
        
        # 3.2. 让 LLM 进行思考 (Thought)
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        
        # --- 清洗数据（进阶处理）---
        # 有时候模型会啰嗦，输出多对 Thought-Action，我们只取第一对
        match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("⚠️ [Warn] 已截断多余的输出，只保留第一步行动。")
        
        print(f"🧠 模型输出:\n{llm_output}\n")
        # 把模型的思考加入历史记录
        prompt_history.append(llm_output)
        
        # 3.3. 解析行动 (Action)
        # 使用正则表达式提取 "Action: " 后面的内容
        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            # 如果没找到 Action，说明模型没按套路出牌，报错并让它重试
            observation = "错误：未能解析到 Action 字段。请严格遵循 'Thought: ... Action: ...' 格式。"
            print(f"👀 Observation: {observation}\n" + "="*40)
            prompt_history.append(f"Observation: {observation}")
            continue
            
        action_str = action_match.group(1).strip()

        # --- 情况 A：任务完成 ---
        if action_str.startswith("Finish"):
            # 提取 Finish[...] 方括号里的最终答案
            try:
                final_answer = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL).group(1)
            except AttributeError:
                final_answer = action_str.replace("Finish[", "").rstrip("]")
            
            print(f"🎉 任务完成！最终答案:\n{final_answer}")
            break # 退出循环
        
        # --- 情况 B：调用工具 ---
        try:
            # 解析函数名和参数，例如 get_weather(city="北京")
            # 1. 提取函数名
            tool_name = re.search(r"(\w+)\(", action_str).group(1)
            # 2. 提取括号里的参数字符串
            args_str = re.search(r"\((.*)\)", action_str).group(1)
            # 3. 将参数字符串转换为字典，例如 {'city': '北京'}
            kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
        except AttributeError:
             observation = "错误：Action 格式解析失败，请检查是否为 function_name(arg=\"value\") 格式。"
             print(f"👀 Observation: {observation}\n" + "="*40)
             prompt_history.append(f"Observation: {observation}")
             continue

        # 3.4. 执行工具 (Execution)
        if tool_name in available_tools:
            try:
                # 真正调用 Python 函数！
                # **kwargs 是 Python 的解包语法，把字典变成关键字参数
                observation = available_tools[tool_name](**kwargs)
            except Exception as e:
                observation = f"错误：工具执行异常 - {e}"
        else:
            observation = f"错误：未定义的工具 '{tool_name}'"

        # 3.5. 记录观察结果 (Observation)
        # 将工具的运行结果反馈给模型
        observation_str = f"Observation: {observation}"
        print(f"👀 {observation_str}\n" + "="*40)
        prompt_history.append(observation_str)

if __name__ == "__main__":
    run_agent()
