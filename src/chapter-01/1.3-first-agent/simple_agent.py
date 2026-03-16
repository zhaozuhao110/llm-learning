# -*- coding: utf-8 -*-
import json

class SimpleAgent:
    """
    一个简单的智能体，模拟 Thought-Action-Observation 循环。
    为了演示方便，这里我们用简单的规则模拟 LLM 的决策过程，
    在实际应用中，这里会调用 OpenAI 或其他大模型的 API。
    """
    def __init__(self):
        self.history = [] # 保存对话历史
        print("🤖 智能旅行助手已启动...")

    def think(self, user_input, observation=None):
        """
        模拟 LLM 的思考过程 (Thought)
        根据用户输入或上一步的观察结果，决定下一步行动。
        """
        print(f"\n🧠 思考 (Thought): ", end="")
        
        # 场景 1: 刚收到用户请求，还没有天气信息
        if "查询" in user_input and "天气" in user_input and observation is None:
            print("用户想查天气，我需要调用天气工具。")
            return "call_weather_tool", "北京"
            
        # 场景 2: 已经有了天气信息（Observation），需要推荐景点
        elif observation and "晴" in observation:
            print(f"观察到天气是{observation}，适合户外活动，我推荐去故宫。")
            return "call_spot_tool", "户外"
        
        elif observation and ("雨" in observation or "阴" in observation):
             print(f"观察到天气是{observation}，适合室内活动，我推荐去国家博物馆。")
             return "call_spot_tool", "室内"

        # 场景 3: 已经有了景点推荐，任务完成
        elif observation and ("故宫" in observation or "博物馆" in observation):
             print("我已经拿到了景点建议，可以回答用户了。")
             return "final_answer", observation
             
        else:
            return "unknown", None

    def act(self, action_name, action_arg):
        """
        执行动作 (Action)
        """
        print(f"🔨 行动 (Action): 调用工具 [{action_name}] 参数: {action_arg}")
        
        if action_name == "call_weather_tool":
            return self.get_weather(action_arg)
        elif action_name == "call_spot_tool":
            return self.recommend_spot(action_arg)
        elif action_name == "final_answer":
            return f"根据今天的天气情况，我推荐您去：{action_arg}"
        else:
            return "未知动作"

    # --- 模拟的外部工具 (Tools) ---
    def get_weather(self, city):
        """模拟天气查询工具"""
        # 这里可以是真实的 API 调用
        print(f"   (工具运行中... 查询 {city} 天气)")
        return "晴转多云" 

    def recommend_spot(self, category):
        """模拟景点推荐工具"""
        print(f"   (工具运行中... 检索 {category} 景点)")
        if category == "户外":
            return "故宫博物院"
        else:
            return "中国国家博物馆"

    def run(self, user_task):
        """
        运行智能体的主循环
        """
        print(f"\n👤 用户指令: {user_task}")
        self.history.append(f"User: {user_task}")
        
        # Step 1: 思考 -> 查天气
        action, arg = self.think(user_task)
        
        # Step 2: 行动 -> 获得天气观察结果
        observation = self.act(action, arg)
        print(f"👀 观察 (Observation): {observation}")
        
        # Step 3: 再次思考 -> 基于天气推荐景点
        # 这里我们将上一步的观察结果传给思考函数
        action, arg = self.think(user_task, observation)
        
        # Step 4: 再次行动 -> 获得景点建议
        observation = self.act(action, arg)
        print(f"👀 观察 (Observation): {observation}")
        
        # Step 5: 最终思考 -> 生成回答
        action, arg = self.think(user_task, observation)
        final_result = self.act(action, arg)
        
        print(f"\n🤖 最终回答 (Final Answer): {final_result}")

if __name__ == "__main__":
    # 实例化智能体
    agent = SimpleAgent()
    
    # 给定任务
    task = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
    
    # 运行
    agent.run(task)
