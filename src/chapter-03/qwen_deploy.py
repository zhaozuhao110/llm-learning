import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_id: str):
    """
    加载指定的大语言模型和对应的分词器。
    
    Args:
        model_id (str): 模型在 Hugging Face 上的 ID
        
    Returns:
        tuple: (model, tokenizer, device)
    """
    # 针对 MacMini 优化：优先使用 MPS (Metal Performance Shaders) 进行硬件加速，否则使用 CPU
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"当前使用的计算设备: {device}")

    # 加载分词器
    print("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 加载模型并移动到指定设备
    print("正在加载模型权重 (这可能需要一些时间下载)...")
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    print("模型和分词器加载完成！")
    
    return model, tokenizer, device

def generate_chat_response(prompt: str, model, tokenizer, device) -> str:
    """
    根据用户输入的提示词生成对话回复。
    
    Args:
        prompt (str): 用户输入的提示词
        model: 语言模型实例
        tokenizer: 分词器实例
        device (str): 运行设备
        
    Returns:
        str: 模型生成的文本回复
    """
    # 准备对话输入格式，Qwen模型遵循特定的系统模板
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    # 使用分词器的模板格式化输入文本
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入文本，转换为模型需要的 Tensor 格式
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 使用模型生成回答 (max_new_tokens 限制最多生成 512 个 token)
    # 添加温度采样参数：
    # do_sample=True 开启采样模式
    # temperature=0.1 让模型输出更严谨、确定（不容易出现角色扮演的幻觉）
    # top_p=0.9 配合温度控制多样性
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        top_p=0.9
    )

    # 截取模型新生成的部分（由于 generate 会返回包含输入 prompt 的完整序列）
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码生成的 Token ID 为人类可读的文本，并跳过特殊符号
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

def main():
    """
    主函数，执行模型的加载与对话测试。
    """
    # 指定要部署的模型ID (Qwen1.5-0.5B-Chat 参数量约5亿，适合本地轻量级部署测试)
    model_id = "Qwen/Qwen1.5-0.5B-Chat"
    
    # 1. 加载模型
    model, tokenizer, device = load_model_and_tokenizer(model_id)
    
    # 2. 测试对话
    test_prompt = "你好，请用一段简短的话介绍一下你自己，并告诉我你能做些什么。"
    print(f"\n[用户提问]: {test_prompt}")
    
    response = generate_chat_response(test_prompt, model, tokenizer, device)
    print(f"\n[模型回答]:\n{response}")

if __name__ == "__main__":
    main()
