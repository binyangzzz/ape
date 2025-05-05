import asyncio
import pandas as pd
from prompt_evaluator import PromptEvaluator
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
if __name__ == "__main__":
    # 加载训练数据
    df_train = pd.read_csv('test.csv')  # 确保包含question和answer列

    # 模型配置
    target_model_name = "deepseek-chat"
    review_model_name = "deepseek-chat"
    api_key = ""  # 替换为实际API密钥

    # 模型参数配置
    target_model_params = {
        "temperature": 0,
        "max_tokens": 1000
    }

    review_model_params = {
        "temperature": 0,
        "max_tokens": 10
    }

    # 评审提示模板路径
    review_prompt_template_path = 'review_prompt_template.txt'

    # 初始化评估器
    evaluator = PromptEvaluator(
        df_train=df_train,
        target_model_name=target_model_name,
        review_model_name=review_model_name,
        api_key=api_key,
        target_model_params=target_model_params,
        review_model_params=review_model_params,
        review_prompt_template_path=review_prompt_template_path
    )

    # 获取用户输入的提示
    prompt = input("请输入需要评估的提示语: ")

    # 运行评估
    asyncio.run(evaluator.main(prompt))