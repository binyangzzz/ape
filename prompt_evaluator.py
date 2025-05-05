import asyncio
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
import backoff
import aiohttp
import json


class ReviewModelError(Exception):
    """Custom exception for review model errors."""
    pass


class PromptEvaluator:
    def __init__(self, df_train, target_model_name, review_model_name, api_key,
                 target_model_params=None, review_model_params=None,
                 review_prompt_template_path=None):
        """
        初始化评估器
        :param df_train: 训练数据DataFrame
        :param target_model_name: 目标模型名称（DeepSeek）
        :param review_model_name: 评审模型名称（DeepSeek）
        :param api_key: DeepSeek API密钥
        :param target_model_params: 目标模型参数（temperature等）
        :param review_model_params: 评审模型参数
        :param review_prompt_template_path: 评审提示模板路径
        """
        self.df_train = df_train
        self.target_model_name = target_model_name
        self.review_model_name = review_model_name
        self.api_key = api_key
        self.target_model_params = target_model_params or {"temperature": 0.7, "max_tokens": 512}
        self.review_model_params = review_model_params or {"temperature": 0.3, "max_tokens": 128}
        self.review_prompt_template_path = review_prompt_template_path
        self.api_base = "https://api.deepseek.com/v1/chat/completions"

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def generate_response(self, messages, model_name, params):
        """
        通用API请求方法
        :param messages: 消息列表
        :param model_name: 模型名称
        :param params: 模型参数
        :return: 生成的文本
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_name,
            "messages": messages,
            **params
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    self.api_base,
                    headers=headers,
                    json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API请求失败，状态码：{response.status}，错误信息：{error_text}")

                response_json = await response.json()
                return response_json['choices'][0]['message']['content']

    async def generate_target_model_response(self, question, prompt):
        """生成目标模型响应"""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
        return await self.generate_response(messages, self.target_model_name, self.target_model_params)

    async def generate_review_model_response(self, review_prompt):
        """生成评审模型响应"""
        messages = [
            {"role": "user", "content": review_prompt}
        ]
        return await self.generate_response(messages, self.review_model_name, self.review_model_params)

    async def generate_and_review(self, row, prompt):
        try:
            # 生成目标模型响应
            model_response = await self.generate_target_model_response(row["question"], prompt)

            # 读取并格式化评审提示
            with open(self.review_prompt_template_path, 'r') as f:
                review_prompt = f.read().strip().format(
                    model_response=model_response,
                    ground_truth=row['answer']
                )

            # 获取评审结果
            review_result = await self.generate_review_model_response(review_prompt)
            review_result = review_result.strip().lower()

            # 校验响应有效性
            if not model_response:
                raise ReviewModelError("目标模型返回空响应")

            if review_result not in {'true', 'false'}:
                raise ReviewModelError(f"无效评审结果：{review_result}")

            return row.name, model_response, review_result == 'true'

        except Exception as e:
            print(f"处理行 {row.name} 时发生错误：{str(e)}")
            raise

    async def evaluate_prompt(self, prompt):
        """评估提示模板"""
        tasks = [self.generate_and_review(row, prompt) for _, row in self.df_train.iterrows()]

        results = []
        with tqdm_asyncio(total=len(tasks), desc="评估进度") as pbar:
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                pbar.update(1)

        # 处理结果
        evaluation_results = []
        for index, model_response, is_correct in results:
            self.df_train.at[index, 'model_response'] = model_response
            self.df_train.at[index, 'is_correct'] = is_correct
            evaluation_results.append({
                'question': self.df_train.at[index, 'question'],
                'ground_truth': self.df_train.at[index, 'answer'],
                'model_response': model_response,
                'is_correct': is_correct
            })

        # 保存结果
        results_df = pd.DataFrame(evaluation_results)
        results_csv_path = 'evaluation_results.csv'
        results_df.to_csv(results_csv_path, index=False)

        # 计算准确率
        accuracy = self.df_train['is_correct'].mean()
        return accuracy

    async def main(self, prompt):
        """主执行方法"""
        try:
            accuracy = await self.evaluate_prompt(prompt)
            print(f"\n最终准确率：{accuracy:.2%}")
            return accuracy
        except Exception as e:
            print(f"评估过程中断：{str(e)}")
            raise
