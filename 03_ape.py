import asyncio
import os
import pandas as pd
import re
import aiofiles
import datetime
import aioconsole
import aiohttp
import json
from prompt_evaluator import PromptEvaluator
import backoff


class APD:
    def __init__(self, num_prompts, starting_prompt, df_train, metaprompt_template_path,
                 api_key, generation_model_name, generation_params,
                 target_model_name, target_model_params, review_model_name,
                 review_model_params, review_prompt_template_path):
        self.num_prompts = num_prompts
        self.starting_prompt = starting_prompt
        self.df_train = df_train
        self.metaprompt_template_path = metaprompt_template_path
        self.api_key = api_key
        self.generation_model_name = generation_model_name
        self.generation_params = generation_params
        self.api_base = "https://api.deepseek.com/v1/chat/completions"

        # 创建运行记录文件夹
        self.runs_folder = "runs"
        os.makedirs(self.runs_folder, exist_ok=True)
        self.run_folder = self.create_run_folder()
        self.prompt_history = os.path.join(self.run_folder, 'prompt_history.txt')
        self.prompt_history_chronological = os.path.join(self.run_folder, 'prompt_history_chronological.txt')

        # 初始化评估器
        self.prompt_evaluator = PromptEvaluator(
            df_train=df_train,
            target_model_name=target_model_name,
            review_model_name=review_model_name,
            api_key=api_key,
            target_model_params=target_model_params,
            review_model_params=review_model_params,
            review_prompt_template_path=review_prompt_template_path
        )

        self.user_feedback = ""
        self.best_prompt = starting_prompt
        self.best_accuracy = 0.0

    def create_run_folder(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join(self.runs_folder, f'run_{timestamp}')
        os.makedirs(run_folder, exist_ok=True)
        return run_folder

    def create_prompt_subfolder(self, prompt_number):
        prompt_folder = os.path.join(self.run_folder, f'prompt_{prompt_number}')
        os.makedirs(prompt_folder, exist_ok=True)
        return prompt_folder

    def read_and_sort_prompt_accuracies(self, file_path):
        with open(file_path, 'r') as f:
            content = f.read()

        pattern = re.compile(
            r'<PROMPT>\n<PROMPT_TEXT>\n(.*?)\n</PROMPT_TEXT>\n<ACCURACY>\nAccuracy: ([0-9.]+)\n</ACCURACY>\n</PROMPT>',
            re.DOTALL)
        # pattern = re.compile(
        #     r'<PROMPT[^>]*>[\s\r\n]*<PROMPT_TEXT>[\s\r\n]*(.*?)[\s\r\n]*</PROMPT_TEXT>[\s\r\n]*<ACCURACY>[\s\r\n]*Accuracy:\s*([0-9.,]+)%?[\s\r\n]*</ACCURACY>[\s\r\n]*</PROMPT>',
        #     re.DOTALL)
        matches = pattern.findall(content)

        sorted_prompts = sorted(matches, key=lambda x: float(x[1]))
        return sorted_prompts

    def write_sorted_prompt_accuracies(self, file_path, sorted_prompts):
        sorted_prompts_string = ""
        with open(file_path, 'w') as f:
            for prompt, accuracy in sorted_prompts:
                s = f"<PROMPT>\n<PROMPT_TEXT>\n{prompt}\n</PROMPT_TEXT>\n<ACCURACY>\nAccuracy: {accuracy}\n</ACCURACY>\n</PROMPT>\n\n"
                f.write(s)
                sorted_prompts_string += s
        return sorted_prompts_string

    def update_metaprompt(self, file_path, metaprompt_template_path):
        sorted_prompts = self.read_and_sort_prompt_accuracies(file_path)
        sorted_prompts_string = self.write_sorted_prompt_accuracies(file_path, sorted_prompts)

        with open(metaprompt_template_path, 'r') as f:
            metaprompt_template = f.read()

        metaprompt = metaprompt_template.format(
            prompt_scores=sorted_prompts_string,
            human_feedback=self.user_feedback
        )
        return metaprompt

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def generate_with_backoff(self, metaprompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = [
            {"role": "user", "content": metaprompt}
        ]

        payload = {
            "model": self.generation_model_name,
            "messages": messages,
            **self.generation_params
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    self.api_base,
                    headers=headers,
                    json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API请求失败: {response.status} - {error_text}")

                response_json = await response.json()
                return response_json['choices'][0]['message']['content']

    async def main(self):
        prompt_accuracies = []
        best_prompt = self.starting_prompt
        best_accuracy = 0.0

        for i in range(self.num_prompts + 1):
            await aioconsole.aprint("=" * 100)
            await aioconsole.aprint(f"当前迭代轮次: {i}")

            if i == 0:
                new_prompt = self.starting_prompt
                accuracy = await self.prompt_evaluator.evaluate_prompt(new_prompt)
                best_accuracy = accuracy
                prompt_accuracies.append((new_prompt, accuracy))
            else:
                metaprompt = self.update_metaprompt(self.prompt_history, self.metaprompt_template_path)

                try:
                    response_text = await self.generate_with_backoff(metaprompt)
                except Exception as e:
                    await aioconsole.aprint(f"生成失败: {str(e)}")
                    continue

                await aioconsole.aprint("-" * 100)
                await aioconsole.aprint(response_text)
                await aioconsole.aprint("-" * 100)

                # 解析生成的prompt
                match = re.search(r'\[\[(.*?)\]\]', response_text, re.DOTALL)
                new_prompt = match.group(1).strip() if match else None
                if not new_prompt:
                    await aioconsole.aprint("未检测到有效prompt格式")
                    continue

            # 创建prompt子目录
            prompt_folder = self.create_prompt_subfolder(i)

            # 保存prompt内容
            prompt_file_path = os.path.join(prompt_folder, 'prompt.txt')
            async with aiofiles.open(prompt_file_path, 'w') as f:
                await f.write(new_prompt)

            # 评估prompt
            accuracy = await self.prompt_evaluator.evaluate_prompt(new_prompt)

            prompt_accuracies.append((new_prompt, accuracy))
            await aioconsole.aprint("-" * 100)
            await aioconsole.aprint(f"当前准确率: {accuracy:.2%}")
            await aioconsole.aprint("=" * 100)

            # 更新最佳prompt
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_prompt = new_prompt

            # 记录历史
            async with aiofiles.open(self.prompt_history, 'a') as f:
                await f.write(
                    f"<PROMPT>\n<PROMPT_TEXT>\n{new_prompt}\n</PROMPT_TEXT>\n<ACCURACY>\nAccuracy: {accuracy:.4f}\n</ACCURACY>\n</PROMPT>\n\n")

            async with aiofiles.open(self.prompt_history_chronological, 'a') as f:
                await f.write(f"轮次: {i}\nPrompt: {new_prompt}\n准确率: {accuracy:.2%}\n\n")
                await f.write("=" * 100 + "\n")

            # 保存评估结果
            csv_file_path = os.path.join(prompt_folder, 'evaluation_results.csv')
            evaluation_df = pd.DataFrame({
                "question": self.df_train["question"],
                "answer": self.df_train["answer"],
                "model_response": self.df_train["model_response"],
                "is_correct": self.df_train["is_correct"]
            })
            evaluation_df.to_csv(csv_file_path, index=False)

            # 更新排序后的prompt历史
            sorted_prompts = self.read_and_sort_prompt_accuracies(self.prompt_history)
            self.write_sorted_prompt_accuracies(self.prompt_history, sorted_prompts)

        # 输出最终结果
        starting_accuracy = prompt_accuracies[0][1]
        improvement = best_accuracy - starting_accuracy
        await aioconsole.aprint("=" * 100)
        await aioconsole.aprint(f"最佳prompt: \n{best_prompt}")
        await aioconsole.aprint(f"最高准确率: {best_accuracy:.2%}")
        await aioconsole.aprint(f"准确率提升: {improvement:.2%}")


if __name__ == "__main__":
    # 配置参数
    num_prompts = 5
    starting_prompt = "你是一个数学问题解决专家，请简洁准确回答以下几何问题："
    api_key = ""  # 替换为实际API密钥

    # 模型配置
    generation_model_name = "deepseek-chat"
    generation_params = {
        "temperature": 0.1,
        "max_tokens": 2000
    }

    target_model_name = "deepseek-chat"
    target_model_params = {
        "temperature": 0.1,
        "max_tokens": 1000
    }

    review_model_name = "deepseek-chat"
    review_model_params = {
        "temperature": 0.1,
        "max_tokens": 50
    }

    # 文件路径
    metaprompt_template_path = 'metaprompt_template.txt'
    review_prompt_template_path = 'review_prompt_template.txt'
    df_train = pd.read_csv('train.csv')  # 确保包含question和answer列

    # 初始化APD
    apd = APD(
        num_prompts=num_prompts,
        starting_prompt=starting_prompt,
        df_train=df_train,
        metaprompt_template_path=metaprompt_template_path,
        api_key=api_key,
        generation_model_name=generation_model_name,
        generation_params=generation_params,
        target_model_name=target_model_name,
        target_model_params=target_model_params,
        review_model_name=review_model_name,
        review_model_params=review_model_params,
        review_prompt_template_path=review_prompt_template_path
    )

    asyncio.run(apd.main())