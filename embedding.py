import os  # 操作系统接口
from openai import OpenAI  # OpenAI客户端（未使用）
import numpy as np  # 数值计算（未使用）
import aiohttp  # 异步HTTP客户端
import asyncio  # 异步编程
import os  # 重复导入
async def emb(input, model='BAAI/bge-large-zh-v1.5', api_key=os.environ.get('siliconflow_api_key'), url=r"https://api.siliconflow.cn/v1/embeddings", max_retries=3):
    # 构造API请求payload
    payload = {
        "model": model,
        "input": input,
        "encoding_format": "float"
    }
    # 设置请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # 使用异步会话进行API调用
    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries):
            try:
                print('正在发送embedding请求,input:', input)
                # 发送POST请求
                async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        print('embedding请求成功,input:', input)
                        print(len(result['data'][0]['embedding']))
                        return result['data'][0]['embedding']
                    else:
                        # 处理非200状态码
                        print(f"Request failed with status code {response.status}: {await response.text()}")
                        continue
            except asyncio.TimeoutError:
                print("Request timed out.")
                continue
            except Exception as e:
                print(f"An error occurred: {e}")
                continue
        return None
    