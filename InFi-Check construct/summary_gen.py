# -*- coding:utf-8 -*-
import os
import pandas as pd
from tqdm import tqdm
from openai import BadRequestError
from openai import OpenAI
import re, time
import json
from requests.exceptions import RequestException


class ChatGPT_Evaluator():
	def __init__(self, model_name):
		self.client = OpenAI(
			api_key=YOUR_API_KEY,
			base_url=YOUR_BASE_URL
		)
		self.model_name = model_name

	def make_input(self, document, document_words):
		# 系统设定、指令、基础样例
		messages = [
			{
				'role': 'system',
				'content': f'You are a helpful assistant.'
			},
			{
				'role': 'user',
				'content': f'I\'ll provide you with a document. Your task is to write a short summary for this document according to the following requirements:\n1. The length of the summary should be within 100 to {int(min(200, document_words/3+10))} words.\n2. Every sentence in the summary should be directly supported by the content of the document.\n3. For each event, make sure every important entity such as person, location and time is kept in the summary, especially entities that occurs in parallel.\n4. When doing simplification, make sure each complex event or idea remains true to the original meaning. Avoid over-simplification that leads to in-consistency with the origin document.\n\nDocument:\n{document}\n\nPlease directly output the summary without any extra words.'
			}
		]
		return messages

	def get_answer(self, input, document_words):
		label = 'unknown'
		answer = ''
		retries = 0
		while retries < 5:  # 最多重试10次，避免在某个数据上卡死
			try:
				response = self.client.chat.completions.create(
					model=self.model_name,
					messages=input,
					temperature=0.00001
				)
				answer = response.choices[0].message.content
				if 'deepseek-r1' in self.model_name.lower():
					answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
					answer = answer.strip()
				answer = answer.strip()
				# print(answer)
				print('prompt', response.usage.prompt_tokens)
				print('response', response.usage.completion_tokens)
				# 检查长度
				answer_words = len(answer.split(' '))
				print(answer_words)
				if answer_words < 100 or answer_words > min(200, document_words/3+10):
					retries += 1
					continue
				return answer
			except RequestException as e:	# 网络请求失败，等待一定时间后再次尝试
				print(str(e))
				wait = 2 ** retries  # 指数等待，避免短时间多次请求造成拥挤
				print(f"Request failed, retrying in {wait} seconds...")
				time.sleep(wait)
				retries += 1
			except BadRequestError as e:	# 输入被OpenAI判定为不合法数据，无法获得输出
				print(str(e))
				answer = str(e)
				return label + ' ' + answer
			except Exception as e:
				print(str(e))
				answer = str(e)
				return label + ' ' + answer
		return label + ' ' + answer
	
	def process_file(self, input_folder, output_folder):
		for file_name in os.listdir(input_folder):
			if '.txt' not in file_name:
				continue
			with open(os.path.join(input_folder, file_name), 'r', encoding='utf-8') as f:
				document = ''.join(f.readlines())

			document_words = len(document.split(' '))
			print(file_name, document_words)
			# 太短的论文就不要了
			if document_words < 300 or document_words > 1000:
				continue
			if os.path.exists(os.path.join(output_folder, f'{file_name[:-4]}_summary.txt')):
				continue

			message = self.make_input(document, document_words)
			answer = self.get_answer(message, document_words)
			if 'unknown' in answer:
				continue
			# print(answer)
			with open(os.path.join(output_folder, f'{file_name[:-4]}_summary.txt'), 'w', encoding='utf-8') as f:
				f.write(answer)


evaluator = ChatGPT_Evaluator(model_name='deepseek-r1')
evaluator.process_file('selected_dataset/document', 'selected_dataset/new_summary')