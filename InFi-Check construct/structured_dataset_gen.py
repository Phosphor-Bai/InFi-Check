# -*- coding:utf-8 -*-
import os, re
import pandas as pd
from tqdm import tqdm
from openai import BadRequestError
from openai import OpenAI
import re, time
import json
from requests.exceptions import RequestException


def parse_txt_to_dict(txt_content):
	# 初始化字典
	result_dict = {
		"Error Type": "",
		"Method": "",
		"Instruction": "",
		"Format": "",
		"Few Shot": False
	}

	# 使用正则表达式提取Error、Method、Instruction和Format
	error_match = re.search(r'Error Type: (.*)', txt_content)
	if error_match:
		result_dict["Error Type"] = error_match.group(1).strip()
	else:
		print('Failure in error parsing')
		return None

	method_match = re.search(r'Method: (.*)', txt_content)
	if method_match:
		result_dict["Method"] = method_match.group(1).strip()
	else:
		print('Failure in method parsing')
		return None
	
	few_shot_match = re.search(r'Few Shot: (.*)', txt_content)
	if few_shot_match:
		few_shot = few_shot_match.group(1).strip()
		assert few_shot in ['Yes', 'No']
		result_dict["Few Shot"] = True if few_shot == 'Yes' else False
	else:
		print('Failure in few shot parsing')
		return None

	instruction_match = re.search(r'Instruction:([\s\S]*?)Format:', txt_content)
	if instruction_match:
		result_dict["Instruction"] = instruction_match.group(1).strip()
	else:
		print('Failure in instruction parsing')
		return None

	if result_dict['Few Shot']:
		format_keys_match = re.search(r'Format:([\s\S]*?)Example Document:', txt_content)
		if format_keys_match:
			result_dict["Format"] = format_keys_match.group(1).strip()
		else:
			print('Failure in format parsing')
			return None
		example_document_match = re.search(r'Example Document:([\s\S]*?)Example Summary:', txt_content)
		if example_document_match:
			result_dict["Example Document"] = example_document_match.group(1).strip()
		else:
			print('Failure in example document parsing')
			return None
		example_summary_match = re.search(r'Example Summary:([\s\S]*?)Example Output:', txt_content)
		if example_summary_match:
			result_dict["Example Summary"] = example_summary_match.group(1).strip()
		else:
			print('Failure in example summary parsing')
			return None
		example_output_match = re.search(r'Example Output:([\s\S]*)', txt_content)
		if example_output_match:
			output = example_output_match.group(1).strip()
			output = str(eval(output))
			result_dict["Example Output"] = output
		else:
			print('Failure in example output parsing')
			return None
	else:
		format_keys_match = re.search(r'Format:([\s\S]*)', txt_content)
		if format_keys_match:
			result_dict["Format"] = format_keys_match.group(1).strip()
		else:
			print('Failure in format parsing')
			return None

	return result_dict


class ChatGPT_Evaluator():
	def __init__(self):
		self.client = OpenAI(
			api_key=YOUR_API_KEY,
			base_url=YOUR_BASE_URL
		)
		self.read_instructions('summary_gen_prompt')
		print(self.instruction_dict.keys())

	def read_instructions(self, instruction_root):
		instruction_dict = {}
		for root, dirs, files in os.walk(instruction_root):
			for file in files:
				# 获取文件的完整路径
				instruction_file_path = os.path.join(root, file)
				# 获取文件的相对路径（相对于指令根目录）
				relative_path = os.path.relpath(instruction_file_path, instruction_root)
				if 'structured_extrinsic' not in relative_path:
					continue
				with open(instruction_file_path, 'r', encoding='utf-8') as f:
					instruction_text = ''.join(f.readlines())
				result_dict = parse_txt_to_dict(instruction_text)
				if not result_dict:
					print(relative_path, 'Parsing error!')
					continue
				result_dict['Relative Path'] = relative_path
				instruction_dict[result_dict['Error Type']+'|'+result_dict['Method']] = result_dict
				
		self.instruction_dict = instruction_dict
				

	def make_input(self, document, summary, instruction_dict):
		# 系统设定、指令、基础样例
		messages = [
			{
				'role': 'system',
				'content': f'You are a helpful assistant.'
			}
		]
		instruction = instruction_dict['Instruction']
		format = instruction_dict['Format']
		if instruction_dict['Few Shot']:
			# 不能在dict中输入括号
			user_prompt = f"Here is a document with a summary (the summary is given in the structure of a Python list, with each element being a string of a sentence). Please create a fake summary based on the origin summary by the following steps:\n{instruction}\nMake sure the new summary should not be fully supported by the document, and not change any other part in the summary besides those associated with the modification.\n\nYou should only respond in format as described below. Do not return anything else. START YOUR RESPONSE WITH '{{'\n\nReturn the result as a Python dictionary with the following keys:\n{format}\nReplace any line breaks in the values with '\n' so that the dictionary can be parsed using eval()."
			# print(user_prompt)
			messages += [
				{
					'role': 'user',
					'content': user_prompt
				},
				{
					'role': 'assistant',
					'content': "Sure! Please give me the documents and the summaries."
				},
				{
					'role': 'user',
					'content': f"Document:\n{instruction_dict['Example Document']}\n\nSummary:\n{instruction_dict['Example Summary']}"
				},
				{
					'role': 'assistant',
					'content': instruction_dict['Example Output']
				},
				{
					'role': 'user',
					'content': f"Document:\n{document}\n\nSummary:\n{summary}"
				},
			]
		else:
			# 不能在dict中输入括号
			user_prompt = f"Here is a document with a summary. Please create a fake summary based on the origin summary by the following steps:\n{instruction}\nMake sure the new summary should not be fully supported by the document, and not change any other part in the summary besides those associated with the modification.\n\nDocument:\n{document}\n\nSummray:\n{summary}\n\nYou should only respond in format as described below. Do not return anything else. START YOUR RESPONSE WITH '{{'\n\nReturn the result as a Python dictionary with the following keys:\n{format}\n\nMake sure the dictionary can be parsed using eval():\nReplace any line breaks in the values with '\n'.\nWrap each string with double quotes (\"), replace any double quotes (\") inside a string with single quotes (\')."
			# print(user_prompt)
			messages.append(
				{
					'role': 'user',
					'content': user_prompt
				}
			)
		# print(messages)
		return messages

	def get_answer(self, input):
		answer = ''
		retries = 0
		while retries < 3:  # 最多重试3次，避免在某个数据上卡死
			try:
				response = self.client.chat.completions.create(
					model='gpt-4o-2024-11-20',
					messages=input,
					temperature=0.00001
				)
				answer = response.choices[0].message.content
				# print(answer)
				answer = answer.replace('```python', '')
				answer = answer.replace('```json', '')
				answer = answer.replace('```', '')
				start_pos = answer.find('{')
				end_pos = answer.rfind('}') + 1
				assert start_pos >= 0 and end_pos > start_pos
				answer = answer[start_pos:end_pos]
				answer = eval(answer)
				return answer
			except RequestException as e:
				print(str(e))
				wait = 2 ** retries 
				print(f"Request failed, retrying in {wait} seconds...")
				time.sleep(wait)
				retries += 1
			except BadRequestError as e:	# OpenAI illegal check
				print(str(e))
				answer = str(e)
				return 'FAIL TO GENERATE DATA'
			except Exception as e:
				if 'I notice you\'ve shared' in answer:	# Claude toxicity check
					return 'FAIL TO GENERATE DATA'
				print(str(e))
				print('----------------Retry num:', retries)
				retries += 1
		return 'FAIL TO GENERATE DATA'
	
	def process_file(self, base_folder):
		exist_error_list = ['Paul Warnke']	# Error parsing data
		for file_name in os.listdir(os.path.join(base_folder, 'short_reference')):
			if '.txt' not in file_name and '.json' not in file_name:
				continue

			if '.json' in file_name:
				document_name = file_name.replace('_ref.json', '')
			else:
				document_name = file_name.replace('_ref.txt', '')
			print(document_name)
			if document_name in exist_error_list:
				continue

			with open(os.path.join(base_folder, 'document', f'{document_name}.txt'), 'r', encoding='utf-8') as f:
				document = ''.join(f.readlines())
				document = document.replace('\"', '\'')	# 避免生成json之后parse出错

			with open(os.path.join(base_folder, 'short_reference', file_name), 'r', encoding='utf-8') as f:
				summary = ''.join(f.readlines())
				summary = eval(summary)
				if summary['errors']:	# 晚点检查一下miss了多少数据
					continue
				summary = ['\"' + s['summary sentence'].replace('\"', '\'') + '\"' for s in summary['find_support_result']]
				summary = '[' + ', '.join(summary) + ']'	# 否则输入容易parse不好

			for instruction_type, instruction in self.instruction_dict.items():
				output_folder = os.path.join(base_folder, 'short_error_dataset', document_name, instruction['Relative Path'].replace('.txt', ''))
				output_file = os.path.join(output_folder, instruction['Method']+'.txt')
				if os.path.exists(output_file):
					continue
				message = self.make_input(document, summary, instruction)
				answer = self.get_answer(message)
				if not isinstance(answer, dict):
					continue
				os.makedirs(output_folder, exist_ok=True)
				with open(output_file, 'w', encoding='utf-8') as f:
					f.write(str(answer))


evaluator = ChatGPT_Evaluator()
evaluator.process_file('selected_dataset')