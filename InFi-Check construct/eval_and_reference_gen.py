import os
import time
import json
import nltk
from openai import OpenAI
from openai import BadRequestError
from difflib import SequenceMatcher
from requests.exceptions import RequestException

def calculate_similarity(s1, s2):
	"""计算两个字符串的相似度"""
	return SequenceMatcher(None, s1, s2).ratio()


with open('summary_eval_prompt/find_support.txt', 'r', encoding='utf-8') as f:
	find_support_prompt = ''.join(f.readlines())

with open('summary_eval_prompt/find_support_format.txt', 'r', encoding='utf-8') as f:
	find_support_format_prompt = ''.join(f.readlines())

with open('summary_eval_prompt/critics.txt', 'r', encoding='utf-8') as f:
	critics_prompt = ''.join(f.readlines())

with open('summary_eval_prompt/critics_format.txt', 'r', encoding='utf-8') as f:
	critics_format_prompt = ''.join(f.readlines())

with open('summary_eval_prompt/critics_with_revise.txt', 'r', encoding='utf-8') as f:
	critics_with_revise_prompt = ''.join(f.readlines())

with open('summary_eval_prompt/critics_with_revise_format.txt', 'r', encoding='utf-8') as f:
	critics_with_revise_format_prompt = ''.join(f.readlines())


client = OpenAI(
	api_key=YOUR_API_KEY,
	base_url=YOUR_BASE_URL
)

model_list = [	# can be changed to other model list
	{
		'name': "gpt-4o",
		'revise': True
	},
	{
		'name': "Qwen/Qwen2.5-72B-Instruct",
		'revise': False
	},
	{
		'name': "gemini-1.5-flash",
		'revise': False
	}
]


def eval_summary(document, summary, source_description):
	# source_description: wikipedia of XXX / BBC XXX news
	find_support_result = []
	eval_result = []
	errors = []
	retries = 0
	while retries < 5:
		try:
			find_support_completion = client.chat.completions.create(
				model="gpt-4o",
				messages=[
						{'role': 'system', 'content': 'You are an helpful assistant that can strictly finds errors in passages.'}, 
						{'role':'user', 'content':f'{find_support_prompt}\n\nDocument:{document}\nSummary:{summary}\n\n{find_support_format_prompt}'}
					]
			)
			summary_sentences = nltk.sent_tokenize(summary)
			find_support_result = find_support_completion.choices[0].message.content
			if find_support_result[:9] == '```python' and find_support_result[-3:] == '```':
				find_support_result = find_support_result[9:-3].strip()
			if find_support_result[:7] == '```json' and find_support_result[-3:] == '```':
				find_support_result = find_support_result[7:-3].strip()
			find_support_result = find_support_result.replace('```', '')
			find_support_result = eval(find_support_result)
			# print(find_support_result)
			assert isinstance(find_support_result, list)
			if len(find_support_result) == len(summary_sentences):
				for i, sent_result in enumerate(find_support_result):
					assert 'sentences from the document' in sent_result
					assert 'summary sentence' in sent_result
					s = sent_result['summary sentence']
					if s != summary_sentences[i] and calculate_similarity(s, summary_sentences[i]) < 0.8:
						errors.append({
							'error': 'Mismatch summary sentence',
							'info': [
								s, summary_sentences[i]
							]
						})
			else:
				errors.append({
					'error': 'Mismatch sentence number',
					'info': [
						find_support_result, summary_sentences
					]
				})
			find_support_result = [{
				'document': ' '.join(r['sentences from the document']), 
				'summary sentence': r['summary sentence'],
				'reference': r['sentences from the document']
			} for r in find_support_result]
			break
		except RequestException as e:	# 网络请求失败，等待一定时间后再次尝试
			print(str(e))
			wait = 2 ** retries  # 指数等待，避免短时间多次请求造成拥挤
			print(f"Request failed, retrying in {wait} seconds...")
			time.sleep(wait)
			retries += 1
		except BadRequestError as e:	# 输入被OpenAI判定为不合法数据，无法获得输出
			print('bad request error', str(e))
			return 'BAD REQUEST ERROR', []
		except AssertionError as e:
			print('assertion error', str(e))
			retries += 1
		except Exception as e:
			print('other error', str(e))
			retries += 1
	if not find_support_result or retries >= 5:
		return 'CANNOT PARSE RESULT', []

	for r in find_support_result:
		revision = ''
		supported_num = 0
		response_num = 0
		revise_count = 0
		critics_result = None

		while revise_count <= 3:
			for model in model_list:
				retries = 0
				while retries < 5:
					try:
						if model['revise']:
							critics_completion = client.chat.completions.create(
								model=model['name'],
								messages=[
										{'role': 'system', 'content': 'You are an helpful assistant that can strictly finds errors in passages.'}, 
										{'role':'user', 'content':f'{critics_with_revise_prompt.replace("SOURCE_DESCRIPTION", source_description)}\n\n\Pieces from the document:\n{r["document"]}\nSummary sentence:\n{r["summary sentence"]}\n\n{critics_with_revise_format_prompt}'}
									]
							)
							critics_result = critics_completion.choices[0].message.content
							critics_result = critics_result.strip()
							if critics_result[:9] == '```python' and critics_result[-3:] == '```':
								# print('<added markdown>')
								critics_result = critics_result[9:-3].strip()
							if critics_result[:7] == '```json' and critics_result[-3:] == '```':
								# print('<added markdown>')
								critics_result = critics_result[7:-3].strip()
							critics_result = critics_result.replace('```', '')
							critics_result = eval(critics_result)
							response_num += 1
							if critics_result['support or not'] == 'YES':
								supported_num += 1
							else:
								revision = critics_result['summary sentence that is supported']
							break
						else:
							critics_completion = client.chat.completions.create(
								model=model['name'],
								messages=[
										{'role': 'system', 'content': 'You are an helpful assistant that can strictly finds errors in passages.'}, 
										{'role':'user', 'content':f'{critics_prompt.replace("SOURCE_DESCRIPTION", source_description)}\n\n\Pieces from the document:\n{r["document"]}\nSummary sentence:\n{r["summary sentence"]}\n\n{critics_format_prompt}'}
									]
							)
							critics_result = critics_completion.choices[0].message.content
							# print(critics_result)
							critics_result = critics_result.strip()
							if critics_result[:9] == '```python' and critics_result[-3:] == '```':
								# print('<added markdown>')
								critics_result = critics_result[9:-3].strip()
							if critics_result[:7] == '```json' and critics_result[-3:] == '```':
								# print('<added markdown>')
								critics_result = critics_result[7:-3].strip()
							critics_result = critics_result.replace('```', '')

							critics_result = eval(critics_result)
							response_num += 1
							if critics_result['support or not'] == 'YES':
								supported_num += 1
							break
					except RequestException as e:	# 网络请求失败，等待一定时间后再次尝试
						print(model['name'], str(e))
						wait = 2 ** retries  # 指数等待，避免短时间多次请求造成拥挤
						print(f"Request failed, retrying in {wait} seconds...")
						time.sleep(wait)
						retries += 1
					except BadRequestError as e:	# 输入被OpenAI判定为不合法数据，无法获得输出
						print(model['name'], str(e))
						return 'BAD REQUEST ERROR', []
					except Exception as e:
						if critics_result:
							print(critics_result)
						print(model['name'], str(e))
						retries += 1
			if supported_num > 0.5 * response_num:	# 一半以上的模型认为没有问题
				eval_result.append({
					'summary sentence': r['summary sentence'],
					'reference': r['reference']
				})
				break
			else:
				if revision:	# 需要再检验一下这个是不是对的
					r['summary sentence'] = revision
					revision = ''
					supported_num = 0
					response_num = 0
					revise_count += 1
				else:
					revise_count += 1
		if revise_count >= 3:
			return 'ERROR IN REVISION', []
	return eval_result, errors

# source_description = 'wikipedia of Oscar Wilde'
# with open('example_wiki.txt', 'r', encoding='utf-8') as f:
# 	document = ''.join(f.readlines())
# 	document = document.replace('\"', '\'')	# 避免生成json之后parse出错

# with open('example_wiki_summary_2.txt', 'r', encoding='utf-8') as f:
# 	summary = ''.join(f.readlines())
# supported_summary = eval_summary(document, summary, source_description)


document_path = 'selected_dataset/document'
summary_path = 'selected_dataset/new_summary'
save_path = 'selected_dataset/new_supported_summary'
reference_save_path = 'selected_dataset/new_reference'
prefixes = ['business_', 'entertainment_', 'politics_', 'sport_', 'tech_']

for summary_name in os.listdir(summary_path):
	if summary_name[-4:] != '.txt':
		continue
	if os.path.exists(os.path.join(save_path, summary_name.replace('_summary', '_supported_summary'))):
		continue
	with open(os.path.join(summary_path, summary_name), 'r', encoding='utf-8') as f:
		summary = ''.join(f.readlines())
		summary = summary.replace('\"', '\'')
		summary = summary.replace('“', '\'')
		summary = summary.replace('”', '\'')
	document_name = summary_name.replace('_summary', '')
	with open(os.path.join(document_path, document_name), 'r', encoding='utf-8') as f:
		document = ''.join(f.readlines())
		document = document.replace('\"', '\'')	# 避免后续LLM生成json之后parse出错
		document = document.replace('“', '\'')
		document = document.replace('”', '\'')
	document_name = document_name[:-4]
	if any(document_name.startswith(prefix) for prefix in prefixes):
		news_type = document_name.split('_')[1]
		source_description = f'BBC {news_type} news'
	else:
		source_description = f'wikipedia of {document_name}'

	print(source_description)
	eval_result, errors = eval_summary(document, summary, source_description)
	if not isinstance(eval_result, list):
		print(eval_result)
		continue

	supported_summary = ' '.join(
		[r['summary sentence'] for r in eval_result]
	)
	reference_result = {
		'find_support_result': eval_result,
		'errors': errors
	}
	with open(os.path.join(save_path, summary_name.replace('_summary', '_supported_summary')), 'w', encoding='utf-8') as f:
		f.write(supported_summary)
	with open(os.path.join(reference_save_path, summary_name.replace('_summary', '_ref')), 'w', encoding='utf-8') as f:
		json.dump(reference_result, f, indent=4, ensure_ascii=False)

