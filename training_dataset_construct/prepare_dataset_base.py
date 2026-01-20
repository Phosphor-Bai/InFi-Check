import os, re
import json
import random
import jsonlines

random.seed(312)	# 用于复现随机打乱
document_path = '../SIS-Fact construct/selected_dataset/document'
summary_path = '../SIS-Fact construct/selected_dataset/supported_summary'
reference_path = '../SIS-Fact construct/selected_dataset/reference'
error_path = '../SIS-Fact construct/selected_dataset/error_dataset'
sft_path = 'sft_dataset/jsonl'
error_type_dict = {
	'predicate': 'Predicate Error', 
	'entity': 'Entity Error', 
	'circumstance': 'Circumstance Error',
	'co-reference': 'Co-reference Error', 
	'discourse link': 'Discourse Link Error'
}
error_sample_num_dict = {
	'Predicate Error':2, 
	'Entity Error':2, 
	'Circumstance Error':1,
	'Co-reference Error':2, 
	'Discourse Link Error':1
}

positive_response_list = [
	'All the sentences are supported by the origin document.',
	'Every sentence in the summary is fully supported by the original document.',
	'All statements in the summary align with the content of the original document.',
	'The sentences in the summary faithfully reflects the original document.',
	'No discrepancies were found; the summary is entirely consistent with the original document.',
	'All information in the summary is corroborated by the original document.',
	'There is complete alignment between the summary and the original document.',
	'All content in the summary is accurately derived from the original document.',
	'Every detail in the summary is rooted in the original document\'s content.',
	'All elements of the summary are directly traceable to the original document.',
	'The summary adheres completely to the original document\'s information.',
	'The summary does not deviate from the facts presented in the original document.',
	'The information in the summary is thoroughly aligned with the original document.',
	'The summary is fully validated against the original document.',
	'There are no unsupported sentences in the summary; it matches the original document.'
]

with open('summary/prompt/sft_prompt.txt', 'r', encoding='utf-8') as f:
	sft_input_format = ''.join(f.readlines())

empty_counter = {k:0 for k in error_sample_num_dict.keys()}
pos_counter = 0
pos_num = 0
neg_counter = 0
neg_num = 0

def make_base_data(input, output):
	data = {
		"text": f"<|start_header_id|>:{input}\n<|end_header_id|>:{output}"
	}
	return data

def prepare_negative_data(document, title):
	def retrieve_error_set(error_folder, document):
		negative_set = []
		def generate_negative_output(error_dict):
			# ---处理modified element---
			add_mark = True
			modified_element = error_dict['modified element']
			# 有可能没找到可以替换的circumstance
			if 'The meaning has not been altered' in error_dict['explanation'] or 'No wrong information' in error_dict['wrong information'] or 'no wrong information' in error_dict['wrong information']:
				return ''
			# co-reference会说一句话
			if error_dict['error type'] == 'Co-reference Error':
				if 'The subject of the new sentence is' in modified_element:
					modified_element = modified_element.replace('The subject of the new sentence is', '').strip()
				elif 'The new pronoun' in modified_element:
					modified_element = modified_element.replace('The new pronoun', '').strip()
				elif ' ' not in modified_element:	# 应该就是只回答了一个代词
					pass
				else:
					# print('Weird data', modified_element)
					pass
			# circumstance会写成The new A(location/time/circumstance) B used to replace the original A C.
			elif error_dict['error type'] == 'Circumstance Error':
				pattern = r"The new (\w+?) (?:(['\"].*?['\"])|([^\s]+(?:\s+[^\s]+)*)) used to replace the original \1 .+"
				match = re.match(pattern, modified_element)
				if match:
					modified_element = match.group(2) or match.group(3)
				else:
					pattern = r"The new circumstance ([^']+|'[^']*') used to replace"
					match = re.search(pattern, modified_element)
					if match:
						modified_element = match.group(1)
					else:
						pattern = r"The new (.+?) used to replace"
						match = re.search(pattern, modified_element)
						if match:
							modified_element = modified_element[0].lower() + modified_element[1:]	# 开头转为小写
							if modified_element[:4] != 'the':
								modified_element = 'the ' + match.group(1)
							# print(modified_element)
							add_mark = False
						else:
							print('direct circumstance:', modified_element)
			specific_begin = ' Specifically, '
			if not modified_element:
				if 'remove' in error_dict['modification explanation'] or 'Remove' in error_dict['modification explanation']:
					specific_begin = ''	# 因为操作是直接删去，所以不需要Specifically这一句
				else:
					print('Empty modified element!')
					print(error_dict)
			elif len(modified_element) >= 0.9 * len(error_dict['modified text']):	# discourse link和predicate中会出现
				# print('---', error_dict['error type'])
				# print(modified_element)
				# print(error_dict['modified text'])
				specific_begin = ''
				modified_element = ''
			else:
				if modified_element[-1] == '.':		# 去掉句号便于添加引号
					modified_element = modified_element[:-1]
				if add_mark and not (modified_element[0]=="\'" and modified_element[-1] == "\'"):
					# if len(modified_element.split(' ')) >= 2:
					# 	print('need mark?', modified_element)
					modified_element = f"\'{modified_element}\'"
				modified_element += '.'

			# ---处理original text---
			# 形如：The original sentence containing the selected XXX: <S1> The sentence containing the XXX used for replacement: <S2>
			original_text = error_dict['original text in summary']
			if error_dict['method'] == 'merging sentences':
				if 'Sentence 1:' in original_text and 'Sentence 2:' in original_text:
					original_text = original_text.replace('Sentence 1:', '').strip()
					original_text = ' '.join([s.strip() for s in original_text.split('Sentence 2:')])
			elif error_dict['method'] == 'swapping numbers':
				if 'The sentence' not in original_text and 'Sentence in' not in original_text:
					print('Out of rules text', original_text)	# 筛选过，应该没有
				if 'The sentence' in original_text:
					original_text = original_text.split('The sentence')
					assert len(original_text) == 2
					original_text = original_text[0]
					assert 'Sentence in' not in original_text
				elif 'Sentence in' not in original_text:
					original_text = original_text.split('Sentence in')
					assert len(original_text) == 2
					original_text = original_text[0]
					assert 'The sentence' not in original_text
				if ':' in original_text:
					prefix = original_text[:original_text.find(':')]
					assert 'The original' in prefix or 'Original' in prefix
					original_text = original_text[original_text.find(':'):].strip()


			output = f"The following part is not supported by the origin document:\n- Location: The error occurs in the following sentence: {error_dict['modified text']}{specific_begin}{modified_element}\n- Explanation: {error_dict['wrong information']}\n- Correction: {original_text}\n- Error Type: {error_dict['error type']}"
			output += "\n" + "Therefore, the answer is YES."
			return output

		error_counter = {k:[] for k in error_sample_num_dict.keys()}
		for root, dirs, files in os.walk(error_folder):
			for file in files:
				# 获取文件的完整路径
				instruction_file_path = os.path.join(root, file)
				# 获取文件的相对路径（相对于指令根目录）
				relative_path = os.path.relpath(instruction_file_path, error_folder)

				with open(instruction_file_path, 'r', encoding='utf-8') as f:
					error_dict = ''.join(f.readlines())
					error_dict = eval(error_dict)
				# print(relative_path)
				error_dict['error type'] = error_type_dict[relative_path.split('/')[-3]]
				error_dict['method'] = relative_path.split('/')[-2]
				input = f"{sft_input_format}\nDocument:\n{document}\nSummary:\n{error_dict['full text of modified summary']}"
				output = generate_negative_output(error_dict)
				if output:
					global neg_counter, neg_num
					neg_counter += len(output.split(' '))
					neg_num += 1
					error_counter[error_dict['error type']].append(len(negative_set))	# 记录每个错误类型的样例所在位置
					negative_set.append([make_base_data(input, output), error_dict['error type']])
		
		# 挑选出用来补空数据项的额外数据（用某个有2个样例的错误类型匀一个给0个的）
		empty_error_type = [k for k,v in error_counter.items() if len(v)==0]
		double_error_type = [k for k,v in error_counter.items() if len(v)==2]
		assert len(double_error_type) >= len(empty_error_type)
		random.shuffle(double_error_type)
		for i in range(len(empty_error_type)):
			instance = random.choice([0, 1])
			negative_set[error_counter[double_error_type[i]][instance]][1] = empty_error_type[i]
			empty_counter[empty_error_type[i]] += 1

		random.shuffle(negative_set)
		return negative_set
	
	error_folder = os.path.join(error_path, title, 'intrinsic')
	negative_data = retrieve_error_set(error_folder, document)
	return negative_data

def prepare_positive_data(document, summary, reference):
	input = f"{sft_input_format}\nDocument:\n{document}\nSummary:\n{summary}"
	output = ''
	for i, item in enumerate(reference['find_support_result']):
		if len(item['sentences from the document']) == 1:
			output += f"Summary sentence {i+1} is supported by the following sentence:\n- {item['sentences from the document']}\n\n"
		else:
			sentences= '\n- '.join(item['sentences from the document'])
			output += f"Summary sentence {i+1} is supported by the following sentence:{sentences}\n\n"
	output += f"{random.choice(positive_response_list)}\nTherefore, the answer is NO."
	global pos_counter, pos_num
	pos_counter += len(output.split(' '))
	pos_num += 1
	return make_base_data(input, output)

def prepare_sft_data():
	summaries = os.listdir(summary_path)
	full_data = {}
	for summary_filename in summaries:
		with open(os.path.join(summary_path, summary_filename), 'r', encoding='utf-8') as f:
			summary = ''.join(f.readlines())
		title = summary_filename.replace('_supported_summary.txt', '')
		with open(os.path.join(document_path, f'{title}.txt'), 'r', encoding='utf-8') as f:
			document = ''.join(f.readlines())
		if os.path.exists(os.path.join(reference_path, f'{title}_ref.json')):
			with open(os.path.join(reference_path, f'{title}_ref.json'), 'r', encoding='utf-8') as f:
				reference = json.load(f)
			full_data[title] = {
				'positive': prepare_positive_data(document, summary, reference),
				'negative': prepare_negative_data(document, title)
			}
		else:
			full_data[title] = {
				'negative': prepare_negative_data(document, title)
			}
	print(empty_counter)
	return full_data

def negative_shuffle_and_select_data(full_data):
	sft_train_data = []
	sft_valid_data = []
	sft_test_data = []
	total_size = len(full_data)
	print('total size', total_size)
	valid_size = 100
	test_size = 100
	train_size = total_size - valid_size - test_size
	full_keys = list(full_data.keys())
	train_data = {k:full_data[k] for k in full_keys[:train_size]}
	valid_data = {k:full_data[k] for k in full_keys[train_size:train_size+valid_size]}
	test_data = {k:full_data[k] for k in full_keys[train_size+valid_size:]}
	for title in train_data.keys():
		for neg in train_data[title]['negative']:
			neg_data, error_type = neg
			sft_train_data.append(neg_data)
	random.shuffle(sft_train_data)
	for title in valid_data.keys():
		for neg in valid_data[title]['negative']:
			neg_data, error_type = neg
			sft_valid_data.append(neg_data)
	random.shuffle(sft_valid_data)
	for title in test_data.keys():
		for neg in test_data[title]['negative']:
			neg_data, error_type = neg
			sft_test_data.append(neg_data)
	random.shuffle(sft_test_data)
	return sft_train_data, sft_valid_data, sft_test_data


def few_positive_shuffle_and_select_data(full_data, positive_num):
	'''
	使用全部的负例，然后混入一点点正例
	'''
	sft_train_data = []
	sft_valid_data = []
	sft_test_data = []
	positive_train_data = []
	total_size = len(full_data)
	print('total size', total_size)
	valid_size = 100
	test_size = 100
	train_size = total_size - valid_size - test_size
	full_keys = list(full_data.keys())
	train_data = {k:full_data[k] for k in full_keys[:train_size]}
	valid_data = {k:full_data[k] for k in full_keys[train_size:train_size+valid_size]}
	test_data = {k:full_data[k] for k in full_keys[train_size+valid_size:]}
	for title in train_data.keys():
		positive_train_data.append(train_data[title]['positive'])
		for neg in train_data[title]['negative']:
			neg_data, error_type = neg
			sft_train_data.append(neg_data)
	sft_train_data += random.choices(positive_train_data, k=positive_num)
	random.shuffle(sft_train_data)
	for title in valid_data.keys():
		for neg in valid_data[title]['negative']:
			neg_data, error_type = neg
			sft_valid_data.append(neg_data)
	random.shuffle(sft_valid_data)
	for title in test_data.keys():
		for neg in test_data[title]['negative']:
			neg_data, error_type = neg
			sft_test_data.append(neg_data)
	random.shuffle(sft_test_data)
	return sft_train_data, sft_valid_data, sft_test_data

def shuffle_and_select_data(full_data, round_num=5):
	'''
	修改：改为只训一轮，每个正例匹配五个错误类型各一个负例（shuffle过所以方法均等）
	------
	1:1选取正负例，要求如下：
	1. 确保每轮数据中，每种错误类型的负例数量均匀分布。
	2. 确保每个正例在每轮中对应一个负例，且在五轮中使用的负例各不相同，并覆盖所有错误类型。
	3. 允许某些正例缺失一部分错误类型，此时允许重复选择同一错误类型，但仍然不许选择相同的负例（这部分在挑选样例的时候就做了，错误类型存成了重复使用的类型（实际数据中的错误类型没变））
	'''
	sft_train_data = {i:[] for i in range(round_num)}
	sft_valid_data = []
	sft_test_data = []
	total_size = len(full_data)
	print('total size', total_size)
	valid_size = 100
	test_size = 100
	train_size = total_size - valid_size - test_size
	full_keys = list(full_data.keys())
	random.shuffle(full_keys)
	train_data = {k:full_data[k] for k in full_keys[:train_size]}
	valid_data = {k:full_data[k] for k in full_keys[train_size:train_size+valid_size]}
	test_data = {k:full_data[k] for k in full_keys[train_size+valid_size:]}

	# maximum_num_for_error_type = (train_size+len(error_type_dict.keys())-1) // len(error_type_dict.keys())
	maximum_num_for_error_type = train_size	# 每个正例对于各个类型的错误都匹配一次
	for round in range(round_num):
		error_counter = {v:0 for k,v in error_type_dict.items()}
		for title in train_data.keys():
			if 'positive' in train_data[title]:
				sft_train_data[round].append(train_data[title]['positive'])
		# 不循环的话可能后面有些样例较少的一直由于顺序原因取不到
		while sum([v for k,v in error_counter.items()]) < train_size * len(error_type_dict.keys()):
			# print(error_counter)
			for title in train_data.keys():
				for neg in train_data[title]['negative']:
					neg_data, error_type = neg
					if error_counter[error_type] < maximum_num_for_error_type:
						sft_train_data[round].append(neg_data)
						error_counter[error_type] += 1
						train_data[title]['negative'].remove(neg)
						break
		random.shuffle(sft_train_data[round])
		print(error_counter)

	# maximum_num_for_error_type = (valid_size+len(error_type_dict.keys())-1) // len(error_type_dict.keys())	# 每个样例只取一个负例
	error_counter = {v:0 for k,v in error_type_dict.items()}
	for i, title in enumerate(valid_data.keys()):
		if 'positive' in valid_data[title]:
			sft_valid_data.append(valid_data[title]['positive'])
		for neg in valid_data[title]['negative']:
			neg_data, error_type = neg
			if error_counter[error_type] == i:	# 每个document每个错误类型只测一次
				sft_valid_data.append(neg_data)
				error_counter[error_type] += 1

			# if error_counter[error_type] < maximum_num_for_error_type:
			# 	sft_valid_data.append(neg_data)
			# 	error_counter[error_type] += 1
			# 	break
	print(error_counter)
	random.shuffle(sft_valid_data)

	# maximum_num_for_error_type = (test_size+len(error_type_dict.keys())-1) // len(error_type_dict.keys())	# 每个样例只取一个负例
	error_counter = {v:0 for k,v in error_type_dict.items()}
	for i, title in enumerate(test_data.keys()):
		if 'positive' in test_data[title]:
			sft_test_data.append(test_data[title]['positive'])
		for neg in test_data[title]['negative']:
			neg_data, error_type = neg
			if error_counter[error_type] == i:	# 每个document每个错误类型只测一次
				sft_test_data.append(neg_data)
				error_counter[error_type] += 1
			# if error_counter[error_type] < maximum_num_for_error_type:
			# 	sft_test_data.append(neg_data)
			# 	error_counter[error_type] += 1
			# 	break
	print(error_counter)
	random.shuffle(sft_test_data)

	return sft_train_data, sft_valid_data, sft_test_data

def prepare_dataset_full():
	full_sft_data = prepare_sft_data()
	round_num = 1
	sft_train_data, sft_valid_data, sft_test_data = shuffle_and_select_data(full_data=full_sft_data, round_num=round_num)
	for round, round_data in sft_train_data.items():
		print('Train set size:', len(round_data))
		if round_num == 1:
			with jsonlines.open(os.path.join(sft_path, f'summary_sft_train_pos1neg5_with_ref.jsonl'), mode='w') as writer:
				for item in round_data:
					writer.write(item)
		else:
			with jsonlines.open(os.path.join(sft_path, f'summary_sft_train_with_ref_round{round}.jsonl'), mode='w') as writer:
				for item in round_data:
					writer.write(item)
	with jsonlines.open(os.path.join(sft_path, f'summary_sft_valid_with_ref.jsonl'), mode='w') as writer:
		print('Valid set size:', len(sft_valid_data))
		for item in sft_valid_data:
			writer.write(item)
	with jsonlines.open(os.path.join(sft_path, f'summary_sft_test_with_ref.jsonl'), mode='w') as writer:
		print('Test set size:', len(sft_test_data))
		for item in sft_test_data:
			writer.write(item)
	with jsonlines.open('example_summary_sft_data_with_ref.jsonl', mode='w') as writer:
		for item in sft_train_data[0][:10]:
			writer.write(item)

def prepare_dataset_few_positive(positive_num):
	full_sft_data = prepare_sft_data()
	sft_train_data, sft_valid_data, sft_test_data = few_positive_shuffle_and_select_data(full_sft_data, positive_num=positive_num)
	with jsonlines.open(os.path.join(sft_path, f'pos{positive_num}_summary_sft_train.jsonl'), mode='w') as writer:
		print(len(sft_train_data))
		for item in sft_train_data:
			writer.write(item)

def prepare_dataset_negative():
	full_sft_data = prepare_sft_data()
	sft_train_data, sft_valid_data, sft_test_data = negative_shuffle_and_select_data(full_data=full_sft_data)
	with jsonlines.open(os.path.join(sft_path, f'neg_only_summary_sft_train.jsonl'), mode='w') as writer:
		print(len(sft_train_data))
		for item in sft_train_data:
			writer.write(item)
	with jsonlines.open(os.path.join(sft_path, f'neg_only_summary_sft_valid.jsonl'), mode='w') as writer:
		print(len(sft_valid_data))
		for item in sft_valid_data:
			writer.write(item)
	with jsonlines.open(os.path.join(sft_path, f'neg_only_summary_sft_test.jsonl'), mode='w') as writer:
		print(len(sft_test_data))
		for item in sft_test_data:
			writer.write(item)

prepare_dataset_full()
print('Average positive output length:', float(pos_counter)/pos_num)
print('Average negative output length:', float(neg_counter)/neg_num)