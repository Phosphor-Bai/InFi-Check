import os
import random

file_names = os.listdir('dataset/raw_detnet_wiki_en_test')
random.shuffle(file_names)

types = ['BUS', 'GOV', 'HEA', 'LAW', 'LIF', 'MIL', 'GEN']
counter = {t:0 for t in types}

for file_name in file_names:
	file_type = file_name.split('_')[1]
	assert file_type in types
	if counter[file_type] > 100:
		continue
	lines = open(os.path.join('dataset/raw_detnet_wiki_en_test', file_name), 'r', encoding='utf-8').readlines()
	doc_line = lines[0]
	doc_name = doc_line.split('\t')[2]

	find_end_headword = False
	temp_wiki_document = ''
	para_num = 0
	for line in lines[1:]:
		line = line.strip()
		# print(line)
		if not find_end_headword:
			if line == '#e-headword':
				find_end_headword = True
				in_infobox = False
				in_para = False
			continue	# headword里面也有sent
		if in_infobox:	# infobox的不是正文内容，省去
			if '#e-infobox' in line:
				in_infobox = False
			continue
		if '#s-infobox' in line:
			in_infobox = True
			continue

		if not in_para:	# 开头可能重复一下标题
			if '#s-para' in line:
				in_para = True
				temp_para = ''
			continue

		if '#s-para' in line:
			temp_para = ''
		if '#s-sent' in line:
			line = line.split('\t')
			if not temp_para:
				temp_para = line[3]
			else:
				temp_para += ' ' + line[3]
		if '#e-para' in line:
			para_num += 1
			if not temp_wiki_document:
				temp_wiki_document = temp_para
			else:
				temp_wiki_document += '\n' + temp_para

	if len(temp_wiki_document.split(' ')) < 300 or len(temp_wiki_document.split(' ')) > 1000:
		continue
	doc_name = doc_name.replace('/', ' ')
	with open(os.path.join('selected_dataset/document', f'{doc_name}.txt'), 'w', encoding='utf-8') as f:
		f.write(temp_wiki_document)
	counter[file_type] += 1

print(counter)