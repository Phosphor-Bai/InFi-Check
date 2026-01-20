import os
import random


def process_file(input_folder, folder_type, output_folder):
	counter = 0
	document_list = os.listdir(os.path.join(input_folder, folder_type))
	random.shuffle(document_list)
	for file_name in document_list:
		if '.txt' not in file_name:
			continue
		print(folder_type, file_name)
		with open(os.path.join(input_folder, folder_type, file_name), 'r', encoding='utf-8') as f:
			lines = f.readlines()
			lines = [l.strip() for l in lines]
			lines = [l for l in lines if l]
			title = lines[0].strip().replace('/', '')
			document = '\n'.join(lines[1:])

		document_words = len(document.split(' '))
		# 太短的论文就不要了
		if document_words < 300 or document_words > 1000:
			continue
		else:
			with open(os.path.join(output_folder, f'{folder_type}_{title}.txt'), 'w', encoding='utf-8') as f:
				f.write(document)
			counter += 1
			if counter > 150:
				return

input_path = 'dataset/BBC News Articles'
output_path = 'selected_dataset/document'
folder_types = os.listdir(input_path)
for folder_type in folder_types:
	process_file(input_path, folder_type, output_path)