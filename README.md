# InFi-Check
Original source for [InFi-Check: Interpretable and Fine-Grained Fact-Checking of LLMs](https://arxiv.org/abs/2601.06666).

Our code consists of two parts: first generate all the data based on the InFi-Check pipeline, then use the data to generate jsonlines training dataset.

## InFi-Check construct
Run this folder by:
1. Download BBC News Dataset (https://huggingface.co/datasets/gopalkalpande/bbc-news-summary) and DetNet Wikipedia Dataset (https://github.com/yumoxu/detnet) to `dataset` folder.
2. Use `dataset_bbc.py` and `dataset_detnet_wiki.py` to clean the data.
3. Run `summary_gen.py` to generate summary for each document.
4. Run `eval_and_reference_gen.py` to refine summary and extract grounding document sentences.
5. Run `structured_dataset_gen.py` to generate error data based on InFi-Check.

## training_dataset_construct
Run `prepare_dataset_base.py` to parse the data into jsonlines structure.
