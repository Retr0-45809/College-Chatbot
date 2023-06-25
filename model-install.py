from transformers import BertTokenizerFast, BertForQuestionAnswering

tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")

model = BertForQuestionAnswering.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")

model.save_pretrained("bert-model")
tokenizer.save_pretrained("bert-model")