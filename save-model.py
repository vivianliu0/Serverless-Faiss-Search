from transformers import (DPRQuestionEncoder, DPRQuestionEncoderTokenizer)

model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base").save_pretrained("./search-lambda/model/")
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base").save_pretrained("./search-lambda/model/")
