# only pretrained model code: 
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# importing the pre-trained model
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# writing the sentence to translate 
article = "my name is sanjana, i am from hyderabad!"
inputs = tokenizer(article, return_tensors="pt")

# translating the sentence
translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["hin_Deva"], max_length=30
)
print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])

# tel_Telu --> for telugu translation 
# hin_Deva --> for hindi translation