---
license: apache-2.0
language:
- ja
- zh
---
from transformers import MarianMTModel, MarianTokenizer

model = MarianMTModel.from_pretrained(“shun89/opus-mt-ja-zh”)

tokenizer = MarianTokenizer.from_pretrained(“shun89/opus-mt-ja-zh”)


text = '高校生の時、毎週土曜日の午後は友達のリナと一緒に図書館で勉強していました。リナは数学が得意で、いつも私の分からない問題を丁寧に教えてくれました。休み時間には、自販機でコーラを買って廊下で話したり、放課後に近くのカフェでケーキを食べながら未来の夢について話したりしていました。今でもその頃の時間がとても懐かしいです。'

inputs = tokenizer(texts, return_tensors="pt",padding=True, truncation=True, max_length=256)

outputs = model.generate(**inputs)

result= " ".join(tokenizer.batch_decode(outputs, skip_special_tokens=True))

print("待翻译语句：",text)

print("翻译结果：",result)