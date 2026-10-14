from transformers import AutoTokenizer, pipeline

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

sentences = [
    "I love this product. The quality is excellent!",
    "The delivery was late and customer service was poor."
]

print("TOKENIZER OUTPUT")

for sentence in sentences:
    encoded = tokenizer(
        sentence,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    print("\nSentence:", sentence)
    print("Tokens:", tokenizer.tokenize(sentence))
    print("Input IDs:", encoded["input_ids"].tolist())
    print("Attention Mask:", encoded["attention_mask"].tolist())

print("\nSENTIMENT ANALYSIS")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

sentiment_sentences = [
    "I really enjoyed using this application.",
    "The service was terrible and disappointing.",
    "The product is okay, nothing special."
]

sentiment_results = sentiment_pipeline(sentiment_sentences)

for sentence, result in zip(sentiment_sentences, sentiment_results):
    print("\nText:", sentence)
    print("Label:", result["label"])
    print("Confidence:", round(result["score"], 4))

print("\nNAMED ENTITY RECOGNITION")
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

ner_sentences = [
    "Sundar Pichai works at Google.",
    "Apple opened an office in London.",
    "Microsoft was founded in the United States."
]

ner_results = ner_pipeline(ner_sentences)

for sentence, entities in zip(ner_sentences, ner_results):
    print("Text:", sentence)
    print("Entities:", entities)