from transformers import pipeline

MODEL_NAME = "mvonwyl/distilbert-base-uncased-finetuned-squad2"
nlp = pipeline('question-answering', model=MODEL_NAME, tokenizer=MODEL_NAME)

def answer_question(question, context):
    return nlp({'question': question, 'context': context})