from transformers import pipeline

def answer_question(question, context, model_name="mvonwyl/distilbert-base-uncased-finetuned-squad2"):
    if not hasattr(answer_question, "nlp"):
        answer_question.nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    return answer_question.nlp({'question': question, 'context': context})