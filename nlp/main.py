from searchable_index import search_contexts 
from questions_answering import answer_question

print("Type your question:")
try:
    while True:
        question = input(">>> ")
        contexts = search_contexts(question)
        context = "".join(contexts)
        answer = answer_question(question, context)
        print(answer)
except KeyboardInterrupt:
    print('Exit...')