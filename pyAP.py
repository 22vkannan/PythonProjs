import spacy
from spacy.kb import KnowledgeBase
from transformers import pipeline
import requests

#loading english nlp model from spacy
nlp = spacy.load("en_core_web_sm")

#intitializing Knowledge Base
kb = KnowledgeBase(vocab=nlp.vocab)


#populating knowledge base with entities and descriptions from an external knowledge graph
def populate_knowledge_base(graph_api_url):
    response = requests.get(graph_api_url)
    if response.status_code == 200:
        data = response.json()
        for entity in data['entities']:
            kb.add_entity(entity=entity['id'], freq=entity['frequency'], description=entity['description'])
        kb.set_entities(entity_list=kb.get_entity_strings())
        return True
    else:
        print("Failed to fetch data from the knowledge graph API.")
        return False


#loading pre-trained question pipeline from transformers
qa_pipeline = pipeline("question-answering")


#using spacy to extract question entities
def extract_entities(question):
    doc = nlp(question)
    return [ent.text for ent in doc.ents]


#matching question entities with knowledge base entities
def match_entities(question_entities):
    for entity in question_entities:
        entity_match = kb.get_candidates(entity)
        if entity_match:
            return entity_match[0]  # Return the first matching entity
    return None


#gneerating answer based on matched entity
def generate_answer(question, matched_entity):
    if matched_entity:
        answer = qa_pipeline(question=question, context=matched_entity["description"])
        return answer["answer"]
    return None


#handling user queries
def main():
    #using wikidata external knowledge api
    graph_api_url = "https://query.wikidata.org/sparql"

    #populating entities in knowledge base
    if not populate_knowledge_base(graph_api_url):
        return

    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == "exit":
            break

        #Extracting entities from question
        question_entities = extract_entities(question)

        #matching question entities with entities from knowledge base
        matched_entity = match_entities(question_entities)

        #answer generation
        answer = generate_answer(question, matched_entity)

        #printing final answer
        if answer:
            print("Answer:", answer)
        else:
            print("Sorry, I couldn't find any relevant information.")


if __name__ == "__main__":
    main()
