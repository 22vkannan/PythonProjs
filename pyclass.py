import spacy
from spacy import displacy

#english ner model from spacy
nlp_en = spacy.load("en_core_web_sm")

#multilanguage model from spacy
nlp_multi = spacy.load("xx_ent_wiki_sm")  # Replace "xx" with the language code, e.g., "es" for Spanish

# Sample text for NER in English
text_en = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California."

#sample text
text_multi = "El presidente de Estados Unidos, Joe Biden, visitó México la semana pasada."

doc_en = nlp_en(text_en)

doc_multi = nlp_multi(text_multi)

#extracting entities from processed documents
entities_en = [(ent.text, ent.label_) for ent in doc_en.ents]
entities_multi = [(ent.text, ent.label_) for ent in doc_multi.ents]

#visualizing these entities in English
displacy.serve(doc_en, style="ent")

# Visualize named entities in another language
displacy.serve(doc_multi, style="ent")

#printing entities, labels in English
for entity, label in entities_en:
    print(f"Entity (English): {entity}, Label: {label}")

#printing named entities
for entity, label in entities_multi:
    print(f"Entity (Other Language): {entity}, Label: {label}")
