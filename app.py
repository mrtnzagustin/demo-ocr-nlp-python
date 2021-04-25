# coding: utf8
import hug
import spacy
import tika
tika.TikaClientOnly = True # Mark tika python as client online (no localhost server will be mounted)
from tika import parser
import os

print("Loading Models...")
MODELS = {
    "es_core_news_sm": spacy.load("es_core_news_sm")
}
print("Models Loaded!")


def get_model_desc(nlp, model_name):
    """Get human-readable model name, language name and version."""
    lang_cls = spacy.util.get_lang_class(nlp.lang)
    lang_name = lang_cls.__name__
    model_version = nlp.meta["version"]
    return f"{lang_name} - {model_name} (v{model_version})"


@hug.get("/models")
def models():
    return {
        "models": {name: get_model_desc(nlp, name) for name, nlp in MODELS.items()},
        "labels": {name: nlp.pipe_labels for name, nlp in MODELS.items()},
    }


@hug.get("/ner/text")
def ner(text: str):
    """Get entities for file specified"""
    
    try:
        # Obtain the spanish model
        nlp = MODELS["es_core_news_sm"]
        
        # Executes nlp
        doc = nlp(text)

        return {
                "error": False,
                "data": {
                    "nounPhrases": [chunk.text for chunk in doc.noun_chunks],
                    "verbs": [token.lemma_ for token in doc if token.pos_ == "VERB"],
                    "entitites": [ { "label": entity.label_, "text": entity.text } for entity in doc.ents ]
                }
            }
    except Exception as exception:
        return {
            "error": True,
            "message": exception.__str__()
        }


@hug.get("/ner/file")
def ner(filePath: str):
    """Get entities for file specified"""
    
    try:
        # Obtain the spanish model
        nlp = MODELS["es_core_news_sm"]
        
        # Parse of file with apache tika (metadata and content)
        parsed = parser.from_file(filePath, f'{os.getenv("APACHE_TIKA_SERVER")}/tika')
        
        # Get the metadata of the pdf file
        # metadata = parsed['metadata']
        
        # Executes nlp over content
        doc = nlp(parsed['content'])
        
        return {
                "error": False,
                "data": {
                    "nounPhrases": [chunk.text for chunk in doc.noun_chunks],
                    "verbs": [token.lemma_ for token in doc if token.pos_ == "VERB"],
                    "entitites": [ { "label": entity.label_, "text": entity.text } for entity in doc.ents ]
                }
            }
    except Exception as exception:
        return {
            "error": True,
            "message": exception.__str__()
        }
    
