from typing import List, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT


app = FastAPI()


SBERT_MODEL_NAME = 'distilbert-base-nli-mean-tokens'
sentence_transformer_model = SentenceTransformer(SBERT_MODEL_NAME, device="cpu")
keybert_model = KeyBERT(model=sentence_transformer_model)


def get_keybert_model():
    return keybert_model


class KeywordRequest(BaseModel):
    text: str
    top_n: Optional[int] = 10
    ngram_lo: Optional[int] = 1
    ngram_hi: Optional[int] = 2
    diversity: Optional[float] = 0.6


class KeywordResponse(BaseModel):
    keywords: List[str]
    scores: List[float]


@app.get('/')
def read_root():
    return {'hello': 'world'}


@app.post('/keywords')
def create_keywords(
    keyword_request: KeywordRequest, 
    keybert_model = Depends(get_keybert_model)
):

    try_top_n = keyword_request.top_n
    keywords_and_scores = []
    while len(keywords_and_scores) == 0:
        keywords_and_scores = keybert_model.extract_keywords(
            keyword_request.text,
            keyphrase_ngram_range=(keyword_request.ngram_lo, keyword_request.ngram_hi),
            stop_words='english',
            use_mmr=True,
            diversity=keyword_request.diversity,
            top_n=try_top_n,
        )
        try_top_n -= 1
    return KeywordResponse(
        keywords=[el[0] for el in keywords_and_scores],
        scores=[el[1] for el in keywords_and_scores],
    )