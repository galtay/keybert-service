# keybert-service

service to extract keywords from text using keybert

* https://download.pytorch.org/whl/torch_stable.html
* https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/

## Downlload sentence transformer model and load from disk

* get zip file (e.g. distilbert-base-nli-mean-tokens.zip)
* unzip to directory `unzip distilbert-base-nli-mean-tokens.zip -d distilbert-base-nli-mean-tokens`

``` python
from sentence_transformers import SentenceTransformer
st = SentenceTransformer('/path/to/distilbert-base-nli-mean-tokens')
```
