from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone, ServerlessSpec

class EmbeddingModel:
    def __init__(self):
        self.model_name = os.environ.get('MODEL_NAME', 'intfloat/e5-base-v2')
        self.pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        self.index_name = os.environ.get('INDEX_NAME', 'semantic-search')
        self._model = None
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self._pinecone_init()
    
    def _pinecone_init(self):
        model = self._load_model()
        sample_embedding = model.encode("query: sample text")
        dimension = len(sample_embedding)

        if self.index_name not in [i.name for i in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        return self.pc.Index(self.index_name)
        
    def _load_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def get_embedding(self, text):
        model = self._load_model()
        return model.encode(f"query: {text}")
    
    def add_embedding(self, id, text, embedding=None):
        if embedding is None:
            embedding = self.get_embedding(text)
        
        vector = embedding.tolist()
        self.index.upsert(vectors=[{"id": id, "values": vector}])
        
    def search(self, query, top_k=5):
        embedding = self.get_embedding(query)
        vector = embedding.tolist()

        results = self.index.query(vector=vector, top_k=top_k)
        return results

default_model = EmbeddingModel()
get_embedding = default_model.get_embedding
search = default_model.search
add_embedding = default_model.add_embedding
