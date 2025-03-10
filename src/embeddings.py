from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional

class EmbeddingModel:
    def __init__(self, 
                 model_name=None, 
                 pinecone_api_key=None, 
                 index_name=None):
        self.model_name = model_name or os.environ.get('MODEL_NAME', 'intfloat/e5-base-v2')
        self.pinecone_api_key = pinecone_api_key or os.environ.get('PINECONE_API_KEY')
        self.index_name = index_name or os.environ.get('INDEX_NAME', 'semantic-search')
        self._model = None
        self._pc = None
        self._index = None
    
    @property
    def pc(self):
        if self._pc is None:
            if not self.pinecone_api_key:
                raise ValueError("PINECONE_API_KEY environment variable is not set")
            self._pc = Pinecone(api_key=self.pinecone_api_key)
        return self._pc
    
    @property
    def index(self):
        if self._index is None:
            self._index = self._pinecone_init()
        return self._index
    
    def _pinecone_init(self):
        try:
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
        except Exception as e:
            raise
        
    def _load_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def get_embedding(self, text: str) -> List[float]:
        model = self._load_model()
        return model.encode(f"query: {text}")
    
    def add_embedding(self, id: str, text: str, metadata: Dict[str, Any] = None, embedding: Optional[List[float]] = None) -> None:
        if embedding is None:
            embedding = self.get_embedding(text)
        
        vector = embedding.tolist() if not isinstance(embedding, list) else embedding
        vector_data = {"id": id, "values": vector}
        if metadata:
            vector_data["metadata"] = metadata
            
        try:
            self.index.upsert(vectors=[vector_data])
        except Exception as e:
            raise
        
    def search(self, query: str, top_k: int = 5, filter: Dict = None) -> Dict:
        embedding = self.get_embedding(query)
        vector = embedding.tolist() if not isinstance(embedding, list) else embedding

        try:
            search_params = {"top_k": top_k}
            if filter:
                search_params["filter"] = filter
                
            results = self.index.query(vector=vector, **search_params)
            return results
        except Exception as e:
            raise

def create_embedding_model(model_name=None, pinecone_api_key=None, index_name=None):
    return EmbeddingModel(
        model_name=model_name,
        pinecone_api_key=pinecone_api_key,
        index_name=index_name
    )

class EmbeddingService:
    def __init__(self, embedding_model=None):
        self.model = embedding_model or create_embedding_model()
    
    def get_embedding(self, text):
        return self.model.get_embedding(text)
        
    def search(self, query, top_k=5, filter=None):
        return self.model.search(query, top_k=top_k, filter=filter)
        
    def add_embedding(self, id, text, metadata=None, embedding=None):
        return self.model.add_embedding(id, text, metadata=metadata, embedding=embedding)
