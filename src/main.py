from flask import Flask, request, jsonify
from embeddings import EmbeddingService
import uuid

app = Flask(__name__)
embedding_service = EmbeddingService()

@app.route('/')
def api_root():
    return 'Welcome to the Semantic Search API'

@app.route('/embed', methods=['POST'])
def api_embed():
    data = request.get_json()
    query = data['query']
    
    embedding = embedding_service.get_embedding(query)
    id = str(uuid.uuid4())
    embedding_service.add_embedding(id, query, embedding=embedding)
    
    return jsonify({ 
        "response": f"Embedded query: {query}", 
        "status": 200, 
        "embedding": embedding.tolist(),
        "id": id
    })

@app.route('/search', methods=['POST'])
def api_search():
    data = request.get_json()
    query = data.get('query')
    top_k = data.get('top_k', 5)
    
    if not query:
        return jsonify({"error": "No query provided", "status": 400}), 400
    
    results = embedding_service.search(query, top_k=top_k)
    
    return jsonify({
        "response": f"Search results for: {query}",
        "status": 200,
        "results": results.to_dict()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')