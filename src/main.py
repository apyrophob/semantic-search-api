from flask import Flask
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def api_root():
    return 'Welcome'

@app.route('/embed', methods=['POST'])
def api_embed():
    data = request.get_json()
    query = data['query']
    
    # embedding logic here
    
    return jsonify({ "response": f"Embedded query: {query}", "status": 200 })

@app.route('/search')
def api_search():
    return 'Search'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')