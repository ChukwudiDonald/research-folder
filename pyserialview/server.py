
from flask import Flask

app = Flask(__name__)

@app.route('/test', methods=['GET', 'POST'])
def home():
    return {"message": "Hello, Flask server is running!"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# cloudflared tunnel --url http://localhost:5000

