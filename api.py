from flask import Flask, request, jsonify
from scripts.process_image import process_image  # import the specific function from your ML model

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    image = request.files['image']
    results = process_image(image)  # use your ML model to process the image
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

