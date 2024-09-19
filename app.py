from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load your data
df_pivot = pd.read_csv('poll_data_winner.csv')


@app.route('/data')
def get_data():
    # Convert DataFrame to JSON and return
    return jsonify(df_pivot.to_dict(orient='records'))


@app.route('/')
def serve_html():
    # Serve the HTML file
    return send_from_directory('/', 'index.html')


if __name__ == '__main__':
    app.run(debug=True)
