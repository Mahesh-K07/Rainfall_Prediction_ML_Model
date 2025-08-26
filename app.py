from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('rainfall_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        
        features = [float(x) for x in request.form.values()]
        final = [np.array(features)]
        prediction = model.predict(final)
        return render_template('index.html', prediction_text=f'Predicted Rainfall: {prediction[0]:.2f} mm')
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)

