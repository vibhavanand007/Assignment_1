from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Handle form submission    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting user inputs (TV, Radio, Newspaper)
        tv_budget = float(request.form['TV'])
        radio_budget = float(request.form['Radio'])
        newspaper_budget = float(request.form['Newspaper'])
        
        # Prepare input data (only 3 features now)
        input_features = np.array([[tv_budget, radio_budget, newspaper_budget]])

        # Make prediction
        predicted_sales = model.predict(input_features)[0]

        return render_template(
            'index.html',
            prediction_text=f'Estimated Sales: ${predicted_sales:,.2f}'
        )
    except Exception as e:
        return render_template('index.html', error_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
