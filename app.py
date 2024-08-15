from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('random_forest_model.pkl')  # or use pickle.load()
# List of possible locations
locations = [
    'Bahria Town Karachi',
    'Cantt',
    'Clifton',
    'DHA Defence',
    'Federal B Area',
    'Gulistan-e-Jauhar',
    'Gulshan-e-Iqbal Town',
    'Korangi',
    'Malir',
    'Nazimabad',
    'North Karachi',
    'Shah Faisal Town',
    'Tariq Road',
    'University Road'
]

@app.route('/check')
def check():
    return("working")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        # Extract input data
        baths = data.get('baths')
        bedrooms = data.get('bedrooms')
        area_sq_yards = data.get('AreaSqYards')
        location = data.get('location')

        # Validate input
        if location not in locations:
            return jsonify({'error': 'Invalid location entered. Please select a valid location.'}), 400

        # Create a dictionary for one-hot encoding
        location_dict = {f'location_{loc}': 0 for loc in locations}
        location_dict[f'location_{location}'] = 1

        # Combine all data into one dictionary
        user_data = {
            'baths': baths,
            'bedrooms': bedrooms,
            'AreaSqYards': area_sq_yards,
            **location_dict
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([user_data])

        # Check the DataFrame columns
        print("DataFrame columns:", input_df.columns)

        # Check if the DataFrame has all required columns
        required_columns = pd.Index(['baths', 'bedrooms', 'AreaSqYards'] + [f'location_{loc}' for loc in locations])
        missing_columns = required_columns.difference(input_df.columns)

        if not missing_columns.empty:
            return jsonify({'error': f'Missing columns: {", ".join(missing_columns)}'}), 500

        # Predict
        predicted_price = model.predict(input_df)[0]
        return jsonify({'predicted_price': predicted_price})

    except Exception as e:
        print("Error:", e)  # Debug print
        return jsonify({'error': str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=8080)
