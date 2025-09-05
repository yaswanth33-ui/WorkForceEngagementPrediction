from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from .utils import predict_single_employee

app = Flask(__name__)

CORS(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('app.html')


@app.route('/api/verify_engagement', methods=['POST', 'GET'])
def verify_engagement():
    if request.method == 'POST':
        try:
            employee_data = request.get_json()
            result = predict_single_employee(employee_data)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        # For GET requests, return the example data structure
        return jsonify({
            "example_data": {
                'absence_reason_code': 4,
                'absence_month': 1,
                'weekday_code': 1,
                'season_indicator': 1,
                'commute_cost': 50.0,
                'commute_distance': 20.0,
                'years_at_company': 5.0,
                'daily_workload': 10.0,
                'performance_target': 85.0,
                'disciplinary_action': 1,
                'education_level': 2,
                'alcohol_consumption': 0,
                'tobacco_use': 0,
                'body_weight_kg': 70.0,
                'body_height_cm': 175.0,
                'bmi_score': 25.0
            }
        })


if __name__ == '__main__':
    app.run(port=5000, debug=True)