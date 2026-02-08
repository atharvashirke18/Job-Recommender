from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from collections import Counter

# Import your existing FastJobRecommender class
# Adjust the import path based on where your inference file is located
# For example, if your file is named 'inference.py' in the same directory:
from recommend_jobs import FastJobRecommender

# Or if it's in a different location, adjust the path accordingly:
# import sys
# sys.path.append('/path/to/your/model/directory')
# from your_inference_file import FastJobRecommender

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend


def convert_to_native_types(obj):
    """
    Recursively convert NumPy and pandas types to native Python types
    for JSON serialization
    """
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # Handle scalar NumPy types
        return obj.item()
    else:
        return obj


# Initialize the recommender once when the server starts
print("Initializing Job Recommender System...")
recommender = FastJobRecommender(model_dir='models', data_dir='datasets')
print("✅ System ready!\n")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Job Recommender API is running',
        'total_jobs': len(recommender.jobs_df)
    }), 200


@app.route('/api/recommend', methods=['POST'])
def recommend_jobs():
    """
    Main recommendation endpoint

    Expected JSON body:
    {
        "skills": "Python, Machine Learning, SQL, Docker",
        "experience_years": 5,
        "expected_salary": 120000,
        "preferred_location": "Remote",  // optional, defaults to "Remote"
        "top_n": 10  // optional, defaults to 10
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        if 'skills' not in data:
            return jsonify({'error': 'skills field is required'}), 400

        if 'experience_years' not in data:
            return jsonify({'error': 'experience_years field is required'}), 400

        if 'expected_salary' not in data:
            return jsonify({'error': 'expected_salary field is required'}), 400

        # Parse skills
        skills_input = data['skills']
        if isinstance(skills_input, str):
            candidate_skills = [s.strip() for s in skills_input.split(',') if s.strip()]
        elif isinstance(skills_input, list):
            candidate_skills = [str(s).strip() for s in skills_input if str(s).strip()]
        else:
            return jsonify({'error': 'skills must be a comma-separated string or array'}), 400

        if not candidate_skills:
            return jsonify({'error': 'At least one skill is required'}), 400

        # Parse experience years
        try:
            experience_years = int(data['experience_years'])
            if experience_years < 0 or experience_years > 50:
                return jsonify({'error': 'experience_years must be between 0 and 50'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'experience_years must be a valid integer'}), 400

        # Parse expected salary
        try:
            expected_salary = int(data['expected_salary'])
            if expected_salary <= 0:
                return jsonify({'error': 'expected_salary must be a positive number'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'expected_salary must be a valid integer'}), 400

        # Optional parameters
        preferred_location = data.get('preferred_location', 'Remote')
        top_n = data.get('top_n', 10)

        try:
            top_n = int(top_n)
            if top_n < 1 or top_n > 100:
                top_n = 10
        except (ValueError, TypeError):
            top_n = 10

        # Get recommendations
        recommendations = recommender.recommend_jobs_batch(
            candidate_skills=candidate_skills,
            experience_years=experience_years,
            expected_salary=expected_salary,
            preferred_location=preferred_location,
            top_n=top_n
        )

        # Convert all NumPy/pandas types to native Python types
        recommendations = convert_to_native_types(recommendations)

        # Calculate summary statistics
        if recommendations:
            avg_prob = float(np.mean([r['selection_probability'] for r in recommendations]))
            high_prob_count = sum(1 for r in recommendations if r['selection_probability'] >= 0.7)
            medium_prob_count = sum(1 for r in recommendations if 0.5 <= r['selection_probability'] < 0.7)
            low_prob_count = sum(1 for r in recommendations if r['selection_probability'] < 0.5)

            # Most common missing skills
            all_missing = []
            for rec in recommendations:
                all_missing.extend(rec['missing_skills'])

            from collections import Counter
            missing_counter = Counter(all_missing)
            top_skills_to_learn = [
                {'skill': skill, 'frequency': count}
                for skill, count in missing_counter.most_common(10)
            ]
        else:
            avg_prob = 0
            high_prob_count = 0
            medium_prob_count = 0
            low_prob_count = 0
            top_skills_to_learn = []

        response = {
            'success': True,
            'input': {
                'skills': candidate_skills,
                'experience_years': experience_years,
                'expected_salary': expected_salary,
                'preferred_location': preferred_location
            },
            'recommendations': recommendations,
            'summary': {
                'total_recommendations': len(recommendations),
                'average_selection_probability': float(avg_prob),
                'high_probability_jobs': high_prob_count,
                'medium_probability_jobs': medium_prob_count,
                'low_probability_jobs': low_prob_count,
                'top_skills_to_learn': top_skills_to_learn
            }
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"Error in recommend_jobs: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/api/skills', methods=['GET'])
def get_available_skills():
    """Get list of all available skills in the system"""
    try:
        if recommender.all_skills:
            return jsonify({
                'success': True,
                'skills': list(recommender.all_skills),
                'total_skills': len(recommender.all_skills)
            }), 200
        else:
            return jsonify({
                'success': True,
                'skills': [],
                'total_skills': 0
            }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/jobs/stats', methods=['GET'])
def get_job_stats():
    """Get statistics about available jobs"""
    try:
        stats = {
            'total_jobs': len(recommender.jobs_df),
            'locations': recommender.jobs_df['location'].value_counts().to_dict(),
            'experience_levels': recommender.jobs_df['experience_level'].value_counts().to_dict(),
            'salary_range': {
                'min': int(recommender.jobs_df['salary_min'].min()),
                'max': int(recommender.jobs_df['salary_max'].max()),
                'avg_min': int(recommender.jobs_df['salary_min'].mean()),
                'avg_max': int(recommender.jobs_df['salary_max'].mean())
            }
        }

        # Convert NumPy types to native Python types
        stats = convert_to_native_types(stats)

        return jsonify({
            'success': True,
            'stats': stats
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)