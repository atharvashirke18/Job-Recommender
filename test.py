"""
Simple test client for the Job Recommender API
"""
import requests
import json

API_BASE_URL = 'http://localhost:5000/api'


def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "=" * 80)
    print("Testing Health Check Endpoint")
    print("=" * 80)

    response = requests.get(f'{API_BASE_URL}/health')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_recommendations():
    """Test the recommendations endpoint"""
    print("\n" + "=" * 80)
    print("Testing Recommendations Endpoint")
    print("=" * 80)

    # Example request
    data = {
        "skills": "Python, Machine Learning, SQL, Docker, AWS",
        "experience_years": 5,
        "expected_salary": 120000,
        "preferred_location": "Remote",
        "top_n": 5
    }

    print(f"\nRequest Data:")
    print(json.dumps(data, indent=2))

    response = requests.post(f'{API_BASE_URL}/recommend', json=data)
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()

        print(f"\n✅ Success! Found {result['summary']['total_recommendations']} recommendations")
        print(f"\nSummary:")
        print(f"  - Average Selection Probability: {result['summary']['average_selection_probability'] * 100:.1f}%")
        print(f"  - High Probability Jobs (≥70%): {result['summary']['high_probability_jobs']}")
        print(f"  - Medium Probability Jobs (50-70%): {result['summary']['medium_probability_jobs']}")
        print(f"  - Low Probability Jobs (<50%): {result['summary']['low_probability_jobs']}")

        print(f"\nTop 3 Recommended Jobs:")
        for i, job in enumerate(result['recommendations'][:3], 1):
            print(f"\n  {i}. {job['title']} at {job['company']}")
            print(f"     Location: {job['location']}")
            print(f"     Salary: ${job['salary_min']:,} - ${job['salary_max']:,}")
            print(f"     Match Score: {job['match_score'] * 100:.1f}%")
            print(f"     Selection Probability: {job['selection_probability'] * 100:.1f}%")
            print(f"     Matching Skills: {', '.join(job['matching_skills'][:5])}")
            if job['missing_skills']:
                print(f"     Missing Skills: {', '.join(job['missing_skills'][:3])}")

        if result['summary']['top_skills_to_learn']:
            print(f"\nTop Skills to Learn:")
            for skill in result['summary']['top_skills_to_learn'][:5]:
                print(f"  - {skill['skill']} (needed for {skill['frequency']} jobs)")
    else:
        print(f"Error: {response.json()}")


def test_skills():
    """Test the skills endpoint"""
    print("\n" + "=" * 80)
    print("Testing Skills Endpoint")
    print("=" * 80)

    response = requests.get(f'{API_BASE_URL}/skills')
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Total Skills: {result['total_skills']}")
        print(f"Sample Skills: {', '.join(result['skills'][:10])}")


def test_job_stats():
    """Test the job statistics endpoint"""
    print("\n" + "=" * 80)
    print("Testing Job Statistics Endpoint")
    print("=" * 80)

    response = requests.get(f'{API_BASE_URL}/jobs/stats')
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        stats = result['stats']

        print(f"\nTotal Jobs: {stats['total_jobs']:,}")

        print(f"\nTop Locations:")
        for location, count in list(stats['locations'].items())[:5]:
            print(f"  - {location}: {count:,} jobs")

        print(f"\nExperience Levels:")
        for level, count in stats['experience_levels'].items():
            print(f"  - {level}: {count:,} jobs")

        print(f"\nSalary Range:")
        print(f"  - Min: ${stats['salary_range']['min']:,}")
        print(f"  - Max: ${stats['salary_range']['max']:,}")
        print(f"  - Avg Min: ${stats['salary_range']['avg_min']:,}")
        print(f"  - Avg Max: ${stats['salary_range']['avg_max']:,}")


def test_error_handling():
    """Test error handling"""
    print("\n" + "=" * 80)
    print("Testing Error Handling")
    print("=" * 80)

    # Test missing required fields
    print("\n1. Testing missing skills:")
    response = requests.post(f'{API_BASE_URL}/recommend', json={
        "experience_years": 5,
        "expected_salary": 120000
    })
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {response.json()}")

    # Test invalid experience
    print("\n2. Testing invalid experience:")
    response = requests.post(f'{API_BASE_URL}/recommend', json={
        "skills": "Python",
        "experience_years": -5,
        "expected_salary": 120000
    })
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {response.json()}")

    # Test invalid salary
    print("\n3. Testing invalid salary:")
    response = requests.post(f'{API_BASE_URL}/recommend', json={
        "skills": "Python",
        "experience_years": 5,
        "expected_salary": 0
    })
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {response.json()}")


def main():
    print("\n" + "=" * 80)
    print("JOB RECOMMENDER API TEST CLIENT")
    print("=" * 80)
    print("\nMake sure the API server is running on http://localhost:5000")

    try:
        # Run all tests
        test_health_check()
        test_recommendations()
        test_skills()
        test_job_stats()
        test_error_handling()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80 + "\n")

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server.")
        print("Please make sure the server is running on http://localhost:5000")
        print("Run: python app.py\n")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")


if __name__ == "__main__":
    main()