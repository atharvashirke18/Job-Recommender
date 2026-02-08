# Job Recommender API


### 1. Run the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check

**GET** `/api/health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "message": "Job Recommender API is running",
  "total_jobs": 15000
}
```

### 2. Get Job Recommendations

**POST** `/api/recommend`

Get personalized job recommendations.

**Request Body:**
```json
{
  "skills": "Python, Machine Learning, SQL, Docker",
  "experience_years": 5,
  "expected_salary": 120000,
  "preferred_location": "Remote",
  "top_n": 10
}
```

**Parameters:**
- `skills` (required): Comma-separated string or array of skills
- `experience_years` (required): Integer between 0-50
- `expected_salary` (required): Positive integer
- `preferred_location` (optional): String, defaults to "Remote"
- `top_n` (optional): Integer 1-100, defaults to 10

**Response:**
```json
{
  "success": true,
  "input": {
    "skills": ["Python", "Machine Learning", "SQL", "Docker"],
    "experience_years": 5,
    "expected_salary": 120000,
    "preferred_location": "Remote"
  },
  "recommendations": [
    {
      "job_id": "JOB001",
      "title": "Senior Data Scientist",
      "company": "TechCorp",
      "location": "Remote",
      "experience_level": "Senior Level",
      "salary_min": 110000,
      "salary_max": 150000,
      "required_skills": ["Python", "Machine Learning", "SQL", "TensorFlow"],
      "matching_skills": ["Python", "Machine Learning", "SQL"],
      "missing_skills": ["TensorFlow"],
      "skill_match_count": 3,
      "total_required_skills": 4,
      "match_score": 0.85,
      "selection_probability": 0.78
    }
  ],
  "summary": {
    "total_recommendations": 10,
    "average_selection_probability": 0.65,
    "high_probability_jobs": 3,
    "medium_probability_jobs": 5,
    "low_probability_jobs": 2,
    "top_skills_to_learn": [
      {
        "skill": "TensorFlow",
        "frequency": 7
      },
      {
        "skill": "AWS",
        "frequency": 5
      }
    ]
  }
}
```

### 3. Get Available Skills

**GET** `/api/skills`

Get list of all skills in the system.

**Response:**
```json
{
  "success": true,
  "skills": ["Python", "Java", "JavaScript", "..."],
  "total_skills": 500
}
```

### 4. Get Job Statistics

**GET** `/api/jobs/stats`

Get statistics about available jobs.

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_jobs": 15000,
    "locations": {
      "Remote": 5000,
      "New York": 3000,
      "San Francisco": 2000
    },
    "experience_levels": {
      "Mid Level": 6000,
      "Senior Level": 4500,
      "Entry Level": 3000
    },
    "salary_range": {
      "min": 40000,
      "max": 300000,
      "avg_min": 85000,
      "avg_max": 125000
    }
  }
}
```

## React Frontend Integration



### Example using Fetch

```javascript
async function getRecommendations(skills, experienceYears, expectedSalary) {
  const response = await fetch('http://localhost:5000/api/recommend', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      skills: skills,
      experience_years: experienceYears,
      expected_salary: expectedSalary,
      preferred_location: 'Remote',
      top_n: 10
    })
  });
  
  if (!response.ok) {
    throw new Error('API request failed');
  }
  
  return await response.json();
}
```

### React Component Example

```jsx
import React, { useState } from 'react';
import axios from 'axios';

function JobRecommender() {
  const [skills, setSkills] = useState('');
  const [experience, setExperience] = useState(0);
  const [salary, setSalary] = useState(0);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/api/recommend', {
        skills: skills,
        experience_years: parseInt(experience),
        expected_salary: parseInt(salary),
        top_n: 10
      });

      setRecommendations(response.data.recommendations);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to get recommendations');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Skills (comma-separated)"
          value={skills}
          onChange={(e) => setSkills(e.target.value)}
          required
        />
        <input
          type="number"
          placeholder="Years of Experience"
          value={experience}
          onChange={(e) => setExperience(e.target.value)}
          required
        />
        <input
          type="number"
          placeholder="Expected Salary"
          value={salary}
          onChange={(e) => setSalary(e.target.value)}
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Loading...' : 'Get Recommendations'}
        </button>
      </form>

      {recommendations.length > 0 && (
        <div>
          <h2>Recommended Jobs</h2>
          {recommendations.map((job, index) => (
            <div key={job.job_id} className="job-card">
              <h3>{job.title} at {job.company}</h3>
              <p>Location: {job.location}</p>
              <p>Salary: ${job.salary_min.toLocaleString()} - ${job.salary_max.toLocaleString()}</p>
              <p>Match Score: {(job.match_score * 100).toFixed(1)}%</p>
              <p>Selection Probability: {(job.selection_probability * 100).toFixed(1)}%</p>
              <div>
                <strong>Matching Skills:</strong> {job.matching_skills.join(', ')}
              </div>
              {job.missing_skills.length > 0 && (
                <div>
                  <strong>Skills to Learn:</strong> {job.missing_skills.join(', ')}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default JobRecommender;
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (validation errors)
- `404`: Endpoint not found
- `405`: Method not allowed
- `500`: Internal server error

Error response format:
```json
{
  "success": false,
  "error": "Error message",
  "message": "Detailed error description"
}
```

