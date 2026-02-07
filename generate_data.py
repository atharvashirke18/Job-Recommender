import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Expanded skill sets for more diversity
skill_pool = {
    'data_science': [
        'Python', 'R', 'Machine Learning', 'Deep Learning', 'SQL', 'NoSQL',
        'Data Visualization', 'Statistics', 'TensorFlow', 'PyTorch', 'Keras',
        'Pandas', 'NumPy', 'Scikit-learn', 'Matplotlib', 'Seaborn', 'Tableau',
        'Power BI', 'Spark', 'Hadoop', 'ETL', 'Big Data', 'A/B Testing',
        'NLP', 'Computer Vision', 'Time Series', 'Reinforcement Learning',
        'XGBoost', 'LightGBM', 'Feature Engineering', 'MLOps'
    ],
    'web_dev': [
        'JavaScript', 'TypeScript', 'React', 'Angular', 'Vue.js', 'Node.js',
        'Express.js', 'Next.js', 'HTML', 'CSS', 'SASS', 'Webpack', 'Babel',
        'Redux', 'GraphQL', 'REST API', 'Git', 'MongoDB', 'Firebase',
        'Responsive Design', 'Bootstrap', 'Tailwind CSS', 'jQuery', 'Ajax',
        'WebSockets', 'Progressive Web Apps', 'Chrome DevTools', 'Jest',
        'Cypress', 'Webpack', 'npm', 'Yarn'
    ],
    'backend': [
        'Python', 'Java', 'C#', 'Go', 'Rust', 'PHP', 'Ruby', 'Scala',
        'Spring Boot', 'Django', 'Flask', 'FastAPI', '.NET', 'Ruby on Rails',
        'SQL', 'PostgreSQL', 'MySQL', 'Oracle', 'MongoDB', 'Redis', 'Cassandra',
        'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'Microservices',
        'REST API', 'gRPC', 'RabbitMQ', 'Kafka', 'Nginx', 'Apache',
        'OAuth', 'JWT', 'WebSockets', 'GraphQL', 'SOA'
    ],
    'mobile': [
        'React Native', 'Flutter', 'Swift', 'SwiftUI', 'Kotlin', 'Java',
        'Objective-C', 'Dart', 'iOS Development', 'Android Development',
        'Firebase', 'SQLite', 'Core Data', 'Realm', 'UI/UX', 'Material Design',
        'Human Interface Guidelines', 'API Integration', 'Push Notifications',
        'In-App Purchases', 'Maps Integration', 'Camera Integration',
        'Bluetooth', 'ARKit', 'ARCore', 'Xamarin', 'Ionic', 'Cordova'
    ],
    'devops': [
        'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'CI/CD', 'Jenkins',
        'GitLab CI', 'GitHub Actions', 'Terraform', 'Ansible', 'Chef', 'Puppet',
        'Linux', 'Bash', 'Python', 'Monitoring', 'Prometheus', 'Grafana',
        'ELK Stack', 'Splunk', 'Nagios', 'Git', 'Nginx', 'Apache', 'HAProxy',
        'Load Balancing', 'Networking', 'Security', 'CloudFormation', 'Helm'
    ],
    'cloud': [
        'AWS', 'Azure', 'GCP', 'Lambda', 'EC2', 'S3', 'RDS', 'DynamoDB',
        'CloudFront', 'API Gateway', 'ECS', 'EKS', 'CloudWatch', 'IAM',
        'VPC', 'Route 53', 'Serverless', 'Azure Functions', 'Cloud Functions',
        'App Engine', 'BigQuery', 'Cloud Storage', 'Cloud SQL', 'Terraform',
        'CloudFormation', 'ARM Templates', 'Cost Optimization'
    ],
    'security': [
        'Cybersecurity', 'Penetration Testing', 'Ethical Hacking', 'OWASP',
        'Firewalls', 'IDS/IPS', 'SIEM', 'Encryption', 'PKI', 'SSL/TLS',
        'OAuth', 'SAML', 'Zero Trust', 'Compliance', 'ISO 27001', 'SOC 2',
        'GDPR', 'HIPAA', 'Network Security', 'Application Security',
        'Cloud Security', 'Incident Response', 'Threat Intelligence'
    ],
    'blockchain': [
        'Blockchain', 'Ethereum', 'Solidity', 'Smart Contracts', 'Web3.js',
        'Hyperledger', 'Bitcoin', 'Cryptocurrency', 'DApps', 'NFT',
        'Consensus Algorithms', 'Cryptography', 'Truffle', 'Hardhat', 'Remix'
    ]
}

# Expanded job titles
job_titles = {
    'data_science': [
        'Data Scientist', 'Senior Data Scientist', 'Lead Data Scientist',
        'ML Engineer', 'Senior ML Engineer', 'AI Researcher', 'AI Engineer',
        'Data Analyst', 'Senior Data Analyst', 'Business Intelligence Analyst',
        'Data Engineer', 'MLOps Engineer', 'Computer Vision Engineer',
        'NLP Engineer', 'Research Scientist'
    ],
    'web_dev': [
        'Frontend Developer', 'Senior Frontend Developer', 'Lead Frontend Developer',
        'Full Stack Developer', 'Senior Full Stack Developer', 'UI Developer',
        'JavaScript Developer', 'React Developer', 'Angular Developer',
        'Vue Developer', 'Web Developer', 'Frontend Architect'
    ],
    'backend': [
        'Backend Developer', 'Senior Backend Developer', 'Lead Backend Developer',
        'Java Developer', 'Senior Java Developer', 'Python Developer',
        'Senior Python Developer', 'Software Engineer', 'Senior Software Engineer',
        'Backend Architect', 'API Developer', 'Microservices Developer'
    ],
    'mobile': [
        'Mobile Developer', 'Senior Mobile Developer', 'iOS Developer',
        'Senior iOS Developer', 'Android Developer', 'Senior Android Developer',
        'Flutter Developer', 'React Native Developer', 'Mobile Architect',
        'Mobile App Developer', 'Hybrid Mobile Developer'
    ],
    'devops': [
        'DevOps Engineer', 'Senior DevOps Engineer', 'Lead DevOps Engineer',
        'Cloud Engineer', 'Senior Cloud Engineer', 'Site Reliability Engineer',
        'Platform Engineer', 'Infrastructure Engineer', 'Build Engineer',
        'Release Engineer', 'DevOps Architect'
    ],
    'cloud': [
        'Cloud Solutions Architect', 'AWS Solutions Architect', 'Azure Architect',
        'GCP Architect', 'Cloud Engineer', 'Cloud Developer', 'Serverless Developer',
        'Cloud Consultant', 'Cloud Infrastructure Engineer'
    ],
    'security': [
        'Security Engineer', 'Senior Security Engineer', 'Security Analyst',
        'Cybersecurity Analyst', 'Penetration Tester', 'Security Architect',
        'Security Consultant', 'Information Security Manager', 'SOC Analyst'
    ],
    'blockchain': [
        'Blockchain Developer', 'Smart Contract Developer', 'Blockchain Engineer',
        'Blockchain Architect', 'Web3 Developer', 'Solidity Developer'
    ]
}

# Expanded companies (100+ companies)
companies = [
    'Google', 'Microsoft', 'Amazon', 'Meta', 'Apple', 'Netflix', 'Adobe', 'Salesforce',
    'LinkedIn', 'Twitter', 'Uber', 'Airbnb', 'Spotify', 'Tesla', 'Oracle',
    'IBM', 'Intel', 'NVIDIA', 'AMD', 'Qualcomm', 'Cisco', 'VMware', 'Dell',
    'HP', 'Samsung', 'Sony', 'LG', 'Panasonic', 'Toshiba', 'Hitachi',
    'Accenture', 'Deloitte', 'PwC', 'EY', 'KPMG', 'Capgemini', 'Infosys', 'TCS',
    'Wipro', 'Cognizant', 'HCL', 'Tech Mahindra', 'Genpact', 'DXC Technology',
    'SAP', 'Siemens', 'Bosch', 'GE', 'Honeywell', 'Schneider Electric',
    'Goldman Sachs', 'JPMorgan Chase', 'Morgan Stanley', 'Bank of America',
    'Citigroup', 'Wells Fargo', 'HSBC', 'Barclays', 'Credit Suisse',
    'PayPal', 'Square', 'Stripe', 'Coinbase', 'Robinhood', 'Visa', 'Mastercard',
    'Walmart', 'Target', 'Costco', 'Home Depot', 'Lowe\'s', 'Best Buy',
    'Starbucks', 'McDonald\'s', 'Chipotle', 'Domino\'s', 'Yum Brands',
    'Pfizer', 'Johnson & Johnson', 'Moderna', 'AstraZeneca', 'Novartis',
    'SpaceX', 'Blue Origin', 'Boeing', 'Lockheed Martin', 'Northrop Grumman',
    'Zoom', 'Slack', 'Atlassian', 'ServiceNow', 'Workday', 'Snowflake',
    'Databricks', 'MongoDB', 'Redis Labs', 'Elastic', 'Confluent', 'HashiCorp'
]

# Expanded locations (50+ cities)
locations = [
    'Remote', 'San Francisco, CA', 'New York, NY', 'Seattle, WA', 'Austin, TX',
    'Boston, MA', 'Los Angeles, CA', 'Chicago, IL', 'Denver, CO', 'Portland, OR',
    'Atlanta, GA', 'Miami, FL', 'Dallas, TX', 'Houston, TX', 'Phoenix, AZ',
    'San Diego, CA', 'Philadelphia, PA', 'Washington, DC', 'Detroit, MI',
    'Minneapolis, MN', 'Nashville, TN', 'Charlotte, NC', 'Raleigh, NC',
    'London, UK', 'Berlin, Germany', 'Paris, France', 'Amsterdam, Netherlands',
    'Dublin, Ireland', 'Stockholm, Sweden', 'Copenhagen, Denmark', 'Zurich, Switzerland',
    'Toronto, Canada', 'Vancouver, Canada', 'Montreal, Canada', 'Singapore',
    'Tokyo, Japan', 'Seoul, South Korea', 'Hong Kong', 'Sydney, Australia',
    'Melbourne, Australia', 'Bangalore, India', 'Mumbai, India', 'Hyderabad, India',
    'Pune, India', 'Delhi, India', 'Chennai, India', 'Dubai, UAE', 'Tel Aviv, Israel'
]

experience_levels = ['Entry Level', 'Mid Level', 'Senior Level', 'Lead', 'Principal']

# First and last names for realistic candidate names
first_names = [
    'James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph',
    'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica',
    'Sarah', 'Karen', 'Nancy', 'Lisa', 'Margaret', 'Betty', 'Sandra', 'Ashley',
    'Emily', 'Emma', 'Madison', 'Olivia', 'Sophia', 'Ava', 'Isabella', 'Mia',
    'Christopher', 'Daniel', 'Matthew', 'Anthony', 'Mark', 'Donald', 'Steven', 'Paul',
    'Andrew', 'Joshua', 'Kenneth', 'Kevin', 'Brian', 'George', 'Edward', 'Ronald',
    'Raj', 'Priya', 'Amit', 'Neha', 'Rahul', 'Anjali', 'Vikram', 'Pooja',
    'Wei', 'Li', 'Chen', 'Zhang', 'Wang', 'Liu', 'Yang', 'Huang',
    'Mohammed', 'Ali', 'Omar', 'Fatima', 'Aisha', 'Hassan', 'Ibrahim', 'Yusuf'
]

last_names = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
    'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas',
    'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Thompson', 'White', 'Harris',
    'Sharma', 'Patel', 'Kumar', 'Singh', 'Gupta', 'Reddy', 'Nair', 'Iyer',
    'Chen', 'Wang', 'Li', 'Zhang', 'Liu', 'Yang', 'Huang', 'Wu',
    'Kim', 'Park', 'Choi', 'Jung', 'Kang', 'Cho', 'Yoon', 'Jang',
    'Nguyen', 'Tran', 'Le', 'Pham', 'Hoang', 'Vu', 'Dang', 'Bui',
    'Khan', 'Ahmed', 'Ali', 'Hassan', 'Hussein', 'Rahman', 'Abdullah', 'Ibrahim'
]

def generate_jobs(n_jobs=150000, chunk_size=10000):
    print(f"\nGenerating {n_jobs:,} job postings...")
    all_categories = list(skill_pool.keys())
    chunks = []

    for chunk_num in range(0, n_jobs, chunk_size):
        jobs = []
        end = min(chunk_num + chunk_size, n_jobs)

        for i in range(chunk_num, end):
            category = random.choice(all_categories)
            num_skills = random.randint(5, 12)
            skills = random.sample(skill_pool[category], k=min(num_skills, len(skill_pool[category])))

            job = {
                'job_id': f'JOB{i+1:07d}',
                'title': random.choice(job_titles[category]),
                'company': random.choice(companies),
                'location': random.choice(locations),
                'experience_level': random.choice(experience_levels),
                'required_skills': ', '.join(skills),
                'salary_min': random.randint(50, 180) * 1000,
                'salary_max': random.randint(190, 350) * 1000,
                'category': category,
                'posted_date': (datetime.now() - timedelta(days=random.randint(0, 180))).strftime('%Y-%m-%d')
            }
            jobs.append(job)

        chunks.append(pd.DataFrame(jobs))
        print(f"  Generated {end:,}/{n_jobs:,} jobs")

    jobs_df = pd.concat(chunks, ignore_index=True)
    jobs_df.to_csv(os.path.join(DATASET_DIR, 'jobs.csv'), index=False)
    print(f"  ✓ Saved {len(jobs_df):,} jobs to jobs.csv")

    return jobs_df


def generate_candidates(n_candidates=200000, chunk_size=10000):
    print(f"\nGenerating {n_candidates:,} candidate profiles...")
    all_categories = list(skill_pool.keys())
    chunks = []

    for chunk_num in range(0, n_candidates, chunk_size):
        candidates = []
        end = min(chunk_num + chunk_size, n_candidates)

        for i in range(chunk_num, end):
            num_categories = random.randint(1, 3)
            categories = random.sample(all_categories, k=num_categories)
            all_skills = []

            for cat in categories:
                num_skills = random.randint(4, 10)
                all_skills.extend(random.sample(skill_pool[cat], k=min(num_skills, len(skill_pool[cat]))))

            candidate = {
                'candidate_id': f'CAND{i+1:07d}',
                'name': f'{random.choice(first_names)} {random.choice(last_names)}',
                'skills': ', '.join(set(all_skills)),
                'experience_years': random.randint(0, 20),
                'education': random.choice(['Bachelors', 'Masters', 'PhD', 'Bootcamp', 'Self-taught']),
                'preferred_location': random.choice(locations),
                'expected_salary': random.randint(60, 250) * 1000,
                'profile_created': (datetime.now() - timedelta(days=random.randint(0, 730))).strftime('%Y-%m-%d')
            }
            candidates.append(candidate)

        chunks.append(pd.DataFrame(candidates))
        print(f"  Generated {end:,}/{n_candidates:,} candidates")

    candidates_df = pd.concat(chunks, ignore_index=True)
    candidates_df.to_csv(os.path.join(DATASET_DIR, 'candidates.csv'), index=False)
    print(f"  ✓ Saved {len(candidates_df):,} candidates to candidates.csv")

    return candidates_df


def generate_applications(jobs_df, candidates_df, n_applications=200000, chunk_size=10000):
    print(f"\nGenerating {n_applications:,} application records...")
    chunks = []
    exp_map = {'Entry Level': 2, 'Mid Level': 5, 'Senior Level': 8, 'Lead': 12, 'Principal': 15}

    for chunk_num in range(0, n_applications, chunk_size):
        applications = []
        end = min(chunk_num + chunk_size, n_applications)

        for i in range(chunk_num, end):
            job = jobs_df.sample(1).iloc[0]
            candidate = candidates_df.sample(1).iloc[0]

            job_skills = set(job['required_skills'].split(', '))
            candidate_skills = set(candidate['skills'].split(', '))

            skill_match = len(job_skills.intersection(candidate_skills)) / len(job_skills) if len(job_skills) > 0 else 0
            exp_diff = abs(candidate['experience_years'] - exp_map[job['experience_level']])
            exp_match = max(0, 1 - exp_diff / 10)

            location_match = 1.0 if (job['location'] == 'Remote' or job['location'] == candidate['preferred_location']) else 0.7
            match_score = 0.5 * skill_match + 0.3 * exp_match + 0.2 * location_match

            selected = 1 if random.random() < match_score else 0

            application = {
                'application_id': f'APP{i+1:07d}',
                'job_id': job['job_id'],
                'candidate_id': candidate['candidate_id'],
                'selected': selected,
                'match_score': round(match_score, 3),
                'application_date': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
            }
            applications.append(application)

        chunks.append(pd.DataFrame(applications))
        print(f"  Generated {end:,}/{n_applications:,} applications")

    applications_df = pd.concat(chunks, ignore_index=True)
    applications_df.to_csv(os.path.join(DATASET_DIR, 'applications.csv'), index=False)
    print(f"  ✓ Saved {len(applications_df):,} applications to applications.csv")

    return applications_df
# Main execution
if __name__ == "__main__":
    start_time = datetime.now()

    DATASET_DIR = "datasets"
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Generate datasets
    jobs_df = generate_jobs(n_jobs=150000)
    candidates_df = generate_candidates(n_candidates=200000)
    applications_df = generate_applications(jobs_df, candidates_df, n_applications=200000)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE!")
    print("="*80)
    print(f"Jobs:         {len(jobs_df):,}")
    print(f"Candidates:   {len(candidates_df):,}")
    print(f"Applications: {len(applications_df):,}")
    print(f"\nTotal time:   {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"\nFiles created:")
    print(f"  • jobs_large.csv")
    print(f"  • candidates_large.csv")
    print(f"  • applications_large.csv")
    print("="*80)
