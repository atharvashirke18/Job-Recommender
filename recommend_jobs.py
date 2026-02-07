import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple

class FastJobRecommender:
    def __init__(self, model_dir='models', data_dir='datasets'):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model = None
        self.all_skills = None
        self.jobs_df = None
        self.jobs_cache = {}  # Cache for faster lookups
        
        self.load_model()
        self.load_and_index_jobs()
    
    def load_model(self):
        """Load trained model and artifacts"""
        print("Loading model...", end='')
        
        model_path = os.path.join(self.model_dir, 'job_recommender_xgb.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        vocab_path = os.path.join(self.model_dir, 'skill_vocabulary.pkl')
        with open(vocab_path, 'rb') as f:
            self.all_skills = pickle.load(f)
        
        print(" ✓")
    
    def load_and_index_jobs(self):
        """Load all available jobs and create indices for fast lookup"""
        print("Loading jobs...", end='')
        
        # Load from all splits and combine
        jobs_list = []
        for split in ['train', 'cv', 'test']:
            jobs_path = os.path.join(self.data_dir, 'Splits', split, 'jobs.csv')
            if os.path.exists(jobs_path):
                jobs_list.append(pd.read_csv(jobs_path))
        
        self.jobs_df = pd.concat(jobs_list, ignore_index=True)
        self.jobs_df = self.jobs_df.drop_duplicates(subset=['job_id'])
        
        # Pre-process job skills for faster matching
        self.jobs_df['skills_list'] = self.jobs_df['required_skills'].apply(self.extract_skills_list)
        self.jobs_df['skills_set'] = self.jobs_df['skills_list'].apply(set)
        
        print(f" ✓ ({len(self.jobs_df):,} jobs)")
    
    def extract_skills_list(self, skills_str):
        """Convert comma-separated skills to list"""
        if pd.isna(skills_str):
            return []
        return [s.strip() for s in str(skills_str).split(',')]
    
    def create_skill_features(self, job_skills_set, candidate_skills_set, 
                              job_skills_count, candidate_skills_count):
        """Create skill match features - optimized version"""
        common_skills = job_skills_set.intersection(candidate_skills_set)
        skill_match_count = len(common_skills)
        skill_match_ratio = skill_match_count / job_skills_count if job_skills_count > 0 else 0
        candidate_coverage = skill_match_count / candidate_skills_count if candidate_skills_count > 0 else 0
        missing_skills_count = len(job_skills_set - candidate_skills_set)
        missing_skills_ratio = missing_skills_count / job_skills_count if job_skills_count > 0 else 0
        
        return {
            'skill_match_count': skill_match_count,
            'skill_match_ratio': skill_match_ratio,
            'candidate_coverage': candidate_coverage,
            'missing_skills_count': missing_skills_count,
            'missing_skills_ratio': missing_skills_ratio,
            'total_job_skills': job_skills_count,
            'total_candidate_skills': candidate_skills_count
        }
    
    def calculate_match_score(self, job_skills_set, candidate_skills_set, 
                             job_skills_count, exp_match, location_match):
        """Calculate overall match score"""
        skill_match = len(job_skills_set.intersection(candidate_skills_set)) / job_skills_count if job_skills_count > 0 else 0
        
        # Weighted combination
        match_score = 0.5 * skill_match + 0.3 * exp_match + 0.2 * location_match
        return match_score
    
    def recommend_jobs_batch(self, candidate_skills: List[str], experience_years: int, 
                             expected_salary: int, preferred_location: str = 'Remote', 
                             top_n: int = 10) -> List[Dict]:
        """
        Recommend jobs using vectorized operations for speed
        """
        
        candidate_skills_set = set(candidate_skills)
        candidate_skills_count = len(candidate_skills_set)
        
        exp_level_map = {'Entry Level': 2, 'Mid Level': 5, 'Senior Level': 8, 
                        'Lead': 12, 'Principal': 15}
        
        # Pre-filter jobs with at least 1 skill match for speed
        print("  Filtering relevant jobs...", end='')
        self.jobs_df['has_skill_match'] = self.jobs_df['skills_set'].apply(
            lambda x: len(x.intersection(candidate_skills_set)) > 0
        )
        relevant_jobs = self.jobs_df[self.jobs_df['has_skill_match']].copy()
        print(f" {len(relevant_jobs):,} found")
        
        print("  Computing match scores...", end='')
        
        # Vectorized feature computation
        features_list = []
        recommendations = []
        
        for idx, job in relevant_jobs.iterrows():
            job_skills_set = job['skills_set']
            job_skills_count = len(job_skills_set)
            
            # Experience features
            expected_exp = exp_level_map.get(job['experience_level'], 5)
            exp_diff = abs(experience_years - expected_exp)
            exp_match = max(0, 1 - exp_diff / 10)
            exp_over_qualified = 1 if experience_years > expected_exp + 3 else 0
            exp_under_qualified = 1 if experience_years < expected_exp - 2 else 0
            
            # Salary features
            salary_fit = 1.0 if job['salary_min'] <= expected_salary <= job['salary_max'] else 0.0
            salary_below = 1 if expected_salary < job['salary_min'] else 0
            salary_above = 1 if expected_salary > job['salary_max'] else 0
            
            # Location features
            location_match = 1.0 if (job['location'] == 'Remote' or 
                                    job['location'] == preferred_location) else 0.0
            
            # Skill features
            skill_features = self.create_skill_features(
                job_skills_set, candidate_skills_set, 
                job_skills_count, candidate_skills_count
            )
            
            # Calculate match score
            match_score = self.calculate_match_score(
                job_skills_set, candidate_skills_set, 
                job_skills_count, exp_match, location_match
            )
            
            # Overall match as feature
            overall_match = match_score
            
            # Create feature vector
            features = {
                **skill_features,
                'experience_years': experience_years,
                'expected_experience': expected_exp,
                'experience_diff': exp_diff,
                'experience_match': exp_match,
                'exp_over_qualified': exp_over_qualified,
                'exp_under_qualified': exp_under_qualified,
                'expected_salary': expected_salary / 1000,
                'salary_min': job['salary_min'] / 1000,
                'salary_max': job['salary_max'] / 1000,
                'salary_fit': salary_fit,
                'salary_below': salary_below,
                'salary_above': salary_above,
                'location_match': location_match,
                'overall_match': overall_match
            }
            
            features_list.append(features)
            
            # Store for recommendations
            matching_skills = list(job_skills_set.intersection(candidate_skills_set))
            missing_skills = list(job_skills_set - candidate_skills_set)
            
            recommendations.append({
                'job_id': job['job_id'],
                'title': job['title'],
                'company': job['company'],
                'location': job['location'],
                'experience_level': job['experience_level'],
                'salary_min': job['salary_min'],
                'salary_max': job['salary_max'],
                'required_skills': job['skills_list'],
                'match_score': match_score,
                'matching_skills': matching_skills,
                'missing_skills': missing_skills,
                'skill_match_count': len(matching_skills),
                'total_required_skills': job_skills_count
            })
        
        print(" ✓")
        
        # Batch prediction
        print("  Predicting probabilities...", end='')
        features_df = pd.DataFrame(features_list)
        selection_probs = self.model.predict_proba(features_df)[:, 1]
        
        # Add probabilities to recommendations
        for i, rec in enumerate(recommendations):
            rec['selection_probability'] = selection_probs[i]
        
        print(" ✓")
        
        # Sort by selection probability
        recommendations.sort(key=lambda x: x['selection_probability'], reverse=True)
        
        # Clean up
        self.jobs_df.drop('has_skill_match', axis=1, inplace=True, errors='ignore')
        
        return recommendations[:top_n]
    
    def format_recommendation(self, rec: Dict, rank: int) -> str:
        """Format a single recommendation for display"""
        
        output = []
        output.append("=" * 80)
        output.append(f"#{rank} - {rec['title']} at {rec['company']}")
        output.append("=" * 80)
        output.append(f"📍 Location: {rec['location']}")
        output.append(f"💼 Experience Level: {rec['experience_level']}")
        output.append(f"💰 Salary Range: ${rec['salary_min']:,} - ${rec['salary_max']:,}")
        output.append(f"🎯 Match Score: {rec['match_score']*100:.1f}%")
        
        # Skill analysis
        output.append(f"\n📊 Skill Analysis:")
        skill_match_pct = (rec['skill_match_count'] / rec['total_required_skills'] * 100) if rec['total_required_skills'] > 0 else 0
        output.append(f"   Match: {rec['skill_match_count']}/{rec['total_required_skills']} skills ({skill_match_pct:.1f}%)")
        
        if rec['matching_skills']:
            skills_str = ', '.join(rec['matching_skills'][:5])
            output.append(f"   ✅ You Have: {skills_str}")
            if len(rec['matching_skills']) > 5:
                output.append(f"                {', '.join(rec['matching_skills'][5:])}")
        
        if rec['missing_skills']:
            skills_str = ', '.join(rec['missing_skills'][:5])
            output.append(f"   ⚠️  Skills to Learn: {skills_str}")
            if len(rec['missing_skills']) > 5:
                output.append(f"                       {', '.join(rec['missing_skills'][5:])}")
        
        # Selection probability
        prob = rec['selection_probability'] * 100
        output.append(f"\n🎲 Selection Probability: {prob:.1f}%")
        
        if prob >= 70:
            output.append("   💚 Great match! High chance of selection.")
        elif prob >= 50:
            output.append("   💛 Moderate chance. Consider improving skill gaps.")
        else:
            output.append("   ❤️  Lower chance. Focus on skill development.")
        
        return "\n".join(output)
    
    def interactive_recommend(self):
        """Interactive recommendation interface"""
        print("\n" + "=" * 80)
        print("JOB RECOMMENDATION SYSTEM")
        print("=" * 80)
        
        # Get user input
        skills_input = input("\nEnter your skills (comma-separated):\nExample: Python, Machine Learning, SQL, Docker\n> ")
        candidate_skills = [s.strip() for s in skills_input.split(',')]
        
        while True:
            try:
                experience_years = int(input("\nEnter years of experience (0-30): "))
                if 0 <= experience_years <= 30:
                    break
                print("Please enter a value between 0 and 30")
            except ValueError:
                print("Please enter a valid number")
        
        while True:
            try:
                expected_salary = int(input("\nEnter expected salary ($): "))
                if expected_salary > 0:
                    break
                print("Please enter a positive salary")
            except ValueError:
                print("Please enter a valid number")
        
        # Get recommendations
        print("\n🔍 Finding best jobs for you...")
        recommendations = self.recommend_jobs_batch(
            candidate_skills=candidate_skills,
            experience_years=experience_years,
            expected_salary=expected_salary,
            top_n=5
        )
        
        print(f"✅ Found {len(recommendations)} matching jobs!\n")
        
        # Display recommendations
        for idx, rec in enumerate(recommendations, 1):
            print(self.format_recommendation(rec, idx))
            print()
        
        # Summary
        print("=" * 80)
        print("RECOMMENDATION SUMMARY")
        print("=" * 80)
        avg_prob = np.mean([r['selection_probability'] for r in recommendations]) * 100
        print(f"Average Selection Probability: {avg_prob:.1f}%")
        
        high_prob_count = sum(1 for r in recommendations if r['selection_probability'] >= 0.7)
        print(f"High Probability Jobs (≥70%): {high_prob_count}")
        
        # Most common missing skills
        all_missing = []
        for rec in recommendations:
            all_missing.extend(rec['missing_skills'])
        
        if all_missing:
            from collections import Counter
            missing_counter = Counter(all_missing)
            print("\n🎯 Top Skills to Learn (appear most in missing):")
            for skill, count in missing_counter.most_common(5):
                print(f"   • {skill} (needed for {count} jobs)")
        
        print("=" * 80)


def main():
    recommender = FastJobRecommender(model_dir='models', data_dir='datasets')
    recommender.interactive_recommend()


if __name__ == "__main__":
    main()