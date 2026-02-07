import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import os
from datetime import datetime

class JobRecommenderTrainer:
    def __init__(self, data_dir="datasets/Splits"):
        self.data_dir = data_dir
        self.model = None
        self.label_encoders = {}
        self.all_skills = set()
        
    def load_data(self, split='train'):
        """Load jobs, candidates, and applications data"""
        print(f"\n{'='*80}")
        print(f"Loading {split.upper()} data...")
        print(f"{'='*80}")
        
        jobs_path = os.path.join(self.data_dir, split, 'jobs.csv')
        candidates_path = os.path.join(self.data_dir, split, 'candidates.csv')
        applications_path = os.path.join(self.data_dir, split, 'applications.csv')
        
        jobs_df = pd.read_csv(jobs_path)
        candidates_df = pd.read_csv(candidates_path)
        applications_df = pd.read_csv(applications_path)
        
        print(f"✓ Jobs: {len(jobs_df):,}")
        print(f"✓ Candidates: {len(candidates_df):,}")
        print(f"✓ Applications: {len(applications_df):,}")
        
        # Check class distribution
        pos_count = applications_df['selected'].sum()
        neg_count = len(applications_df) - pos_count
        print(f"✓ Positive samples: {pos_count:,} ({pos_count/len(applications_df)*100:.2f}%)")
        print(f"✓ Negative samples: {neg_count:,} ({neg_count/len(applications_df)*100:.2f}%)")
        
        return jobs_df, candidates_df, applications_df
    
    def extract_skills_list(self, skills_str):
        """Convert comma-separated skills to list"""
        if pd.isna(skills_str):
            return []
        return [s.strip() for s in str(skills_str).split(',')]
    
    def build_skill_vocabulary(self, jobs_df, candidates_df):
        """Build complete skill vocabulary from all data"""
        print("\nBuilding skill vocabulary...")
        
        # Extract all unique skills
        job_skills = jobs_df['required_skills'].apply(self.extract_skills_list)
        candidate_skills = candidates_df['skills'].apply(self.extract_skills_list)
        
        for skills_list in job_skills:
            self.all_skills.update(skills_list)
        for skills_list in candidate_skills:
            self.all_skills.update(skills_list)
        
        self.all_skills = sorted(list(self.all_skills))
        print(f"✓ Found {len(self.all_skills)} unique skills")
        
    def create_skill_features(self, job_skills, candidate_skills):
        """Create skill match features"""
        job_skills_set = set(job_skills)
        candidate_skills_set = set(candidate_skills)
        
        # Skill overlap metrics
        common_skills = job_skills_set.intersection(candidate_skills_set)
        skill_match_count = len(common_skills)
        skill_match_ratio = skill_match_count / len(job_skills_set) if len(job_skills_set) > 0 else 0
        
        # Candidate skill coverage
        candidate_coverage = skill_match_count / len(candidate_skills_set) if len(candidate_skills_set) > 0 else 0
        
        # Missing skills
        missing_skills_count = len(job_skills_set - candidate_skills_set)
        missing_skills_ratio = missing_skills_count / len(job_skills_set) if len(job_skills_set) > 0 else 0
        
        return {
            'skill_match_count': skill_match_count,
            'skill_match_ratio': skill_match_ratio,
            'candidate_coverage': candidate_coverage,
            'missing_skills_count': missing_skills_count,
            'missing_skills_ratio': missing_skills_ratio,
            'total_job_skills': len(job_skills_set),
            'total_candidate_skills': len(candidate_skills_set)
        }
    
    def engineer_features(self, jobs_df, candidates_df, applications_df):
        """Create feature matrix from applications"""
        print("\nEngineering features...")
        
        # Merge applications with jobs and candidates
        merged = applications_df.merge(jobs_df, on='job_id', how='left')
        merged = merged.merge(candidates_df, on='candidate_id', how='left')
        
        print(f"✓ Merged {len(merged):,} records")
        
        features_list = []
        
        for idx, row in merged.iterrows():
            if idx % 10000 == 0:
                print(f"  Processing {idx:,}/{len(merged):,}...", end='\r')
            
            # Extract skills
            job_skills = self.extract_skills_list(row['required_skills'])
            candidate_skills = self.extract_skills_list(row['skills'])
            
            # Skill features
            skill_features = self.create_skill_features(job_skills, candidate_skills)
            
            # Experience features
            exp_level_map = {'Entry Level': 2, 'Mid Level': 5, 'Senior Level': 8, 
                           'Lead': 12, 'Principal': 15}
            expected_exp = exp_level_map.get(row['experience_level'], 5)
            exp_diff = abs(row['experience_years'] - expected_exp)
            exp_match = max(0, 1 - exp_diff / 10)
            exp_over_qualified = 1 if row['experience_years'] > expected_exp + 3 else 0
            exp_under_qualified = 1 if row['experience_years'] < expected_exp - 2 else 0
            
            # Salary features
            salary_min = row['salary_min']
            salary_max = row['salary_max']
            expected_salary = row['expected_salary']
            salary_fit = 1.0 if salary_min <= expected_salary <= salary_max else 0.0
            salary_below = 1 if expected_salary < salary_min else 0
            salary_above = 1 if expected_salary > salary_max else 0
            
            # Location features
            location_match = 1.0 if (row['location'] == 'Remote' or 
                                    row['location'] == row['preferred_location']) else 0.0
            
            # Overall match score (as a feature)
            overall_match = (0.5 * skill_features['skill_match_ratio'] + 
                           0.3 * exp_match + 
                           0.2 * location_match)
            
            # Combined features
            features = {
                **skill_features,
                'experience_years': row['experience_years'],
                'expected_experience': expected_exp,
                'experience_diff': exp_diff,
                'experience_match': exp_match,
                'exp_over_qualified': exp_over_qualified,
                'exp_under_qualified': exp_under_qualified,
                'expected_salary': expected_salary / 1000,  # Scale down
                'salary_min': salary_min / 1000,
                'salary_max': salary_max / 1000,
                'salary_fit': salary_fit,
                'salary_below': salary_below,
                'salary_above': salary_above,
                'location_match': location_match,
                'overall_match': overall_match,
                'selected': row['selected']
            }
            
            features_list.append(features)
        
        print(f"\n✓ Created {len(features_list):,} feature vectors")
        
        return pd.DataFrame(features_list)
    
    def train(self, X_train, y_train, X_cv, y_cv):
        """Train XGBoost model with proper hyperparameters for imbalanced data"""
        print("\n" + "="*80)
        print("TRAINING XGBOOST MODEL")
        print("="*80)
        
        # Calculate scale_pos_weight for class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        
        print(f"\nClass imbalance ratio: {scale_pos_weight:.2f}")
        print(f"Positive samples: {pos_count:,}")
        print(f"Negative samples: {neg_count:,}")
        
        # Improved XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'max_depth': 5,  # Reduced to prevent overfitting
            'learning_rate': 0.05,  # Lower learning rate
            'n_estimators': 100,  # Fewer trees initially
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1,  # Increased regularization
            'min_child_weight': 3,  # Increased to prevent overfitting
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
            'random_state': 42,
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'early_stopping_rounds': 20  # Stop if no improvement
        }
        
        print("\nModel Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Separate early stopping parameter
        early_stopping_rounds = params.pop('early_stopping_rounds')
        
        # Create and train model
        self.model = xgb.XGBClassifier(**params)
        
        print(f"\nTraining on {len(X_train):,} samples...")
        print(f"Validation on {len(X_cv):,} samples...\n")
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_cv, y_cv)],
            verbose=10
        )
        
        print("\n✓ Training complete!")
        
    def evaluate(self, X, y, dataset_name='Test'):
        """Evaluate model performance"""
        print("\n" + "="*80)
        print(f"{dataset_name.upper()} SET EVALUATION")
        print("="*80)
        
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        accuracy = accuracy_score(y, y_pred)
        
        # Handle cases where precision/recall might fail
        try:
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
        except:
            precision = recall = f1 = 0.0
            
        auc = roc_auc_score(y, y_pred_proba)
        
        print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {auc:.4f}")
        
        # Feature importance
        print("\n" + "-"*80)
        print("TOP 10 IMPORTANT FEATURES")
        print("-"*80)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(10).iterrows():
            print(f"{row['feature']:30s} {row['importance']:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def save_model(self, output_dir='models'):
        """Save trained model and artifacts"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("SAVING MODEL")
        print("="*80)
        
        # Save XGBoost model
        model_path = os.path.join(output_dir, 'job_recommender_xgb.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved to {model_path}")
        
        # Save skill vocabulary
        vocab_path = os.path.join(output_dir, 'skill_vocabulary.pkl')
        with open(vocab_path, 'wb') as f:
            pickle.dump(self.all_skills, f)
        print(f"✓ Skill vocabulary saved to {vocab_path}")
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_skills': len(self.all_skills),
            'model_type': 'XGBoost',
            'version': '2.0'
        }
        metadata_path = os.path.join(output_dir, 'model_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Metadata saved to {metadata_path}")


def main():
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("JOB RECOMMENDATION MODEL - TRAINING PIPELINE V2")
    print("="*80)
    
    # Initialize trainer
    trainer = JobRecommenderTrainer(data_dir="datasets/Splits")
    
    # Load data
    train_jobs, train_candidates, train_applications = trainer.load_data('train')
    cv_jobs, cv_candidates, cv_applications = trainer.load_data('cv')
    test_jobs, test_candidates, test_applications = trainer.load_data('test')
    
    # Build skill vocabulary from ALL data
    all_jobs = pd.concat([train_jobs, cv_jobs, test_jobs], ignore_index=True)
    all_candidates = pd.concat([train_candidates, cv_candidates, test_candidates], ignore_index=True)
    trainer.build_skill_vocabulary(all_jobs, all_candidates)
    
    # Engineer features
    train_features = trainer.engineer_features(train_jobs, train_candidates, train_applications)
    cv_features = trainer.engineer_features(cv_jobs, cv_candidates, cv_applications)
    test_features = trainer.engineer_features(test_jobs, test_candidates, test_applications)
    
    # Prepare training data
    feature_columns = [col for col in train_features.columns if col != 'selected']
    
    X_train = train_features[feature_columns]
    y_train = train_features['selected']
    
    X_cv = cv_features[feature_columns]
    y_cv = cv_features['selected']
    
    X_test = test_features[feature_columns]
    y_test = test_features['selected']
    
    print(f"\nFeature dimensions: {X_train.shape}")
    print(f"Number of features: {len(feature_columns)}")
    
    # Train model
    trainer.train(X_train, y_train, X_cv, y_cv)
    
    # Evaluate
    cv_metrics = trainer.evaluate(X_cv, y_cv, 'Validation')
    test_metrics = trainer.evaluate(X_test, y_test, 'Test')
    train_metrics = trainer.evaluate(X_train, y_train, 'Train')
    
    # Save model
    trainer.save_model()
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Train Accuracy: {train_metrics['accuracy'] * 100:.2f}%")
    print(f"CV Accuracy:    {cv_metrics['accuracy'] * 100:.2f}%")
    print(f"Test Accuracy:  {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()