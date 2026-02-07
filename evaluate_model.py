import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_DIR = "models"
DATA_DIR = "datasets"
LABEL_COL = "selected"


# ---------- Feature Engineering Helpers ----------

def extract_skills_set(skills_str):
    if pd.isna(skills_str):
        return set()
    return set(s.strip() for s in str(skills_str).split(","))


def create_features(df):
    """
    Recreate the SAME features used during training & inference
    """

    exp_level_map = {
        'Entry Level': 2,
        'Mid Level': 5,
        'Senior Level': 8,
        'Lead': 12,
        'Principal': 15
    }

    feature_rows = []

    for _, row in df.iterrows():
        candidate_skills = extract_skills_set(row["skills"])
        job_skills = extract_skills_set(row["required_skills"])

        candidate_skills_count = len(candidate_skills)
        job_skills_count = len(job_skills)

        # Skill matching
        common_skills = candidate_skills.intersection(job_skills)
        skill_match_count = len(common_skills)

        skill_match_ratio = skill_match_count / job_skills_count if job_skills_count else 0
        candidate_coverage = skill_match_count / candidate_skills_count if candidate_skills_count else 0

        missing_skills_count = len(job_skills - candidate_skills)
        missing_skills_ratio = missing_skills_count / job_skills_count if job_skills_count else 0

        # Experience
        expected_exp = exp_level_map.get(row["experience_level"], 5)
        exp_diff = abs(row["experience_years"] - expected_exp)
        exp_match = max(0, 1 - exp_diff / 10)

        exp_over_qualified = 1 if row["experience_years"] > expected_exp + 3 else 0
        exp_under_qualified = 1 if row["experience_years"] < expected_exp - 2 else 0

        # Salary
        salary_fit = 1.0 if row["salary_min"] <= row["expected_salary"] <= row["salary_max"] else 0.0
        salary_below = 1 if row["expected_salary"] < row["salary_min"] else 0
        salary_above = 1 if row["expected_salary"] > row["salary_max"] else 0

        # Location
        location_match = 1.0 if (
            row["location"] == "Remote" or
            row["location"] == row["preferred_location"]
        ) else 0.0

        # Match score (same formula)
        skill_match = skill_match_ratio
        overall_match = (
            0.5 * skill_match +
            0.3 * exp_match +
            0.2 * location_match
        )

        feature_rows.append({
            'skill_match_count': skill_match_count,
            'skill_match_ratio': skill_match_ratio,
            'candidate_coverage': candidate_coverage,
            'missing_skills_count': missing_skills_count,
            'missing_skills_ratio': missing_skills_ratio,
            'total_job_skills': job_skills_count,
            'total_candidate_skills': candidate_skills_count,
            'experience_years': row["experience_years"],
            'expected_experience': expected_exp,
            'experience_diff': exp_diff,
            'experience_match': exp_match,
            'exp_over_qualified': exp_over_qualified,
            'exp_under_qualified': exp_under_qualified,
            'expected_salary': row["expected_salary"] / 1000,
            'salary_min': row["salary_min"] / 1000,
            'salary_max': row["salary_max"] / 1000,
            'salary_fit': salary_fit,
            'salary_below': salary_below,
            'salary_above': salary_above,
            'location_match': location_match,
            'overall_match': overall_match
        })

    return pd.DataFrame(feature_rows)


# ---------- Load Model ----------

def load_model():
    with open(os.path.join(MODEL_DIR, "job_recommender_xgb.pkl"), "rb") as f:
        return pickle.load(f)


# ---------- Load & Prepare Data ----------

def load_split(split):
    apps = pd.read_csv(f"{DATA_DIR}/Splits/{split}/applications.csv")
    candidates = pd.read_csv(f"{DATA_DIR}/Splits/{split}/candidates.csv")
    jobs = pd.read_csv(f"{DATA_DIR}/Splits/{split}/jobs.csv")

    df = apps.merge(candidates, on="candidate_id", how="left") \
             .merge(jobs, on="job_id", how="left")

    y = df[LABEL_COL]
    X = create_features(df)

    return X, y


# ---------- Evaluation ----------

def evaluate(model, split):
    print(f"\n📊 Evaluating {split.upper()} set")

    X, y_true = load_split(split)
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")

    return acc


# ---------- Main ----------

def main():
    print("Loading trained model...")
    model = load_model()
    print("✓ Model loaded")

    cv_acc = evaluate(model, "cv")
    test_acc = evaluate(model, "test")

    print("\n" + "=" * 60)
    print("✅ FINAL EVALUATION")
    print("=" * 60)
    print(f"CV Accuracy   : {cv_acc:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")

    if abs(cv_acc - test_acc) > 0.05:
        print("⚠️  Warning: Possible overfitting")
    else:
        print("✅ Good generalization")


if __name__ == "__main__":
    main()
