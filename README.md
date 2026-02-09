# 🎯 Job Recommendation Engine with XGBoost

**An intelligent ML-powered system that matches candidates with suitable jobs, predicts selection probability, and identifies skill gaps.**

## 🧠 How It Works (Simple Explanation)
### The Big Picture

Imagine you're a job recruiter with 150,000 job postings and you need to find the best matches for a candidate. You'd look at:

1. **Skills**: Do they have what the job needs?
2. **Experience**: Are they at the right level?
3. **Salary**: Do expectations align?
4. **Location**: Are they in the right place?

This system does exactly that, but **automatically and instantly** using machine learning!

### What Makes It Smart?

**Traditional job search:**
- You manually search "Python developer"
- Get 10,000 results
- Spend hours filtering

**This AI system:**
- Analyzes YOUR specific skill combination
- Predicts YOUR chance at EACH job
- Shows you the TOP 10 where you're most likely to succeed
- Tells you EXACTLY which skills to learn for better jobs

---
## 🏗️ System Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📊 jobs.csv              📊 candidates.csv    📊 applications.csv│
│  • 150,000 jobs           • 200,000 candidates  • 200,000 records│
│  • Skills required        • Skills possessed    • Selection history│
│  • Salary range           • Experience          • Match scores   │
│  • Experience level       • Salary expectation  •                │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each candidate-job pair, calculate 21 features:           │
│                                                                 │
│  🎯 SKILL FEATURES (7)                                          │
│     • skill_match_count: How many skills overlap?              │
│     • skill_match_ratio: % of job skills you have              │
│     • missing_skills_count: How many skills to learn           │
│     • candidate_coverage: % of your skills used                │
│                                                                 │
│  💼 EXPERIENCE FEATURES (6)                                     │
│     • experience_diff: Gap between you and job requirement     │
│     • experience_match: How well experience aligns (0-1)       │
│     • exp_over_qualified: Are you too experienced?             │
│     • exp_under_qualified: Not enough experience?              │
│                                                                 │
│  💰 SALARY FEATURES (5)                                         │
│     • salary_fit: Is your expectation in range?                │
│     • salary_below: Expecting less than minimum?               │
│     • salary_above: Expecting more than maximum?               │
│                                                                 │
│  📍 LOCATION FEATURES (1)                                       │
│     • location_match: Remote or matching city?                 │
│                                                                 │
│  🎯 OVERALL MATCH (1)                                           │
│     • Weighted combination of skill, exp, location             │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING MODEL                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🌳 XGBoost Classifier                                          │
│                                                                 │
│  Training Process:                                              │
│  1. Split data: 70% train, 20% validation, 10% test            │
│  2. Handle class imbalance (31% selected, 69% rejected)        │
│  3. Learn patterns across 140,000 training examples            │
│  4. Build 100 decision trees to predict selection              │
│                                                                 │
│  Output: Probability of selection (0-100%)                     │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RECOMMENDATION ENGINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. User enters: Skills, Experience, Salary                    │
│  2. Pre-filter: Only jobs with ≥1 skill match (~40,000)        │
│  3. Calculate features for each filtered job                   │
│  4. Batch predict selection probability                        │
│  5. Sort by probability (highest first)                        │
│  6. Return top 10 recommendations                               │
│                                                                 │
│  Speed: 10-30 seconds (vs 5-10 minutes before optimization)    │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each recommended job:                                      │
│  • Job title, company, location                                │
│  • Salary range                                                 │
│  • Match score (0-100%)                                         │
│  • Selection probability (0-100%)                               │
│  • Skills you have ✅                                           │
│  • Skills to learn ⚠️                                           │
│  • Recommendation (💚 great, 💛 moderate, ❤️ learn more)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Basic Usage (3 Commands)

```bash
# 1. Train the model
python train_model.py

# 2. Get job recommendations (interactive)
python recommend_jobs.py


### Example Session

```bash
$ python recommend_jobs.py

Loading model... ✓
Loading jobs... ✓ (150,000 jobs)

Enter your skills (comma-separated):
> Python, Machine Learning, Pandas

Enter years of experience (0-30): 3

Enter expected salary ($): 120000

🔍 Finding best jobs for you...
  Filtering relevant jobs... 38,726 found
  Computing match scores... ✓
  Predicting probabilities... ✓
✅ Found 10 matching jobs!

#1 - Data Scientist at Google
📍 Location: Remote
💼 Experience Level: Mid Level
💰 Salary Range: $110,000 - $180,000
🎯 Match Score: 78.5%

📊 Skill Analysis:
   Match: 5/6 skills (83.3%)
   ✅ You Have: Python, Machine Learning, Pandas, SQL, NumPy
   ⚠️  Skills to Learn: TensorFlow

🎲 Selection Probability: 92.3%
   💚 Great match! High chance of selection.
```

---

## 🔧 Detailed Components
### 1. Data Generation (`generate_data.py`)
**Purpose**: Creates realistic synthetic dataset

**What it generates:**
- **150,000 jobs** across 8 categories (Data Science, Web Dev, Backend, Mobile, DevOps, Cloud, Security, Blockchain)
- **200,000 candidates** with diverse skill combinations
- **200,000 applications** with selection outcomes based on match quality

**Key insight**: Selection isn't random - better matches have higher selection rates

### 2. Data Splitting (`split_datasets.py`)
**Purpose**: Prepares data for machine learning
**How it splits:**
```
Total Data (200,000 applications)
│
├─ 70% Training (140,000)   → Model learns from this
│
├─ 20% Validation (40,000)  → Tunes model during training
│
└─ 10% Test (20,000)        → Final performance evaluation
```
**Why this matters**: Prevents overfitting - the model can't "memorize" test data

### 3. Model Training (`train_model.py`)
**Purpose**: Teaches the AI to predict selections

**The Learning Process:**
```python
# For each of 140,000 training applications:
# 1. Calculate features
features = {
    'skill_match_ratio': 0.67,      # Has 4 of 6 required skills
    'experience_match': 0.85,        # 5 years exp, needs 6
    'salary_fit': 1.0,               # Salary expectation in range
    'location_match': 1.0,           # Location matches
    ...
}

# 2. Check actual outcome
selected = True  # This candidate was selected

# 3. Model learns the pattern
# "When skill_match=0.67, exp_match=0.85, salary fits, location matches
#  → 85% chance of selection"
```

**After 140,000 examples**, the model recognizes:
- High skill match + good experience = likely selection
- Low skill match + wrong level = unlikely selection
- Salary mismatch = reduces chances
- Remote jobs = more forgiving on location

**Handles Class Imbalance:**
- 69% applications are rejected
- 31% are selected
- Uses `scale_pos_weight=2.22` so model doesn't just predict "reject" always

### 4. Recommendation Engine (`recommend_jobs.py`)
**Purpose**: Finds your best job matches FAST

**Speed Optimization Trick:**
```python
# SLOW WAY (5-10 minutes): 
for each of 150,000 jobs:
    calculate_features()
    predict()

# FAST WAY (10-30 seconds):
jobs_with_any_skill_match = filter(jobs, has_matching_skill)  # ~40,000 jobs
features_for_all = calculate_features_batch()  # All at once
probabilities = model.predict_batch(features_for_all)  # All at once
```

**Result**: 10-20x faster without losing accuracy!

### Feature Importance (What Matters Most)
```
1. overall_match (20.2%)        ← Combined score most important
2. missing_skills_ratio (19.1%) ← How many skills you lack
3. experience_diff (13.5%)      ← Experience gap
4. experience_match (9.7%)      ← How well experience fits
5. skill_match_ratio (4.0%)     ← % of required skills you have
```

**Insight**: The model weighs **missing skills** and **experience fit** most heavily.

---

## 🔬 Technical Details
### Why XGBoost?

**XGBoost** (eXtreme Gradient Boosting) is chosen because:

1. **Handles mixed features well**: We have numeric (salary), categorical (experience level), and boolean (location match) features
2. **Built-in regularization**: Prevents overfitting on 140,000 training samples
3. **Fast predictions**: Can score 40,000+ jobs in seconds
4. **Feature importance**: Tells us what matters most in selection
5. **Handles imbalanced data**: 31% selected vs 69% rejected - XGBoost handles this natively

### Model Architecture

```
XGBoost = Ensemble of 100 Decision Trees

Each tree learns a different pattern:

Tree 1: "If skill_match > 0.8 → high probability"
Tree 2: "If exp_diff < 2 AND salary_fit=1 → high probability"
Tree 3: "If missing_skills > 5 → low probability"
...
Tree 100: "If overall_match > 0.7 → high probability"

Final prediction = Average of all 100 trees
```

### Feature Engineering Rationale
**Why 21 features?**

Each feature captures a different aspect of "fit":

- **Skill features** (7): Raw match isn't enough - we need ratio, coverage, missing count
- **Experience features** (6): Both absolute (years) and relative (diff from expected) matter
- **Salary features** (5): Not just "in range" - also "below minimum" and "above maximum" matter
- **Location feature** (1): Simple binary - matches or doesn't
- **Overall match** (1): Pre-computed weighted score helps model converge faster

**Why these weights? (50% skill, 30% exp, 20% location)**

Based on hiring best practices:
- Skills are most important (can you do the job?)
- Experience matters but can be compensated (can learn on job)
- Location least important (remote work increasing)

### Hyperparameter Choices

```python
{
    'max_depth': 5,              # Prevents overfitting (default: 6)
    'learning_rate': 0.05,       # Slower = more accurate (default: 0.1)
    'n_estimators': 100,         # Balance speed/accuracy (default: 100)
    'scale_pos_weight': 2.22,    # Handles 69% negative / 31% positive split
    'gamma': 1,                  # Regularization (default: 0)
    'reg_alpha': 0.1,            # L1 regularization
    'reg_lambda': 1.0,           # L2 regularization
}
```
**These were tuned to prevent overfitting** while maintaining predictive power.

## 🎓 Learning Resources
### Understanding XGBoost
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [How XGBoost Works (Simple)](https://www.youtube.com/watch?v=OtD8wVaFm6E)

### Understanding ML Evaluation Metrics
- **Accuracy**: % of correct predictions
- **Precision**: Of predicted "yes", how many were actually "yes"
- **Recall**: Of actual "yes", how many did we catch
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Ability to distinguish between classes 

### Why Class Imbalance Matters
When 69% of applications are rejected, a "dumb" model that always predicts "reject" gets 69% accuracy! We use `scale_pos_weight` to force the model to learn the harder task of predicting "selected".

---
## 🚀 Future Enhancements

Possible improvements:

1. **Resume Parsing**: Upload resume, extract skills automatically
2. **Deep Learning**: Use neural networks for better pattern recognition
3. **Collaborative Filtering**: "Candidates like you also applied to..."
4. **Real-time Job Feed**: Integrate with LinkedIn/Indeed APIs
5. **Career Path**: "To reach senior level, learn these 5 skills"
6. **Salary Prediction**: "Based on your skills, expect $X-$Y"
7. **Interview Prep**: Generate questions based on missing skills
---




