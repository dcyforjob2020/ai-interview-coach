# AI Interview Coach - Qwen2.5 project

AI-driven interview simulation and feedback system using Qwen2.5.

## Roadmap
- [x] Dataset Building & Preparation
- [x] Student Answer Generation (Test Set)
- [x] Baseline Feedback Generation (Test Set)
- [x] Baseline Feedback Scoring with Qwen3.5-9B (Judge)
- [x] Preference Candidate Generation (RLAIF - Training Set)
- [x] Preference Pair Selection with Qwen3.5-9B (RLAIF - Training Set)
- [x] SFT & DPO Training with QLoRA (RLAIF)
- [x] New Model Feedback Generation & Evaluation
- [x] Baseline vs New Model Comparison

## Experimental Results

The trained model (DPO) was evaluated against the baseline model (`Qwen/Qwen2.5-3B-Instruct`) on the fixed test set of 200 examples.

| Model | Average Score (1-20) | SD |
| :--- | :--- | :--- |
| **Baseline (Qwen2.5-3B-Instruct)** | 16.86 | 2.37 |
| **New Model (SFT + DPO)** | **17.03** | 1.85 |

### Statistical Significance
The +0.17 average improvement is **not statistically significant**. Tested with a
paired comparison over the 200 examples (`eval/statistical_tests.py`):

| Test | Statistic | p-value | Significant (α=0.05)? |
| :--- | :--- | :--- | :--- |
| Paired t-test | t = 0.81 | 0.42 | No |
| Wilcoxon signed-rank | W = 1868.5 | 0.70 | No |
| Sign test (non-tie pairs) | 48 vs 40 | 0.46 | No |
| Cohen's d (effect size) | 0.057 | — | Negligible |

Win/loss/tie on overall score: **new model wins 48, baseline wins 40, ties 112**.

### Why the Difference Is Not Significant
- **Ceiling effect**: The judge already rates the baseline very highly — 65% of
  baseline outputs score ≥18/20, and 191/200 examples score ≥15 for *both*
  models. There is little headroom for the trained model to improve.
- **Coarse judge resolution**: Scores cluster on a few discrete values (mostly
  15 and 18), and the five rubric dimensions almost always move together. This
  adds noise and hides fine-grained differences.
- **Offsetting wins and losses**: Real gains are cancelled by new regressions.
  The new model recovers several catastrophic baseline failures (e.g. three
  cases go from 1 → 18 by fixing hallucinated "missing" concepts), but it also
  introduces a few new technical errors (e.g. `test_0194` drops 18 → 1 after
  falsely claiming `@Override` throws a runtime error). See
  `eval/failure_analysis.py`.

### Test-Set Caveat
The current test set is **not as diverse as intended**: all 200 student answers
are labeled `partially_correct` (the five answer types described below are not
represented), and category/difficulty metadata is mostly `Unknown`. Rebuilding
the test set with the full range of answer types would give training a fairer
chance to show measurable gains.

### Observed Qualitative Improvements (not statistically confirmed)
- **Hallucination reduction**: Recovers baseline cases that wrongly claimed a
  student missed a concept actually present in their answer.
- **Educational depth**: More concrete examples (e.g. explaining memoization
  with Fibonacci) and clearer next steps in winning cases.

## Detailed Experimental Plan

### 1. Dataset Source and Size
The dataset is built from two software engineering interview question sources: `Software Questions` and `ai_interview_questions`.

After preprocessing, the data is split into:
- Training set: approximately 4,653 examples.
- Test set: 200 held-out examples.

Each example contains an interview question, a reference answer, metadata such as category and difficulty, and a simulated student answer.

The test set is kept fixed throughout the experiment. The same test questions and the same student answers are used for both baseline evaluation and final model evaluation. Student answers are not regenerated during the comparison stage.

### 2. Baseline Evaluation Setup
The baseline model is `Qwen/Qwen2.5-3B-Instruct`.

For each example in the fixed test set, the model generates coaching feedback based on:
- The interview question.
- The reference answer.
- The fixed student answer.

The output is saved as baseline feedback. The baseline feedback is then scored by the judge model, `Qwen/Qwen3.5-9B`, using the same evaluation rubric that will later be used for the trained model. This creates the baseline performance score before fine-tuning.

### 3. Student Answer Simulation Setup
Student answers are generated before training and evaluation. They are designed to simulate a first-year CS Master's student answering software engineering interview questions.

The simulated answers include different answer quality types:
- Correct but incomplete.
- Partially correct.
- Incorrect.
- Too vague.
- Verbose but unfocused.

Once generated, these student answers are fixed. They are not regenerated for baseline evaluation, preference candidate generation, or final model evaluation. This ensures that any score difference comes from changes in the coaching feedback model, not from changes in student answers.

### 4. Preference Candidate Generation and Bias Control
For the training set, `Qwen/Qwen2.5-3B-Instruct` generates two coaching feedback candidates for each fixed student answer:
- `feedback_a`
- `feedback_b`

To avoid making one side consistently better by design, the system uses randomized but reproducible feedback styles. The A/B assignment is not tied to a fixed quality level, and the model records metadata such as:
- Feedback style.
- Generation temperature.
- Candidate model name.

This reduces the risk that the preference model learns superficial patterns such as always preferring `feedback_b`, longer answers, or one specific feedback style.

### 5. Judge Rubric and Preference Selection
The judge model is `Qwen/Qwen3.5-9B`.

For baseline scoring, it evaluates each feedback response using a rubric based on:
- Technical correctness.
- Specificity.
- Helpfulness.
- Actionability.
- Suitability for software engineering interview coaching.

For preference selection, the judge compares `feedback_a` and `feedback_b` and selects the better one. The prompt explicitly instructs the judge not to prefer an answer because of:
- A/B position.
- Length.
- Formatting.
- Tone alone.
- Style label.

The selected response becomes the `chosen` feedback, and the other response becomes the `rejected` feedback for later RLAIF/DPO training.

### 6. SFT Dataset Construction
Construct a supervised fine-tuning dataset from high-quality examples. The SFT dataset uses the interview question, reference answer, and fixed student answer as input, and uses selected high-quality coaching feedback as the target output.

This prepares the model to learn the general format and behavior of interview coaching before preference optimization.

### 7. SFT with QLoRA
Fine-tune `Qwen/Qwen2.5-3B-Instruct` using supervised fine-tuning with QLoRA.

QLoRA is used to reduce GPU memory cost by training lightweight adapter weights instead of fully fine-tuning all model parameters. The SFT model becomes the intermediate improved model.

### 8. DPO Training with QLoRA (RLAIF)
Use the preference pairs selected by `Qwen/Qwen3.5-9B` to train the model with Direct Preference Optimization.

The model learns to prefer feedback that is more technically correct, specific, helpful, actionable, and suitable for interview coaching. QLoRA is again used to make training feasible with limited compute.

### 9. New Model Evaluation
Use the trained model on the same held-out `data/test.jsonl` set used in the baseline stage. The test questions and student answers remain fixed and are not regenerated.

For each test example, generate new coaching feedback using the trained model. Then use `Qwen/Qwen3.5-9B` to score the new feedback using the same rubric as the baseline evaluation.

### 10. Baseline vs New Model Comparison
Compare baseline scores and new model scores on the same 200 test examples.

The comparison measures whether training improved feedback quality. Evaluation metrics include:
- Average score improvement.
- Technical correctness.
- Specificity.
- Helpfulness.
- Actionability.
- Interview coaching suitability.
- Win rate between baseline and trained model feedback.

### 11. Small-scale Human Evaluation
Conduct a small human evaluation to validate whether the AI judge scores are reasonable.

Human evaluators compare a subset of baseline and trained model feedback. They judge which feedback is more useful, accurate, and actionable for a student preparing for software engineering interviews. This provides additional evidence beyond automated scoring.

## Model Roles

- **Base/Policy Model**: `Qwen/Qwen2.5-3B-Instruct`
  - Used for student answer simulation and generating coach feedback candidates.
  - This is the model that will be fine-tuned.
- **Judge Model**: `Qwen/Qwen3.5-9B`
  - Used for feedback scoring and preference selection (RLAIF).

## Trained Models
The following QLoRA adapters are available on Hugging Face:
- **SFT Adapter**: [Hank122222222/ai-interview-coach-sft-adapter](https://huggingface.co/Hank122222222/ai-interview-coach-sft-adapter)
- **DPO Adapter**: [Hank122222222/ai-interview-coach-dpo-adapter](https://huggingface.co/Hank122222222/ai-interview-coach-dpo-adapter)

## Methodology Details

### Student Simulation Profile
To test the coach's effectiveness, we simulate answers from a specific persona:
- **Level**: First-year CS Master's student.
- **Characteristics**: Strong fundamentals but limited industry experience; prone to omitting edge cases or using imprecise terminology.
- **Answer Types**: Correct but incomplete, partially correct, incorrect, too vague, or verbose but unfocused.

### Scoring Rubric (1-20 Scale)
The judge model evaluates feedback across five dimensions:
1. **Technical Correctness**: Accuracy of the technical advice.
2. **Specificity**: Depth of identification of strengths and gaps.
3. **Helpfulness**: Potential for student improvement.
4. **Actionability**: Clarity of next steps.
5. **Interview Coaching Quality**: Suitability for an interview context.

*Note: Scores of 18-20 are reserved for "Exceptional" feedback with concrete code snippets or master-class guidance.*

### RLAIF Feedback Styles
During training data generation, we prompt the base model with diverse styles to explore the preference space:
- Technical precision focus.
- Interview communication focus.
- Balanced coaching (strengths/weaknesses/advice).
- Supportive yet strict tone.

## Dataset Schema

### Train/Test Base Data (`data/train.jsonl`, `data/test.jsonl`)
```json
{
  "id": "train_0001",
  "question": "...",
  "reference_answer": "...",
  "category": "...",
  "difficulty": "...",
  "student_answer": "...",
  "student_answer_type": "...",
  "source": "..."
}
```

### Baseline Feedback (`baseline/baseline_outputs.jsonl`)
Generated from the test set using the base model.
```json
{
  "id": "test_0001",
  "baseline_feedback": "...",
  "baseline_model": "Qwen/Qwen2.5-3B-Instruct",
  "...": "base_fields"
}
```

### Feedback Scoring (`baseline/baseline_scores.jsonl`)
Scored by the judge model on a 1-20 scale.
```json
{
  "id": "test_0001",
  "technical_correctness": 15,
  "specificity": 14,
  "helpfulness": 16,
  "actionability": 15,
  "interview_coaching_quality": 15,
  "overall_score": 15.0,
  "reason": "Concise and technically accurate with clear improvement steps.",
  "judge_model": "Qwen/Qwen3.5-9B",
  "feedback_field": "baseline_feedback"
}
```

### Preference Candidates (`train/preference_candidates.jsonl`)
Generated from the training set using the base model. Each record contains two candidate feedback responses for the same fixed student answer.
```json
{
  "id": "train_0001",
  "question": "...",
  "reference_answer": "...",
  "category": "...",
  "difficulty": "...",
  "student_answer": "...",
  "student_answer_type": "...",
  "source": "...",
  "feedback_a": "...",
  "feedback_b": "...",
  "feedback_a_style": "...",
  "feedback_b_style": "...",
  "feedback_a_temperature": 0.75,
  "feedback_b_temperature": 0.65,
  "candidate_model": "Qwen/Qwen2.5-3B-Instruct"
}
```

### Preference Pairs (`train/preference_pairs.jsonl`)
The final RLAIF training data with labeled preferences, generated by the judge model.
```json
{
  "id": "train_0001",
  "chosen_feedback": "...",
  "rejected_feedback": "...",
  "winner": "a",
  "feedback_a_avg_score": 15.5,
  "feedback_b_avg_score": 12.0,
  "judge_model": "Qwen/Qwen3.5-9B",
  "...": "base_and_candidate_fields"
}
```

## Workflow & Commands

### 1. Setup
Install dependencies and verify model access:
```bash
pip install -r requirements.txt
python setup_model.py
```
*Note: For Google Colab users, use the provided `.ipynb` notebooks in `eval/` and `train/` for optimized environment setup.*

### 2. Baseline Evaluation
Score the existing baseline feedback to establish a benchmark:
```bash
python eval/score_feedback.py \
    --input baseline/baseline_outputs.jsonl \
    --output baseline/baseline_scores.jsonl \
    --feedback-field baseline_feedback
```

### 3. RLAIF: Preference Data Generation
Generate candidate pairs for the training set:
```bash
python train/generate_candidates.py \
    --input data/train.jsonl \
    --output train/preference_candidates.jsonl
```
*Or use `train/generate_candidates.ipynb` on Colab.*

### 4. RLAIF: Preference Labeling
Use the judge model to select the better candidate:
```bash
python eval/score_preferences.py \
    --input train/preference_candidates.jsonl \
    --output train/preference_pairs.jsonl
```
*Or use `eval/score_preferences.ipynb` on Colab.*

### 5. Fine-Tuning: SFT
1. Build the SFT dataset from preference pairs:
```bash
python train/sft/build_sft_dataset.py
```
2. Run SFT training:
```bash
python train/sft/train_sft_qlora.py
```

### 6. Fine-Tuning: DPO
1. Build the DPO dataset from preference pairs:
```bash
python train/dpo/build_dpo_dataset.py
```
2. Run DPO training:
```bash
python train/dpo/train_dpo_qlora.py
```

### 7. Comparison & Analysis
After both models are scored, compare them and analyze the results:
```bash
# Average scores, per-dimension deltas, win/loss/tie
python eval/compare_scores.py

# Statistical significance: paired t-test, Wilcoxon, sign test, Cohen's d
# -> writes eval/statistical_tests.json
python eval/statistical_tests.py

# Failure analysis: categorize regressions/gains, breakdowns, top examples
# -> writes eval/failure_analysis.json, failure_regressions.csv, failure_gains.csv
python eval/failure_analysis.py
```
