import numpy as np
import pandas as pd
import collections
import string
import re
import os
import time
import json
from datetime import datetime
from tqdm import tqdm

# ==========================================
# 1. TEXT NORMALIZATION & UTILS
# ==========================================

def normalize_text(s):
    """Standard SQuAD normalization."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_json_score(llm_response):
    """
    Robustly extracts JSON from LLM response even if it chatters.
    Expected format: {"score": 0.8, "reason": "..."}
    """
    try:
        # Try finding the first '{' and last '}'
        start = llm_response.find('{')
        end = llm_response.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = llm_response[start:end]
            data = json.loads(json_str)
            return float(data.get("score", 0.0)), data.get("reason", "No reason provided")
    except Exception as e:
        print(f"⚠️ JSON Parse Error in Judge: {e}")
    return 0.0, "Parse Error"

def sleep_and_count(api_calls, sleep_interval):
    if api_calls > 0 and api_calls % 5 == 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("api_calls.txt", "a") as f:
            f.write(f"[{timestamp}] API Calls: {api_calls}\n")
    elif api_calls > 0 and api_calls % 15 == 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("api_calls.txt", "a") as f:
            f.write(f"[{timestamp}] API Calls: {api_calls}\n")
        time.sleep(15)
    else:
        time.sleep(sleep_interval)
    
    api_calls += 1

    return api_calls

# ==========================================
# 2. DETERMINISTIC METRICS (Math-based)
# ==========================================

def calculate_exact_match(prediction, ground_truth):
    return int(normalize_text(prediction) == normalize_text(ground_truth))

def calculate_f1_score(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_context_similarity(retrieved_docs, gold_context):
    gold_tokens = set(normalize_text(gold_context).split())
    if not gold_tokens:
        return 0.0
    best_jaccard = 0.0
    for doc in retrieved_docs:
        retrieved_tokens = set(normalize_text(doc['text']).split())
        intersection = gold_tokens.intersection(retrieved_tokens)
        union = gold_tokens.union(retrieved_tokens)
        score = 0.0 if not union else len(intersection) / len(union)
        if score > best_jaccard: best_jaccard = score
    return best_jaccard

def calculate_hit_rate(retrieved_docs, ground_truth_title):
    for doc in retrieved_docs:
        if doc['title'].strip().lower() == ground_truth_title.strip().lower():
            return 1
    return 0

# ==========================================
# 3. THE PRO LLM JUDGE (Chain-of-Thought)
# ==========================================

class LLMJudge:
    def __init__(self, groq_client, model="llama-3.3-70b-versatile"):
        self.client = groq_client
        self.model = model

    def evaluate_faithfulness(self, context, answer):
        """
        Check if the answer is grounded in the provided context.
        Prioritizes detecting hallucinations.
        """
        prompt = f"""
        You are an expert Fact-Checking AI.
        Task: Rate the 'Faithfulness' of the Answer based strictly on the Context.

        [Context Start]
        {context[:4000]}
        [Context End]

        [Answer]
        {answer}

        Instructions:
        1. Break the Answer down into atomic statements/claims.
        2. For each claim, search for supporting evidence in the Context.
        3. If a claim is NOT found in the context (even if it is factually true in the real world), it is considered a Hallucination.
        4. If the answer refuses to answer ("I don't know") because context is missing, score it 1.0.

        Scoring Criteria:
        - 1.0: Every single claim is supported by the context.
        - 0.8: Mostly supported, but contains minor details not in text.
        - 0.5: Half supported, half hallucinated.
        - 0.0: Major hallucinations or contradicts context.

        Response Format (JSON):
        {{"reason": "1. Claim 'X' supported by '...'. 2. Claim 'Y' not found...", "score": <float>}}
        """
        return self._call_judge(prompt)

    def evaluate_relevance(self, question, answer):
        """
        Check if the answer addresses the user's intent.
        """
        prompt = f"""
        You are an expert Evaluator AI.
        Task: Rate 'Answer Relevance' regarding the User Question.

        [User Question]
        {question}

        [System Answer]
        {answer}

        Instructions:
        1. Analyze the intent of the User Question.
        2. Determine if the Answer directly addresses this intent.
        3. Penalize answers that are verbose, evasive, or start with unnecessary fillers ("Here is some information about...").
        4. Ignore factual correctness (faithfulness is checked elsewhere); focus on *responsiveness*.

        Scoring Criteria:
        - 1.0: Perfect, direct, and concise answer to the specific question.
        - 0.7: Answers the question but includes unnecessary fluff.
        - 0.5: Tangentially related or misses the core intent.
        - 0.0: Completely irrelevant or refuses to answer a valid question.

        Response Format (JSON):
        {{"reason": "Step-by-step analysis of how well the intent was met...", "score": <float>}}
        """
        return self._call_judge(prompt)

    def evaluate_context_utility(self, question, context):
        """
        Check if the retrieved context actually contained the answer (Recall).
        """
        prompt = f"""
        You are an expert Search Relevance AI.
        Task: Rate 'Context Utility'. Does the retrieved text contain the answer to the question?

        [Question]
        {question}

        [Retrieved Context]
        {context[:4000]}

        Instructions:
        1. Read the Question carefully.
        2. Scan the Context for the specific answer.
        3. Determine if the context contains *sufficient* information to answer fully.

        Scoring Criteria:
        - 1.0: The context contains the exact answer and supporting details.
        - 0.7: The context contains the answer but requires some deduction or is slightly messy.
        - 0.4: The context is related to the topic but does not answer the specific question.
        - 0.0: The context is irrelevant noise.

        Response Format (JSON):
        {{"reason": "Identify the exact sentence in context that answers the question...", "score": <float>}}
        """
        return self._call_judge(prompt)

    def evaluate_negative_rejection(self, question, answer, is_impossible):
        """
        Check if the model correctly refused to answer an impossible question.
        Only runs if the ground truth says the question is impossible.
        """
        if not is_impossible:
            return 1.0, "N/A (Question was possible)"
            
        prompt = f"""
        You are a strict Logic Evaluator.
        Task: Rate 'Negative Rejection'. The user asked a question that is IMPOSSIBLE to answer based on the available context.

        [Question]
        {question}

        [AI Answer]
        {answer}

        Instructions:
        1. The correct behavior is to REFUSE to answer (e.g., "I don't know", "The context does not contain this info").
        2. The incorrect behavior is to hallucinate an answer or use outside knowledge.

        Scoring Criteria:
        - 1.0: Clearly states the information is missing.
        - 0.5: Refuses but provides "related" irrelevant info.
        - 0.0: Attempts to answer the question using outside knowledge or hallucinations.

        Response Format (JSON):
        {{"reason": "Did the AI admit ignorance?...", "score": <float>}}
        """
        return self._call_judge(prompt)
        
    def evaluate_coherence(self, answer):
        """
        Check style, grammar, and conciseness.
        """
        prompt = f"""
        You are a professional Editor.
        Task: Rate 'Coherence & Style' of the generated text.

        [Answer]
        {answer}

        Instructions:
        1. Check for grammatical correctness.
        2. Check for "hallucination loops" (repeating the same sentence).
        3. Check for formatting (lists, paragraphs) that improves readability.
        4. Penalize robotic or overly verbose introductions.

        Scoring Criteria:
        - 1.0: Fluent, professional, concise, and well-structured.
        - 0.7: Grammatically correct but slightly wordy or clunky.
        - 0.4: Hard to read, poor formatting, or repetitive.
        - 0.0: Incoherent or gibberish.

        Response Format (JSON):
        {{"reason": "Specific critique of style...", "score": <float>}}
        """
        return self._call_judge(prompt)

    def evaluate_answer_similarity(self, ground_truth, generated_answer):
        """
        Judge: Does the generated answer mean the same thing as the ground truth?
        Handles semantic equivalence (e.g., "100%" vs "All of it").
        """
        # If no ground truth exists, we cannot evaluate similarity.
        if not ground_truth:
            return 1.0, "N/A (No ground truth provided)"

        prompt = f"""
        You are an expert Semantic Match Evaluator.
        Task: Rate the 'Semantic Similarity' between the Ground Truth (GT) and the Generated Answer (Gen).

        [Ground Truth]
        {ground_truth}

        [Generated Answer]
        {generated_answer}

        Instructions:
        1. Normalize the content: Treat "100 meters" and "100m" as identical. Treat "2023-01-01" and "Jan 1st, 2023" as identical.
        2. Check for Entailment: Does the Generated Answer *entail* the Ground Truth?
           - If Gen includes the GT + extra correct details -> HIGH SCORE.
           - If Gen includes the GT + contradictory details -> LOW SCORE.
           - If Gen is vague ("It was a good year") vs specific GT ("1999") -> LOW SCORE.
        3. Ignore style/verbosity: "The answer is 5" is equal to "5".

        Scoring Criteria:
        - 1.0: Exact semantic match or fully entails the GT (e.g., GT="Paris", Gen="Paris, France").
        - 0.8: Correct but adds slight unnecessary ambiguity or minor fluff.
        - 0.5: Partial match. Covers some of the GT but misses key constraints (e.g., GT="Red and Blue", Gen="Red").
        - 0.0: Incorrect, contradictory, or completely unrelated.

        Response Format (JSON):
        {{"reason": "Step 1: Analyze GT intent. Step 2: Compare with Gen...", "score": <float>}}
        """
        return self._call_judge(prompt)

    def _call_judge(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"} 
            )
            content = completion.choices[0].message.content
            return extract_json_score(content)
        except Exception as e:
            return 0.0, f"Error: {str(e)}"

# ==========================================
# 4. THE EVALUATOR ENGINE
# ==========================================

class RAGEvaluator:
    def __init__(self, retriever_instance=None, generator_func=None, groq_client=None):
        self.retriever = retriever_instance
        self.generator = generator_func
        self.judge = LLMJudge(groq_client) if groq_client else None
        
    def run_experiment(self, 
                       dataset, 
                       experiment_name="experiment", 
                       rag_type="baseline", 
                       model_name="llama-3.3-70b-versatile", 
                       sample_size=50,
                       target_rpm=30,
                       output_dir="./data/results",
                       use_llm_judge=False,
                       judge_model="llama-3.3-70b-versatile"):
        
        # Setup
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{experiment_name}_{timestamp}"
        
        # Dynamic Sleep (Add buffer for Judge's extra calls)
        sleep_interval = (60.0 / target_rpm) * 1.2
        api_calls = 0 
        print(f"⏱️ Target RPM: {target_rpm} | Sleep Base: {sleep_interval:.2f}s")
        print(f"Model: {model_name}, Judge: {judge_model}")
        
        if self.judge: self.judge.model = judge_model

        # Sampling
        if sample_size and sample_size < len(dataset):
            print(f"🎲 Sampling {sample_size} questions...")
            np.random.seed(26)
            indices = np.random.choice(len(dataset), sample_size, replace=False)
            eval_data = [dataset[i] for i in indices]
        else:
            eval_data = dataset

        print(f"🚀 Experiment: {experiment_name} | Judge: {use_llm_judge}")
        
        results = []
        
        for record in tqdm(eval_data, desc="Evaluating"):
            query = record['question']
            gold_title = record['title']
            gold_context = record['context']
            is_impossible = record.get('is_impossible', False)
            
            # Robust Answer Extraction
            gold_answers = []
            answers_data = record['answers']
            if isinstance(answers_data, dict) and 'text' in answers_data:
                gold_answers = answers_data['text'] 
            elif isinstance(answers_data, list):
                gold_answers = [a['text'] for a in answers_data if 'text' in a]

            try:
                # --- A. GENERATION ---
                start_time = time.time()
                # print(model_name)
                generated_answer, retrieved_docs = self.generator(query, self.retriever, llm_model=model_name)
                latency = time.time() - start_time
                
                # --- B. DETERMINISTIC METRICS ---
                hit_rate = calculate_hit_rate(retrieved_docs, gold_title)
                context_sim = calculate_context_similarity(retrieved_docs, gold_context)
                
                if not gold_answers:
                    em_score = 0; f1_score = 0
                else:
                    em_score = max([calculate_exact_match(generated_answer, ans) for ans in gold_answers])
                    f1_score = max([calculate_f1_score(generated_answer, ans) for ans in gold_answers])
                
                # --- C. LLM JUDGE METRICS ---
                metrics_log = {}
                
                ctx_text = "\n".join([d['text'] for d in retrieved_docs])

                if use_llm_judge and self.judge:
                    
                    # 1. Faithfulness
                    f_score, f_reason = self.judge.evaluate_faithfulness(ctx_text, generated_answer)
                    api_calls = sleep_and_count(api_calls, sleep_interval)
                    
                    # 2. Relevance
                    r_score, r_reason = self.judge.evaluate_relevance(query, generated_answer)
                    api_calls = sleep_and_count(api_calls, sleep_interval)
                    
                    # 3. Context Utility (Did we find good stuff?)
                    u_score, u_reason = self.judge.evaluate_context_utility(query, ctx_text)
                    api_calls = sleep_and_count(api_calls, sleep_interval)
                    
                    # 4. Negative Rejection (Only if impossible)
                    if is_impossible:
                        n_score, n_reason = self.judge.evaluate_negative_rejection(query, generated_answer, is_impossible)
                        api_calls = sleep_and_count(api_calls, sleep_interval)
                    else:
                        n_score, n_reason = np.nan, "N/A"
                        
                    # 5. Coherence
                    c_score, c_reason = self.judge.evaluate_coherence(generated_answer)
                    api_calls = sleep_and_count(api_calls, sleep_interval)

                    best_gt = gold_answers[0] if gold_answers else ""
                    s_score, s_reason = self.judge.evaluate_answer_similarity(best_gt, generated_answer)
                    api_calls = sleep_and_count(api_calls, sleep_interval)

                    metrics_log = {
                        "judge_faithfulness": f_score,
                        "judge_relevance": r_score,
                        "judge_utility": u_score,
                        "judge_neg_rejection": n_score,
                        "judge_coherence": c_score,
                        "judge_similarity": s_score,
                        "judge_reasoning": f"Faith: {f_reason} | Util: {u_reason} | Rel: {r_reason} | Coherence: {c_reason} | Sim: {s_reason}"
                    }
                else:
                    # Fill blanks if judge is off
                    metrics_log = {k: 0.0 for k in ["judge_faithfulness", "judge_relevance", "judge_utility", "judge_coherence", "judge_similarity"]}
                    metrics_log["judge_neg_rejection"] = np.nan
                    metrics_log["judge_reasoning"] = ""

                # --- D. LOGGING ---
                row = {
                    "run_id": run_id,
                    "experiment_name": experiment_name,
                    "rag_type": rag_type,
                    "model_name": model_name,
                    "judge_model": judge_model,
                    "timestamp": timestamp,
                    "question": query,
                    "is_impossible": is_impossible,
                    "generated_answer": generated_answer,
                    "gold_answers": str(gold_answers),
                    "gold_context_preview": gold_context[:50] + "...",
                    "retrieved_titles": [d['title'] for d in retrieved_docs],
                    "latency_seconds": round(latency, 2),
                    
                    # Deterministic
                    "retrieval_hit_rate": hit_rate,
                    "context_similarity": round(context_sim, 4),
                    "exact_match": em_score,
                    "f1_score": round(f1_score, 4),
                    
                    # Judge
                    **metrics_log,
                    "retrieved_context_text": ctx_text
                }
                results.append(row)
                api_calls = sleep_and_count(api_calls, sleep_interval)
                
            except Exception as e:
                print(f"⚠️ Error: {e}")
                api_calls = sleep_and_count(api_calls, sleep_interval)
                continue

        # Save & Summary
        df = pd.DataFrame(results)
        if df.empty: return df, {}
        
        filename = f"{output_dir}/eval_{experiment_name}_{rag_type}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        # Summary Stats
        summary = {
            "Total": len(df),
            "Avg F1": df['f1_score'].mean(),
            "Avg Exact Match": df['exact_match'].mean(),
            "Avg Hit Rate": df['retrieval_hit_rate'].mean(),
            "Avg Faithfulness": df['judge_faithfulness'].mean(),
            "Avg Relevance": df['judge_relevance'].mean(),
            "Avg Context Utility": df['judge_utility'].mean(),
            "Avg Coherence": df['judge_coherence'].mean(),
            "Avg Semantic Similarity": df['judge_similarity'].mean(),
            "Total API Calls": api_calls,
        }
        
        print("\n📊 EXPERIMENT SUMMARY")
        print("="*40)
        for k, v in summary.items(): print(f"{k:<25}: {v:.4f}")
        print("="*40)
        print(f"💾 Saved: {filename}")
        return df, summary

    def evaluate_batch(self, 
                      results_path_or_df, 
                      target_rpm=30,
                      judge_model="llama-3.3-70b-versatile"):
        """
        Runs the LLM Judge on a pre-generated CSV or DataFrame.
        Only calculates missing metrics (where value is 0.0 or NaN).
        """
        import ast

        # 1. Load Data
        if isinstance(results_path_or_df, str):
            print(f"📂 Loading results from: {results_path_or_df}")
            df = pd.read_csv(results_path_or_df)
        else:
            df = results_path_or_df.copy()
            
        if self.judge: 
            self.judge.model = judge_model
        else:
            print("⚠️ No Judge initialized! Pass a groq_client to RAGEvaluator.")
            return df, {}

        # 2. Setup Loop
        sleep_interval = (60.0 / target_rpm) * 1.2
        api_calls = 0
        print(f"⏱️  Refining Evaluations | Target RPM: {target_rpm} | Model: {judge_model}")
        
        # Ensure columns exist
        required_judge_cols = [
            "judge_faithfulness", "judge_relevance", "judge_utility", 
            "judge_neg_rejection", "judge_coherence", "judge_similarity", "judge_reasoning"
        ]
        for col in required_judge_cols:
            if col not in df.columns:
                df[col] = 0.0 if col != "judge_reasoning" else ""
                
        # 3. Iterate
        updated_rows = []
        
        # Convert DataFrame to dicts for easier iteration/update
        records = df.to_dict('records')
        
        for i, row in enumerate(tqdm(records, desc="Judging")):
            # Check if we need to run the judge (if faithfulness is 0, we assume it wasn't run)
            # You could add more granular checks
            needs_eval = (row.get('judge_faithfulness', 0) == 0)
            
            if needs_eval:
                try:
                    query = row.get('question', "")
                    gen_ans = row.get('generated_answer', "")
                    ctx_text = row.get('retrieved_context_text', "")
                    is_impossible = row.get('is_impossible', False)
                    
                    # Parse gold answers safely
                    gold_field = row.get('gold_answers', "[]")
                    if isinstance(gold_field, str):
                        try:
                            gold_answers = ast.literal_eval(gold_field)
                        except:
                            gold_answers = []
                    else:
                        gold_answers = gold_field

                    # --- Run Metrics ---
                    
                    # 1. Faithfulness
                    f_score, f_reason = self.judge.evaluate_faithfulness(ctx_text, gen_ans)
                    api_calls = sleep_and_count(api_calls, sleep_interval)

                    # 2. Relevance
                    r_score, r_reason = self.judge.evaluate_relevance(query, gen_ans)
                    api_calls = sleep_and_count(api_calls, sleep_interval)

                    # 3. Context Utility
                    u_score, u_reason = self.judge.evaluate_context_utility(query, ctx_text)
                    api_calls = sleep_and_count(api_calls, sleep_interval)

                    # 4. Negative Rejection
                    if is_impossible:
                        n_score, n_reason = self.judge.evaluate_negative_rejection(query, gen_ans, is_impossible)
                        api_calls = sleep_and_count(api_calls, sleep_interval)
                    else:
                        n_score, n_reason = np.nan, "N/A"

                    # 5. Coherence
                    c_score, c_reason = self.judge.evaluate_coherence(gen_ans)
                    api_calls = sleep_and_count(api_calls, sleep_interval)

                    # 6. Similarity
                    best_gt = gold_answers[0] if gold_answers else ""
                    s_score, s_reason = self.judge.evaluate_answer_similarity(best_gt, gen_ans)
                    api_calls = sleep_and_count(api_calls, sleep_interval)

                    # Update Row
                    row['judge_model'] = judge_model
                    row['judge_faithfulness'] = f_score
                    row['judge_relevance'] = r_score
                    row['judge_utility'] = u_score
                    row['judge_neg_rejection'] = n_score
                    row['judge_coherence'] = c_score
                    row['judge_similarity'] = s_score
                    row['judge_reasoning'] = f"Faith: {f_reason} | Util: {u_reason} | Rel: {r_reason} | Coherence: {c_reason} | Sim: {s_reason}"
                    
                except Exception as e:
                    print(f"⚠️ Error evaluating row {i}: {e}")
                    api_calls = sleep_and_count(api_calls, sleep_interval)
            
            updated_rows.append(row)

        # 4. Save & Summarize
        result_df = pd.DataFrame(updated_rows)
        
        # Construct filename
        if isinstance(results_path_or_df, str):
            base, ext = os.path.splitext(results_path_or_df)
            new_filename = f"{base}_judged{ext}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"judged/re_evaluated_results_{timestamp}.csv"
            print(f"Saving evaluated results to {new_filename}")
            
        result_df.to_csv(new_filename, index=False)
        
        # Summary Stats
        summary = {
            "Total": len(result_df),
            "Avg Faithfulness": result_df['judge_faithfulness'].mean(),
            "Avg Relevance": result_df['judge_relevance'].mean(),
            "Avg Context Utility": result_df['judge_utility'].mean(),
            "Avg Coherence": result_df['judge_coherence'].mean(),
            "Avg Semantic Similarity": result_df['judge_similarity'].mean(),
            "Total API Calls": api_calls,
        }
        
        print("\n📊 RE-EVALUATION SUMMARY")
        print("="*40)
        for k, v in summary.items(): print(f"{k:<25}: {v:.4f}")
        print("="*40)
        print(f"💾 Saved: {new_filename}")
        
        return result_df, summary