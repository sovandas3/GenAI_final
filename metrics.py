# Delete everything in Evalution_matric.py and paste ONLY this:
from typing import List, Dict
from sacrebleu import sentence_bleu
from rouge_score import rouge_scorer
import re

class GenAIEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def simple_tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [word for word in text.split() if word]

    def bleu_score(self, reference: str, candidate: str) -> float:
        ref_tokens = [self.simple_tokenize(reference)]
        cand_tokens = self.simple_tokenize(candidate)
        try:
            return sentence_bleu(cand_tokens, ref_tokens).score / 100.0
        except:
            return 0.0

    def rouge_score(self, reference: str, candidate: str) -> Dict[str, float]:
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {k: v.fmeasure for k, v in scores.items()}
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def evaluate_generation(self, reference_texts: List[str], generated_texts: List[str]) -> Dict[str, float]:
        if len(reference_texts) != len(generated_texts):
            raise ValueError("Input lengths must match")
        bleu_scores = [self.bleu_score(ref, gen) for ref, gen in zip(reference_texts, generated_texts)]
        rouge_scores_list = [self.rouge_score(ref, gen) for ref, gen in zip(reference_texts, generated_texts)]
        
        return {
            "average_bleu": sum(bleu_scores) / len(bleu_scores),
            "average_rouge_1": sum(rs.get('rouge1', 0.0) for rs in rouge_scores_list) / len(rouge_scores_list),
            "average_rouge_2": sum(rs.get('rouge2', 0.0) for rs in rouge_scores_list) / len(rouge_scores_list),
            "average_rouge_l": sum(rs.get('rougeL', 0.0) for rs in rouge_scores_list) / len(rouge_scores_list),
            "n_samples": len(reference_texts)
        }
