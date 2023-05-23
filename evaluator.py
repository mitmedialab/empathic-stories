import numpy as np
from rouge import Rouge
import scipy.stats as stats
from evaluate import load
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

import logging
logger = logging.getLogger(__name__)

class Evaluator():
    def __init__(self, device, df_test_pairs, df_test_stories):
        self.rouge = Rouge()
        self.bertscore = load("bertscore", device=device)
        self.meteor = load('meteor', device=device)
        story_pair_dict = {}
        pair_ids = df_test_pairs['pairs'].apply(lambda x: [int(_) for _ in eval(x)])
        for i in range(0, len(df_test_pairs)):
            tup = pair_ids[i]
            if tup[0] not in story_pair_dict:
                story_pair_dict[tup[0]] = [(tup[1], df_test_pairs['similarity_empathy_human_AGG'][i])]
            else:
                story_pair_dict[tup[0]].append((tup[1], df_test_pairs['similarity_empathy_human_AGG'][i]))
            if tup[1] not in story_pair_dict:
                story_pair_dict[tup[1]] = [(tup[0], df_test_pairs['similarity_empathy_human_AGG'][i])]
            else:
                story_pair_dict[tup[1]].append((tup[0], df_test_pairs['similarity_empathy_human_AGG'][i]))        
        for tup in story_pair_dict:
            story_pair_dict[tup] = sorted(
                story_pair_dict[tup],
                key=lambda x: x[1],
                reverse = True
            )
        self.df_test_stories = df_test_stories
        self.df_test_pairs = df_test_pairs
        self.story_pair_dict = story_pair_dict
        self.qids = [_ for _ in sorted(list(self.story_pair_dict.keys()))]


    def get_bertscore(self, predictions, references):
        results = self.bertscore.compute(predictions=predictions, references=references, lang="en", rescale_with_baseline = True, model_type="microsoft/deberta-xlarge-mnli")
        return np.mean(results["precision"]), np.mean(results["recall"]), np.mean(results["f1"])

    def get_meteor(self, predictions, references):
        results = self.meteor.compute(predictions=predictions, references=references)
        return results["meteor"]

    def get_pearson(self, predictions, references):
        return stats.pearsonr(predictions, references)

    def get_spearman(self, predictions, references):
        return stats.spearmanr(predictions, references)

    def get_rouge_scores(self, predictions, references):
        return self.rouge.get_scores(hyps=predictions, refs=references, avg=True)

    def get_bleu_scores(self, predictions, references):
        bs = [nltk.translate.bleu_score.sentence_bleu([ref], hyp) for ref, hyp in zip(references.apply(lambda x: word_tokenize(x)), predictions.apply(lambda x: word_tokenize(x)))]
        return np.mean(bs)
    
    def precision_score(self, y_true, y_pred):
        return precision_score(y_true, y_pred)
        
    def recall_score(self, y_true, y_pred):
        return recall_score(y_true, y_pred)
    
    def f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred)
    
    def accuracy_score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    
    def confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)
    
    def compute_metrics(self, labels, model=None, scores=None):
        cosine_scores = np.asarray(scores)

        labels = np.asarray(labels)
        output_scores = {}
        for short_name, name, scores, reverse in [['cossim', 'Cosine-Similarity', cosine_scores, True]]:
            acc, acc_threshold = self.find_best_acc_and_threshold(scores, labels, reverse)
            f1, precision, recall, f1_threshold = self.find_best_f1_and_threshold(scores, labels, reverse)
            ap = average_precision_score(labels, scores * (1 if reverse else -1))

            logger.info("Accuracy with {}:           {:.2f}\t(Threshold: {:.4f})".format(name, acc * 100, acc_threshold))
            logger.info("F1 with {}:                 {:.2f}\t(Threshold: {:.4f})".format(name, f1 * 100, f1_threshold))
            logger.info("Precision with {}:          {:.2f}".format(name, precision * 100))
            logger.info("Recall with {}:             {:.2f}".format(name, recall * 100))
            logger.info("Average Precision with {}:  {:.2f}\n".format(name, ap * 100))

            output_scores[short_name] = {
                'accuracy' : acc,
                'accuracy_threshold': acc_threshold,
                'f1': f1,
                'f1_threshold': f1_threshold,
                'precision': precision,
                'recall': recall,
                'ap': ap
            }
        return output_scores

    def find_best_acc_and_threshold(self, scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(labels == 0)

        for i in range(len(rows)-1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i+1][0]) / 2

        return max_acc, best_threshold

    def find_best_f1_and_threshold(self, scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows)-1):
            score, label = rows[i]
            nextract += 1

            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return best_f1, best_precision, best_recall, threshold
    
    def get_retrieval_results(self, similarity_column, embeddings = None):
        story_pair_dict = self.story_pair_dict
        qids = self.qids
        docids = []
        human_scores = []
        for _ in sorted(list(story_pair_dict.keys())):
            docids.append([_[0] for _ in story_pair_dict[_]])
            human_scores.append([_[1] for _ in story_pair_dict[_]])

        precision = []
        kendall_taus = []
        weighted_taus = []
        spearmans = []

        new_story_pair_dict = {}
        reversed1 = {}
        for i in story_pair_dict:
            mapping = {}
            mapping2 = {}
            rank = 1
            for j in range(len(story_pair_dict[i])-1):
                tup1 = story_pair_dict[i][j]
                tup2 = story_pair_dict[i][j+1]
                mapping2[tup1[0]] = rank
                if rank not in mapping:
                    mapping[rank] = [tup1[0]]
                else:
                    mapping[rank].append(tup1[0])
                if tup1[1] != tup2[1]:
                    rank += 1
                if j == len(story_pair_dict[i]) - 2:
                    mapping2[tup2[0]] = rank
                    if rank not in mapping:
                        mapping[rank] = [tup2[0]]
                    else:
                        mapping[rank].append(tup2[0])
            new_story_pair_dict[i] = mapping
            reversed1[i] = mapping2

        story_pair_dict_machine = {}
        for key1 in story_pair_dict:
            for key2, human_score in story_pair_dict[key1]:
                pair_id = str(tuple(sorted([key1, key2])))
                machine_score = self.df_test_pairs[self.df_test_pairs['pairs'] == pair_id][similarity_column].iloc[0]       
                if key1 in story_pair_dict_machine:
                    story_pair_dict_machine[key1].append((key2, machine_score))
                else:
                    story_pair_dict_machine[key1] = [(key2, machine_score)]

        for tup in story_pair_dict_machine:
            story_pair_dict_machine[tup] = sorted(
                story_pair_dict_machine[tup],
                key=lambda x: x[1],
                reverse = True
            )

        new_story_pair_dict_machine = {}
        reversed2 = {}
        for i in story_pair_dict_machine:
            mapping = {}
            mapping2 = {}
            rank = 1
            for j in range(len(story_pair_dict_machine[i])-1):
                tup1 = story_pair_dict_machine[i][j]
                tup2 = story_pair_dict_machine[i][j+1]
                mapping2[tup1[0]] = rank
                if rank not in mapping:
                    mapping[rank] = [tup1[0]]
                else:
                    mapping[rank].append(tup1[0])
                if tup1[1] != tup2[1]:
                    rank += 1
                if j == len(story_pair_dict_machine[i]) - 2:
                    mapping2[tup2[0]] = rank
                    if rank not in mapping:
                        mapping[rank] = [tup2[0]]
                    else:
                        mapping[rank].append(tup2[0])
            new_story_pair_dict_machine[i] = mapping
            reversed2[i] = mapping2

        for key in sorted(list(reversed1.keys())):
            human = reversed1[key]
            human2 = new_story_pair_dict[key]
            machine = reversed2[key]
            machine2 = new_story_pair_dict_machine[key]
            mapping = sorted(list(reversed1[key].keys()))
            r1 = [reversed1[key][k] for k in mapping]
            r2 =[reversed2[key][k] for k in mapping]
            if len(human) > 0:
                p = 0
                for key2 in human2[1]:
                    if key2 in machine2[1]:
                        p = 1
                        break
                precision.append(p)
            spearman = stats.spearmanr(r1, r2).correlation
            kendall_tau = stats.kendalltau(r1, r2).correlation
            if not np.isnan(spearman):
                spearmans.append(spearman)
            if not np.isnan(kendall_tau):
                kendall_taus.append(kendall_tau)  

        precision_final = np.mean(precision)
        kendall_tau_final = np.mean(kendall_taus)
        spearman_final = np.mean(spearmans)

        return precision_final, kendall_tau_final, spearman_final
