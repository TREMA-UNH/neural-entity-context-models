from typing import List, Dict

import pytrec_eval


def get_metric(qrels: str, run: str, metric: str = 'map') -> float:

    # Read the qrel file
    with open(qrels, 'r') as f_qrel:
        qrel_dict = pytrec_eval.parse_qrel(f_qrel)

    # Read the run file
    with open(run, 'r') as f_run:
        run_dict = pytrec_eval.parse_run(f_run)

    # Evaluate
    evaluator = pytrec_eval.RelevanceEvaluator(qrel_dict, pytrec_eval.supported_measures)
    results = evaluator.evaluate(run_dict)
    mes = {}
    for _, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            mes[measure] = pytrec_eval.compute_aggregated_measure(measure,
                                                                  [query_measures[measure]
                                                                   for query_measures in results.values()])
    return mes[metric]


def get_mrr(qrels: str, trec: str, metric: str = 'mrr_cut_10') -> float:
    k = int(metric.split('_')[-1])

    qrel = {}
    with open(qrels, 'r') as f_qrel:
        for line in f_qrel:
            qid, _, did, label = line.strip().split()
            if qid not in qrel:
                qrel[qid] = {}
            qrel[qid][did] = int(label)

    run = {}
    with open(trec, 'r') as f_run:
        for line in f_run:
            qid, _, did, _, _, _ = line.strip().split()
            if qid not in run:
                run[qid] = []
            run[qid].append(did)

    mrr = 0.0
    for qid in run:
        rr = 0.0
        for i, did in enumerate(run[qid][:k]):
            if qid in qrel and did in qrel[qid] and qrel[qid][did] > 0:
                rr = 1 / (i + 1)
                break
        mrr += rr
    mrr /= len(run)
    return mrr
