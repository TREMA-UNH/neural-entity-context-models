import json
import sys
import tqdm
import os
import argparse
from typing import List, Dict, Set, Tuple

def create_data(
        mode: str,
        run: Dict,
        qrels: Dict[str, List[str]],
        query_annotations: Dict[str, str],
        name2id: Dict[str, str],
        embeddings: Dict[str, List[float]]
) -> List[str]:

    data: List[str] = []

    for query_id, query_text in tqdm.tqdm(query_annotations.items(), total=len(query_annotations)):
        if query_id in run and query_id in qrels:
            retrieved_entities: List[str] = list(run[query_id].keys())
            relevant_entities: List[str] = qrels[query_id]
            non_relevant_entities: List[str] = [entity for entity in retrieved_entities if entity not in relevant_entities]
            entity_data: Dict = run
            if mode == 'pairwise':
                data.extend(to_pairwise_data(
                    query_id,
                    query_text,
                    relevant_entities,
                    non_relevant_entities[:len(relevant_entities)],
                    entity_data,
                    embeddings
                ))
            else:
                data.extend(to_pointwise_data(
                    query_entities,
                    relevant_entities,
                    non_relevant_entities[:len(relevant_entities)],
                    embeddings
                ))

    return data

def get_entity_neighbors(entity_data: Dict,
        query_id: str,
        entity_id: str) -> List[Dict]:
    
    neighbors_data = []
    
    if query_id in entity_data:
        if entity_id in entity_data[query_id]:
            for para_id, para_data in entity_data[query_id][entity_id].items():
                temp_para_data = dict()
                temp_para_data['paraid'] = para_id
                temp_para_data['paratext'] = para_data[0]
                temp_para_data['parascore'] = para_data[1]
                neighbors_data.append(temp_para_data)

    return neighbors_data


def to_pairwise_data(
        query_id: str,
        query_text: str,
        relevant_entities: List[str],
        non_relevant_entities: List[str],
        entity_data: Dict,
        embeddings: Dict[str, List[float]]
) -> List[str]:

    query = {'query_id': query_id,
            'query_text': query_text}


    entity_pairs: List[Tuple[str, str]] = [(a, b) for a in relevant_entities for b in non_relevant_entities if a != b]

    data: List[str] = []
    for pos_entity, neg_entity in entity_pairs:
        if pos_entity in embeddings and neg_entity in embeddings:
        
            pos_ent_neighbors_data = get_entity_neighbors(entity_data, query_id, pos_entity)
            neg_ent_neighbors_data = get_entity_neighbors(entity_data, query_id, neg_entity)

            pos_ent = {
                'entity_id': pos_entity,
                'entity_embedding': embeddings[pos_entity],
                'entity_neighbors': pos_ent_neighbors_data
            }
            neg_ent = {
                'entity_id': neg_entity,
                'entity_embedding': embeddings[neg_entity],
                'entity_neighbors': neg_ent_neighbors_data
            }
            data.append(json.dumps({
                'query': query,
                'ent_pos': pos_ent,
                'ent_neg': neg_ent
            }))
    return data



def to_pointwise_data(
        query_entities: List[str],
        relevant_entities: List[str],
        non_relevant_entities: List[str],
        embeddings: Dict[str, List[float]]

) -> List[str]:
    data: List[str] = []

    query = [{'entity_id': query_entity,'entity_embedding': embeddings[query_entity]}
        for query_entity in query_entities if query_entity in embeddings]

    if len(query) == 0:
        return []


    for ent_pos in relevant_entities:
        if ent_pos in embeddings:
            entity = {
                'entity_id': ent_pos,
                'entity_embedding': embeddings[ent_pos]
            }
            data.append(json.dumps({
                'query': query,
                'entity': entity,
                'label': 1
            }))

    for ent_neg in non_relevant_entities:
        if ent_neg in embeddings:
            entity = {
                'entity_id': ent_neg,
                'entity_embedding': embeddings[ent_neg]
            }
            data.append(json.dumps({
                'query': query,
                'entity': entity,
                'label': 0
            }))
    return data

def write_to_file(data: List[str], out_file: str) -> None:
    with open(out_file, 'a') as f:
        for item in data:
            f.write("%s\n" % item)


def read_tsv(file: str) -> Dict[str, str]:
    with open(file, 'r') as f:
        return dict((line.split('\t')[0], line.split('\t')[1].strip())  for line in f)


def load_entity_file(input_file: str) -> Dict:
    entity_data = dict()
    with open(input_file, 'r') as reader:
        entity_data = json.loads(reader.read())

    return entity_data


def load_file(file_path: str) -> Dict[str, List[str]]:
    rankings: Dict[str, List[str]] = {}
    with open(file_path, 'r') as file:
        for line in file:
            line_parts = line.split(" ")
            query_id = line_parts[0]
            entity_id = line_parts[2]
            entity_list: List[str] = rankings[query_id] if query_id in rankings else []
            entity_list.append(entity_id)
            rankings[query_id] = entity_list
    return rankings


def main():
    parser = argparse.ArgumentParser("Create a training file.")
    parser.add_argument("--mode", help='One of (pairwise|pointwise).', required=True, type=str)
    parser.add_argument("--entity-json", help='Entity json file.', required=True, type=str)
    parser.add_argument("--qrels", help='Entity ground truth file.', required=True, type=str)
    parser.add_argument("--query-annotations", help='File containing query annotations.', required=True, type=str)
    #parser.add_argument("--entity-name2id", help='EntityName --> EntityId mappings.', required=True, type=str)
    parser.add_argument("--entity-embeddings", help='Entity embedding file.', required=True, type=str)
    parser.add_argument("--output", help='Directory where to save.', required=True, type=str)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])


    print('Loading run file....')
    run: Dict = load_entity_file(args.entity_json)
    print('[Done].')

    print('Loading qrels file....')
    qrels: Dict[str, List[str]] = load_file(args.qrels)
    print('[Done].')

    print('Loading query annotations...')
    query_annotations: Dict[str, str] = read_tsv(args.query_annotations)
    print('[Done].')

    print('Loading entity embeddings...')
    with open(args.entity_embeddings, 'r') as f:
        embeddings: Dict[str, List[float]] = json.load(f)
    print('[Done].')

    save: str = args.save + '/' + 'train.' + args.mode + '.jsonl'

    data: List[str] = create_data(
        mode=args.mode,
        run=run,
        qrels=qrels,
        query_annotations=query_annotations,
        embeddings=embeddings
    )

    print('Writing to file...')
    write_to_file(data, save)
    print('[Done].')


if __name__ == '__main__':
    main()
