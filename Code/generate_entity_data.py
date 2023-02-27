import json
import argparse
import tqdm
from typing import List, Dict

def _read_jsonl_file(input_file:str)->List:
    para_data = []
    with open(input_file, 'r') as reader:
        for item in reader:
            para_data.append(json.loads(item))
    return para_data

def _generate_entity_data(input_file:str) -> Dict:
    '''
    Output: {queryid1: {entityid1: {paraid1: [paratext, para_norm_score],
                                  paraid2: [paratext, para_norm_score]},
                       entityid2: {paraid3: [paratext, para_norm_score],
                                   paraid1: [paratext, para_norm_score]}
                      },
             queryid2: {entityid1: {paraid: [paratext,para_norm_score]}}
            }
    '''

    entity_data = dict()

    para_data = _read_jsonl_file(input_file)

    for para in tqdm.tqdm(para_data,total=len(para_data)):
        outlinks = para['outlinkids']
        outlinks = list(filter(lambda x: x != '', outlinks))
        if len(outlinks) > 0:
            if para['queryid'] in entity_data:
                query_entity_data = entity_data[para['queryid']]
                para_norm_score = para['score']/len(outlinks)
                for entity in outlinks:
                    if entity in query_entity_data:
                        temp_neighbor_dict = query_entity_data[entity]
                        if para['paraid'] not in temp_neighbor_dict:
                            temp_neighbor_dict[para['paraid']] = [para['paratext'], para_norm_score]
                            query_entity_data[entity] = temp_neighbor_dict
                    else:
                        temp_neighbor_dict = dict()
                        temp_neighbor_dict[para['paraid']] = [para['paratext'], para_norm_score]
                        query_entity_data[entity] = temp_neighbor_dict
                entity_data[para['queryid']] = query_entity_data
            else:
                temp_ent_data = dict()
                para_norm_score = para['score']/len(outlinks)
                for entity in outlinks:
                    temp_neighbor_dict = dict()
                    temp_neighbor_dict[para['paraid']] = [para['paratext'], para_norm_score]
                    temp_ent_data[entity] = temp_neighbor_dict
                entity_data[para['queryid']] = temp_ent_data

    return entity_data

def _write_to_json_file(data:Dict,
        output_file: str) -> None:
    with open(output_file, 'w') as writer:
        writer.write(json.dumps(data))


def main(input_file: str,
        output_file: str):
    _ent_data = _generate_entity_data(input_file)
    _write_to_json_file(_ent_data, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create entity data file")
    parser.add_argument("--input",help="input paragraph jsonl file used to generate entity data",required=True)
    parser.add_argument("--output",help="output entity data json file",required=True)
    args = parser.parse_args()
    main(args.input, args.output)
