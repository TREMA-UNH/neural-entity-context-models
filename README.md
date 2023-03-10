# neural-entity-context-models

This repository is for the paper **Neural Entity Context Models** submitted to SIGIR 2023 Short Paper Track.

```Code``` contains the code of the Neural ECM models.

```Runs``` contains the run files generated by the ECM models.

## Graph Relevance Network


To train the GRN model, please run the following command:

``` 
python train.py --train $data/$mode/train.pairwise.jsonl --save-dir $data/$mode --dev $data/$mode/test.jsonl --save GRN.bin --run $data/$mode/test.run --query-in-emb 50 --ent-in-emb 100 --para-in-emb 50 --model-type pairwise --use-cuda --cuda $cuda --epoch 50 --batch-size 1000 --seed 91453 --layer-flag 1
```

## Graph Attention Network


To train the GAT model, please run the following command:

``` 
python train.py --train $data/$mode/train.pairwise.jsonl --save-dir $data/$mode --dev $data/$mode/test.jsonl --save GAT.bin --run $data/$mode/test.run --query-in-emb 50 --ent-in-emb 100 --para-in-emb 50 --model-type pairwise --use-cuda --cuda $cuda --epoch 50 --batch-size 1000 --seed 91453 --layer-flag 2
```

## GRN-ECM


To train the GRN-ECM model, please run the following command:

``` 
python train.py --train $data/$mode/train.pairwise.jsonl --save-dir $data/$mode --dev $data/$mode/test.jsonl --save GRN-ECM.bin --run $data/$mode/test.run --query-in-emb 1 --ent-in-emb 1 --para-in-emb 1 --model-type pairwise --use-cuda --cuda $cuda --epoch 50 --batch-size 1000 --seed 91453 --layer-flag 3
```



