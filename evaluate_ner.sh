#!/usr/bin/env bash

for i in {0..11}
do
    echo "Evaluating the model # $i"
    python -m allennlp.run evaluate /Users/daniel/ideaProjects/allennlp/ner_output_train/model.tar.gz $1 --weights-file /Users/daniel/ideaProjects/allennlp/ner_output_train/model_state_epoch_$i.th
done
