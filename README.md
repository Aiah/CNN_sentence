# My Version of CNN Sentence Classification in Theano
### Requirements
Code is written in Python (2.7) and requires Theano (0.7).

Using the pre-trained `word2vec` vectors will also require downloading the binary file from
https://code.google.com/p/word2vec/


### Data Preprocessing
To process the raw data, run

```
python process_data.py path
```

where path points to the word2vec binary file (i.e. `GoogleNews-vectors-negative300.bin` file). 
This will create a pickle object called `mr.p` in the same folder, which contains the dataset
in the right format.

Note: This will create the dataset with different fold-assignments than was used in the paper.
You should still be getting a CV score of >81% with CNN-nonstatic model, though.

### Using the GPU
```
 THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cer_main.py
```


json：
``` json
/*
hidden_units: 100: feature map 个数， 2：最终分类个数
static: static代表embbeding不变
word_vectors: word2vec or rand
*/
```