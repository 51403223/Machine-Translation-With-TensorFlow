Machine Translation with Tensorflow

VERSION 1
1) attention_model_v1
_ Encoder has 2 layers: 1st layer is bidiretional lstm and 2nd layer is 2 stacked lstms
_ Decoder has 1 lstm with attention
_ Encoder pass both cell state and last hidden state to Decoder as initial state at first decode step (time=0)
2) infer_attention_model_v1
    a) Greedy Search
    * with decay learning rate after every 4 epochs
    _ tst2012: bleu= 20.476983485168752, max_order=4, smooth=False
    _ tst2013: bleu= 23.507148735156456, max_order=4, smooth=False
    * learning rate fixed to 1.0
    _ tst2012: bleu= 18.770609325156386, max_order=4, smooth=False
    _ tst2013: bleu= 20.645434410841183, max_order=4, smooth=False

VERSION 2
Same graph as version 1
Difference: just last hidden state of Encoder pass to Decoder at first step, cell state of Decoder will be zeros