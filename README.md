# MOSLingo
Out-of-distribution detection for large semantic text classification for TU/e course Research topic of data mining (course code: 2AMM20)

This is an OOD detection framework named `MOSLingo` for intent detection task in natural language processing. Our idea comes from [MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space](https://arxiv.org/abs/2105.01879). We make experiments on natural language understanding field and offer some optimizations based on original framework.

Different from the grouping strategy in `MOS`, `MOSLingo` utilize four intent detection datasets as IND data and two as OOD data. The datasets cover areas such as financial, medical, programming, etc.

## Usage
### Data Preparation
We have offered processed data in dataset folder. See `README.md` in dataset folder for more.

### Grouped-softmax and ungrouped-softmax Finetuning
For grouped softmax, you should change dataset paths and training configurations.
Then you could run `finetune_MOSLingo_grouped.py` directly by
```
python finetune_MOSLingo_grouped.py
```

For ungrouped softmax, you should change dataset paths and training configurations.
Then you could run `finetune_MOSLingo_ungrouped.py` directly by
```
python finetune_MOSLingo_ungrouped.py
```

### Evaluation
for grouped / ungrouped evaluation, you should change dataset paths and configurations.
Then you could run eval python file directly by
```
python eval_MOSLingo_grouped.py
```

```
python eval_MOSLingo_ungrouped.py
```
