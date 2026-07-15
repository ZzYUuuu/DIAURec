# DIAURec

This is the PyTorch implementation for the paper:

> Yu Zhang, Yiwen Zhang*, Yi Zhang, Lei Sang. **"Dual-Intent Space Representation Optimization for Recommendation"**. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'26). [[Paper](https://arxiv.org/abs/2604.09087)]

## Environment Setting

```python
python == 3.8.18
pytorch == 2.1.0 (cuda:12.1)
scipy == 1.10.1
numpy == 1.24.3
tqdm == 4.65.0
```

## Examples

We evaluate DIAURec on several benchmark recommendation datasets.
DIAURec optimizes collaborative representations by introducing prototype and distribution intent spaces.

## Run

The running command is as follows:

```python
%run encoder/train_encoder.py --model 'diaurec' --dataset 'dataset'
```

Please replace `'dataset'` with the name of the dataset you want to run.

## Acknowledgement

To ensure fair comparisons and maintain consistency, our model training framework, the LLM-generated user and item profiles, and their corresponding embedding representations are mainly adapted from the following repository:

https://github.com/HKUDS/RLMRec

We sincerely thank the authors for releasing their code and for their valuable contributions to the open-source community.

## Citation

If you find this work helpful, please cite it:

```bibtex
@inproceedings{Yu_DIAURec_2026,
  title = {Dual-Intent Space Representation Optimization for Recommendation},
  author = {Zhang, Yu and Zhang, Yiwen and Zhang, Yi and Sang, Lei},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2026},
  doi = {10.1145/3805712.3809551},
  pages = {2420--2430},
  numpages = {11}
}
```
