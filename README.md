# Main references
* Attention is all you need - https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
* Transformer Tutorial - https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
# Masterplan
1. Train transformer on translation tasks for fixed number of epochs
2. Apply pruning algorithms
3. Analyze BLEU score before and after pruning

# To Do - Beginning
* [x] Integrate Encoder, Decoder, Seq2Seq Model
* [x] Add Multimodal Translation Dataset (WMT16 Multi30k Dataset)
* [x] Add BLEU Metric
* [x] Add Training and Evaluation Loop
* [ ] Add more Datasets (e.g. WMT16 News)
    - [x] Choose framework for dataloaders (HuggingFace vs. Torchtext)
    - [x] add WMT14 by TranslationDataset class and test
    - [x] add WMT14 by WMT14 class and test
