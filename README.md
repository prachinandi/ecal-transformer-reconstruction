# ecal-transformer-reconstruction
While reading the LHCb calorimeter performance and calibration papers, I realised I needed more understanding of how energy reconstruction actually works. So I built a small synthetic ECAL setup with the help of AI, where I simulate showers, tokenize calorimeter cells and compare a classical sum and caliberation baseline with a transformer based regressor. The goal was not to replicate LHCb exactly but to understand the reconstruction problem and wanted to get started with this as a part of learning milestone to get familiar with the proj.

### Pipeline
1. Synthetic ECAL shower generation on 2D grid (Gaussian-ish shower + fluctuations + noise)
2. Tokenization keep top K energy cells per event: tokens: [x_norm, y_norm, log_e, e] + mask 
3. Baseline reconstruction: E_pred = a * sum(e_i) + b (learned calibration)
4. Transformer regressor: encoder only Transformer over sparse tokens

Ref: https://arxiv.org/pdf/2008.11556



