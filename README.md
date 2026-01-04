# Viewport-Unaware Full-Reference Omnidirectional Image Quality Assessment with Inter-Patch and Sequence Similarity
[Jiebin Yan], [Zhiyong Liu], [Junjie Chen], [Xiaoyu Xu], [Pengfei Chen],  [Yuming Fang]

## Database:JUFE-10K

## :hammer_and_wrench: Usage

### ERP Patch Extraction
If you want to retrain the IPSS model, using JUFE-10K database or another database, you first need to prepare ERP patch.
```
run sampling_alter.py
```
### Training IPSS
Modify the configuration in config.py
- Modify training and test dataset path
