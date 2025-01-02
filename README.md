# synesis
Tools for holistic representation evaluation and understanding

[Read the Docs](https://synesis.rtfd.io)

*synesis*: unification, to bring (something) together that thereby gives meaning

## Run
For all, `--nolog` disables logging to Weights & Biases.
### Informativeness
**Downstream**: Predict factor of variation directly.
```
python -m synesis.informativeness.downstream -f <feature_name> -d <dataset_name> (-t <task_name>)
```
### Equivariance
**Parameters**: Predict transformation parameters given two features, one of them being a transformed version of the other.
```
python -m synesis.equivariance.parameters -tf <transform_name> -f <feature_name> -d <dataset_name> (-t <task_name>)
```
**Features**: Predict feature of transformed data given original feature and transformation parameters.
```
python -m synesis.equivariance.features -tf <transform_name> -f <feature_name> -d <dataset_name> (-t <task_name>)
```

## Extract features
**Feature extraction** helps speed up downstream training.
```
python -m synesis.extract -f <feature> -d <dataset> (-b <batch_size>)
```

## Develop
Before working with the repo, please install requirements (or ruff and pre-commit) and run
```pre-commit install``` to install the pre-commit ruff formatting hook.

Before committing new pretrained models/feature extractors, run `pytest tests/test_feature_extraction`. Before committing new datasets run `pytest tests/test_datasets`. For now, you might need to manually add the feature/dataset name in the test file's pytest fixture.

### Adding datasets
**Follow the structure of one of the existing datasets.**
1. place under `synesis/datasets/`
2. filename should be lowercase version of class name *(helps make it automatically available to all other components)*

**The necessary `__init__` parameters are:**
1. `feature` (str) which is the name of the feature extractor that is going to be used
2. `root` (str, Path) which should ideally contain a default path
3. `split` Optional[str] which will be `[None, "train", "validation", "test"]`  
4. `feature_config` which overrrides the default config
5. in datasets with variable length items: `itemization`, which is a bool to decide whether to return variable length items or items with equal-length subitems

**The dataset should have the following attributes:**
> consult existing datasets to copy implementations
1. `self.root`
2. `self.raw_data_paths` list of e.g. audio paths
3. `self.feature_paths` e.g. list of embedding paths will be, based on the feature extractor name provided (see example in other datasets)
4. `self.paths` which will be raw_data_paths or feature_paths based on the item_format provided (["raw", "feature"])
5. `self.labels` list of encoded labels
6. `self.label_encoder` which will be either the LaberEncoder or MultiLabelBinarizer from `sklearn`

**Some processes that need to be implemented:**
> consult existing datasets to copy implementations
1. Use default feature config and override with provided one
```python
from config.features import feature_configs

if not feature_config:
    feature_config = feature_configs[feature]
self.feature_config = feature_config
```
2. `__getitem__` should return a raw data item or feature based on self.`item_format`
3. `__getitem__` returns tensors on CPU, unless variable-length-item dataset, in which case if `itemization==True`, item is list of tensor subitems

4. If variable-length-item dataset, `synesis.datasets.dataset_utils.load_track` automatically does this for audio. If it's for other modality, you might need to implement on your own.

### Adding feature extractors (/pretrained models)
**Follow existing feature extractor implementations.**
1. place under `synesis/features/`
2. filename should be lowercase version of class name *(helps make it automatically available to all other components)*
3. if pretrained model, place weights in `models/pretrained/*.pt` with same lowercase name
4. the extractor should have a forward method that deals with batched and channeled data (b, c, feature_dims)
5. ideally, it should return unchanneled data (b, feature_dim)
6. add an entry to `config/features` with the feature name (same as class name) that contains at least `feature_dim`. For now, some adjustments might need to be made for other parameters...
