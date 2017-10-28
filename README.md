# Cancer-detection

### Dataset Structure
```
|__eye_test
   |__Annotations
      |__*.npy
   |__Images
      |__*.jpeg
   |__ImageSets
      |__train.txt
```

### Running the test
* Go to **code** directory
```
cd ROOT_DIR/code
```
* Generate **pupils2017.csv**
```
python -m data_scripts.load_data
```
* Generate TensorFlow checkpoints
```
python -m engine.train_cnn
```
* Test accuracy
```
python -m engine.test_cnn
```
