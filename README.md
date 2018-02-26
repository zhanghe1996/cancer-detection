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

### Codes
* data_scripts/download_image: Download image from eyesnap database using the json file
* data_scripts/load_data: Generate csv file for future use of training and testing. With the flag --test, testing data will be generated as data/pupils2017_test.csv. 
Each line in the csv file is in the folloing format: First 32*32*3 features store 32*32 pixels for a pupil. The next 32*32*3 features store 32*32 pixels for the corresponding eye. The folloing string is the ObjectId in the json file. The next string indicate if this is a left eye or right eye. The next integer indicate if this eye is normal or not (0: normal, 1: otherwise).
* data_scripts/pupils2017: Load data from csv file. No need to worry about this file.
* data_scripts/pupils2017_dataset: Divide data into training and validation set. No need to worry about this file.
* engine/train_cnn: Code for training process. The trained model will be saved in ROOT_DIR/checkpoint after running
```
cd ROOT_DIR/code
python -m engine.train_cnn
```
Models after 200, 400, 600, 800, and 1000 iterations will be saved (training is preformed a lot of rounds). 
No need to worry about the code.
* engine/test_cnn: Output testing performance of the trained model. 
```
cd ROOT_DIR/code
python -m engine.test_cnn
```
The precision and accuracy on the validation set will be tested on the models after 200, 400, 600, 800, and 1000 iterations.



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
