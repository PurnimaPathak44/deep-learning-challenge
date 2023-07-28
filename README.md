Overview of the Analysis 

The purpose of the analysis is to help the nonprofit foundation Alphabet Soup select the charity applicants for funding with the best chance of success in their ventures. To achieve this purpose, I am using machine learning and neural networks to use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.
Alphabet Soup’s business team has provided a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:





• EIN and NAME—Identification columns
• APPLICATION_TYPE—Alphabet Soup application type
• AFFILIATION—Affiliated sector of industry
• CLASSIFICATION—Government organization classification
• USE_CASE—Use case for funding
• ORGANIZATION—Organization type
• STATUS—Active status
• INCOME_AMT—Income classification
• SPECIAL_CONSIDERATIONS—Special considerations for application
• ASK_AMT—Funding amount requested
• IS_SUCCESSFUL—Was the money used effectively





To achieve our goal, I started with preprocessing the data.


Preprocessing the data


The process of data analysis began with cleaning the data. There were 2 columns- EIN and NAME —Identification columns which were not relevant in our analysis, so I began with dropping those 2 columns.
 Next thing was to identify and assign the target variable for the model which was the IS_SUCCESSFUL column and the rest of the columns were identified as the features for the model. After that the unique values for each column were determined for binning purpose. The number of data points for each unique value was used to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then it was checked if the binning was successful.


pd.get_dummies() was used to encode categorical variables. Then the preprocessed data was split into training and testing datasets.


Compile, Train, and Evaluate the Model



I attempted to compile the neuron network with 80 in the first layer and 30 in the second layer. The activation used was “relu” for both the hidden layers as it is non-linear data and “relu” activation performs better at non-linear data, and “sigmoid” activation was used for the output layer.



Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 80)                3520      
                                                                 
 dense_1 (Dense)             (None, 30)                2430      
                                                                 
 dense_2 (Dense)             (None, 1)                 31        
                                                                 
=================================================================
Total params: 5,981
Trainable params: 5,981
Non-trainable params: 0
_________________________________________________________________




The result of the evaluation using the test data yielded the below-



268/268 - 0s - loss: 0.5543 - accuracy: 0.7277 - 481ms/epoch - 2ms/step
Loss: 0.5543038249015808, Accuracy: 0.7276967763900757




The accuracy is at 72.77 percent, which is less than what we would like to see. So, the next process I took was to optimize it.








Optimization




I wanted to optimize the data and I wanted to see how it would work if I did not drop the “NAME” column. So, this time I started with dropping only the “EIN” column. This time I added 3 hidden layers. The first layer had 8 neurons, the second layer had 16 and the third layer had 24 neurons.

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 8)                 3576      
                                                                 
 dense_1 (Dense)             (None, 16)                144       
                                                                 
 dense_2 (Dense)             (None, 24)                408       
                                                                 
 dense_3 (Dense)             (None, 1)                 25        
                                                                 
=================================================================
Total params: 4,153
Trainable params: 4,153
Non-trainable params: 0
_________________________________________________________________



 This largely improved the performance in comparison to the initial training model that had an accuracy of 72%. This time the accuracy increased to 79%.



268/268 - 1s - loss: 0.4417 - accuracy: 0.7903 - 790ms/epoch - 3ms/step
Loss: 0.4417283833026886, Accuracy: 0.7903206944465637






Summary


 In short, Deep neural network machine learning model was really helpful in producing a model to predict if the charity applicants will produce successful results or not.
Adding the “Name” column was very important that yielded better accuracy than the targeted 75% which proves that the shape of the dataset matters a lot to improve accuracy and it is very important to keep in mind what columns are we dropping while pre-processing.




