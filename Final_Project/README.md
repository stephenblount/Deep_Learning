# Abstract:

Drowsy driving accounts for nearly 10% of all motor vehicle accidents per year in the United States. This equates to roughly 100,000 accidents, 71,000 injuries, and 1,550 deaths per year. The goal of this project is to implement a drowsiness detection program using convolutional neural networks (CNN) and computer vision. A CNN was trained using the MRL eye dataset to classify an eye as open or closed. Once an adequately accurate model was constructed. OpenCV and Haar cascade classifiers were used to detect the face and eyes of a person in real time from a web camera. The area of the eyes were isolated and converted into the correct format to be sent to the CNN model. The CNN model determined if the right and left eye were open or closed independently of each other. If both eyes are closed a counter increases. If both eyes are open or one is open the counter decreases or remains zero. If the counter reaches 15 a police siren is played to awaken the drowsy driver.

The data for the CNN was obtained from the Media Research Lab (MRL). The dataset contained 84,898 images of eyes. The dataset was annotated with the following properties: 

Subject ID: Data was collected from 37 different persons (33 male & 4 female)

Gender: male or female

Glasses: no or yes

Eye State: closed or open

Reflection: none, small &  big

Lighting Condition: bad or good

Sensor ID: RealSense, IDS, Aptina (different resolutions)

After the data was collected it was sorted into different folders for training and testing to be batch accessed. When training the neural network all images were resized to match the smallest image in the set 66 by 66 pixels. The images were collected using infrared cameras to allow for data collection at night. The infrared images still contained three channel inputs, but appeared grayscale when printed. The decision was made to convert every image to grayscale to reduce the amount of data that needed to be stored and processed. Additionally, before the images were sent into the model to be trained, they were scaled by a factor of 255 to normalize the inputs. The model was trained using 50 epochs and the learning rate was reduced on plateau to allow the model to converge at a minimum. With 50 epochs, accuracy eventually became asymptotic and the model training process was halted. The model achieved an accuracy of 96.5% on the testing data. 

After a neural network was constructed, OpenCV (Open Computer Vision) was used for object detection and classification in real time. Once the program had access to a web camera, each frame of the webcam video was converted into an image. The haar cascade classifiers were used for object detection of the face, right eye, and left eye. The right and left eyes were isolated and formatted to be sent into the CNN for eye state classification. A counter was created to track the progress of the classifier. If both eyes were closed the counter increased by one. If both eyes or either eye were open the counter decreased by one to a threshold of zero. If the counter was able to reach fifteen a police siren sound would play and the counter would reset to zero. 

The drowsiness detection system works well, but has its faults. The haar classifier seems to have trouble distinguishing between right and left eyes and will occasionally predict both eyes as right and both as left at the same time (e.g. predicting there are four eyes). Additionally, the CNN struggles in different light settings (e.g. too dark or too light)
