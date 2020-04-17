###############################################################################################################################################
#Welcome, today i am going to show you a demo of implementing a simple digit recognition using open cv and deep neural networks
# this is the main code 
# Coded by Hemanta Ingle, MSEE, SJSU, CA,hemanta.ingle@sjsu.edu, 
# Code Status- Compiled and run successfully 
# Note- Tensorflow version 2 or above will cause some library errors, suggested to downgrade if above version 2
#  Reference -https://dev.to/noorkhokhar10, the three working models in our code have been taken from Noors Github repository
################################################################################################################################################

# First we will import all the necessary libraries for the image recognition
import numpy as np  # a numerical library that has many useful built in math functions
import input_data
import cv2 # Computer Vision library 
import Digit_Recognizer_DL # importing Deep Learning model 
import Digit_Recognizer_LR # Logistic Regression Machine Learning Model
import Digit_Recognizer_NN # Feedforward fully connected neural network


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False) # read the data sets
    data = mnist.train.next_batch(8000)  # training batch
    train_x = data[0] # element from list 0 is input training data 
    Y = data[1] # element from list 1 is desired training output 
    train_y = (np.arange(np.max(Y) + 1) == Y[:, None]).astype(int) # convert data into integer
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    tb = mnist.train.next_batch(2000) #test batch
    Y_test = tb[1] # desired output for test
    X_test = tb[0] # input for test
    # 0.00002-92
    # 0.000005-92, 93 when 200000 190500

    #d1 = Digit_Recognizer_LR.model(train_x.T, train_y.T, Y, X_test.T, Y_test, num_iters=1500, alpha=0.05, print_cost=True)
    #w_LR = d1["w"]
    #b_LR = d1["b"]

    #d2 = Digit_Recognizer_NN.model_nn(train_x.T, train_y.T, Y, X_test.T, Y_test, n_h=100, num_iters=1500, alpha=0.05,print_cost=True)
    dims = [784, 100, 80, 50, 10]
    d3 = Digit_Recognizer_DL.model_DL(train_x.T, train_y.T, Y, X_test.T, Y_test, dims, alpha=0.5, num_iterations=1100, print_cost=True)


    cap = cv2.VideoCapture(0) # capture the video from the usb cam

    while (cap.isOpened()): 
        ret, img = cap.read()# returning an image from the video source
        img, contours, thresh = get_img_contour_thresh(img)
        ans1 = ''
        ans2 = ''
        ans3 = ''
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea) # max function iterates over contour area and determines a contour
            if cv2.contourArea(contour) > 2500:
                # print(predict(w_from_model,b_from_model,contour))
                x, y, w, h = cv2.boundingRect(contour)#Let (x,y) be the top-left coordinate of the     rectangle and (w,h) be its width and height
                # newImage = thresh[y - 15:y + h + 15, x - 15:x + w +15]
                newImage = thresh[y:y + h, x:x + w]
                newImage = cv2.resize(newImage, (28, 28))# Resizing the image to 28*28 pixels
                newImage = np.array(newImage)# create an array of that image
                newImage = newImage.flatten() # Create an array of inputs 756
                newImage = newImage.reshape(newImage.shape[0], 1)
                #ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage) # get the prediction from the Logistic regression model
                #ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage) # get the prediction from the Neural Nettwork model
                ans3 = Digit_Recognizer_DL.predict(d3, newImage)  # get the prediction from the Deep Learning model

        x, y, w, h = 0, 0, 300, 300 # Display the boxed that will show the predicted output
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320),
                  #  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350),
                   # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", img)
        cv2.imshow("Contours", thresh)
        k = cv2.waitKey(10)
        if k == 27:
            break


def get_img_contour_thresh(img):
    x, y, w, h = 0, 0, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1

main()
