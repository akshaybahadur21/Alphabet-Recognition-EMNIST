import numpy as np
import cv2
import Alphabet_Recognizer_DL
import Alphabet_Recognizer_LR
import Alphabet_Recognizer_NN
from collections import deque
from mnist import MNIST


def main():
    emnist_data = MNIST(path='gzip\\', return_type='numpy')
    emnist_data.select_emnist('letters')
    x_orig, y_orig = emnist_data.load_training()

    train_x = x_orig[0:3000, :]
    Y = y_orig.reshape(y_orig.shape[0], 1)
    Y = Y[0:3000, :]
    Y = Y[:, 0]
    train_y = (np.arange(np.max(Y) + 1) == Y[:, None]).astype(int)

    X_test = x_orig[3000:3500, :]
    Y_test = y_orig.reshape(y_orig.shape[0], 1)
    Y_test = Y_test[3000:3500, :]
    Y_test = Y_test[:, 0]

    letter_count = {0: 'CHECK', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
                    11: 'k',
                    12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v',
                    23: 'w',
                    24: 'x', 25: 'y', 26: 'z', 27: 'CHECK'}

    d1 = Alphabet_Recognizer_LR.model(train_x.T, train_y.T, Y, X_test.T, Y_test, num_iters=800, alpha=0.00009,
                                      print_cost=True)
    w_LR = d1["w"]
    b_LR = d1["b"]

    d2 = Alphabet_Recognizer_NN.model_nn(train_x.T, train_y.T, Y, X_test.T, Y_test, n_h=100, num_iters=4000,
                                         alpha=0.005,
                                         print_cost=True)

    dims = [784, 100, 80, 50, 27]
    d3 = Alphabet_Recognizer_DL.model_DL(train_x.T, train_y.T, Y, X_test.T, Y_test, dims, alpha=0.05,
                                         num_iterations=900,
                                         print_cost=True)

    cap = cv2.VideoCapture(0)
    Lower_green = np.array([110, 50, 50])
    Upper_green = np.array([130, 255, 255])
    pts = deque(maxlen=512)
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    digit = np.zeros((200, 200, 3), dtype=np.uint8)
    ans1 = 0
    ans2 = 0
    ans3 = 0

    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imgHSV, Lower_green, Upper_green)
        blur = cv2.medianBlur(mask, 15)
        blur = cv2.GaussianBlur(blur, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        center = None

        if len(cnts) >= 1:
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 250:
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)
                M = cv2.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                pts.appendleft(center)
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 10)
                    cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 5)
        elif len(cnts) == 0:
            if len(pts) != []:
                blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                blur1 = cv2.medianBlur(blackboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
                if len(blackboard_cnts) >= 1:
                    cnt = max(blackboard_cnts, key=cv2.contourArea)
                    print(cv2.contourArea(cnt))
                    if cv2.contourArea(cnt) > 2000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        digit = blackboard_gray[y:y + h, x:x + w]
                        newImage = cv2.resize(digit, (28, 28))
                        newImage = np.array(newImage)
                        newImage = newImage.flatten()
                        newImage = newImage.reshape(newImage.shape[0], 1)
                        ans1 = Alphabet_Recognizer_LR.predict_for_cv(w_LR, b_LR, newImage)
                        ans2 = Alphabet_Recognizer_NN.predict_nn_for_cv(d2, newImage)
                        ans3 = Alphabet_Recognizer_DL.predict_for_cv(d3, newImage)
            pts = deque(maxlen=512)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Logistic Regression : " + str(letter_count[ans1]), (10, 410),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Shallow Network :  " + str(letter_count[ans2]), (10, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Deep Network :  " + str(letter_count[ans3]), (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", img)
        # cv2.imshow("Contours", thresh)
        k = cv2.waitKey(10)
        if k == 27:
            break


main()
