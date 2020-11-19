import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

low_threshold = 90
high_threshold = 150
kernel_size = 3
rho = 1
theta = np.pi/180
threshold = 3
minLineLength = 50
MaxGap = 100
lower_blue = np.array([245,245,245])
upper_blue = np.array([255,255,255])

cap = cv.VideoCapture('solidWhiteRight.mp4')
while True:
    ret, image = cap.read()
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    mask = cv.inRange(image, lower_blue, upper_blue)
    res = cv.bitwise_and(image, image, mask=mask)



    region_of_interest_vertices = [
        (0, height),
        (width / 2, 300),
        (width, height)
    ]


    def region_of_interest(img, vertices):
        mask = cv.inRange(img, lower_blue, upper_blue)
        channel_count = img.shape[2]
        match_mask_color = (255,) * channel_count
        cv.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv.bitwise_and(img, img, mask=mask)
        return masked_image


    cropped_image = region_of_interest(image,
                                       np.array([region_of_interest_vertices], np.int32), )

    blurred_gray = cv.GaussianBlur(res, (kernel_size, kernel_size), 0)
    canny_edgyy = cv.Canny(blurred_gray, low_threshold, high_threshold)
    line_image = np.copy(image) * 0
    lines = cv.HoughLinesP(canny_edgyy, rho, theta, threshold, np.array([]),
                           minLineLength, MaxGap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    color_edges = np.dstack((canny_edgyy, canny_edgyy, canny_edgyy))
    lines_edges = cv.addWeighted(color_edges, 0.8, line_image, 1, 0)
    cv.imshow('what is this??',res)
    cv.imshow('cropped image', cropped_image)
    cv.imshow('gray', gray)
    cv.imshow('Blur', blurred_gray)
    cv.imshow('frame', canny_edgyy)
    cv.imshow('Lines',lines_edges)
    if cv.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
