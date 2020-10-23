'''
Last modified time: 2019/07/25
OpenCV face detection,
function: face_detection, box_reduce, face_cut
'''
import cv2
import os
import numpy as np


# Create the cascade
# Please confirm that the cascPath's file exists.
cascPath = "./cv2_data/frontalface_xml/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


def face_detection(image):
    '''
    OpenCV face detection

    input:

        image: gray or RGB(BGR) image

    output:

        find face: bool, find a face or no.
        face local: list, the Largest area face bounding box: [left, top, right, bottom]
    '''

    # Detect faces in the gray_image
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # print('faces', faces)
    # print("Found {0} faces!".format(len(faces)))

    find_face = False
    face_local = []
    if len(faces) > 0:
        find_face = True

        max_size = 120 * 120
        for (x, y, w, h) in faces:      # find max area face
            if w * h >= max_size:
                max_size = w * h
                face_local = list((x, y, x + w, y + h))

    return find_face, face_local


def face_cut(f_lc, img):
    '''
    input:

        f lc: face bounding box: [left, top, right, bottom]
        img: source image (BGR or Gray)

    output:

        image face: a face image (BGR, Gray match source image)
    '''

    image_face = img[f_lc[1] : f_lc[3], f_lc[0] : f_lc[2]]

    return image_face


def box_reduce(f_lc, cut_percent=30):
    '''
    Let face bounding box more close to the face,
    cut percent 30% while reduce the area by 30%

    input:

        f lc: face bounding box: [left, top, right, bottom]
        cut percent: reduce percent(Default: 30%)

    output:

        f lc: list, face bounding box: [left, top, right, bottom]
    '''

    box_x_len = f_lc[2] - f_lc[0]
    box_y_len = f_lc[3] - f_lc[1]
    cut_x_len = int(box_x_len * (cut_percent / 100))
    cut_y_len = int(box_y_len * (cut_percent / 100))
    f_lc[0] = f_lc[0] + int(cut_x_len / 2)
    f_lc[1] = f_lc[1] + int(cut_y_len / 2)
    f_lc[2] = f_lc[2] - int(cut_x_len / 2)
    f_lc[3] = f_lc[3] - int(cut_y_len / 2)

    return f_lc


if __name__ == "__main__":

    ## read one image ###
    imagePath = './cv2_data/20190501_104336.png'
    print(imagePath)
    image = cv2.imread(imagePath)
    success, local = face_detection(image)
    if success:
        # print(local)
        local = box_reduce(local)
        cv2.rectangle(
            image, (local[0], local[1]), (local[2], local[3]), (0, 255, 0), 4, cv2.LINE_AA)
    else:
        print("No face in picture.")

    cv2.imshow("Frame", image)

    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
