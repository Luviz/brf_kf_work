#! /usr/bin/python
import cv2 as cv
from sys import argv


def processMeter(img, k=91):
    # blur the img
    img = img.copy()
    b_img = cv.GaussianBlur(img, (k, k), 500)
    img_HSV = cv.cvtColor(b_img, cv.COLOR_BGR2HSV_FULL)

    saturation = img_HSV[:, :, 1]
    value = img_HSV[:, :, 2]

    _, thresh1 = cv.threshold(saturation, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
    _, thresh2 = cv.threshold(value, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
    out_img = thresh2 | thresh1
    threshold = 200
    canny_img = cv.Canny(out_img, threshold, threshold * 2)
    return canny_img


def selectMeterLocation(img):
    new_img = processMeter(img)
    contours, _ = cv.findContours(new_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    mapArea = {
        w * h: (x, y, w, h) for x, y, w, h in [cv.boundingRect(cnt) for cnt in contours]
    }
    return mapArea[max(mapArea.keys())]


def corpImg(img, x, y, w, h):
    return img[y : y + h, x : x + w]


def main(src: str, dest_path: str):

    img = cv.imread(src)

    x, y, w, h = selectMeterLocation(img)
    sel_img = corpImg(img, x, y, w, h)

    cv.imwrite(f"{dest_path}/{src.split('/')[-1]}", sel_img)


if __name__ == "__main__":
    src = argv[1]
    dest_path = argv[2]
    main(src, dest_path)
