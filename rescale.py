import cv2 as cv

capture = cv.VideoCapture('Videos/dog.mp4')


cv.destroyAllWindows()

def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)


def changeRes(width, height):
    capture.set(3, width)
    capture.set(4, height)

changeRes(640, 480)




while True:
    isTrue, frame = capture.read()
    resized_frame = rescaleFrame(frame)
    cv.imshow('Video', frame)

    cv.rectangle(frame, (0, 0), (250, 250), (0, 255, 0), thickness=2)
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()