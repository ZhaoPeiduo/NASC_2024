import cv2
import imutils
import numpy as np
from skimage.filters import threshold_local
from datetime import date

"""
Modified image scanner from https://github.com/Manu10744/image2scan
"""

class ImageScanner:
    """ Scanner that applies edge detection in order to scan an ordinary image into a grayscale scan
        while positioning the point of view accordingly if needed. """

    def __init__(self, image, box_mode=True):
        """
        :param image: Path to the image to scan
        :param show_results: Specifies whether to show intermediate results in GUI windows or not
        """
        self.image = image
        self.drawing_box = False
        self.box_start = None
        self.user_defined_contours = []
        self.box_mode = box_mode

    def __reset(self):
        self.box_start = None
        self.user_defined_contours = []

    def scan(self):
        """ Searches for an rectangular object in the given image and saves the scan result of that object
        in the destination directory as pdf file """
        screenContours = self.__get_contours()
        self.__reset()
        return self.__transform_and_scan(screenContours)

    def __get_contours(self):
        """ Transforms the image colors to black and white in a way so that only the edges become clearly visible. """
        cv2_image = cv2.imread(self.image)
        cv2_image = imutils.resize(cv2_image, height=500)

        if self.box_mode:
            window_name = "Select a box region to capture text"
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, self.__draw_box, cv2_image)
        else:
            window_name = "Select 4 Points and click on 'X'"
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, self.__select_points, cv2_image)

        while len(self.user_defined_contours) != 4:
            cv2.imshow(window_name, cv2_image)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

        # Transform the user defined points into a numpy array which openCV expects
        screenCnt = np.array(self.user_defined_contours)
        return screenCnt

    def __select_points(self, event, x, y, flags, image):
        """ Event Handler for click events which lets the user define 4 points in order to determine the
        object to be scanned when OpenCV itself failed to detect 4 edges
        :param x:  x-coordinate of the clicked point
        :param y:  y-coordinate of the clicked point
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.drawMarker(image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR,
                           markerSize=10, thickness=1, line_type=cv2.LINE_AA)

            self.user_defined_contours.append([x, y])
    
    def __draw_box(self, event, x, y, flags, image):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_box = True
            self.box_start = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_box:
                temp_box_end = (x, y)
                image_copy = image.copy()
                cv2.rectangle(image_copy, self.box_start, temp_box_end, (0, 255, 0), 2)
                cv2.imshow('Preview of box', image_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing_box = False
            box_end = (x, y)
            cv2.rectangle(image, self.box_start, box_end, (0, 255, 0), 2)
            self.user_defined_contours = [self.box_start,
                     (box_end[0], self.box_start[1]),
                     (box_end),
                     (self.box_start[0], box_end[1])
                    ]

    def __transform_and_scan(self, screenCnt):
        """ Transforms the perspective to a top-down view and creates the scan from the transformed image. """
        cv2_image = cv2.imread(self.image)
        ratio = cv2_image.shape[0] / 500.0
        transformed = self.__four_point_transform(cv2_image, screenCnt.reshape(4, 2) * ratio)

        transformed_grayscaled = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
        threshold = threshold_local(transformed_grayscaled, 7, offset=10, method="gaussian")
        transformed_grayscaled = (transformed_grayscaled > threshold).astype("uint8") * 255

        return transformed_grayscaled

    def __order_points(self, pts):
        # initialzie a list of coordinates that will be ordered such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def __four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them individually
        rect = self.__order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct the set of destination points to obtain a
        # "birds eye view",(i.e. top-down view) of the image, again specifying points in the top-left, top-right,
        # bottom-right, and bottom-left order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped