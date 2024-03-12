import cv2
from PIL import Image
import numpy as np
import easyocr

class Scanner:
    def __init__(self, grayscale=False, edge_detect=False, thres_block_size=15, thres_C=20):
        self.grayscale = grayscale
        self.edge_detect = edge_detect
        self.thres_block_size = thres_block_size
        self.thres_C= thres_C

    def select_region(self, image_path, threshold=False):
        start_x = 0
        start_y = 0
        end_x = 0
        end_y = 0
        cropping = False
        
        def mouse_event(event, x, y, flags, param):
            nonlocal start_x, start_y, end_x, end_y, cropping
            if event == cv2.EVENT_LBUTTONDOWN:
                start_x, start_y, end_x, end_y = x, y, x, y
                cropping = True
            elif event == cv2.EVENT_MOUSEMOVE:
                if cropping:
                    end_x, end_y = x, y
                    image_copy = image.copy()
                    cv2.rectangle(image_copy, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                    cv2.imshow("image", image_copy)
            elif event == cv2.EVENT_LBUTTONUP:
                end_x, end_y = x, y
                cropping = False
                # Draw rectangle around the selected region
                image_copy = image.copy()
                cv2.rectangle(image_copy, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                cv2.imshow("image", image_copy)
                cv2.destroyAllWindows()
                
        image = cv2.imread(image_path)

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_event)
        cv2.imshow("image", image)
        cv2.waitKey(0)

        image = image[start_y:end_y, start_x:end_x]

        if threshold:
            image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.thres_block_size, self.thres_C)

        return image
    
class TextExtractor:
    def __init__(self, grayscale=False, edge_detect=False, thres_block_size=15, thres_C=20):
        self.scanner = Scanner(grayscale, edge_detect, thres_block_size, thres_C)
        self.reader = easyocr.Reader(['ja'])
    
    def extract_question_and_options(self, image, num_options=4, threshold=False):
        selected_region = self.scanner.select_region(image,threshold=threshold)
        result = self.reader.readtext(selected_region)
        result = [x[1] for x in result]
        suggested_options = result[-num_options:]
        suggested_question = ' '.join(result[:-num_options])
        return suggested_question, suggested_options