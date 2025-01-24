import pyautogui
import cv2
import numpy as np
import pytesseract
import re
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class VideoSampler:

    def __init__ (self, width=1920, height=1080):
        
        self.width, self.height = width, height
        self.img_x, self.img_y = 256, 256
        self.prev_speed_1 = 0
        self.prev_speed_2 = 0
        self.prev_speed_3 = 0
        self.changed = 0
        self.conv_model = None
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def sample(self, region=None):
        self.screenshot = pyautogui.screenshot(region=region)
        
        # resize the frame to 360p
        # self.screenshot = self.screenshot.resize((1920, 1080))

        self.frame = np.array(self.screenshot)
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # save the state
        # cv2.imwrite('state.png', self.state)
        # reshape the state to 256
        self.state = cv2.resize(self.frame, (256, 256))
        return self.state
    
    def get_speed(self):
        speed_img = self.frame[180:220, 1780:1870]
        # invert image
        # _, speed_img = cv2.threshold(speed_img, 254, 255, cv2.THRESH_BINARY)
        inverted_image = cv2.bitwise_not(speed_img)
        #show image
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'        
        text = pytesseract.image_to_string(inverted_image, config=custom_config)
        # print('Extracted Speed:', text)
        numbers = re.findall(r'\d+', text)
        number = int(numbers[0]) if numbers else 0
        if abs(number - self.prev_speed_1) > 20 and self.changed < 6:
            number = self.prev_speed_1
            self.changed += 1
        else:
            self.changed = 0
        speed = np.mean([number, self.prev_speed_1, self.prev_speed_2, self.prev_speed_3])
        self.prev_speed_3 = self.prev_speed_2
        self.prev_speed_2 = self.prev_speed_1
        self.prev_speed_1 = number
        #     if number%10 == 1:
        # print('Speed', speed)
        return speed
        
    def filter_edges_for_roads(self, edges, min_length=160):
        """
        Filter edges to identify those likely corresponding to roads based on orientation and length.

        Parameters:
        - edges: List of edge segments, where each segment is represented by its endpoints.
        - min_length: Minimum length of an edge segment to be considered a road (in pixels).

        Returns:
        - List of edge segments identified as roads.
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        road_edges = []

        for contour in contours:
            # Approximate the contour to get a simpler polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the contour is long enough to be considered a road
            if cv2.arcLength(approx, True) >= min_length:
                # Check orientation of the contour (simple horizontal check)
                x, y, w, h = cv2.boundingRect(contour)
                if h > w:  # Adjust this condition based on the expected orientation of roads
                    road_edges.append(approx)

        return road_edges
        
    def find_road(self):
        # apply blur
        img = cv2.GaussianBlur(self.state, (5, 5), 0)
        
        #detect edges
        edges = cv2.Canny(img, 50, 150)
        filtered_edges = self.filter_edges_for_roads(edges, min_length=300)
        # mask = np.zeros_like(edges)
        
        # define a region of interest
        # polygon = np.array([[(0, self.height), (0, 0), (self.width, 0), (self.width, self.height)]])

        # cv2.fillPoly(mask, polygon, 255)
        # masked_edges = cv2.bitwise_and(filtered_edges, mask)
        
        #find road lines
        


        # lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    
        # Draw the filtered_edges on a new image
        line_image = np.zeros_like(self.frame)
        for edge in filtered_edges:
            cv2.drawContours(line_image, [edge], -1, (0, 255, 0), 2)



        #save image
        cv2.imwrite('road.png', line_image)

        return self.frame 

    def close(self):
        pass
    

if __name__ == '__main__':
    sampler = VideoSampler()
    cv2.namedWindow('Road', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Road', 1280, 720)
    while True:
        sampler.sample((0, 0, 1920, 1080))
        sampler.find_road()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
