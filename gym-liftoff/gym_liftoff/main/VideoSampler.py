import pyautogui
import cv2
import numpy as np
import pytesseract
import re

class VideoSampler:

    def __init__ (self):
        
        self.height, self.width = pyautogui.size()

        self.conv_model = None


    def sample(self, region=None):
        self.screenshot = pyautogui.screenshot(region=region)
        # resize the frame to 360p
        # self.screenshot = self.screenshot.resize((640, 360))

        self.frame = np.array(self.screenshot)
        self.state = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # save the state
        cv2.imwrite('state.png', self.state)
        return self.state
    
    def get_speed(self):
        speed_img = self.state[180:220, 1780:1860]
        _, binary_image = cv2.threshold(speed_img, 254, 255, cv2.THRESH_BINARY)
        inverted_image = cv2.bitwise_not(binary_image)
        cv2.imwrite('speed.png', inverted_image)
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'        
        text = pytesseract.image_to_string(inverted_image, config=custom_config)
        print('Extracted Speed:', text)
        numbers = re.findall(r'\d+', text)
        print('Extracted Numbers:', numbers)

    def close(self):
        pass
    

if __name__ == '__main__':
    sampler = VideoSampler()
