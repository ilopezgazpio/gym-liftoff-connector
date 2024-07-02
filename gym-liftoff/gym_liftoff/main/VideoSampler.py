import pyautogui
import cv2
import numpy as np
import pytesseract
import re

class VideoSampler:

    def __init__ (self):
        
        self.height, self.width = pyautogui.size()
        self.prev_speed_1 = 0
        self.prev_speed_2 = 0
        self.prev_speed_3 = 0
        self.changed = 0
        self.conv_model = None


    def sample(self, region=None):
        self.screenshot = pyautogui.screenshot(region=region)
        # resize the frame to 360p
        # self.screenshot = self.screenshot.resize((1920, 1080))

        self.frame = np.array(self.screenshot)
        self.state = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # save the state
        # cv2.imwrite('state.png', self.state)
        return self.state
    
    def get_speed(self):
        speed_img = self.state[180:220, 1780:1870]
        # invert image
        # _, speed_img = cv2.threshold(speed_img, 254, 255, cv2.THRESH_BINARY)
        inverted_image = cv2.bitwise_not(speed_img)
        #show image
        cv2.imshow('Speed', inverted_image)
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'        
        text = pytesseract.image_to_string(inverted_image, config=custom_config)
        # print('Extracted Speed:', text)
        numbers = re.findall(r'\d+', text)
        number = int(numbers[0]) if numbers else 0
        if abs(number - self.prev_speed_1) > 20 and self.changed < 6:
            number = self.prev_speed_1
            self.changed+=1
        else:
            self.changed = 0
        speed = np.mean([number, self.prev_speed_1, self.prev_speed_2, self.prev_speed_3])
        self.prev_speed_3 = self.prev_speed_2
        self.prev_speed_2 = self.prev_speed_1
        self.prev_speed_1 = number
        # if number:
        #     if number%10 == 1:
        print('Speed', speed)
        return speed
        

    def close(self):
        pass
    

if __name__ == '__main__':
    sampler = VideoSampler()
    cv2.namedWindow('Speed', cv2.WINDOW_NORMAL)
    while True:
        sampler.sample()
        sampler.get_speed()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
