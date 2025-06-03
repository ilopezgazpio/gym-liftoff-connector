'''
Code is based on https://github.com/iosonofabio/virtual_gamepad and uinput official examples

For the moment, running as root is mandatory (see uinput examples where this is fixed)

sudo modprobe uinput

NOTE: whoever runs the script must have access to /dev/uinput. You can either run it as root/sudo, or give your user priviledges via a group, e.g.

sudo groupadd uinput
sudo usermod -aG uinput "$USER"
sudo chmod g+rw /dev/uinput
sudo chgrp uinput /dev/uinput
'''

from collections import defaultdict
import uinput
import time
import logging

logger = logging.getLogger(__name__)

events = (
    uinput.BTN_SOUTH,
    uinput.BTN_EAST,
    uinput.BTN_C,
    uinput.BTN_NORTH,
    uinput.BTN_WEST,
    uinput.BTN_Z,
    uinput.BTN_TL,
    uinput.BTN_TR,
    uinput.BTN_TL2,
    uinput.BTN_TR2,
    uinput.BTN_SELECT,
    uinput.BTN_START,
    uinput.BTN_MODE,
    uinput.BTN_THUMBL,
    uinput.BTN_THUMBR,
    uinput.BTN_TRIGGER_HAPPY1,
    uinput.BTN_TRIGGER_HAPPY2,
    uinput.BTN_TRIGGER_HAPPY3,
    uinput.BTN_TRIGGER_HAPPY4,
    uinput.BTN_TRIGGER_HAPPY5,
    uinput.BTN_TRIGGER_HAPPY6,
    uinput.BTN_TRIGGER_HAPPY7,
    uinput.BTN_TRIGGER_HAPPY8,
    uinput.ABS_X  + (0, 2047, 7, 127),
    uinput.ABS_Y  + (0, 2047, 7, 127),
    uinput.ABS_Z  + (0, 2047, 7, 127),
    uinput.ABS_RX + (0, 2047, 7, 127),
    uinput.ABS_RY + (0, 2047, 7, 127),
    uinput.ABS_RZ + (0, 2047, 7, 127),
    uinput.ABS_THROTTLE + (0, 2047, 7, 127)
)

class VirtualGamepad:

    action_map = {
        ""
        }

    def __init__(self):
        self.device = uinput.Device(
            events,
            vendor=0x1209,
            product=0x4f54,
            version=0x111,
            name="RadioMaster TX16S Joystick (connector)",
        )
        logger.info("Virtual Gamepad {} created..... OK")

    def throttle(self, value):
        stick = uinput.ABS_Y
        self.__emit__(stick, value)

    def zeroThrottle(self):
        self.throttle(0)

    def centerThrottle(self):
        self.throttle(2047 // 2)

    def fullThrottle(self):
        self.throttle(2047)

    def yaw(self, value):
        stick = uinput.ABS_X
        self.__emit__(stick, value)

    def zeroYaw(self):
        self.yaw(0)

    def centerYaw(self):
        self.yaw(2047 // 2)

    def fullYaw(self):
        self.yaw(2047)

    def roll(self, value):
        stick = uinput.ABS_RX
        self.__emit__(stick, value)

    def zeroRoll(self):
        self.roll(0)

    def centerRoll(self):
        self.roll(2047 // 2)

    def fullRoll(self):
        self.roll(2047)

    def pitch(self, value):
        stick = uinput.ABS_RY
        self.__emit__(stick, value)

    def zeroPitch(self):
        self.pitch(2047)

    def centerPitch(self):
        self.pitch(2047 // 2)

    def fullPitch(self):
        self.pitch(0)

    def act(self, action):
        self.throttle(int(action[0]))
        self.yaw(int(action[1]))
        self.roll(int(action[2]))
        self.pitch(int(action[3]))

    def reset(self):
        self.zeroThrottle()
        self.centerYaw()
        self.centerRoll()
        self.centerPitch()

    def __emit__(self, event, value):
        self.device.emit(event, value)

    def close(self):
        self.device.destroy()





if __name__ == '__main__':
    device = VirtualGamepad()

    # #wait for controller to be ready
    # time.sleep(5)
    # # Start throttle
    # device.throttle(0)
    # time.sleep(2)
    # device.throttle(2047)
    # time.sleep(2)
    # device.throttle(0)
    # time.sleep(2)
    # device.throttle(2047)

    # # Start yaw
    # device.yaw(0)
    # time.sleep(2)
    # device.yaw(2047)
    # time.sleep(2)

    # # Start roll
    # device.roll(0)
    # time.sleep(2)
    # device.roll(2047)
    # time.sleep(2)

    # # Start pitch
    # device.pitch(0)
    # time.sleep(2)
    # device.pitch(2047)
    # time.sleep(2)


    '''
    device.emit(uinput.BTN_TR, 1)
    device.emit(uinput.BTN_THUMBL, 1)
    device.emit(uinput.BTN_THUMBR, 1)
    device.emit(uinput.ABS_Y, 0)                    # Zero Y
    device.emit(uinput.ABS_Y, 255)                  # Max Y
    device.emit(uinput.ABS_X, 0)                    # Zero X
    device.emit(uinput.ABS_X, 255)                  # Max X
    '''
