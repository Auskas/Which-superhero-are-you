#!/usr/bin/python3
# vibrator.py - a vibrator class for android

# Current Plyer release has an issue with android vibrator.
# This module fixes it.

from jnius import autoclass, cast
from plyer.platforms.android import SDK_INT
from plyer.platforms.android import activity

class Vibrator():

    def __init__(self):
        Context = autoclass("android.content.Context")
        vibrator_service = activity.getSystemService(Context.VIBRATOR_SERVICE)
        self.vibrator = cast("android.os.Vibrator", vibrator_service)
        if SDK_INT >= 26:
            self.VibrationEffect = autoclass("android.os.VibrationEffect")

        
    def vibrate(self, duration):
        # duration in milliseconds.
        if SDK_INT >= 26:
            self.vibrator.vibrate(self.VibrationEffect.createOneShot(int(duration), 
                                  self.VibrationEffect.DEFAULT_AMPLITUDE))
        else:
            self.vibrator.vibrate(int(duration))


    