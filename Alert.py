import pygame
import cv2

class AlertSystem:
    def __init__(self, sound_file="./beep-warning-6387.mp3"):
        pygame.mixer.init()
        self.sound_file = sound_file
        self.sound_playing = False
        self.warning_visible = False
    
    def playsound(self, state):
        """1 = play sound, 0 = stop sound"""
        if state == 1 and not self.sound_playing:
            try:
                pygame.mixer.music.load(self.sound_file)
                pygame.mixer.music.play(loops=-1)
                self.sound_playing = True
            except Exception as e:
                print(f"Error playing sound: {e}")
        elif state == 0 and self.sound_playing:
            pygame.mixer.music.stop()
            self.sound_playing = False
    
    def dispWarn(self, frame, state):
        """1 = show warning on frame, 0 = hide warning. Returns modified frame."""
        if state == 1:
            h, w = frame.shape[:2]
            warning_text = "WARNING"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            
            (text_w, text_h), _ = cv2.getTextSize(warning_text, font, font_scale, thickness)
            x = (w - text_w) // 2
            y = (h + text_h) // 2
            
            cv2.rectangle(frame, (x - 10, y - text_h - 10), (x + text_w + 10, y + 10), (0, 0, 255), -1)
            cv2.putText(frame, warning_text, (x, y), font, font_scale, (255, 255, 255), thickness)
            self.warning_visible = True
        else:
            self.warning_visible = False
        return frame
    
    def stop(self):
        """Stop all alerts"""
        self.playsound(0)
        self.warning_visible = False