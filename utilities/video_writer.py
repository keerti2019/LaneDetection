import cv2


# Saves video from images
class VideoWriter:
    def __init__(self, file_name, frames_per_sec, width, height):
        self.height = height
        self.width = width
        self.frames_per_sec = frames_per_sec
        self.file_name = file_name
        self.out = cv2.VideoWriter(self.file_name,
                                   cv2.VideoWriter_fourcc(*'avc1'),
                                   self.frames_per_sec,
                                   (self.width, self.height))
        print("Output Video {} | FPS {} | Width {} | Height {}.".format(self.file_name, self.frames_per_sec, self.width, self.height))

    def add_image(self, image):
        self.out.write(image)

    def close(self):
        print("Closing video stream {}".format(self.file_name))
        self.out.release()
