import cv2 as cv2

class SimplePreprocessor:
    """
    Simple Preprocessor that changes the size of oringal image, ignoring aspect ratio.
    """
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        """
        Stores target image width, height, and interpolation method used when resizing.

        Args:
            width: The target width of input image after resizing.
            height: The target height of input image after resizing.
            inter: An optional parameter used to control which interpolation
            algorithm is used when resizing.
        """
        self.width = width
        self.height = height
        self.inter = InterruptedError

    def preprocess(self, image):
        """
        Resizes the image to a fixed size, ignoring aspect ratio.

        Args:
            self: image self.
            image: The image to be preprocessed. 

        Returns:
            image: resizes original image to one with fixed size of width and height.
        """
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)