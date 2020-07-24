import cv2
import numpy as np
import os


class SimpleDatasetLoader:
    """Pre processes data"""
    def __init__(self, preprocessors=None):
        """
        Constructor of SimpleDataSetLoader.

        Args:
            preprocessors: Optionally pass in a list of image preprocessors
            that can be sequentially applied to a given input image.
        """
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        """
        Initialises the list of features and labels.

        Args:
            imagePaths: A list specifying the file paths to the images in our
            dataset.
            verbose: Verbose level can be used to print updates to console to
            monitor how many images has processed.
        """
        data = []   # the images
        labels = []     # class labels for the images

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming that the path
            # has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            # check t osee if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our preprocessed image as a "feature vector" by updating
            # the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show an update every 'verbose' images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f"[INFO] processed {i + 1}/{len(imagePaths)}")

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))
