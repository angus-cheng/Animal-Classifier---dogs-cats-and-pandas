# k-NN algorithm implementation
from sklearn.neighbors import KNeighborsClassifier
# helper utility to convert labels represented as strings to integers where
# there is one unique integer per class label
from sklearn.preprocessing import LabelEncoder
# creates training and test splits
from sklearn.model_selection import train_test_split
# utility function used to evalaute performance of classifier
from sklearn.metrics import classification_report
from preprocessing.SimplePreprocessor import SimplePreprocessor
from datasets.SimpleDatasetLoader import SimpleDatasetLoader
from imutils import paths
import argparse

# construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
# required path to where input image dataset resides
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
# optional number of k neighbours to apply with k-NN algorithm
ap.add_argument("-k", "--neighbors", type=int, default=1,
                help="# of nearest neighbors for classification")
# optional number of concurrent jobs to run when computing distance between an
# input data point and the training set
ap.add_argument("-j", "--jobs", type=int, default=1,
                help="""# of jobs for k-NN distance (-1 uses all avaialable
                cores)""")
args = vars(ap.parse_args())

# retrieve list of images to analyse
print("[INFO] loading images...")
# grab file path to all images of dataset
imagePaths = list(paths.list_images(args["dataset"]))

# initialise the image preprocessor, load dataset from disk, and reshape the
# data matrix
# resize images to 32x32
sp = SimplePreprocessor(32, 32)
# initialises dataset loader and implies sp will be applied to every image
sdl = SimpleDatasetLoader(preprocessors=[sp])
# loads actual image dataset from disk with array shape of (3000, 32, 32, 3)
# 3000 = images , 32 x 32 = pixels, 3 = rgb channels
(data, labels) = sdl.load(imagePaths, verbose=500)
# flattens 32 x 32 x 3 images into (3000, 3072) array
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
# computes number of bytes array consumes (MB)
print("[INFO] feature matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

# encode labels as integers where each class has one unique integer
# cat = 0, dog = 1, panda = 2
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition data and labels into training and testing splits using 75% of the
# data for training and the remaining 25% for testing
# X = data points, Y = class labels
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25,
                                                  random_state=42)

# train and evaluate k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
# initialise KNeighborsClassifier class
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
                             n_jobs=args["jobs"])
# trains classifier - k-NN stores data internally to create predictions on the
# test set by computing distance between input data and trainX data
model.fit(trainX, trainY)
# evaluate classifier with testY class labels, predicted class labels and
# names of class labels
print(classification_report(testY, model.predict(testX),
      target_names=le.classes_))
