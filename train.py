import tensorflow as tf
import os
from scipy.io import loadmat
from pyimagesearch import config
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import time

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# # Optionally, you can limit the GPU memory to a specific fraction (e.g., 90%)
# # Uncomment and set memory_limit in MiB if needed
# gpus = tf.config.experimental.list_physical_devices('GPU')
# try:
#     for gpu in gpus:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpu,
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(0.8*4096))])  # Set to 4096 MiB (4 GB)
# except RuntimeError as e:
#     print(e)

# CPU RUN
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# import tensorflow as tf

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print("Running on CPU...")


# initialize the list of data (images), class labels, target bounding
# box coordinates, and image paths
print("[INFO] loading dataset...")
data = []
labels = []
bboxes = []
imagePaths = []

for category in os.listdir(config.IMAGES_PATH):
    category_image_dir = os.path.join(config.IMAGES_PATH, category)
    category_annotation_dir = os.path.join(config.ANNOTS_PATH, category)
    
    if os.path.isdir(category_image_dir) and os.path.isdir(category_annotation_dir):
        for image_file in os.listdir(category_image_dir):
            image_name, _ = os.path.splitext(image_file) # splits image at '.jpg'
            annotation_name = image_name.replace('image_', 'annotation_')
            annotation_file = annotation_name + '.mat'
            annotation_path = os.path.join(category_annotation_dir, annotation_file)
            
            if os.path.exists(annotation_path):
                image_path = os.path.join(category_image_dir, image_file)
                annotation_data = loadmat(annotation_path)
                (startX, startY, endX, endY) = annotation_data['box_coord'][0]
                image = cv2.imread(image_path)
                (h, w) = image.shape[:2]
                # scale the bounding box coordinates relative to the spatial dimensions of the input image
                startX = float(startX) / w
                startY = float(startY) / h
                endX = float(endX) / w
                endY = float(endY) / h

                # load the image and preprocess it
                image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
                image = tf.keras.preprocessing.image.img_to_array(image)

                # update our list of data, class labels, bounding_boxes and filenames
                data.append(image)
                labels.append(category)
                bboxes.append((startX, startY, endX, endY))
                imagePaths.append(image_path)


# convert the data, class labels, bounding boxes, and image paths to
# NumPy arrays, scaling the input pixel intensities from the range [0, 255] to [0, 1]
data = np.array(data, dtype='float32') / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes, dtype='float32')
imagePaths = np.array(imagePaths)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# only there are only two labels in the dataset, then we need to use
# Keras/TensorFlow's utility function as well
if len(lb.classes_) == 2:
	labels = tf.keras.utils.to_categorical(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing

split = train_test_split(data, labels, bboxes, imagePaths,
                         test_size=0.2, random_state=42)

# unpacj the data split
(X_train, X_test) = split[:2]
(y_train, y_test) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

# write the testing image paths to disk so that we can use then
# when evaluating/testing our object detector
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testPaths))
f.close()

# load the VGG16 network, ensuring the head FC layers are left off
vgg = tf.keras.applications.VGG16(weights='imagenet', include_top=False,
                                  input_tensor=tf.keras.layers.Input(shape=(224, 224, 3)))

# freeze all VGG layers so they will *not* be updated during the training process
vgg.trainable = False

# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = tf.keras.layers.Flatten()(flatten)

# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = tf.keras.layers.Dense(128, activation="relu", kernel_initializer='he_normal')(flatten)
bboxHead = tf.keras.layers.Dense(64, activation="relu", kernel_initializer='he_normal')(bboxHead)
bboxHead = tf.keras.layers.Dense(32, activation="relu", kernel_initializer='he_normal')(bboxHead)
bboxHead = tf.keras.layers.Dense(4, activation="sigmoid",
	name="bounding_box")(bboxHead)
# construct a second fully-connected layer head, this one to predict
# the class label
softmaxHead = tf.keras.layers.Dense(512, activation="relu", kernel_initializer='he_normal')(flatten)
softmaxHead = tf.keras.layers.Dropout(0.5)(softmaxHead)
softmaxHead = tf.keras.layers.Dense(512, activation="relu", kernel_initializer='he_normal')(softmaxHead)
softmaxHead = tf.keras.layers.Dropout(0.5)(softmaxHead)
softmaxHead = tf.keras.layers.Dense(len(lb.classes_), activation="softmax",
	name="class_label")(softmaxHead)
# put together our model which accept an input image and then output
# bounding box coordinates and a class label
model = tf.keras.models.Model(
	inputs=vgg.input,
	outputs=(bboxHead, softmaxHead))

# define a dictionary to set the loss methods -- categorical
# cross-entropy for the class label head and mean absolute error for the bounding box head
losses = {
	"class_label": "categorical_crossentropy",
	"bounding_box": "mean_squared_error",
}
# define a dictionary that specifies the weights per loss (both the
# class label and bounding box outputs will receive equal weight)
lossWeights = {
	"class_label": 1.0,
	"bounding_box": 1.0
}
# initialize the optimizer, compile the model, and show the model summary
opt = tf.keras.optimizers.Nadam(learning_rate=config.INIT_LR)
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
print(model.summary())

# construct a dictionary for our target training outputs
trainTargets = {
	"class_label": y_train,
	"bounding_box": trainBBoxes
}
# construct a second dictionary, this one for our target testing
# outputs
testTargets = {
	"class_label": y_test,
	"bounding_box": testBBoxes
}

# Create TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{str(int(time.time()))}')
# Create EarlyStopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3, 
                                                           restore_best_weights=True)

# train the network for bounding box regression and class label
# prediction
print("[INFO] training model...")
history = model.fit(
	X_train, y_train,
	validation_data=(X_test, y_test),
	batch_size=config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS,
    callbacks=[tensorboard_callback, early_stopping_callback],
	verbose=1)
# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH)
# serialize the label binarizer to disk
print("[INFO] saving label binarizer...")
f = open(config.LB_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()

# plot the total loss, label loss, and bounding box loss
lossNames = ["loss", "class_label_loss", "bounding_box_loss"]
N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
# loop over the loss names
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(N, history.history[l], label=l)
	ax[i].plot(N, history.history["val_" + l], label="val_" + l)
	ax[i].legend()
# save the losses figure and create a new figure for the accuracies
plt.tight_layout()
plotPath = os.path.sep.join([config.PLOTS_PATH, "losses.png"])
plt.savefig(plotPath)
plt.close()

# create a new figure for the accuracies
plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["class_label_accuracy"],
	label="class_label_train_acc")
plt.plot(N, history.history["val_class_label_accuracy"],
	label="val_class_label_acc")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
# save the accuracies plot
plotPath = os.path.sep.join([config.PLOTS_PATH, "accs.png"])
plt.savefig(plotPath)