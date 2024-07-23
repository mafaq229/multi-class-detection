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

# Function to load and preprocess images and annotations
def load_and_preprocess_image(image_path, annotation_path):
    annotation_data = loadmat(annotation_path.numpy().decode('utf-8'))
    (startX, startY, endX, endY) = annotation_data['box_coord'][0]
    
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0

    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    startX = tf.cast(startX, tf.float32) / tf.cast(w, tf.float32)
    startY = tf.cast(startY, tf.float32) / tf.cast(h, tf.float32)
    endX = tf.cast(endX, tf.float32) / tf.cast(w, tf.float32)
    endY = tf.cast(endY, tf.float32) / tf.cast(h, tf.float32)

    bbox = tf.convert_to_tensor([startX, startY, endX, endY], dtype=tf.float32)
    
    return image, bbox

def load_and_preprocess_wrapper(image_path, annotation_path, label):
    image, bbox = tf.py_function(func=load_and_preprocess_image, inp=[image_path, annotation_path], Tout=[tf.float32, tf.float32])
    image.set_shape((224, 224, 3))
    bbox.set_shape((4,))
    return image, bbox, label

# Create datasets
def create_dataset(image_paths, annotation_paths, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, annotation_paths, labels))
    dataset = dataset.shuffle(len(image_paths))
    dataset = dataset.map(load_and_preprocess_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# initialize the list of data (images), class labels, target bounding
# box coordinates, and image paths
print("[INFO] loading dataset...")
data = []
labels = []
bboxes = []
imagePaths = []
annotationPaths = []

for category in os.listdir(config.IMAGES_PATH):
    category_image_dir = os.path.join(config.IMAGES_PATH, category)
    category_annotation_dir = os.path.join(config.ANNOTS_PATH, category)
    
    if os.path.isdir(category_image_dir) and os.path.isdir(category_annotation_dir):
        for image_file in os.listdir(category_image_dir):
            image_name, _ = os.path.splitext(image_file)
            annotation_name = image_name.replace('image_', 'annotation_')
            annotation_file = annotation_name + '.mat'
            annotation_path = os.path.join(category_annotation_dir, annotation_file)
            
            if os.path.exists(annotation_path):
                image_path = os.path.join(category_image_dir, image_file)
                data.append(image_path)
                annotationPaths.append(annotation_path)
                labels.append(category)

# Convert the labels to numerical values
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
train_imagePaths, test_imagePaths, train_annotationPaths, test_annotationPaths, y_train, y_test = train_test_split(data, annotationPaths, labels, test_size=0.2, random_state=42)

# Create training and testing datasets
batch_size = config.BATCH_SIZE
train_dataset = create_dataset(train_imagePaths, train_annotationPaths, y_train, batch_size)
test_dataset = create_dataset(test_imagePaths, test_annotationPaths, y_test, batch_size)

# write the testing image paths to disk so that we can use then
# when evaluating/testing our object detector
print("[INFO] saving testing image paths...")
with open(config.TEST_PATHS, "w") as f:
    f.write("\n".join(test_imagePaths))

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
bboxHead = tf.keras.layers.Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

# construct a second fully-connected layer head, this one to predict
# the class label
softmaxHead = tf.keras.layers.Dense(512, activation="relu", kernel_initializer='he_normal')(flatten)
softmaxHead = tf.keras.layers.Dropout(0.5)(softmaxHead)
softmaxHead = tf.keras.layers.Dense(512, activation="relu", kernel_initializer='he_normal')(softmaxHead)
softmaxHead = tf.keras.layers.Dropout(0.5)(softmaxHead)
softmaxHead = tf.keras.layers.Dense(len(lb.classes_), activation="softmax", name="class_label")(softmaxHead)

# put together our model which accept an input image and then output
# bounding box coordinates and a class label
model = tf.keras.models.Model(inputs=vgg.input, outputs=(bboxHead, softmaxHead))

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

# Create TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{str(int(time.time()))}')
# Create EarlyStopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_class_label_accuracy', patience=3, restore_best_weights=True)

# train the network for bounding box regression and class label
# prediction
print("[INFO] training model...")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=config.NUM_EPOCHS,
    callbacks=[tensorboard_callback, early_stopping_callback],
    verbose=1
)

# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH)
# serialize the label binarizer to disk
print("[INFO] saving label binarizer...")
with open(config.LB_PATH, "wb") as f:
    f.write(pickle.dumps(lb))

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
