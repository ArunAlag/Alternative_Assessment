import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers, models
from pycocotools.coco import COCO
import random

# Check and enable GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

# ------------------------------------------------------------------
# 1) DATASET PATHS
# ------------------------------------------------------------------
train_folder_path = "train/"
valid_folder_path = "valid/"
test_folder_path = "test/"
train_coco_json = "_annotations_train.coco.json"
valid_coco_json = "_annotations_valid.coco.json"
test_coco_json = "_annotations_test.coco.json"

# ------------------------------------------------------------------
# 2) LOAD COCO ANNOTATIONS
# ------------------------------------------------------------------
def load_coco_annotations(folder_path, coco_json):
    coco = COCO(os.path.join(folder_path, coco_json))
    images = []
    masks = []

    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(folder_path, img_info['file_name'])
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Load image (256x256, grayscale, normalized)
        img = tf.keras.preprocessing.image.load_img(
            img_path, color_mode="grayscale", target_size=(256, 256)
        )
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0

        # Create mask
        mask = np.zeros((img.shape[0], img.shape[1]))
        for ann in anns:
            binary_mask = coco.annToMask(ann)
            binary_mask = tf.cast(binary_mask, tf.float32)
            if binary_mask.shape != (img.shape[0], img.shape[1]):
                binary_mask = tf.image.resize(binary_mask[..., None], (img.shape[0], img.shape[1]))
                binary_mask = tf.squeeze(binary_mask).numpy()
            mask = np.maximum(mask, binary_mask)

        mask = tf.image.resize(mask[..., None], (256, 256))
        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Load data
X_train, y_train = load_coco_annotations(train_folder_path, train_coco_json)
X_val, y_val = load_coco_annotations(valid_folder_path, valid_coco_json)
X_test, y_test = load_coco_annotations(test_folder_path, test_coco_json)

# Ensure correct shape (expand dims again if needed)
y_train = np.expand_dims(y_train[..., 0], axis=-1)
y_val   = np.expand_dims(y_val[..., 0], axis=-1)
y_test  = np.expand_dims(y_test[..., 0], axis=-1)

# ------------------------------------------------------------------
# 3) DATA AUGMENTATION
# ------------------------------------------------------------------
data_gen_args = dict(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen  = ImageDataGenerator(**data_gen_args)

# Custom data generator
class CustomDataGenerator(Sequence):
    def __init__(self, image_generator, mask_generator):
        self.image_generator = image_generator
        self.mask_generator  = mask_generator

    def __len__(self):
        return len(self.image_generator)

    def __getitem__(self, index):
        x = self.image_generator[index]
        y = self.mask_generator[index]
        return x, y

# Prepare data generators
seed = 42
image_generator = image_datagen.flow(X_train, batch_size=8, seed=seed)  # smaller batch
mask_generator  = mask_datagen.flow(y_train, batch_size=8, seed=seed)
train_generator = CustomDataGenerator(image_generator, mask_generator)

# ------------------------------------------------------------------
# 4) BUILD U-NET++ MODEL
# ------------------------------------------------------------------
def conv_block(inputs, filters, kernel_size=(3,3), padding="same", activation="relu"):
    x = layers.Conv2D(filters, kernel_size, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x

def encoder_block(inputs, filters):
    x = conv_block(inputs, filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(inputs)
    x = layers.concatenate([x, skip_features])
    x = conv_block(x, filters)
    return x

def unet_plus_plus(input_shape=(256, 256, 1), num_classes=1):
    inputs = layers.Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = conv_block(p4, 1024)
    b1 = layers.Lambda(lambda x: tf.nn.dropout(x, rate=0.1))(b1)

    # Decoder
    d4 = decoder_block(b1, s4, 512)
    d4 = layers.Lambda(lambda x: tf.nn.dropout(x, rate=0.1))(d4)
    d3 = decoder_block(d4, s3, 256)
    d2 = decoder_block(d3, s2, 128)
    d1 = decoder_block(d2, s1, 64)

    outputs = layers.Conv2D(num_classes, (1, 1), padding="same", activation="sigmoid")(d1)
    model = models.Model(inputs, outputs, name="U_Net_PlusPlus")
    return model

# ------------------------------------------------------------------
# 5) CUSTOM METRICS/LOSSES
# ------------------------------------------------------------------
def iou_score(y_true, y_pred, smooth=1e-7):
    """
    Intersection over Union = (TP) / (TP + FP + FN)
    We'll threshold y_pred at 0.5 for the predicted mask.
    """
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def dice_coefficient(y_true, y_pred, smooth=1e-7):
    """
    Dice = (2 * TP) / (2 * TP + FP + FN)
    We'll threshold y_pred at 0.5 for the predicted mask.
    """
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def bce_dice_loss(y_true, y_pred):
    """
    Combined BCE + (1 - Dice).
    This helps address class imbalance by focusing on overlap.
    """
    bce  = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_coefficient(y_true, y_pred)
    return bce + (1.0 - dice)

# ------------------------------------------------------------------
# 6) TRAIN THE MODEL
# ------------------------------------------------------------------
strategy = tf.distribute.MirroredStrategy()  # If you have multiple GPUs
with strategy.scope():
    model = unet_plus_plus(input_shape=(256, 256, 1), num_classes=1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=bce_dice_loss,  # <-- Use BCE + Dice
        metrics=["accuracy", iou_score, dice_coefficient]
    )

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("unetplusplus_best_model.keras", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7)
]

history = model.fit(
    train_generator,
    epochs=50,  # more epochs so we can see if it converges
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

model.save("unetplusplus_final_model.keras")

# ------------------------------------------------------------------
# 7) EVALUATE ON TEST SET
# ------------------------------------------------------------------
test_loss, test_accuracy, test_iou, test_dice = model.evaluate(X_test, y_test, batch_size=8)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test IOU: {test_iou}")
print(f"Test Dice: {test_dice}")

# ------------------------------------------------------------------
# 8) PLOT METRICS
# ------------------------------------------------------------------
epochs_range = range(1, len(history.history['loss']) + 1)

plt.figure(figsize=(20, 5))

# (a) LOSS PLOT
plt.subplot(1, 4, 1)
plt.plot(epochs_range, history.history['loss'], label='Train Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Val Loss')
plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
plt.title('Loss (BCE + 1 - Dice)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# (b) ACCURACY PLOT
plt.subplot(1, 4, 2)
plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Val Accuracy')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# (c) IOU PLOT
plt.subplot(1, 4, 3)
plt.plot(epochs_range, history.history['iou_score'], label='Train IOU')
plt.plot(epochs_range, history.history['val_iou_score'], label='Val IOU')
plt.axhline(y=test_iou, color='r', linestyle='--', label='Test IOU')
plt.title('IOU')
plt.xlabel('Epoch')
plt.ylabel('IOU')
plt.legend()

# (d) DICE PLOT
plt.subplot(1, 4, 4)
plt.plot(epochs_range, history.history['dice_coefficient'], label='Train Dice')
plt.plot(epochs_range, history.history['val_dice_coefficient'], label='Val Dice')
plt.axhline(y=test_dice, color='r', linestyle='--', label='Test Dice')
plt.title('Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.legend()

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 9) VISUALIZE PREDICTIONS (COLORED OVERLAY)
# ------------------------------------------------------------------
def visualize_prediction(img, true_mask, pred_mask, threshold=0.5):
    """
    img: single image array of shape (256,256,1) in [0,1]
    true_mask: ground-truth mask array (256,256,1)
    pred_mask: predicted mask array (256,256,1)
    threshold: for binarizing predictions
    """
    img_rgb = np.repeat(img, 3, axis=-1)  # grayscale -> 3-channel
    img_rgb = np.clip(img_rgb, 0, 1)

    pred_binary = (pred_mask > threshold).astype(np.float32)

    # color overlays
    alpha = 0.4
    overlay_pred = img_rgb.copy()
    overlay_true = img_rgb.copy()

    # predicted mask -> red
    overlay_pred[..., 0] = np.where(pred_binary[..., 0] == 1, 1.0, overlay_pred[..., 0])
    overlay_pred[..., 1] = np.where(pred_binary[..., 0] == 1, 0.0, overlay_pred[..., 1])
    overlay_pred[..., 2] = np.where(pred_binary[..., 0] == 1, 0.0, overlay_pred[..., 2])

    # true mask -> green
    overlay_true[..., 0] = np.where(true_mask[..., 0] == 1, 0.0, overlay_true[..., 0])
    overlay_true[..., 1] = np.where(true_mask[..., 0] == 1, 1.0, overlay_true[..., 1])
    overlay_true[..., 2] = np.where(true_mask[..., 0] == 1, 0.0, overlay_true[..., 2])

    blended_pred = (1 - alpha) * img_rgb + alpha * overlay_pred
    blended_true = (1 - alpha) * img_rgb + alpha * overlay_true

    # Plot side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img.squeeze(), cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(blended_true)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(blended_pred)
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    plt.show()

# ------------------------------------------------------------------
# 10) TEST EXAMPLE VISUALIZATION
# ------------------------------------------------------------------
idx = random.randint(0, len(X_test) - 1)
sample_img       = X_test[idx]
sample_true_mask = y_test[idx]
sample_pred_mask = model.predict(sample_img[np.newaxis, ...])[0]

visualize_prediction(sample_img, sample_true_mask, sample_pred_mask)
