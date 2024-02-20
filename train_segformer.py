import tensorflow as tf
import tensorflow_io as tfio
import os
import glob
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm
import rasterio
import matplotlib.pyplot as plt
from transformers import TFSegformerForSemanticSegmentation


batch_size = 4
epochs = 20

#Check GPU
print(tf.config.list_physical_devices('GPU'))

# Paths to your data directories
images_dir ='/share/projects/erasmus/dfc/Track1/train/images'
labels_dir = '/share/projects/erasmus/dfc/Track1/train/labels'
image_size = 512

### DATA LOADER ###
# a list to collect paths of images
image_path = [os.path.join(images_dir,file) for root, _, files in os.walk(images_dir) for file in files ]
images = sorted(image_path)
# a list to collect paths of masks
mask_path = [os.path.join(labels_dir, file) for root, _, files in os.walk(labels_dir) for file in files]
masks = sorted(mask_path)

print(f"Length of image paths list: {len(images)}")
print(f"Length of mask paths list: {len(masks)}")
print("\n")

auto = tf.data.AUTOTUNE



def parse_image(file_path):
    image = tifffile.imread(file_path)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = tf.image.resize(image, [image_size, image_size])
    # Ensure the image has 6 channels
    image.set_shape([image_size, image_size, 6])
    return image

def parse_mask(file_path):
    mask = tf.io.read_file(file_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [image_size, image_size], method='nearest')
    mask = tf.squeeze(mask)  # Ensure it's a 2D tensor for the mask
    mask.set_shape([image_size, image_size])
    return mask

def load_data(image_path, mask_path):
    image = parse_image(image_path)
    mask = parse_mask(mask_path)
    image = tf.transpose(image, (2, 0, 1))  # Adjust dimensions if necessary
    return {"pixel_values": image, "labels": mask}

# Convert the file paths lists into TensorFlow datasets
image_ds = tf.data.Dataset.from_tensor_slices(images)
mask_ds = tf.data.Dataset.from_tensor_slices(masks)


dataset = tf.data.Dataset.zip((image_ds, mask_ds))
# Convert the dataset to a tensor
dataset_tensor = tf.data.experimental.get_single_element(dataset.batch(len(images_subset)))

# Apply tf.map_fn
processed_dataset_tensor = tf.map_fn(lambda x: load_data(x[0].numpy().decode('utf-8'), x[1].numpy().decode('utf-8')), dataset_tensor, dtype={"pixel_values": tf.float32, "labels": tf.uint8})
# Convert the result back to a dataset
processed_dataset = tf.data.Dataset.from_tensor_slices(processed_dataset_tensor)
processed_dataset = processed_dataset.cache().batch(4).prefetch(auto)

#Creating train and test

total_batches = len(images)/batch_size # change this
train_size = int(0.8 * total_batches)
test_size = total_batches - train_size

train_ds = processed_dataset.take(train_size)
test_ds = processed_dataset.skip(train_size)

print(f"Shape of training dataset:{train_ds.element_spec}")
print(f"Shape of test dataset: {test_ds.element_spec}")

#### COMPILING MODEL ####
model_checkpoint = "nvidia/mit-b0"
id2label = {0: "background", 1: "water"}
label2id = {label: id for id, label in id2label.items()}
num_labels = len(id2label)
model = TFSegformerForSemanticSegmentation.from_pretrained(
    model_checkpoint,
    num_channels=6,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

lr = 0.00006
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer, run_eagerly=True)

### SET UP TRAINING ###


history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs,
)

### PLOTTING ####
# Extracting the metrics
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)  # 'epochs' is the variable used in 'model.fit()'


plt.figure(figsize=(12, 6))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()