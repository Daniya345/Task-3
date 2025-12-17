# %%
# Cats vs Dogs SVM for Kaggle-style dataset
# train/  -> cat.n.jpg, dog.n.jpg
# test1/  -> 1.jpg, 2.jpg, ...
# sampleSubmission.csv -> id,label (template)

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from tqdm import tqdm

print("Python executable:", sys.executable)

# ---------- PATH CONFIG ----------
root_dir = r"C:\Task3"             # folder where this .py file and train/test1 live
train_dir = os.path.join(root_dir, "train")
test_dir  = os.path.join(root_dir, "test1")
submission_template = os.path.join(root_dir, "sampleSubmission.csv")
submission_out = os.path.join(root_dir, "my_submission_svm.csv")

IMG_SIZE = (64, 64)
MAX_TRAIN_IMAGES = 4000   # total images (cats + dogs) to use for training

classes = ["cat", "dog"]  # 0=cat, 1=dog

print("Train dir:", train_dir)
print("Test dir :", test_dir)
print("Template :", submission_template)
print("Output   :", submission_out)

# %%
# ---------- LOAD TRAIN IMAGES FROM train/ (BALANCED) ----------
X = []
y = []

if not os.path.isdir(train_dir):
    raise RuntimeError(f"Train folder does not exist: {train_dir}")

all_files = [
    f for f in os.listdir(train_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Split files into cat and dog lists first
cat_files = [f for f in all_files if "cat" in f.lower()]
dog_files = [f for f in all_files if "dog" in f.lower()]

print("Total cat files:", len(cat_files))
print("Total dog files:", len(dog_files))

if len(cat_files) == 0 or len(dog_files) == 0:
    raise RuntimeError("train/ must contain BOTH cat.* and dog.* files.")

# Decide how many per class to use (balanced + limited)
if MAX_TRAIN_IMAGES is not None:
    per_class = min(len(cat_files), len(dog_files), MAX_TRAIN_IMAGES // 2)
else:
    per_class = min(len(cat_files), len(dog_files))

print("Using", per_class, "cats and", per_class, "dogs")

# Randomly sample per_class files from each class
cat_files = np.random.choice(cat_files, size=per_class, replace=False)
dog_files = np.random.choice(dog_files, size=per_class, replace=False)

selected_files = [(f, 0) for f in cat_files] + [(f, 1) for f in dog_files]

for fname, label in tqdm(selected_files, desc="Loading balanced train images"):
    fpath = os.path.join(train_dir, fname)
    try:
        img = imread(fpath)
        img_resized = resize(img, IMG_SIZE, anti_aliasing=True)
        X.append(img_resized)
        y.append(label)
    except Exception:
        continue

X = np.array(X)
y = np.array(y)

print("Loaded train images shape:", X.shape)
print("Train labels shape:", y.shape)
print("Label counts (0=cat, 1=dog):", dict(zip(*np.unique(y, return_counts=True))))

if len(X) == 0:
    raise RuntimeError("No train images loaded after balancing.")

# %%
# ---------- PREVIEW SOME TRAIN IMAGES ----------
def show_samples(X, y, classes, n=6):
    if len(X) == 0:
        print("No images loaded.")
        return
    n = min(n, len(X))
    plt.figure(figsize=(8, 4))
    for i in range(n):
        idx = np.random.randint(0, len(X))
        plt.subplot(2, (n + 1) // 2, i + 1)
        plt.imshow(X[idx])
        plt.title(classes[y[idx]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_samples(X, y, classes, n=6)

# %%
# ---------- HOG FEATURE EXTRACTION FOR TRAIN ----------
HOG_ORIENTATIONS = 8
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

print("HOG settings:")
print("orientations:", HOG_ORIENTATIONS)
print("pixels_per_cell:", HOG_PIXELS_PER_CELL)
print("cells_per_block:", HOG_CELLS_PER_BLOCK)

def extract_hog_features(images, visualize_example=True, example_index=0):
    hog_features = []
    hog_example = None

    for i, img in enumerate(tqdm(images, desc="Extracting HOG (train)")):
        gray = rgb2gray(img)
        if visualize_example and i == example_index:
            feat, hog_img = hog(
                gray,
                orientations=HOG_ORIENTATIONS,
                pixels_per_cell=HOG_PIXELS_PER_CELL,
                cells_per_block=HOG_CELLS_PER_BLOCK,
                block_norm="L2-Hys",
                visualize=True,
            )
            hog_example = (gray, hog_img)
        else:
            feat = hog(
                gray,
                orientations=HOG_ORIENTATIONS,
                pixels_per_cell=HOG_PIXELS_PER_CELL,
                cells_per_block=HOG_CELLS_PER_BLOCK,
                block_norm="L2-Hys",
                visualize=False,
            )
        hog_features.append(feat)

    return np.array(hog_features), hog_example

X_hog, hog_example = extract_hog_features(X, visualize_example=True, example_index=0)
print("HOG feature matrix shape (train):", X_hog.shape)

if len(X_hog) == 0:
    raise RuntimeError("No HOG features; something went wrong after loading images.")

# %%
# ---------- VISUALIZE HOG FOR ONE TRAIN IMAGE ----------
if hog_example is not None:
    gray_img, hog_img = hog_example
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_img, cmap="gray")
    plt.title("Grayscale image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(hog_img, cmap="gray")
    plt.title("HOG visualization")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("No HOG example available.")

# %%
# ---------- TRAIN / VALIDATION SPLIT ----------
X_train, X_val, y_train, y_val = train_test_split(
    X_hog, y, test_size=0.2, random_state=42, stratify=y
)

print("Train set:", X_train.shape, "Validation set:", X_val.shape)

# %%
# ---------- TRAIN SVM (LINEAR FOR SPEED) ----------
svm_clf = SVC(kernel="linear", C=1.0)
svm_clf.fit(X_train, y_train)

y_val_pred = svm_clf.predict(X_val)

acc = accuracy_score(y_val, y_val_pred)
print("Validation accuracy:", acc)

# %%
# ---------- VALIDATION METRICS TABLE ----------
report_dict = classification_report(
    y_val, y_val_pred, target_names=classes, output_dict=True
)
df_report = pd.DataFrame(report_dict).transpose()

print("Validation metrics (precision / recall / f1 / support):")
print(df_report[["precision", "recall", "f1-score", "support"]])

df_report  # shows as a table in Interactive Window

# %%
# ---------- CONFUSION MATRIX (VALIDATION) ----------
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

fig, ax = plt.subplots(figsize=(4, 4))
disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
plt.title("SVM Confusion Matrix (validation)")
plt.tight_layout()
plt.show()

# %%
# ---------- SHOW SOME TRAIN PREDICTIONS ----------
def show_predictions(X_all, X_features, y_all, model, classes, n=6):
    if len(X_all) == 0:
        print("No images to show.")
        return
    n = min(n, len(X_all))
    idxs = np.random.choice(len(X_all), size=n, replace=False)
    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(idxs):
        img = X_all[idx]
        feat = X_features[idx].reshape(1, -1)
        pred = model.predict(feat)[0]
        plt.subplot(2, (n + 1) // 2, i + 1)
        plt.imshow(img)
        plt.title(f"True: {classes[y_all[idx]]}\nPred: {classes[pred]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_predictions(X, X_hog, y, svm_clf, classes, n=6)

# %%
# ---------- LOAD TEST1 IMAGES (NUMBERED) ----------
test_images = []
test_ids = []

if not os.path.isdir(test_dir):
    raise RuntimeError(f"Test folder does not exist: {test_dir}")

test_files = [
    f for f in os.listdir(test_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

def numeric_key(name):
    base = os.path.splitext(name)[0]
    return int(base) if base.isdigit() else base

test_files = sorted(test_files, key=numeric_key)

print(f"Total image files found in test1: {len(test_files)}")

for fname in tqdm(test_files, desc="Loading test images"):
    fpath = os.path.join(test_dir, fname)
    base = os.path.splitext(fname)[0]
    if not base.isdigit():
        continue
    img_id = int(base)

    try:
        img = imread(fpath)
        img_resized = resize(img, IMG_SIZE, anti_aliasing=True)
        test_images.append(img_resized)
        test_ids.append(img_id)
    except Exception:
        continue

test_images = np.array(test_images)
print("Loaded test images shape:", test_images.shape)
print("Number of test ids:", len(test_ids))

# %%
# ---------- HOG FEATURES FOR TEST ----------
def extract_hog_features_no_vis(images):
    hog_features = []
    for img in tqdm(images, desc="Extracting HOG (test)"):
        gray = rgb2gray(img)
        feat = hog(
            gray,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm="L2-Hys",
            visualize=False,
        )
        hog_features.append(feat)
    return np.array(hog_features)

if len(test_images) > 0:
    X_test_hog = extract_hog_features_no_vis(test_images)
    print("HOG feature matrix shape (test):", X_test_hog.shape)
else:
    X_test_hog = np.array([])
    print("No test images loaded.")

# %%
# ---------- PREDICT ON TEST1 AND CREATE SUBMISSION CSV ----------
if len(X_test_hog) == len(test_ids) and len(test_ids) > 0:
    test_preds = svm_clf.predict(X_test_hog).astype(int)

    submission_df = pd.DataFrame({
        "id": test_ids,
        "label": test_preds
    }).sort_values("id")

    submission_df.to_csv(submission_out, index=False)
    print("Submission saved to:", submission_out)
    print(submission_df.head())
else:
    print("Skipping submission: mismatch or no test data.")
