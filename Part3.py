# Introduction to Artificial Intelligence
# Fashion MNIST Dataset
# Adrian Quiros, Luis Baeza

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Settings
TEST_SIZE     = 0.20
RANDOM_STATE  = 2025
SAVE_PLOTS    = True
USE_RIDGE     = False      
RIDGE_ALPHA   = 1.0        
COEF_FIGSIZE  = (3.2, 3.2) 
OUTPUT_PREFIX = "lr_fashion_mnist_"

# Load data
pickle_candidates = [
    "fashion_mnist_dataset.pickle",
    "fasion_mnist_dataset.pickle"
]

pickle_path = None
for p in pickle_candidates:
    if os.path.exists(p):
        pickle_path = p
        break

if pickle_path is None:
    raise FileNotFoundError(
        "Could not find 'fashion_mnist_dataset.pickle' (or the misspelled 'fasion_mnist_dataset.pickle') "
        "in the current directory."
    )

with open(pickle_path, "rb") as f:
    train_data, train_labels, test_data, test_labels = pickle.load(f)

# Ensure arrays are float and scale to [0,1]
train_data = train_data.astype(np.float32)
test_data  = test_data.astype(np.float32)
maxval = 255.0 if train_data.max() > 1.0 else float(train_data.max())
if maxval == 0:
    raise ValueError("Max pixel value is 0 — dataset appears empty or corrupted.")
train_data /= maxval
test_data  /= maxval

# Shapes / sanity
n_train, n_feat = train_data.shape
n_test  = test_data.shape[0]
print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}, #features = {n_feat}")

# Split held-out validation
X_tr, X_va, y_tr_raw, y_va_raw = train_test_split(
    train_data, train_labels,
    test_size=TEST_SIZE,
    shuffle=True,
    random_state=RANDOM_STATE
)

# One-hot targets for multi-output regression
num_classes = 10
def to_onehot(y, k=num_classes):
    oh = np.zeros((y.shape[0], k), dtype=np.float32)
    oh[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return oh

Y_tr = to_onehot(y_tr_raw, num_classes)
Y_va = to_onehot(y_va_raw, num_classes)

# Fit model
if USE_RIDGE:
    model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
else:
    model = LinearRegression()

model.fit(X_tr, Y_tr)

# Evaluate on validation split (from train)

scores_va = model.predict(X_va)                 # shape: (n_va, 10)
y_pred_va = np.argmax(scores_va, axis=1)

acc_va = accuracy_score(y_va_raw, y_pred_va)
print(f"\nValidation accuracy (split from train): {acc_va:.4f}")

print("\nValidation classification report:")
print(classification_report(y_va_raw, y_pred_va, digits=4))

# Confusion matrix (validation)
cm_va = confusion_matrix(y_va_raw, y_pred_va, labels=np.arange(num_classes))
plt.figure()
plt.imshow(cm_va, interpolation="nearest")
plt.title("Confusion Matrix — Validation")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.tight_layout()
if SAVE_PLOTS: plt.savefig(OUTPUT_PREFIX + "cm_validation.png", dpi=160)
plt.show()

# Final eval on OFFICIAL TEST SET

scores_te = model.predict(test_data)
y_pred_te = np.argmax(scores_te, axis=1)

acc_te = accuracy_score(test_labels, y_pred_te)
print(f"\nTEST accuracy (official test set): {acc_te:.4f}")

print("\nTEST classification report:")
print(classification_report(test_labels, y_pred_te, digits=4))

# Confusion matrix (test)
cm_te = confusion_matrix(test_labels, y_pred_te, labels=np.arange(num_classes))
plt.figure()
plt.imshow(cm_te, interpolation="nearest")
plt.title("Confusion Matrix — Test")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.tight_layout()
if SAVE_PLOTS: plt.savefig(OUTPUT_PREFIX + "cm_test.png", dpi=160)
plt.show()

# Plot coefficient images
coef_mat = model.coef_
if coef_mat.shape != (num_classes, n_feat):
    if coef_mat.shape == (n_feat, num_classes):
        coef_mat = coef_mat.T
    else:
        raise RuntimeError(f"Unexpected coef_ shape: {coef_mat.shape}; expected (10, {n_feat}) or ({n_feat}, 10).")

for cls in range(num_classes):
    coef_img = coef_mat[cls, :].reshape(28, 28)

    plt.figure(figsize=COEF_FIGSIZE)
    plt.imshow(coef_img)
    plt.title(f"Linear Regression Coefficients — Class {cls}")
    plt.axis("off")
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(OUTPUT_PREFIX + f"coef_class_{cls}.png", dpi=160)
    plt.show()

# Quick diagnostics to discuss overfitting / regularization

abs_coef = np.abs(coef_mat).ravel()
near_zero = np.mean(abs_coef < 1e-4)
l2_norms  = np.linalg.norm(coef_mat, axis=1)

print("\n== Coefficient diagnostics ==")
print(f"Proportion of coefficients with |w| < 1e-4: {near_zero:.3f}")
for c, nrm in enumerate(l2_norms):
    print(f"  Class {c}: {nrm:.4f}")

print("\nNotes:")
print("- The coefficient images show which pixels push scores up (bright) or down (dark) for each class.")
print("- If many isolated pixels outside the item silhouette have strong weights, that hints at noise fitting.")
