import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings('ignore')

# SECTION 1: MANUAL DATASET CREATION
def get_manual_datasets():
    datasets = []

    # Dataset 1: Binary, Standard
    X1, y1 = make_classification(n_samples=400, n_features=2, n_informative=2, n_redundant=0,
                                 n_clusters_per_class=1, weights=[0.90, 0.10], flip_y=0.05, random_state=101)
    datasets.append({"data": (X1, y1), "name": "Dataset 1 (Binary-Standard)", "classes": 2})

    # Dataset 2: Binary, Extreme
    X2, y2 = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0,
                                 n_clusters_per_class=1, weights=[0.95, 0.05], flip_y=0.05, random_state=102)
    datasets.append({"data": (X2, y2), "name": "Dataset 2 (Binary-Extreme)", "classes": 2})

    # Dataset 3: Binary, Noisy
    X3, y3 = make_classification(n_samples=400, n_features=2, n_informative=2, n_redundant=0,
                                 n_clusters_per_class=2, weights=[0.85, 0.15], flip_y=0.15, random_state=103)
    datasets.append({"data": (X3, y3), "name": "Dataset 3 (Binary-Noisy)", "classes": 2})

    # Dataset 4: Multi-class
    X4, y4 = make_classification(n_samples=450, n_features=2, n_informative=2, n_redundant=0,
                                 n_clusters_per_class=1, n_classes=3, weights=[0.80, 0.10, 0.10],
                                 flip_y=0.05, random_state=104)
    datasets.append({"data": (X4, y4), "name": "Dataset 4 (Multi-Class A)", "classes": 3})

    # Dataset 5: Multi-class, Complex
    X5, y5 = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0,
                                 n_clusters_per_class=1, n_classes=3, weights=[0.70, 0.15, 0.15],
                                 flip_y=0.10, random_state=105)
    datasets.append({"data": (X5, y5), "name": "Dataset 5 (Multi-Class B)", "classes": 3})

    return datasets

# SECTION 2: CUSTOM OVERSAMPLING ALGORITHM
def custom_oversample(X, y, k=5):
    classes, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)

    print(f"   [ALGORITHM] Analyzing Class Distribution:")
    for cls, count in zip(classes, counts):
        status = "MAJORITY" if count == max_count else "MINORITY"
        print(f"     -> Class {cls}: {count} instances ({status})")

    print(f"   [ALGORITHM] Target size for all classes: {max_count}")

    X_resampled = [X]
    y_resampled = [y]

    for cls, count in zip(classes, counts):
        if count < max_count:
            n_needed = max_count - count
            print(f"     -> ACTION: Upsampling Class {cls} | Generating {n_needed} synthetic samples (KNN + Interpolation)...")

            X_minority = X[y == cls]

            actual_k = min(k, len(X_minority) - 1)
            if actual_k < 1: actual_k = 1

            nbrs = NearestNeighbors(n_neighbors=actual_k + 1).fit(X_minority)
            synthetic_samples = []

            for _ in range(n_needed):
                rand_idx = np.random.randint(0, len(X_minority))
                p = X_minority[rand_idx]

                distances, indices = nbrs.kneighbors(p.reshape(1, -1))
                neighbor_indices = indices[0][1:]

                chosen_neighbor_idx = np.random.choice(neighbor_indices)
                neighbor = X_minority[chosen_neighbor_idx]

                gap = np.random.random()
                diff = neighbor - p
                s = p + gap * diff

                synthetic_samples.append(s)

            X_resampled.append(np.array(synthetic_samples))
            y_resampled.append(np.full(len(synthetic_samples), cls))
        else:
            print(f"     -> ACTION: Class {cls} is already Majority. No changes.")

    X_final = np.vstack(X_resampled)
    y_final = np.hstack(y_resampled)

    print(f"   [ALGORITHM] Oversampling Complete. New Total Size: {len(X_final)} instances.")

    return X_final, y_final

# SECTION 3: VISUALIZATION & METRICS
COLORS = ['#1E90FF', '#FF0000', '#32CD32', '#FF00FF', '#FF8C00']

def plot_roc_curve(y_true, y_probs, n_classes, ax, title):
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], lw=2, alpha=0.8,
                label=f'Class {i} (AUC={roc_val:.2f})')

    fpr_grid = np.linspace(0.0, 1.0, 100)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        mean_tpr += np.interp(fpr_grid, fpr, tpr)
    mean_tpr /= n_classes
    macro_auc = auc(fpr_grid, mean_tpr)

    ax.plot(fpr_grid, mean_tpr, label=f'AVERAGE (AUC={macro_auc:.2f})',
            color='navy', linestyle='--', linewidth=3)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_title(title)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right", fontsize='small')
    ax.grid(True, alpha=0.3)

def plot_dashboard(X_old, y_old, probs_old, X_new, y_new, probs_new, name, n_classes, y_test_old, y_test_new):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plotting Train+Test combined for visualization to show density
    axes[0, 0].set_title(f"{name} - SCATTER (BEFORE)")
    for cls in np.unique(y_old):
        ix = np.where(y_old == cls)
        axes[0, 0].scatter(X_old[ix, 0], X_old[ix, 1], c=COLORS[cls % 5], label=f'Class {cls}', alpha=0.6, edgecolors='white')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    axes[0, 1].set_title(f"{name} - SCATTER (AFTER - Train Set Only)")
    for cls in np.unique(y_new):
        ix = np.where(y_new == cls)
        axes[0, 1].scatter(X_new[ix, 0], X_new[ix, 1], c=COLORS[cls % 5], label=f'Class {cls}', alpha=0.6, edgecolors='white')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    plot_roc_curve(y_test_old, probs_old, n_classes, axes[1, 0], "ROC Curve (BEFORE)")
    plot_roc_curve(y_test_new, probs_new, n_classes, axes[1, 1], "ROC Curve (AFTER)")

    plt.tight_layout()
    plt.show()

# SECTION 4: MAIN EXECUTION AND REPORTING
def main():
    report_data = []
    datasets = get_manual_datasets()

    print(f"{'='*100}")
    print(f"STARTING MIDTERM PROJECT ANALYSIS: {len(datasets)} DATASETS")
    print(f"{'='*100}\n")

    for ds in datasets:
        X, y = ds['data']
        name = ds['name']
        n_cls = ds['classes']
        n_features = X.shape[1]
        n_instances_before = X.shape[0]

        print(f"-> Processing: {name}")
        print(f"   [INFO] Dataset Properties: {n_features} Attributes, {n_instances_before} Total Instances.")

        # 1. First Split (Clean Test Set)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(f"   [STEP 1] Data Split: {len(X_train)} Train samples, {len(X_test)} Test samples.")

        # 2. Train Model on IMBALANCED Train Data (Before Scenario)
        print(f"   [STEP 2] Training Model on IMBALANCED Data (Baseline)...")
        clf_old = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)
        clf_old.fit(X_train, y_train)
        preds_old = clf_old.predict(X_test)
        probs_old = clf_old.predict_proba(X_test)

        # Metrics Before
        try:
            y_bin = label_binarize(y_test, classes=range(n_cls))
            if n_cls == 2 and y_bin.shape[1] == 1: y_bin = np.hstack((1 - y_bin, y_bin))
            auc_old = roc_auc_score(y_bin, probs_old, average='macro', multi_class='ovr')
        except: auc_old = 0.5

        prec_old = precision_score(y_test, preds_old, average='weighted', zero_division=0)
        rec_old = recall_score(y_test, preds_old, average='weighted', zero_division=0)
        f1_old = f1_score(y_test, preds_old, average='weighted', zero_division=0)

        # 3. Oversample ONLY the Training Data (Prevent Data Leakage)
        print(f"   [STEP 3] Applying Custom Oversampling to TRAINING Data...")
        X_train_res, y_train_res = custom_oversample(X_train, y_train)
        n_instances_train_after = X_train_res.shape[0]

        # 4. Train Model on BALANCED Train Data
        print(f"   [STEP 4] Training Model on BALANCED Data...")
        clf_new = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)
        clf_new.fit(X_train_res, y_train_res)

        # 5. Predict on the ORIGINAL Test Data (Real world scenario)
        print(f"   [STEP 5] Evaluating on Test Data...")
        preds_new = clf_new.predict(X_test)
        probs_new = clf_new.predict_proba(X_test)

        # Metrics After
        try:
            y_bin = label_binarize(y_test, classes=range(n_cls))
            if n_cls == 2 and y_bin.shape[1] == 1: y_bin = np.hstack((1 - y_bin, y_bin))
            auc_new = roc_auc_score(y_bin, probs_new, average='macro', multi_class='ovr')
        except: auc_new = 0.5

        prec_new = precision_score(y_test, preds_new, average='weighted', zero_division=0)
        rec_new = recall_score(y_test, preds_new, average='weighted', zero_division=0)
        f1_new = f1_score(y_test, preds_new, average='weighted', zero_division=0)

        # Visualize (Using Train Set for 'After' Scatter to show balancing effect)
        print(f"   [VISUALIZATION] Generating Dashboard...")
        plot_dashboard(X, y, probs_old, X_train_res, y_train_res, probs_new, name, n_cls, y_test, y_test)

        report_data.append({
            "Dataset": name,
            "Features": n_features,
            "Inst(Old)": n_instances_before,
            "Inst(NewTrain)": n_instances_train_after,
            "Precision (Before)": prec_old,
            "Precision (After)": prec_new,
            "Recall (Before)": rec_old,
            "Recall (After)": rec_new,
            "F1-Score (Before)": f1_old,
            "F1-Score (After)": f1_new,
            "AUC-ROC (Before)": auc_old,
            "AUC-ROC (After)": auc_new
        })
        print(f"{'-'*80}\n")

    print(f"\n{'='*30} FINAL PERFORMANCE REPORT {'='*30}")

    df = pd.DataFrame(report_data)

    numeric_cols = ["Features", "Inst(Old)", "Inst(NewTrain)", "Precision (Before)", "Precision (After)",
                    "Recall (Before)", "Recall (After)", "F1-Score (Before)", "F1-Score (After)",
                    "AUC-ROC (Before)", "AUC-ROC (After)"]

    avg_row = df[numeric_cols].mean()
    avg_dict = avg_row.to_dict()
    avg_dict['Dataset'] = 'AVERAGE'

    df = pd.concat([df, pd.DataFrame([avg_dict])], ignore_index=True)

    cols = [
        "Dataset",
        "Features", "Inst(Old)", "Inst(NewTrain)",
        "Precision (Before)", "Precision (After)",
        "Recall (Before)", "Recall (After)",
        "F1-Score (Before)", "F1-Score (After)",
        "AUC-ROC (Before)", "AUC-ROC (After)"
    ]
    df = df[cols]

    print(df.round(2).to_string(index=False))
    print(f"{'-'*100}")

if __name__ == "__main__":
    main()