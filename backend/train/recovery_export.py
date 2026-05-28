import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from PIL import Image
import random

# Konfigürasyon
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "train"
RESULTS_DIR = SCRIPT_DIR / "training_results"
MODELS_DIR = SCRIPT_DIR.parent.parent / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_resnet_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

def main():
    print("[*] Gelişmiş raporlama süreci başlatılıyor...")
    
    # Modelleri Yükle
    model1 = SimpleCNN().to(DEVICE)
    model1.load_state_dict(torch.load(MODELS_DIR / "simple_cnn.pth", map_location=DEVICE, weights_only=True))
    model1.eval()
    
    model2 = get_resnet_model().to(DEVICE)
    model2.load_state_dict(torch.load(MODELS_DIR / "resnet18_forgery.pth", map_location=DEVICE, weights_only=True))
    model2.eval()

    # Veri Hazırla
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=str(DATA_DIR), transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    print(f"[*] {len(dataset)} resim üzerinde detaylı analiz yapılıyor...")
    
    all_labels = []
    all_probs1, all_probs2 = [], []
    all_preds1, all_preds2 = [], []
    sample_images, sample_results = [], []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(DEVICE)
            
            # SimpleCNN Analiz
            out1 = model1(inputs)
            prob1 = torch.sigmoid(out1).cpu().numpy()
            pred1 = (prob1 > 0.5).astype(float)
            
            # ResNet18 Analiz
            out2 = model2(inputs)
            prob2 = torch.sigmoid(out2).cpu().numpy()
            pred2 = (prob2 > 0.5).astype(float)
            
            all_probs1.extend(prob1)
            all_probs2.extend(prob2)
            all_preds1.extend(pred1)
            all_preds2.extend(pred2)
            all_labels.extend(labels.numpy())

            # Görsel örnekler topla (ilk batch'ten)
            if i == 0:
                for j in range(min(10, len(inputs))):
                    sample_images.append(inputs[j].cpu())
                    sample_results.append({
                        'real': dataset.classes[labels[j]],
                        'pred1': dataset.classes[int(pred1[j][0])],
                        'pred2': dataset.classes[int(pred2[j][0])]
                    })

    # --- 1. Gelişmiş Excel Raporu ---
    report1 = classification_report(all_labels, all_preds1, target_names=dataset.classes, output_dict=True)
    report2 = classification_report(all_labels, all_preds2, target_names=dataset.classes, output_dict=True)
    
    with pd.ExcelWriter(RESULTS_DIR / "detailed_performance_report.xlsx") as writer:
        pd.DataFrame(report1).transpose().to_excel(writer, sheet_name='SimpleCNN_Metrics')
        pd.DataFrame(report2).transpose().to_excel(writer, sheet_name='ResNet18_Metrics')
        
        # Karşılaştırma Özeti
        summary_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Macro F1', 'Weighted F1'],
            'SimpleCNN': [report1['accuracy'], report1['macro avg']['f1-score'], report1['weighted avg']['f1-score']],
            'ResNet18': [report2['accuracy'], report2['macro avg']['f1-score'], report2['weighted avg']['f1-score']]
        })
        summary_df.to_excel(writer, sheet_name='Model_Comparison', index=False)

    # --- 2. ROC Eğrisi Grafiği ---
    plt.figure(figsize=(10, 8))
    fpr1, tpr1, _ = roc_curve(all_labels, all_probs1)
    fpr2, tpr2, _ = roc_curve(all_labels, all_probs2)
    plt.plot(fpr1, tpr1, label=f'SimpleCNN (AUC = {auc(fpr1, tpr1):.4f})')
    plt.plot(fpr2, tpr2, label=f'ResNet18 (AUC = {auc(fpr2, tpr2):.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(RESULTS_DIR / "roc_curves.png")

    # --- 3. Örnek Tahminler Görselleştirmesi ---
    plt.figure(figsize=(15, 6))
    for idx in range(len(sample_images)):
        plt.subplot(2, 5, idx+1)
        img = sample_images[idx].permute(1, 2, 0).numpy()
        # Normalizasyonu geri al (görselleştirme için)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        color = 'green' if sample_results[idx]['real'] == sample_results[idx]['pred2'] else 'red'
        plt.title(f"Real: {sample_results[idx]['real']}\nPred: {sample_results[idx]['pred2']}", color=color, fontsize=9)
        plt.axis('off')
    plt.suptitle("Model Tahmin Örnekleri (ResNet18)", fontsize=16)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sample_predictions.png")

    # --- 4. Gelişmiş Confusion Matrix ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(confusion_matrix(all_labels, all_preds1), annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('SimpleCNN Confusion Matrix')
    ax1.set_xticklabels(dataset.classes); ax1.set_yticklabels(dataset.classes)
    
    sns.heatmap(confusion_matrix(all_labels, all_preds2), annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_title('ResNet18 Confusion Matrix')
    ax2.set_xticklabels(dataset.classes); ax2.set_yticklabels(dataset.classes)
    
    plt.savefig(RESULTS_DIR / "detailed_confusion_matrices.png")

    print(f"\n[+] İŞLEM TAMAMLANDI!")
    print(f"1. Excel Raporu: {RESULTS_DIR / 'detailed_performance_report.xlsx'}")
    print(f"2. ROC Eğrileri: {RESULTS_DIR / 'roc_curves.png'}")
    print(f"3. Örnek Tahminler: {RESULTS_DIR / 'sample_predictions.png'}")
    print(f"4. Karmaşıklık Matrisleri: {RESULTS_DIR / 'detailed_confusion_matrices.png'}")

if __name__ == "__main__":
    main()
