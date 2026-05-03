import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image

# 1. Konfigürasyon
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "train"
TEST_DIR = SCRIPT_DIR / "test"
RESULTS_DIR = SCRIPT_DIR / "training_results"
MODELS_DIR = SCRIPT_DIR.parent.parent / "models"
BATCH_SIZE = 32
EPOCHS = 5  # Test için düşük tutuldu, kullanıcı artırabilir
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Veri Hazırlama ve Temizleme
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            print(f"Hata: {self.imgs[index][0]} okunamadı, atlanıyor. {e}")
            # Hatalı resim yerine bir sonraki resmi dene
            return self.__getitem__((index + 1) % len(self))

def get_dataloaders():
    print(f"[*] Eğitim seti taranıyor: {DATA_DIR}...")
    print(f"[*] Test seti taranıyor: {TEST_DIR}...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # Yeni 256x256 resimler için boyut güncellendi
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = SafeImageFolder(root=str(DATA_DIR), transform=transform)
    val_dataset = SafeImageFolder(root=str(TEST_DIR), transform=transform)
    
    print(f"[*] Eğitim setinde {len(train_dataset)}, Test setinde {len(val_dataset)} resim bulundu.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, train_dataset.classes

# 3. Model Tanımları
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
            nn.AdaptiveAvgPool2d((4, 4)),
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
    model = models.resnet18(pretrained=True)
    # Son katmanı ikili sınıflandırma için değiştir
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

# 4. Eğitim Fonksiyonu
def train_model(model, train_loader, val_loader, name="Model"):
    print(f"\n--- {name} Eğitimi Başlıyor ---")
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        processed_batches = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            processed_batches += 1
            
            if (i + 1) % 100 == 0:
                print(f"   > Adım {i+1}/{len(train_loader)} - Batch Loss: {loss.item():.4f}")
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Doğrulama
        model.eval()
        val_loss = 0.0
        correct = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:.4f}")
        
    return model, history, all_labels, all_preds

# 5. Raporlama
def save_reports(history1, history2, labels1, preds1, labels2, preds2):
    # Excel Raporu
    df1 = pd.DataFrame(history1)
    df1['model'] = 'SimpleCNN'
    df2 = pd.DataFrame(history2)
    df2['model'] = 'ResNet18'
    
    report_df = pd.concat([df1, df2])
    report_df.to_excel(RESULTS_DIR / "training_metrics.xlsx", index=False)
    
    # Grafikler
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history1['val_acc'], label='SimpleCNN Val Acc')
    plt.plot(history2['val_acc'], label='ResNet18 Val Acc')
    plt.title('Doğrulama Başarımı (Validation Accuracy)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history1['train_loss'], label='SimpleCNN Loss')
    plt.plot(history2['train_loss'], label='ResNet18 Loss')
    plt.title('Eğitim Kaybı (Training Loss)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "performance_plots.png")
    
    # Confusion Matrix
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    cm1 = confusion_matrix(labels1, preds1)
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues')
    plt.title('SimpleCNN Confusion Matrix')
    
    plt.subplot(1, 2, 2)
    cm2 = confusion_matrix(labels2, preds2)
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens')
    plt.title('ResNet18 Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices.png")
    print(f"\nRaporlar kaydedildi: {RESULTS_DIR}")

def main():
    train_loader, val_loader, classes = get_dataloaders()
    
    # Model 1: SimpleCNN
    model1 = SimpleCNN()
    model1, hist1, labels1, preds1 = train_model(model1, train_loader, val_loader, name="SimpleCNN")
    torch.save(model1.state_dict(), MODELS_DIR / "simple_cnn.pth")
    
    # Model 2: ResNet18
    model2 = get_resnet_model()
    model2, hist2, labels2, preds2 = train_model(model2, train_loader, val_loader, name="ResNet18")
    torch.save(model2.state_dict(), MODELS_DIR / "resnet18_forgery.pth")
    
    # ONNX Export (ResNet18 is better, so export it)
    try:
        dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE)
        torch.onnx.export(model2, dummy_input, MODELS_DIR / "deepfake_detector.onnx", 
                          input_names=['input'], output_names=['output'], opset_version=12)
        print(f"[+] ONNX model başarıyla dışa aktarıldı.")
    except Exception as e:
        print(f"[!] ONNX dışa aktarma hatası: {e}. Ancak modeller .pth olarak kaydedildi.")
    
    save_reports(hist1, hist2, labels1, preds1, labels2, preds2)

if __name__ == "__main__":
    main()
