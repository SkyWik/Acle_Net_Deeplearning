import torch
from torch.utils.data import DataLoader
from config import DATA_DIR, BATCH_SIZE, DEVICE, MODEL_SAVE_PATH, COGNITIVE_LOAD_CLASSES
from dataset import UIChartDataset
from model import ACLENetCNN
import numpy as np

def evaluate_model():
    print("--- Evaluating ACLE-Net Deep Learning Model ---")
    dataset = UIChartDataset(root_dir=DATA_DIR, task="cognitive_load")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = ACLENetCNN(num_classes=len(COGNITIVE_LOAD_CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    print(f"Overall Evaluation Accuracy: {accuracy:.2f}%")
    print("\nClass-wise Metrics:")
    for idx, cls_name in enumerate(COGNITIVE_LOAD_CLASSES):
        cls_mask = (np.array(all_labels) == idx)
        if np.sum(cls_mask) > 0:
            cls_acc = np.mean(np.array(all_preds)[cls_mask] == idx) * 100
            print(f" - {cls_name}: {cls_acc:.2f}%")

if __name__ == "__main__":
    evaluate_model()
