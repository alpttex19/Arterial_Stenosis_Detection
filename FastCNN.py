import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights
import torchvision
import os
import numpy as np
from DetectData import CustomDataset, get_transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    for idx, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if idx % print_freq == 0:
            print(f"Epoch: [{epoch}] Iteration: [{idx}/{len(data_loader)}] Loss: {losses.item()}")

def evaluate(model, data_loader, device):
    model.eval()
    save_dir = "images_with_boxes"
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (images, targets) in enumerate(data_loader):
            if idx >= 20:  # 只显示第一张图像
                break
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for i, (image, output) in enumerate(zip(images, outputs)):
                # 从 GPU tensor 转换为 CPU numpy array
                image_np = image.permute(1, 2, 0).cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)  # 恢复到 0-255 范围

                # 创建绘图对象
                fig, ax = plt.subplots(1)
                ax.imshow(image_np)

                # 绘制预测框
                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                # print(boxes, labels, scores)
                
                for box, label, score in zip(boxes, labels, scores):
                    # if score >= 0.1:  # 只显示置信度大于 0.5 的框
                    x_min, y_min, x_max, y_max = box
                    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    plt.text(x_min, y_min, f"{label}: {score:.2f}", color='yellow', fontsize=12, backgroundcolor='black')

                # 保存结果图像
                plt.axis('off')
                plt.savefig(os.path.join(save_dir, f"output_{idx}_{i}.png"), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 1  # 根据你数据集的类别数调整
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    root = "./stenosis_data/train"
    dataset = CustomDataset(root, get_transform())
    dataset_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    num_epochs = 20
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, dataset_loader, device, epoch)
        lr_scheduler.step()

        # 评估
        evaluate(model, dataset_loader, device)
    
    torch.save(model.state_dict(), "./pth/fasterrcnn_resnet50_fpn.pth")

if __name__ == "__main__":
    main()
