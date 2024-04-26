import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data import Stenosis_Dataset
from combined_loss import CombinedLoss
from UNet import UNet
import os 
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0], device=device))
combined_loss_fn = CombinedLoss(ce_weight=torch.tensor([1.0, 10.0], device=device), dice_weight=1.0, iou_weight=1.0)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


def train(epoch_num, output_path, train_loader_1, train_loader_2, val_loader):
    train_log = []
    f1_score_log = []
    for epoch in range(epoch_num):
        loss_epoch_log = []
        print(f"EPOCH: {epoch}/{epoch_num}")
        model.train()
        if epoch < 0: #epoch_num // 2:
            for batch_idx, (inputs, masks) in enumerate(train_loader_1):
                inputs, masks = inputs.to(device), masks.to(device)
                optimizer.zero_grad()
                pred_masks = model(inputs)
                # loss = criterion(pred_masks, masks)
                loss = combined_loss_fn(pred_masks, masks)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print(f"batch_idx: {batch_idx}, train_loss: {loss.item()}")
                    loss_epoch_log.append(loss.item())
                    # 假设 inputs 和 pred_masks 是形状为 [batch_size, channels, height, width] 的张量
                    inputs_images = inputs.detach().cpu()
                    pred_masks_images = pred_masks.detach().cpu()
                    masks_images = masks.detach().cpu()
                    for i, img in enumerate(masks_images):
                        img = img.float()
                        torchvision.utils.save_image(img, f'{output_path}/{epoch}_masks_{i}.png')
                    # 保存 inputs
                    for i, img in enumerate(inputs_images):
                        torchvision.utils.save_image(img, f'{output_path}/{epoch}_inputs_{i}.png')
                    # 保存 pred_masks
                    for i, img in enumerate(pred_masks_images):
                        img = img.float()
                        torchvision.utils.save_image(img, f'{output_path}/{epoch}_pred_masks_{i}.png')
                    # 保存 masks
                
            lr_scheduler.step()

        else:
            for batch_idx, (inputs, masks) in enumerate(train_loader_2):
                inputs, masks = inputs.to(device), masks.to(device)
                optimizer.zero_grad()
                pred_masks = model(inputs)
                # loss = criterion(pred_masks, masks)
                loss = combined_loss_fn(pred_masks, masks)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print(f"batch_idx: {batch_idx}, train_loss: {loss.item()}")
                    loss_epoch_log.append(loss.item())
            lr_scheduler.step()
        average_loss = sum(loss_epoch_log) / len(loss_epoch_log)
        train_log.append(average_loss)
        
        model.eval()
        with torch.no_grad():
            total_samples, total_f1_score = 0, 0
            for batch_idx, (inputs, masks) in enumerate(val_loader):
                inputs, masks = inputs.to(device), masks.to(device)
                pred_masks = model(inputs)
                pred_masks = pred_masks[:, 1, :, :] > pred_masks[:, 0, :, :]
                tp = (masks * pred_masks).sum()
                fp = ((1 - masks) * pred_masks).sum()
                fn = (masks * ~pred_masks).sum()
                f1 = tp / (tp + 0.5 * (fp + fn))
                total_samples += inputs.size(0)
                total_f1_score += f1 * inputs.size(0)
            print(f"val_f1_score: {total_f1_score / total_samples}")
        f1_score_log.append((total_f1_score / total_samples).cpu().numpy())

    # 以epoch序号为x轴，train_log为y轴，绘制折线图
    # 以epoch序号为x轴，f1_score_log为y轴，绘制折线图
    plt.plot(range(len(train_log)), train_log)
    plt.plot(range(len(f1_score_log)), f1_score_log)
    plt.xlabel("epoch")
    plt.ylabel("f1_score, train_loss")
    plt.savefig(f"f1_score_{epoch_num}.png")
    plt.clf()  # 清除当前图像
    # 保存模型
    torch.save(model.state_dict(), f"model_{epoch_num}.pth")
    # 释放显存
    torch.cuda.empty_cache()

# 在测试集上测试模型
def test(epoch_num, test_output_path, test_loader):
    model.load_state_dict(torch.load(f"model_{epoch_num}.pth"))
    model.eval()
    with torch.no_grad():
        total_samples, total_f1_score = 0, 0
        for batch_idx, (inputs, masks) in enumerate(test_loader):
            inputs, masks = inputs.to(device), masks.to(device)
            pred_masks = model(inputs)
            pred_masks = pred_masks[:, 1, :, :] > pred_masks[:, 0, :, :]
            pred_masks_images = pred_masks.detach().cpu()
            for i, img in enumerate(pred_masks_images):
                img = img.float()
                torchvision.utils.save_image(img, f'{test_output_path}/{batch_idx}_{i}.png')
            tp = (masks * pred_masks).sum()
            fp = ((1 - masks) * pred_masks).sum()
            fn = (masks * ~pred_masks).sum()
            f1 = tp / (tp + 0.5 * (fp + fn))
            total_samples += inputs.size(0)
            total_f1_score += f1 * inputs.size(0)
        print(f"test_f1_score: {total_f1_score / total_samples}")



if __name__ == "__main__":
    # 定义你想要创建的文件夹的路径
    output_path = "./intermediate_results"
    test_output_path = "./test_results"
    # 使用 os.makedirs 创建文件夹
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(test_output_path, exist_ok=True)

    train_set_1 = Stenosis_Dataset(mode="train_1")
    train_loader_1 = DataLoader(train_set_1, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    train_set_2 = Stenosis_Dataset(mode="train_2")
    train_loader_2 = DataLoader(train_set_2, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    val_set = Stenosis_Dataset(mode="val")
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4, drop_last=False)
    test_set = Stenosis_Dataset(mode="test")
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4, drop_last=False)
    
    epoch_num = 11

    train(epoch_num, output_path, train_loader_1, train_loader_2, val_loader)
    test(epoch_num, test_output_path, test_loader)

