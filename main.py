import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data import Stenosis_Dataset
from combined_loss import CombinedLoss
from Unet import Unet
from ResUnet import ResUnet
from AttentionUnet import Attention_UNet
import os 
from tqdm import tqdm
import torchvision
import argparse
from data_augmentation import SSDAugmentation, SSDBaseTransform



def train(epoch_num, output_path, batch_size):
    train_set_1 = Stenosis_Dataset(mode="train", transform=SSDAugmentation())
    train_loader_1 = DataLoader(train_set_1, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    train_set_2 = Stenosis_Dataset(mode="train", transform=SSDBaseTransform())
    train_loader_2 = DataLoader(train_set_2, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_set = Stenosis_Dataset(mode="val", transform=SSDBaseTransform())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
    train_log = []
    f1_score_log = []
    for epoch in range(epoch_num):
        loss_epoch_log = []
        print(f"EPOCH: {epoch}/{epoch_num}")
        model.train()
        if (epoch < (0)): # epoch_num//2)):
            train_loader = train_loader_1
        else:
            train_loader = train_loader_2
        pbar = tqdm(train_loader)
        for batch_idx, (inputs, masks, image_names) in enumerate(pbar):
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            prob_masks = model(inputs)
            loss = criterion(prob_masks, masks)
            loss.backward()
            # combined_loss, tversky_loss, ce_loss = combined_loss_fn(pred_masks, masks)
            # combined_loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                # pbar.set_postfix({"train_loss": combined_loss.item(), 
                #                   "celoss":ce_loss.item(), "tversky_loss":tversky_loss.item()})
                loss_epoch_log.append(loss.item())
                average_loss = sum(loss_epoch_log) / len(loss_epoch_log)
                pbar.set_postfix({"train_loss": loss.item(), "accum_loss": average_loss})
                train_log += loss_epoch_log
            """  
            # 假设 inputs 和 pred_masks 是形状为 [batch_size, channels, height, width] 的张量
            pred_masks = prob_masks[:, 1, :, :] > prob_masks[:, 0, :, :]
            inputs_images = inputs.detach().cpu()
            pred_masks_images = pred_masks.detach().cpu()
            masks_images = masks.detach().cpu()
            for i, img in enumerate(masks_images):
                img = img.float()
                torchvision.utils.save_image(img, f'{output_path}/{image_names[i]}_masks.png')
            # 保存 inputs
            for i, img in enumerate(inputs_images):
                torchvision.utils.save_image(img, f'{output_path}/{image_names[i]}_inputs.png')
            # 保存 pred_masks
            for i, img in enumerate(pred_masks_images):
                img = img.float()
                torchvision.utils.save_image(img, f'{output_path}/{image_names[i]}_pred.png')
            """
            # 保存 masks
        lr_scheduler.step()
        
        model.eval()
        with torch.no_grad():
            total_samples, total_f1_score = 0, 0
            for batch_idx, (inputs, masks, _) in enumerate(tqdm(val_loader)):
                inputs, masks = inputs.to(device), masks.to(device)
                prob_masks = model(inputs)
                pred_masks = prob_masks[:, 1, :, :] > prob_masks[:, 0, :, :]
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
    torch.save(model.state_dict(), f"./pth/atten_model_{epoch_num}.pth")
    # 释放显存
    torch.cuda.empty_cache()

# 在测试集上测试模型
def test(test_output_path, pretrained_model, batch_size):
    test_set = Stenosis_Dataset(mode="test", transform=SSDBaseTransform())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
    model.load_state_dict(torch.load(pretrained_model))
    model.eval()
    with torch.no_grad():
        total_samples, total_f1_score = 0, 0
        for batch_idx, (inputs, masks, image_names) in enumerate(tqdm(test_loader)):
            inputs, masks = inputs.to(device), masks.to(device)
            prob_masks = model(inputs)
            pred_masks = prob_masks[:, 1, :, :] > prob_masks[:, 0, :, :]
            
            pred_masks_images = pred_masks.detach().cpu()
            masks_images = masks.detach().cpu()
            inputs_images = inputs.detach().cpu()
            for i, img in enumerate(pred_masks_images):
                img = img.float()
                torchvision.utils.save_image(img, f'{test_output_path}/{image_names[i][:-4]}_pred.png')
            for i, img in enumerate(masks_images):
                img = img.float()
                torchvision.utils.save_image(img, f'{test_output_path}/{image_names[i][:-4]}_mask.png')
            for i, img in enumerate(inputs_images):
                torchvision.utils.save_image(img, f'{test_output_path}/{image_names[i][:-4]}_input.png')
            
            tp = (masks * pred_masks).sum()
            fp = ((1 - masks) * pred_masks).sum()
            fn = (masks * ~pred_masks).sum()
            f1 = tp / (tp + 0.5 * (fp + fn))
            total_samples += inputs.size(0)
            total_f1_score += f1 * inputs.size(0)
        print(f"test_f1_score: {total_f1_score / total_samples}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./intermediate_results")
    parser.add_argument("--test_output_path", type=str, default="./test_results")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pretrained_model", type=str, default='./pth/model_10.pth')
    args = parser.parse_args()
    # 使用 os.makedirs 创建文件夹
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.test_output_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Attention_UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 50.0], device=device))
    combined_loss_fn = CombinedLoss(ce_weight=torch.tensor([1.0, 50.0], device=device), dice_weight=2.0, iou_weight=2.0, device=device)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4,gamma=0.1, last_epoch=-1)

    if args.mode == "train":
        train(args.epochs, args.output_path, args.batch_size)
    else:
        test(args.test_output_path, args.pretrained_model, args.batch_size)

