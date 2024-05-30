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
import numpy as np
import argparse
from data_augmentation import SSDAugmentation, SSDBaseTransform
from fast_cnn import get_model_instance_segmentation



def train(epoch_num, output_path, batch_size):
    train_set_1 = Stenosis_Dataset(mode="train", transform=SSDAugmentation())
    train_loader_1 = DataLoader(train_set_1, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    train_set_2 = Stenosis_Dataset(mode="train", transform=SSDBaseTransform())
    train_loader_2 = DataLoader(train_set_2, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_set = Stenosis_Dataset(mode="val", transform=SSDBaseTransform())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
    train_log = []
    f1_score_log = []
    detect_model.load_state_dict(torch.load("./pth/fasterrcnn_resnet50_fpn.pth"))
    detect_model.eval()
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
            ########################################################
            # images_detect = list(img.to(device) for img in inputs)
            # bounding_boxes = detect_model(images_detect)
            # masked_preds = torch.zeros_like(prob_masks)
            # for i, (image, mask, bbox, prob_mask, image_name) in enumerate(zip(images_detect,masks, bounding_boxes, prob_masks, image_names)):
            #     # 留下预测框的区域
            #     masked_preds[i] = mask_the_pred(bbox, prob_mask)
            #########################################################
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
    plt.savefig(f"atten_f1_score_{epoch_num}.png")
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
    detect_model.load_state_dict(torch.load("./pth/fasterrcnn_resnet50_fpn.pth"))
    model.eval()
    detect_model.eval()
    with torch.no_grad():
        total_samples, total_f1_score = 0, 0
        for batch_idx, (images, masks, image_names) in enumerate(tqdm(test_loader)):
            images, masks = images.to(device), masks.to(device)
            prob_masks = model(images)
            images_detect = list(img.to(device) for img in images)
            bounding_boxes = detect_model(images_detect)
            for i, (image, mask, bbox, prob_mask, image_name) in enumerate(zip(images_detect,masks, bounding_boxes, prob_masks, image_names)):
                pred_mask = prob_mask[1, :, :] > prob_mask[0, :, :]
                # 留下预测框的区域
                # masked_pred_mask = pred_mask.long()
                masked_pred_mask = mask_the_mask(bbox, pred_mask)

                torchvision.utils.save_image(masked_pred_mask.detach().cpu().float(), f'{test_output_path}/{image_name[:-4]}_pred.png')
                torchvision.utils.save_image(mask.detach().cpu().float(), f'{test_output_path}/{image_name[:-4]}_mask.png')
                torchvision.utils.save_image(image.detach().cpu().float(), f'{test_output_path}/{image_name[:-4]}_input.png')
                # save_pred_mask(masked_pred_mask, mask, image, image_name, test_output_path)

                tp = (mask * masked_pred_mask).sum()
                fp = ((1 - mask) * masked_pred_mask).sum()
                fn = (mask * (1-masked_pred_mask)).sum()
                f1 = tp / (tp + 0.5 * (fp + fn))
                total_samples += image.size(0)
                total_f1_score += f1 * image.size(0)
        print(f"test_f1_score: {total_f1_score / total_samples}")


def mask_the_pred(bbox, prob_masks):
    # 绘制预测框
    boxes = bbox['boxes']
    labels = bbox['labels']
    scores = bbox['scores']
    # 如果有多个置信度大于 0.7 的框，则全部绘制
    flag = False
    ou = torch.zeros_like(prob_masks[0])
    masked_prob_mask = torch.zeros_like(prob_masks)
    for box, score in zip(boxes, scores):
        if score >= 0.5:
            flag = True
            x_min, y_min, x_max, y_max =map(int, box)
            ou[y_min:y_max, x_min:x_max] = 1
    if flag == True:
        masked_prob_mask[0] = prob_masks[0] + (1-ou)
        masked_prob_mask[1] = prob_masks[1] * ou
    # 否则绘制最高分数的框
    elif len(boxes) > 0 and flag == False:
        highest_score_index = torch.argmax(scores).item()   # 找到最高分数的索引
        highest_score_box = boxes[highest_score_index]

        # 绘制最高分数的框
        x_min, y_min, x_max, y_max = map(int, highest_score_box)  # 将框的坐标转换为整数
        # 初始化一个与 pred_masks 相同形状的全零数组
        zero_mask = torch.zeros_like(prob_masks[0])
        zero_mask[y_min:y_max, x_min:x_max] = 1
        masked_prob_mask[0] = prob_masks[0] + (1-zero_mask)
        masked_prob_mask[1] = prob_masks[1] * zero_mask
        # 只保留框内的区域
    else:
        masked_prob_mask = prob_masks

    return masked_prob_mask


def mask_the_mask(bbox, pred_mask):
    # 绘制预测框
    boxes = bbox['boxes']
    labels = bbox['labels']
    scores = bbox['scores']
    # 如果有多个置信度大于 0.7 的框，则全部绘制
    flag = False
    ou = torch.zeros_like(pred_mask).long()
    for box, score in zip(boxes, scores):
        if score >= 0.5:
            flag = True
            x_min, y_min, x_max, y_max =map(int, box)
            ou[y_min:y_max, x_min:x_max] = 1
    if flag == True:
        masked_pred_mask = pred_mask * ou
    # 否则绘制最高分数的框
    elif len(boxes) > 0 and flag == False:
        highest_score_index = torch.argmax(scores).item()   # 找到最高分数的索引
        highest_score_box = boxes[highest_score_index]

        # 绘制最高分数的框
        x_min, y_min, x_max, y_max = map(int, highest_score_box)  # 将框的坐标转换为整数
        # 初始化一个与 pred_masks 相同形状的全零数组
        masked_pred_mask = torch.zeros_like(pred_mask).long()
        # 只保留框内的区域
        masked_pred_mask[y_min:y_max, x_min:x_max] = pred_mask[y_min:y_max, x_min:x_max]
    else:
        masked_pred_mask = torch.zeros_like(pred_mask).long()

    return masked_pred_mask




def save_pred_mask(masked_pred_mask, mask, image, image_name, test_output_path):
    pred_mask_cpu = masked_pred_mask.detach().cpu().numpy()
    mask_cpu = mask.detach().cpu().numpy()
    input_image_cpu = image.detach().cpu().numpy().squeeze()
    # print(input_image_cpu)
    # print(pred_mask_cpu)
    # print(mask_cpu)
    # 创建一个新的图形
    fig, ax = plt.subplots()
    # 显示原始图像
    ax.imshow(input_image_cpu, cmap='gray')
    # 创建一个与 input_image_cpu 形状相同的覆盖层
    mask_overlay = np.zeros_like(input_image_cpu)
    pred_mask_overlay = np.zeros_like(input_image_cpu)
    # # 将 mask 和 pred_mask 的位置设置为对应的颜色
    mask_overlay[mask_cpu > 0.5] = [100]  # 红色
    pred_mask_overlay[pred_mask_cpu > 0.5] = [200]  # 黄色
    # # 显示覆盖层，设置透明度为 50%
    ax.imshow(mask_overlay, alpha=0.1)
    ax.imshow(pred_mask_overlay, alpha=0.1)
    # 去除坐标轴
    ax.axis('off')
    # 保存结果图像
    plt.savefig(f'{test_output_path}/{image_name[:-4]}_combined.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./intermediate_results")
    parser.add_argument("--test_output_path", type=str, default="./test_results")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pretrained_model", type=str, default='./pth/atten_model_10.pth')
    args = parser.parse_args()
    # 使用 os.makedirs 创建文件夹
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.test_output_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Attention_UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 50.0], device=device))
    combined_loss_fn = CombinedLoss(ce_weight=torch.tensor([1.0, 50.0], device=device), dice_weight=2.0, iou_weight=2.0, device=device)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4,gamma=0.1)

    num_classes = 2  # 根据你数据集的类别数调整
    detect_model = get_model_instance_segmentation(num_classes)
    detect_model.to(device)

    if args.mode == "train":
        train(args.epochs, args.output_path, args.batch_size)
    else:
        test(args.test_output_path, args.pretrained_model, args.batch_size)

