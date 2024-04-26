import torch
import torch.nn as nn

# 定义联合损失函数
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=None, dice_weight=1.0, iou_weight=1.0):
        super(CombinedLoss, self).__init__()
        # 定义交叉熵损失函数
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight)
        # 初始化权重
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight

    # 定义 Dice 损失函数
    def dice_loss(self, pred, target):
        smooth = 1e-6
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_coeff = (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice_coeff

    # 定义 IoU 损失函数
    def iou_loss(self, pred, target):
        smooth = 1e-6
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou_score = (intersection + smooth) / (union + smooth)
        return 1 - iou_score

    # 前向计算
    def forward(self, pred, target):
        # 计算交叉熵损失
        ce_loss = self.ce_loss(pred, target)
        
        # 计算 Dice 损失
        # print(f"Prediction shape: {pred.shape}, Target shape: {target.shape}")
        pred = torch.argmax(pred, dim=1) # 将预测转换为类别
        dice_loss = self.dice_loss(pred, target)
        
        # 计算 IoU 损失
        iou_loss = self.iou_loss(pred, target)
        
        # 加权组合
        combined_loss = ce_loss + self.dice_weight * dice_loss + self.iou_weight * iou_loss
        
        return combined_loss

# 示例代码
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 假设 pred 是模型输出，target 是目标分割图
    pred = torch.randn((2, 2, 256, 256), requires_grad=True)  # 假设模型输出
    target = torch.randint(0, 2, (2, 256, 256)).long()  # 假设目标分割图
    pred = pred.to(device)
    target = target.to(device)
    # 创建联合损失函数实例
    combined_loss_fn = CombinedLoss(ce_weight=torch.tensor([1.0, 100.0], device='cuda'), dice_weight=1.0, iou_weight=1.0)
    combined_loss_fn = combined_loss_fn.to(device)  # 确保损失函数在同一设备上
    # 计算联合损失
    loss = combined_loss_fn(pred, target)
    print(f"联合损失：{loss.item()}")
