import torch
import torch.nn as nn
import torch.nn.functional as F

class MutilTverskyLoss(nn.Module):
    """
        multi label TverskyLoss loss with weighted
        Y_pred: [None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_pred is softmax result
        Y_gt:[None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_gt is one hot result
        alpha: tensor shape (C,) where C is the number of classes,eg:[0.1,1,1,1,1,1]
        :return:
        """

    def __init__(self, alpha, beta):
        super(MutilTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = 1.e-5

    def forward(self, y_pred_logits, y_true):
        y_pred = torch.softmax(y_pred_logits, dim=1)
        Batchsize, Channel = y_pred.shape[0], y_pred.shape[1]
        y_pred = y_pred.float().contiguous().view(Batchsize, Channel, -1)
        y_true = y_true.long().contiguous().view(Batchsize, -1)
        y_true = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
        y_true = y_true.permute(0, 2, 1)  # H, C, H*W
        assert y_pred.size() == y_true.size()
        bg_true = 1 - y_true
        bg_pred = 1 - y_pred
        tp = torch.sum(y_pred * y_true, dim=(0, 2))
        fp = torch.sum(y_pred * bg_true, dim=(0, 2))
        fn = torch.sum(bg_pred * y_true, dim=(0, 2))
        tversky = 1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        mask = y_true.sum((0, 2)) > 0
        tversky *= mask.to(tversky.dtype)
        return (tversky * self.alpha).sum() / torch.count_nonzero(mask)


# 定义联合损失函数
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=None, dice_weight=1.0, iou_weight=1.0, device = None):
        super(CombinedLoss, self).__init__()
        # 定义交叉熵损失函数
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight)
        self.mutiltverskyloss = MutilTverskyLoss(0.5, 0.5)
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
        tversky_loss = self.mutiltverskyloss(pred, target)
        # 加权组合
        combined_loss = ce_loss + tversky_loss # + self.dice_weight * dice_loss + self.iou_weight * iou_loss
        
        return combined_loss, tversky_loss, ce_loss

# 示例代码
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 假设 pred 是模型输出，target 是目标分割图
    pred = torch.randn((2, 2, 256, 256), requires_grad=True)  # 假设模型输出
    target = torch.randint(0, 2, (2, 256, 256)).long()  # 假设目标分割图
    pred = pred.to(device)
    target = target.to(device)
    # 创建联合损失函数实例
    combined_loss_fn = CombinedLoss(ce_weight=torch.tensor([1.0, 10.0], device=device), dice_weight=2.0, iou_weight=2.0)
    combined_loss_fn = combined_loss_fn.to(device)  # 确保损失函数在同一设备上
    # 计算联合损失
    combined_loss, ce_loss, dice_loss, iou_loss = combined_loss_fn(pred, target)
    combined_loss.backward()
    print(f"联合损失：{combined_loss.item()}, 交叉熵损失：{ce_loss.item()}, dice损失：{dice_loss.item()}, iou损失：{iou_loss.item()}")
