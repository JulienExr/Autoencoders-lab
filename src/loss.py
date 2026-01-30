import torch
import torchvision.models as models


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        self.slice1 = torch.nn.Sequential(*[vgg[x] for x in range(4)])     # relu1_2
        self.slice2 = torch.nn.Sequential(*[vgg[x] for x in range(4, 9)])  # relu2_2

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, input, target):
        input = (input + 1.0) / 2.0
        target = (target + 1.0) / 2.0

        feat_input1 = self.slice1(input)
        feat_target1 = self.slice1(target)
        loss1 = torch.nn.functional.l1_loss(feat_input1, feat_target1)
        feat_input2 = self.slice2(feat_input1)
        feat_target2 = self.slice2(feat_target1)
        loss2 = torch.nn.functional.l1_loss(feat_input2, feat_target2)

        return loss1 + loss2
        