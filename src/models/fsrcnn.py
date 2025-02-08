import torch
from torch import nn


class FSRCNN_ESM(nn.Module):
    def __init__(self, scale_factor=8):
        super(FSRCNN_ESM, self).__init__()

        # 特征提取
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # 降维
        self.shrinking = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.PReLU()
        )

        # 映射层
        self.mapping = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(16, 12, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU()  # 增加一层映射
        )

        # 上采样（反卷积）
        self.deconvolution = nn.ConvTranspose2d(
            12, 1, kernel_size=9, stride=scale_factor, padding=4, output_padding=scale_factor - 1
        )

        # 细化层
        self.refinement = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        # 前向传播
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.deconvolution(x)
        x = self.refinement(x)
        return x


# 测试模型结构
if __name__ == "__main__":
    model = FSRCNN_ESM(scale_factor=8)
    print(model)

    # 测试输入
    test_input = torch.randn(1, 1, 28, 28)  # 单通道，28x28输入
    test_output = model(test_input)
    print(f"输出尺寸: {test_output.shape}")