import numpy as np


class CustomConv2D:
    def __init__(self, num_filters, kernel_size):
        self.num_filters = num_filters
        self.kernel_size = kernel_size

    def forward(self, input):
        self.input = input
        num_samples, input_height, input_width, num_channels = input.shape
        # Kernel size: 3x3, 1 input channel, 32 filters
        self.filters = np.random.randn(3, 3, 1, 32)

        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1
        self.output = np.zeros(
            (num_samples, output_height, output_width, self.num_filters))

        for i in range(output_height):
            for j in range(output_width):
                output_region = input[:, i:i +
                                      self.kernel_size, j:j + self.kernel_size, :]
                self.output[:, i, j, :] = np.sum(
                    output_region * self.filters, axis=(1, 2, 3))

        return self.output


# Sử dụng CustomConv2D
custom_conv_layer = CustomConv2D(num_filters=32, kernel_size=3)
input_data = np.random.rand(1, 28, 28, 1)  # 1 sample, 28x28 image, 1 channel
output_data = custom_conv_layer.forward(input_data)
print(output_data.shape)  # In ra kích thước của đầu ra
