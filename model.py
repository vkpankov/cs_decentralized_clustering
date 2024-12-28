import torch
import torch.nn as nn
import torch.nn.functional as F


# For original space without 2D grid projection
class AgentsGaussianCompressModel(nn.Module):
    def __init__(self, input_size=1000, compressed_size=256):
        super(AgentsGaussianCompressModel, self).__init__()
        self.input_size = input_size
        self.compressed_size = compressed_size // 2
        variance = 1 / (self.compressed_size)
        self.agents_measurement_matrix = nn.Parameter(
            torch.randn(self.compressed_size, input_size, dtype=torch.float32)
            * variance
        )

    def forward(self, x):
        device = x.device
        measurement_matrix = self.agents_measurement_matrix.to(device)
        compressed_data = torch.matmul(measurement_matrix, x)
        return compressed_data


# For 2d image projected data
class GaussianCompressModel(nn.Module):
    def __init__(self, chunk_size=(320, 320), compressed_size=(16, 16), channels=1):
        super(GaussianCompressModel, self).__init__()
        self.chunk_size = chunk_size
        self.compressed_size = compressed_size
        self.channels = channels
        variance = 1 / (compressed_size[0] * compressed_size[1])
        chunk_size_flat = chunk_size[0] * chunk_size[1]
        compressed_size_flat = compressed_size[0] * compressed_size[1]
        self.measurement_matrix = nn.Parameter(
            torch.randn(compressed_size_flat, chunk_size_flat, dtype=torch.float32)
            * variance
        )

    def forward(self, x):
        device = x.device
        n, c, h, w = x.shape
        compressed_h = int(h * self.compressed_size[0] / self.chunk_size[0])
        compressed_w = int(w * self.compressed_size[1] / self.chunk_size[1])
        compressed_image = torch.zeros(n, c, compressed_h, compressed_w, device=device)

        measurement_matrix = self.measurement_matrix.to(device)

        for i in range(0, h, self.chunk_size[0]):
            for j in range(0, w, self.chunk_size[1]):
                if i + self.chunk_size[0] <= h and j + self.chunk_size[1] <= w:
                    chunk = x[
                        :, :, i : i + self.chunk_size[0], j : j + self.chunk_size[1]
                    ]
                    chunk_flat = chunk.reshape(n * c, -1)
                    compressed_chunk_flat = torch.matmul(
                        chunk_flat, measurement_matrix.T
                    )
                    compressed_chunk = compressed_chunk_flat.reshape(
                        n, c, self.compressed_size[0], self.compressed_size[1]
                    )
                    ci = i * self.compressed_size[0] // self.chunk_size[0]
                    cj = j * self.compressed_size[1] // self.chunk_size[1]
                    compressed_image[
                        :,
                        :,
                        ci : ci + self.compressed_size[0],
                        cj : cj + self.compressed_size[1],
                    ] = compressed_chunk

        return compressed_image


def build_model(
    chunk_size=(128, 128),
    compressed_size=(16, 16),
    clusters_number=10,
    agents_space=False,
):
    if agents_space:
        compress_model = AgentsGaussianCompressModel(
            input_size=chunk_size[0] * chunk_size[1],
            compressed_size=compressed_size[0] * compressed_size[1],
        )
    else:
        compress_model = GaussianCompressModel(
            chunk_size=chunk_size, compressed_size=compressed_size
        )

    model = torch.nn.Sequential(
        compress_model,
        nn.Flatten(start_dim=1),
        nn.ReLU(),
        nn.BatchNorm1d(compressed_size[0] * compressed_size[1]),
        nn.Linear(compressed_size[0] * compressed_size[1], 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2 * clusters_number),
        nn.Unflatten(1, (2, clusters_number)),
    ).cuda()
    return model
