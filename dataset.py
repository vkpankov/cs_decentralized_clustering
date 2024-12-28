import numpy as np
import torch
import torch


class AgentsDataset(object):

    @staticmethod
    def generate_clusters_and_mask(min_clusters, max_clusters, min_distance, bounds):
        num_clusters = np.random.randint(min_clusters, max_clusters + 1)
        clusters = []

        for _ in range(num_clusters):
            centroid = np.random.uniform(low=bounds[0], high=bounds[1]).astype(
                np.float32
            )
            while any(np.linalg.norm(centroid - c) < min_distance for c in clusters):
                centroid = np.random.uniform(low=bounds[0], high=bounds[1]).astype(
                    np.float32
                )
            clusters.append(centroid)

        padded_clusters = np.pad(
            np.array(clusters).astype(np.float32),
            pad_width=((0, max_clusters - num_clusters), (0, 0)),
            mode="constant",
        )

        cluster_mask = np.zeros((max_clusters,))
        cluster_mask[:num_clusters] = 1

        return padded_clusters, cluster_mask, num_clusters

    def __init__(
        self, agents_num, dim=2, max_clusters=5, min_distance=0.15, ds_size=1000000
    ) -> None:
        self.from_x = 0
        self.to_x = 1
        self.total_agents = agents_num
        self.m = self.total_agents // 2  # compress_size
        self.dim = dim
        self.ds_size = ds_size
        self.max_clusters = max_clusters
        self.min_distance = min_distance

    def __getitem__(self, index):
        clusters_means, clusters_mask, num_clusters = (
            AgentsDataset.generate_clusters_and_mask(
                min_clusters=self.max_clusters,
                max_clusters=self.max_clusters,
                min_distance=self.min_distance,
                bounds=[(0.05, 0.05), (0.95, 0.95)],
            )
        )

        agents_per_cluster = np.random.multinomial(
            self.total_agents, pvals=[1.0 / num_clusters] * num_clusters
        ).astype(np.int32)
        agents = None
        cluster_ids = []
        for cluster in range(num_clusters):

            correlation = np.random.uniform(low=-1, high=1)
            std_dev = np.random.uniform(low=0.0002, high=0.002)
            rand_matrix = np.random.rand(2, 2)
            sym_matrix = rand_matrix + rand_matrix.T
            scaled_matrix = std_dev * sym_matrix
            scaled_matrix[0, 1] = scaled_matrix[1, 0] = std_dev * correlation

            cluster_agents = np.random.multivariate_normal(
                mean=clusters_means[cluster],
                cov=scaled_matrix,
                size=agents_per_cluster[cluster],
            ).astype(np.float32)
            if agents is None:
                agents = cluster_agents
            else:
                agents = np.concatenate((agents, cluster_agents))
            cluster_ids.extend([cluster for ind in range(len(cluster_agents))])
        idx = list(range(len(agents)))
        np.random.shuffle(idx)
        agents[:] = agents[idx]

        return (
            agents,
            clusters_means,
            clusters_mask.astype(np.float32),
            agents_per_cluster,
        )

    def __len__(self):
        return self.ds_size


class GridAgentsDataset(AgentsDataset):
    def __init__(self, *args, grid_size=320, **kwargs):
        super(GridAgentsDataset, self).__init__(*args, **kwargs)
        self.grid_size = grid_size

    def __getitem__(self, index):
        agents, clusters_means, clusters_mask, numpercluster = super(
            GridAgentsDataset, self
        ).__getitem__(index)

        # Create an empty grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Scaling factor to map agent coordinates to grid
        scale = self.grid_size / (self.to_x - self.from_x)

        for agent in agents:
            x, y = agent
            x = int(x * scale)
            y = int(y * scale)
            x = max(min(x, self.grid_size - 1), 0)
            y = max(min(y, self.grid_size - 1), 0)
            grid[x, y] += 1

        targets = []
        for cluster_mean, mask, csize in zip(
            clusters_means, clusters_mask, numpercluster
        ):
            if mask == 0:
                continue

            cx, cy = cluster_mean * scale
            cx = max(min(cx, self.grid_size - 1), 0)
            cy = max(min(cy, self.grid_size - 1), 0)
            width = height = scale * 3 * np.random.uniform(0.0001, 0.001)
            targets.append([1, cx, cy, csize, width, height])

        targets = np.array(targets, dtype=np.float32)

        return (
            torch.tensor(grid).unsqueeze(0),
            agents,
            clusters_means,
            torch.tensor(targets),
        )
