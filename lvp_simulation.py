"""
python lvp_simulation.py \
    --grid_size 128 \
    --agents_num 1000 \
    --max_clusters 10 \
    --batch_size 32 \
    --comp_size 12 12 \
    --step_size 0.2 \
    --num_it_list 10 15 20 25 30 35 40 45 50 100 \
    --num_simulations 30 \
    --graph_type "random" \
    --model_ckpt "/Users/vikentiy/Documents/cs_decentralized_clustering/checkpoints/last_c10_p12x12.pt" \
    --pickle_path "results_12x12.pkl"
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle

from model import build_model
from dataset import GridAgentsDataset

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run decentralized clustering experiments (using LVP simulation and pretrained prediction model)."
    )
    parser.add_argument("--grid_size", type=int, default=128,
                        help="Grid size for the test dataset.")
    parser.add_argument("--agents_num", type=int, default=1000,
                        help="Number of agents in the test dataset.")
    parser.add_argument("--max_clusters", type=int, default=10,
                        help="Maximum number of clusters.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for data loader.")
    parser.add_argument("--comp_size", nargs=2, type=int, default=[12, 12],
                        help="(Height, Width) of the compressed representation.")
    parser.add_argument("--step_size", type=float, default=0.2,
                        help="Step size for consensus updates.")
    parser.add_argument("--num_it_list", nargs="+", type=int, default=[200],
                        help="List of iteration values to run the consensus process.")
    parser.add_argument("--num_simulations", type=int, default=10,
                        help="Number of repeated simulations for each num_it.")
    parser.add_argument("--graph_type", type=str, default="circular",
                        choices=["circular", "random"],
                        help="Type of graph topology to use: 'circular' or 'random'.")
    parser.add_argument("--neg_val", type=int, default=-100,
                        help="Negative threshold for removing edges in adjacency matrix (unused here, but kept).")
    parser.add_argument("--model_ckpt", type=str, default="last_c10_p12x12.pt",
                        help="Path to a pre-trained model checkpoint.")
    parser.add_argument("--pickle_path", type=str, default="pickled_12x12_mae_costs.pkl",
                        help="File path to save the pickled results.")

    return parser.parse_args()



def get_local_measurement(compress_model, agent_state):
    grid_size = 128
    grid_repr = np.zeros((grid_size,grid_size), dtype=np.float32)
    scale = grid_size 
    x = int(agent_state[0] * scale)
    y = int(agent_state[1] * scale)
    x = max(min(x, grid_size - 1), 0)
    y = max(min(y, grid_size - 1), 0)
    grid_repr[x, y] = 1 
    return compress_model(torch.from_numpy(grid_repr).unsqueeze(0).unsqueeze(0).cpu()).squeeze(0).squeeze(0).detach().cpu().numpy()

def build_random_graph(n,p):
    G = nx.erdos_renyi_graph(n, p)
    for node in G.nodes():
        if len(list(G.neighbors(node))) == 0:
            while True:
                target = np.random.randint(0, n)
                if target != node:
                    G.add_edge(node, target)
                    break
    return G

def get_global_measurement(
        rec_model,
        input_agents,
        num_it=100,
        comp_size=(12, 12),
        step_size=0.2,
        graph_type="circular",
        grid_size=128,
    ):

    num_agents = input_agents.shape[0]
    flat_dim = comp_size[0] * comp_size[1]

    global_y = np.zeros((num_agents, flat_dim), dtype=np.float32)
    
    if graph_type == "circular":
        G = nx.cycle_graph(num_agents)
    elif graph_type == "random":
        G = nx.erdos_renyi_graph(num_agents, 0.02)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    adjacency = nx.to_numpy_array(G)

    comm_cost = 0

    adjacency = nx.to_numpy_array(G)  # Convert to adjacency matrix
    comm_cost = 0
    for x in range(0,num_it):
        for i, agent in enumerate(input_agents):   
            if x > 0:
                nbs = np.argwhere(adjacency[i]).flatten()  # Neighbors indices as a flat array
                differences = global_y[nbs] - global_y[i]
                scaled_diffs = step_size * adjacency[i, nbs][:, np.newaxis] * differences
                global_y[i] += scaled_diffs.sum(axis=0)
                comm_cost += len(nbs)*global_y[i].shape[-1]
            else:
                global_y[i] = get_local_measurement(rec_model[0:2], agent)
    total_meas = 1000*global_y[0].astype(np.float32)
    return total_meas, comm_cost


def compute_accuracy_and_mae(batch_res, batch_true_cl):

    distances = torch.cdist(batch_res, batch_true_cl, p=1)
    min_distances = torch.min(distances, dim=2).values  # shape [batch_size, 10]

    mae_per_batch = torch.mean(min_distances, dim=1)
    average_mae = torch.mean(mae_per_batch).item()

    correct_predictions = (min_distances <= 0.05 * 128).float()
    accuracy_per_batch = torch.mean(correct_predictions, dim=1)
    average_accuracy = torch.mean(accuracy_per_batch).item()

    return average_mae, average_accuracy


def main():
    args = parse_arguments()

    val_dataset = GridAgentsDataset(
        grid_size=args.grid_size,
        agents_num=args.agents_num,
        max_clusters=args.max_clusters
    )

    rec_model = build_model(compressed_size=tuple(args.comp_size))
    rec_model.load_state_dict(torch.load(args.model_ckpt, map_location="cpu"))
    rec_model.eval().cpu()

    cost_dict = {}
    mae_dict = {}

    mae_lvp = []
    costs = []

    for num_it_val in tqdm(args.num_it_list):
        cost_dict[num_it_val] = []
        mae_dict[num_it_val] = []

        for i in range(args.num_simulations):
            data, agents, _, target = val_dataset.__getitem__(i)
            # LVP-based global measurement
            total_meas, comm_cost = get_global_measurement(
                rec_model,
                agents,
                num_it=num_it_val,  
                comp_size=tuple(args.comp_size),
                step_size=args.step_size,
                graph_type=args.graph_type
            )
            resp2 = rec_model[2:](torch.from_numpy(total_meas).cpu().unsqueeze(0)).detach().cpu()
            mmae, acc_lvp = compute_accuracy_and_mae(target[:,1:3].cpu().unsqueeze(0), resp2.swapaxes(-1,-2).cpu())
            if mmae > args.grid_size:
                mmae = args.grid_size
            mae_lvp.append(mmae)
            mae_dict[num_it_val].append(mmae)
            cost_dict[num_it_val].append(comm_cost)
        print(f"Num it: {num_it_val}, MAE: {np.mean(mae_lvp)}+-{np.std(mae_lvp)}")


    with open(args.pickle_path, "wb") as f:
        pickle.dump({"mae_dict": mae_dict, "cost_dict": cost_dict}, f)
    print(f"Results saved to {args.pickle_path}")

if __name__ == "__main__":
    main()
