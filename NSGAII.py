import random
import torch
import numpy as np
import math
import pygmo as pg

import Algorithm

class NSGAII(Algorithm.CentralisedAlgorithm):
    def __init__(self, alg_config_filename, domain_name, rover_config_filename, data_filename):
        super().__init__(alg_config_filename, domain_name, rover_config_filename, data_filename)

    def evolve(self, gen=0, traj_write_freq=100):
        """Evolve the population using NSGA-II."""
        # Perform rollout and assign fitness to each individual
        for ind in self.pop:
            # Reset the fitness
            ind.reset_fitness()
            # Conduct rollout
            trajectory, fitness_dict = self.interface.rollout(ind.joint_policy)
            # Compute the trajectory;s entropy
            traj_entropy = self.compute_entropy(trajectory)
            print("trajectory entropy: ", traj_entropy)
            
            # Distribute entropy into fitness components
            weights = {0: 0.1, 1: 0.1}  # Entropy scaling factors for each objective
            for f in fitness_dict:
                fitness_dict[f] += weights[f] * traj_entropy
            
            #fitness_dict = fitness_dict + beta * traj_entropy

            if len(fitness_dict) != self.num_objs:
                raise ValueError(f"[NSGA-II] Expected {self.num_objs} objectives, but got {len(fitness_dict)}.")
            # Store the rollout trajectory
            ind.trajectory = trajectory
            # Store fitness
            for f in fitness_dict:
                ind.fitness[f] = -fitness_dict[f] # NOTE: The fitness sign is flipped to match Pygmo convention
            
            # Add this individual's data to the logger
            self.data_logger.add_data(key='gen', value=gen)
            self.data_logger.add_data(key='id', value=ind.id)
            self.data_logger.add_data(key='fitness', value=ind.fitness)
            if gen == self.num_gens - 1 or gen % traj_write_freq == 0:
                self.data_logger.add_data(key='trajectory', value=ind.trajectory)
            else:
                self.data_logger.add_data(key='trajectory', value=None)
            self.data_logger.write_data()
        
        # Sort the population according to fitness
        sorted_indices = pg.sort_population_mo(points=[ind.fitness for ind in self.pop])
        fitness_tuples = [tuple(ind.fitness) for ind in self.pop]

        # Keep the top half
        sorted_indices = sorted_indices[:len(sorted_indices)//2]
        
        parent_set = [self.pop[i] for i in sorted_indices]
        # Create empty offpring set
        offspring_set = []

        # Fill up the offspring set to the pop_size via offspring-creation
        while len(parent_set) + len(offspring_set) < self.pop_size:
            # Select 2 parents via binary tournament
            idx1, idx2 = random.sample(range(len(sorted_indices)), 2) # Sample two indices from the list
            parent1 = parent_set[min(idx1, idx2)] # choose the lower (more fit) option
            idx1, idx2 = random.sample(range(len(sorted_indices)), 2) # Sample two indices from the list
            parent2 = parent_set[min(idx1, idx2)] # choose the lower (more fit) option
            # Get the offsprings by crossing over these Individuals
            offspring1, offspring2 = self.utils.crossover(parent1, parent2, self.glob_ind_counter)
            # Mutate the offsprings by adding noise
            offspring1.mutate()
            offspring2.mutate()
            # Add to the offspring set
            offspring_set.extend([offspring1, offspring2])
            # Update the global id counter
            self.glob_ind_counter += 2
        
        # Set the population to the parent + offspring set
        self.pop = parent_set
        self.pop.extend(offspring_set)

        random.shuffle(self.pop) # NOTE: This is so that equally dominnat offpsrings in later indices don't just get thrown out
    
    def compute_entropy(self, trajectory):
        """
        Computes the average entropy of the joint position over the episode 
        using KNN entropy estimation.
        """
        # Safety check for empty trajectory
        if not trajectory or len(trajectory) == 0:
            return 0.0

        num_agents = len(trajectory)
        episode_length = len(trajectory[0])

        # ---------------------------------------------------------
        # 1. Process trajectory to extract Joint Positions
        # Target Shape: (Episode_Length, Num_Agents * Position_Dim)
        # ---------------------------------------------------------
        joint_positions = []

        for t in range(episode_length):
            timestep_pos = []
            for agent_i in range(num_agents):
                # Extract position from the specific agent at specific time
                pos = trajectory[agent_i][t]['position']
                
                # Handle inconsistent data types (numpy array vs list vs np.float32)
                if isinstance(pos, np.ndarray):
                    pos = pos.tolist()
                
                # Ensure all elements are standard python floats
                # (The example data contained mixed lists of int and np.float32)
                pos = [float(x) for x in pos]
                
                timestep_pos.extend(pos)
            
            joint_positions.append(timestep_pos)

        # Convert to torch tensor for efficient matrix math
        # Shape: (N, D) where N = Timesteps, D = Joint Dimension
        data = torch.tensor(joint_positions, dtype=torch.float32)

        # ---------------------------------------------------------
        # 2. KNN Entropy Estimation
        # ---------------------------------------------------------
        N = data.shape[0]
        k = 5  # The 'k' for k-Nearest Neighbors
        
        # Edge case: If episode is shorter than k, we cannot compute k-th neighbor
        if N <= k:
            return 0.0

        # Compute pairwise Euclidean distances matrix
        # Shape: (N, N)
        dists = torch.cdist(data, data, p=2)

        # Find the k-th nearest neighbor for each point.
        # We look for the (k+1) smallest values because the 0-th neighbor 
        # is the point itself (distance = 0).
        # values[:, k] extracts the distance to the actual k-th neighbor.
        knn_dists = dists.topk(k + 1, largest=False).values[:, k]

        # ---------------------------------------------------------
        # 3. Compute Entropy Value
        # ---------------------------------------------------------
        # The Kozachenko-Leonenko estimator states H is proportional to mean(log(distance)).
        # We add a small epsilon to prevent log(0) if agents return to exact same pixel.
        epsilon = 1e-12 
        log_dists = torch.log(knn_dists + epsilon)
        
        # Average over the episode
        avg_entropy = torch.mean(log_dists).item()

        return avg_entropy