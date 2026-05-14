import random
import torch
import numpy as np
import math
import pygmo as pg
import os

import Algorithm

ENTROPY = True

class NSGAII(Algorithm.CentralisedAlgorithm):
    def __init__(self, alg_config_filename, domain_name, rover_config_filename, data_filename, beta):
        super().__init__(alg_config_filename, domain_name, rover_config_filename, data_filename)
        self.beta = beta

    def evolve(self, gen=0, traj_write_freq=100):
        """Evolve the population using NSGA-II."""
        # Perform rollout and assign fitness to each individual
        for ind in self.pop:
            # Reset the fitness
            ind.reset_fitness()
            # Conduct rollout with halfway reward tracking
            trajectory, fitness_dict, reward_at_halfway = self.interface.rollout(ind.joint_policy, return_halfway_reward=True)
            # Compute the trajectory's entropy
            traj_entropy = self.compute_entropy(trajectory)
            #print("trajectory entropy: ", traj_entropy)
            
            # Transform fitness to interval rewards: [R(0-T/2), R(T/2,T)]
            # where R(T/2,T) = total_reward - reward_at_halfway
            weights = {0: self.beta, 1: self.beta}  # Entropy scaling factors for each objective

            first_half = reward_at_halfway.get(0, 0)
            total = fitness_dict.get(0, 0)
            second_half = total - first_half

            # Store raw fitness values (before entropy shaping)
            raw_fitness_dict = {
                0: first_half,
                1: second_half
            }

            # Apply entropy shaping as a bonus (unchanged but applied to the interval rewards)
            shaped_fitness_dict = {
                0: first_half + weights[0] * traj_entropy,
                1: second_half + weights[1] * traj_entropy
            }

            if len(shaped_fitness_dict) != self.num_objs:
                raise ValueError(f"[NSGA-II] Expected {self.num_objs} objectives, but got {len(shaped_fitness_dict)}.")
            # Store the rollout trajectory
            ind.trajectory = trajectory
            # Store fitness
            for f in shaped_fitness_dict:
                ind.fitness[f] = -shaped_fitness_dict[f] # NOTE: The fitness sign is flipped to match Pygmo convention
                ind.raw_fitness[f] = -raw_fitness_dict[f]  # unshaped, for evaluation

            # Add this individual's data to the logger
            self.data_logger.add_data(key='gen', value=gen)
            self.data_logger.add_data(key='id', value=ind.id)
            self.data_logger.add_data(key='fitness', value=ind.fitness)
            self.data_logger.add_data(key='raw_fitness', value=ind.raw_fitness)
            self.data_logger.add_data(key='traj_entropy', value=traj_entropy)
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

        # Minimal checkpoint: save latest population state so experiments can be resumed.
        try:
            run_dir = self.interface.rover_env.data_dir
            if run_dir is None and getattr(self, 'data_logger', None) and getattr(self.data_logger, 'target_filename', None):
                run_dir = os.path.dirname(self.data_logger.target_filename)
            if run_dir is not None:
                ckpt = {
                    'gen': gen,
                    'glob_ind_counter': self.glob_ind_counter,
                    'num_gens': self.num_gens,
                    'pop_size': self.pop_size,
                    'population': [],
                    'traj_write_freq': traj_write_freq,
                }
                for ind in self.pop:
                    ind_entry = {
                        'id': ind.id,
                        'fitness': ind.fitness,
                        'raw_fitness': ind.raw_fitness,
                        'policies': [p.state_dict() for p in ind.joint_policy]
                    }
                    ckpt['population'].append(ind_entry)

                latest_path = os.path.join(run_dir, 'latest_checkpoint.pth')
                tmp_path = latest_path + '.tmp'
                torch.save(ckpt, tmp_path)
                try:
                    os.replace(tmp_path, latest_path)
                except Exception:
                    # best-effort move
                    os.rename(tmp_path, latest_path)
        except Exception:
            # Don't interrupt evolution if checkpointing fails
            pass
    
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
        #epsilon = 1e-12
        epsilon = 1 # Shift distance so that log values are not negative 
        log_dists = torch.log(knn_dists + epsilon)
        
        # Average over the episode
        avg_entropy = torch.mean(log_dists).item()

        return avg_entropy