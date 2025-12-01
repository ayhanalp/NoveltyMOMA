import random
import torch
import numpy as np
import math
import pygmo as pg

import Algorithm

class NSGAII(Algorithm.CentralisedAlgorithm):
    def __init__(
        self,
        alg_config_filename,
        domain_name,
        rover_config_filename,
        data_filename,
        intrinsic_type="none",          # "none", "entropy", "novelty"
        intrinsic_usage="reward"        # "reward", "objective"
    ):
        super().__init__(alg_config_filename, domain_name, rover_config_filename, data_filename)
        
        self.intrinsic_type = intrinsic_type
        self.intrinsic_usage = intrinsic_usage

        self.archive = []  # For novelty

    def evolve(self, gen=0, traj_write_freq=100):
        """Evolve the population using NSGA-II."""
        # Perform rollout and assign fitness to each individual
        for ind in self.pop:
            # Reset the fitness
            ind.reset_fitness()
            # Condcut rollout
            trajectory, fitness_dict = self.interface.rollout(ind.joint_policy)
            
            ############ Intrinsic Goodness ############
            intrinsic_val = None
            
            if self.intrinsic_type != "none":
                # Extract positions for intrinsic reward (one agent or aggregated - use Agent 0 for now)
                pos_traj = np.array([step["position"] for step in trajectory[0]])  # shape (T, D)
                intrinsic_val = self.compute_intrinsic(pos_traj)
                
                # Calculate novelty or entropy and either distribute across all rewards or store as another objective
                match self.intrinsic_usage:
                    case "reward":
                        beta = 0.3
                        fitness_dict["reward"] += beta * intrinsic_val

                    case "objective":
                        fitness_dict["intrinsic"] = intrinsic_val

            # Add mean state to archive if novelty
            if self.intrinsic_type == "novelty":
                self.archive.append(pos_traj.mean(axis=0))

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
        """Computes the kNN entropy of a single agent's trajectory"""

        if len(trajectory) == 0:
            return torch.tensor(1.0)

        episodic_memory_tensor = torch.stack(trajectory, dim=0)

        entropy = 0.0
        for state_i in episodic_memory_tensor:
            s_dist = torch.cdist(state_i.unsqueeze(0), episodic_memory_tensor, p=2,
                                 compute_mode='use_mm_for_euclid_dist').squeeze(0).sort()[0]
            s_dist = np.array(s_dist)

            # self.args.k = 5  # was the default
            # k = params.args.k
            k = 5
            k_H = k + 1
            for k_i in range(k_H):
                if len(s_dist) == k_i:
                    break
                else:
                    entropy += math.log(s_dist[k_i] + 1)

        return entropy / len(trajectory)
    
    def compute_novelty(self, trajectory, archive, k=10):
        """Reduces trajectory to mean state and computes knn distance (novelty) to all other archived trajectories """
        # Compute trajectory mean state
        traj_tensor = torch.stack(trajectory, dim=0)  # shape: (T, D)
        mean_states = traj_tensor.mean(dim=0).cpu().numpy()  # shape: (D,)

        if len(archive) == 0:
            return 0.0
        else:
            # Euclidean distance to all mean states in archive
            dists = [np.linalg.norm(mean_states - past_mean) for past_mean in archive]

            # k nearest distances
            k = min(k, len(dists))
            nearest_distances = sorted(dists)[:k]
            novelty = float(np.mean(nearest_distances))

        return novelty
    
    def compute_intrinsic(self, trajectory):
        """Compute entropy or novelty based on intrinsic_type."""
        match self.intrinsic_type:
            case "none":
                return None
            case "entropy":
                return self.compute_entropy(trajectory)
            case "novelty":
                return self.compute_novelty(trajectory, self.archive)
            case _:
                raise ValueError(f"Unknown intrinsic_type: {self.intrinsic_type}")