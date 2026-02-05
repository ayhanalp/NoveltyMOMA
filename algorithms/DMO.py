import random
import numpy
import pygmo as pg

import Algorithm
import Individual

class DMO(Algorithm.CentralisedAlgorithm):
    def __init__(self, alg_config_filename, domain_name, rover_config_filename, data_filename):
        super().__init__(alg_config_filename, domain_name, rover_config_filename, data_filename)

    def evolve(self, gen=0, traj_write_freq=100):
        """Evolve the population using NSGA-II."""
        # Perform rollout and assign fitness to each individual
        for ind in self.pop:
            # Reset the fitness
            ind.reset_fitness()
            # Condcut rollout
            trajectory, fitness_dict = self.interface.rollout(ind.joint_policy)
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

        # Get the nondominated fronts from the population
        ndfs, _, _, _ = pg.fast_non_dominated_sorting(points=[ind.fitness for ind in parent_set])

        # An empty DMO value matrix: pop_size * team_size
        dmo_values = numpy.zeros((len(parent_set), self.team_size)) # NOTE: element [i, j] is individual i, policy j
        # Process each Individual according to nondominated fronts
        for ndf in ndfs:
            # get the hypervolume for points on this ndf
            hypervol = pg.hypervolume(points=[parent_set[i].fitness for i in ndf]).compute([0.000001 for _ in range(self.num_objs)])
            for ind_idx in ndf:
                # Individual we'll be operating on
                op_ind = parent_set[ind_idx]
                # Process each policy in this individual's joint policy
                for p_idx in range(len(op_ind.joint_policy)):
                    # Trajectory with this policy's experience excluded
                    cf_traj = [op_ind.trajectory[i] for i in range(len(op_ind.joint_policy)) if i != p_idx]
                    # Fitness of counterfactual trajectory
                    cf_fitness = [-1 for _ in range(self.num_objs)]
                    cf_fitness_neg = self.interface.evaluate_trajectory(cf_traj)
                    for f in cf_fitness_neg:
                        cf_fitness[f] = -cf_fitness_neg[f] # NOTE: The fitness sign is flipped to match Pygmo convention
                    # Counterfactual ndf (list of fitnesses)
                    cf_ndf_fitnesses = [parent_set[i].fitness if i != ind_idx else cf_fitness for i in ndf]
                    # Counterfactual hypervolume with counterfactual ndf
                    cf_hypervol = pg.hypervolume(points=cf_ndf_fitnesses).compute([0.000001 for _ in range(self.num_objs)])
                    # Assign the dmo value
                    dmo_values[ind_idx][p_idx] = hypervol - cf_hypervol

        # Fill up the offspring set to the pop_size via offspring-creation
        while len(parent_set) + len(offspring_set) < self.pop_size:
            # Compound joint policies that will be used to create offsprings
            parent_jp1, parent_jp2 = [], []
            # Two parents per policy in the offspring
            for policy_idx in range(self.team_size):
                # Select 2 parents via binary tournament
                idx1, idx2 = random.sample(range(len(dmo_values)), 2) # Sample two potential jp indices
                policy_lvl_parent1 = parent_set[idx1] if dmo_values[idx1][policy_idx] > dmo_values[idx2][policy_idx] else parent_set[idx2] # Pick parent with greate DMO value for this policy_idx
                idx1, idx2 = random.sample(range(len(dmo_values)), 2) # Sample two potential jp indices
                policy_lvl_parent2 = parent_set[idx1] if dmo_values[idx1][policy_idx] > dmo_values[idx2][policy_idx] else parent_set[idx2] # Pick parent with greate DMO value for this policy_idx
                # Copy the policy at this index to the parent policies
                parent_jp1.append(policy_lvl_parent1.joint_policy[policy_idx])
                parent_jp2.append(policy_lvl_parent2.joint_policy[policy_idx])
            # Create compound parents with these compound joint policies
            parent1 = Individual.Individual(joint_policy=parent_jp1,
                                            config_filename=self.config_filename, 
                                            num_agents=self.team_size, 
                                            input_size=self.interface.get_state_size(), 
                                            output_size=self.interface.get_action_size(), 
                                            id=-1, 
                                            num_objs=self.num_objs)
            parent2 = Individual.Individual(joint_policy=parent_jp2,
                                            config_filename=self.config_filename, 
                                            num_agents=self.team_size, 
                                            input_size=self.interface.get_state_size(), 
                                            output_size=self.interface.get_action_size(), 
                                            id=-1, 
                                            num_objs=self.num_objs)
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