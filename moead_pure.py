import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz, trapz
import differential as df
from pygmo import hypervolume
from scipy.spatial.distance import cdist
import time
import csv

def time_passed(start, end):
  elapsed_time = end - start

  hours = int(elapsed_time / 3600)
  minutes = int((elapsed_time % 3600) / 60)
  seconds = int((elapsed_time % 3600) % 60)

  print(f"Vrijeme za obuku: {hours} sati, {minutes} minuta, {seconds} sekundi")


t_span = np.linspace(0, 149, num=150)

def objective_function1(u, init_cond):

    sol = solve_ivp(lambda t, y: df.system(t, y, df.input_func(u, t)), [0, 149], init_cond, method='RK45', t_eval=np.linspace(0, 149, num=150))

    T = sol.y[1]
    N = sol.y[0]

    penalty_factor = 100  
    penalty = penalty_factor * np.maximum(np.zeros(len(N)), 0.75 - N)  
    T += penalty

    penalty_factor = 100  
    penalty = penalty_factor * np.maximum(np.zeros(len(N)), -T)  
    T += penalty

    integrated_T = trapz(T, np.linspace(0, 149, num=150))

    return integrated_T, sol.y

def objective_function2(u, states):
    T = states[1]
    N = states[0]
    penalty_factor = 100  
    penaltyN = penalty_factor * np.maximum(np.zeros(len(N)), 0.75 - N)  

    penaltyT = penalty_factor * np.maximum(np.zeros(len(N)), -T)  

    expanded_u = np.repeat(u, 10)
    expanded_u += penaltyN
    expanded_u += penaltyT

    integrated_u = trapz(expanded_u, t_span)

    return integrated_u

# Define the MOEA/D class
class MOEAD:
    def __init__(self, num_individuals, num_dimensions, num_objectives, num_neighbors, search_space, max_generations, init_cond):
        self.num_individuals = num_individuals
        self.num_dimensions = num_dimensions
        self.num_objectives = num_objectives
        self.num_neighbors = num_neighbors
        self.search_space = search_space
        self.max_generations = max_generations
        self.init_cond = init_cond
        self.population = []
        self.weights = self.generate_weights()
        self.z_ref = [500, 500]
        self.external_archive = []
        self.all_archive = []
        self.lossT = []
        self.lossu = []
        self.distances = []


    def generate_weights(self):
        weights = np.random.uniform(0, 1, (self.num_individuals, self.num_objectives))
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        return weights

    def initialize_population(self, num_individuals):
        u_possible = [0, 0.1]
        for _ in range(num_individuals):
            individual = {
                'position': [u_possible[round(random.uniform(self.search_space[0], self.search_space[1]))] for _ in range(self.num_dimensions)],
                'objective_values': None,
                'neighbors': None,
                'states': None
            }
            self.population.append(individual)

    def evaluate_objective_functions(self, individual):
        x = individual['position']
        objective1, states = objective_function1(x, self.init_cond)
        objective2 = objective_function2(x, states)
        return [objective1, objective2], states

    def update_neighbors(self):
        for i in range(self.num_individuals):
            distances = [np.linalg.norm(self.weights[i] - self.weights[j]) for j in range(self.num_individuals)]
            sorted_indices = sorted(range(self.num_individuals), key=lambda k: distances[k])
            individual_neighbors = sorted_indices[1:self.num_neighbors + 1]
            self.population[i]['neighbors'] = individual_neighbors

    def generate_offspring(self, individual):
        donor_indices = random.sample(individual['neighbors'], 3)
        scaling_factor = 0.5
        donor_vectors = [self.population[i]['position'] for i in donor_indices]
        base_vector = donor_vectors[0]
        differential_vector = np.subtract(donor_vectors[1], donor_vectors[2])
        mutant_vector = base_vector + scaling_factor * differential_vector
        mutant_vector = [min(max(p, self.search_space[0]), self.search_space[1]) for p in mutant_vector]
        # mutant_vector = [0 if p<0.05 else 0.1 for p in mutant_vector]

        offspring_vector = self.perform_crossover(individual['position'], mutant_vector, 0.2)
        offspring = {'position':offspring_vector}
        return offspring
    
    def perform_crossover(self, target_vector, mutant_vector, crossover_rate):
        crossover_mask = [random.random() < crossover_rate for _ in range(self.num_dimensions)]
        trial_vector = [mutant_vector[i] if crossover_mask[i] else target_vector[i] for i in range(self.num_dimensions)]

        return trial_vector


    def update_population(self, offspring):
        for i, individual in enumerate(self.population):
            if all(offspring['objective_values'][j] < individual['objective_values'][j] for j in range(self.num_objectives)):
                self.population[i] = offspring

    def calculate_aggregation(self, objective_values, weight_vector):
        g_values = []
        for m in range(self.num_objectives):
            g_values.append(weight_vector[m] * np.abs(objective_values[m] - self.z_ref[m]))
        return max(g_values)

    
    def is_dominating(self, y, y_prime):
        return all(obj1 <= obj2 for obj1, obj2 in zip(y, y_prime)) and any(obj1 < obj2 for obj1, obj2 in zip(y, y_prime))
    
    def init_archive(self):
        for individual in self.population:
            if not any(all(individual['objective_values'][i] > p['objective_values'][i] for i in range(self.num_objectives)) for p in self.population):
                self.external_archive.append(individual)
    
    def plot_pareto_front(self):
        objective1_values = [individual['objective_values'][0] for individual in self.external_archive]
        objective2_values = [individual['objective_values'][1] for individual in self.external_archive]

        plt.scatter(objective1_values, objective2_values)
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Pareto Front')
        plt.show()

        objective1_values = [individual['objective_values'][0] for individual in self.all_archive]
        objective2_values = [individual['objective_values'][1] for individual in self.all_archive]

        plt.scatter(objective1_values, objective2_values)
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Sve arhivirane jedinke na pareto frontu')
        plt.show()

    def plot_states(self, states):

        plt.plot(self.distances)
        plt.xlabel('Generacija')
        plt.ylabel('Najbliža udaljenost od izvora')
        plt.title('Udaljenosti')
        plt.grid(True)
        plt.show()

        plt.figure()

        plt.plot(states[1],label='Tumorske ćelije')
        plt.plot(states[0], label='Normalne ćelije')
        plt.plot(states[2], label='Imune ćelije')
        plt.xlabel('vreme (u danima)')
        plt.title('Stanje u organizmu')
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure()

        plt.plot(self.lossT, label='Loss T')
        plt.plot(self.lossu, label='Loss u')
        plt.xlabel('Generacija')
        plt.grid(True)
        plt.legend()
        plt.show()

        export = [self.distances, list(states[1]), list(states[0]), list(states[2]), self.lossT, self.lossu]

        return export


    def calculate_crowding_distances(self, particles):
        num_objectives = 2
        num_solutions = self.num_individuals

        crowding_distances = [0.0] * num_solutions

        for objective_index in range(num_objectives):
            sorted_front = sorted(particles, key=lambda x: x['objective_values'][objective_index])
            crowding_distances[0] = crowding_distances[num_solutions - 1] = float('inf')

            objective_min = sorted_front[0]['objective_values'][objective_index]
            objective_max = sorted_front[num_solutions - 1]['objective_values'][objective_index]

            if objective_max == objective_min:
                continue

            for i in range(1, num_solutions - 1):
                crowding_distances[i] += (sorted_front[i + 1]['objective_values'][objective_index] -
                                        sorted_front[i - 1]['objective_values'][objective_index]) / (objective_max - objective_min)

        return crowding_distances
    
    def update_archive(self):
        new_archive = []
        for particle in self.population:
            if not any(all(particle['objective_values'][i] > p['objective_values'][i] for i in range(len(p['objective_values']))) for p in self.external_archive):
                new_archive.append(particle)
        if new_archive==[]:
            self.external_archive = self.external_archive
        else:
            self.external_archive = new_archive

    def run(self):
        self.initialize_population(self.num_individuals)
        self.update_neighbors()

        for i, individual in enumerate(self.population):
            individual['objective_values'], individual['states'] = self.evaluate_objective_functions(individual)
        self.init_archive()


        for _ in range(self.max_generations):
            for i, individual in enumerate(self.population):
                # print(self.external_archive)
                offspring = self.generate_offspring(individual)
                offspring['objective_values'], states = self.evaluate_objective_functions(offspring)
                offspring['states'] = states

                # Update reference point set
                for m in range(self.num_objectives):
                    if offspring['objective_values'][m] <= self.z_ref[m]:
                        self.z_ref[m] = offspring['objective_values'][m]

                # Update neighborhood
                for k in individual['neighbors']:
                    g_child = self.calculate_aggregation(offspring['objective_values'], self.weights[k])
                    g_parent = self.calculate_aggregation(individual['objective_values'], self.weights[k])
                    if g_child <= g_parent:
                        self.population[k]['position'] = offspring['position']
                        self.population[k]['objective_values'] = offspring['objective_values']
                        self.population[k]['states'] = offspring['states']

            self.update_archive()
            for individual in self.external_archive:
                self.all_archive.append(individual) 


            # Closest distance
            objective_values = [particle['objective_values'] for particle in self.external_archive]

            distances = cdist(objective_values, np.array([[0.0, 0.0]]), metric='euclidean')

            closest_index = np.argmin(distances)
            closest_point = objective_values[closest_index]
            distance_to_origin = np.linalg.norm(closest_point)

            self.distances.append(distance_to_origin)

            best_individual = self.external_archive[closest_index]

            self.lossT.append(best_individual['objective_values'][0])
            self.lossu.append(best_individual['objective_values'][1])

            threshold = 0.1
            crowding_distances = self.calculate_crowding_distances(self.population)
            sorted_front = [x for _, x in sorted(zip(crowding_distances, self.population), key=lambda pair: pair[0], reverse=True)]
            self.population = []
            self.population = sorted_front[:int((1-threshold)*self.num_individuals)]
            self.initialize_population(int(0.1*self.num_individuals))
            for i, individual in enumerate(self.population):
                individual['objective_values'], individual['states'] = self.evaluate_objective_functions(individual)
            self.update_neighbors()




# Usage example
num_individuals = 150
num_dimensions = 15
num_objectives = 2
num_neighbors = 10
search_space = (0, 1)
max_generations = 100

N0 = 0.9
T0 = 0.25
I0 = 0.25

init_cond = [N0, T0, I0]


moead = MOEAD(num_individuals, num_dimensions,num_objectives, num_neighbors, search_space, max_generations, init_cond)
start = time.time()
moead.run()
end = time.time()

moead.plot_pareto_front()

# Print the non-dominated solutions in the final population
print("Non-dominated Solutions:")
for individual in moead.external_archive:
    print(individual['position'], "Objective Values:", individual['objective_values'])


objective_values = [particle['objective_values'] for particle in moead.external_archive]

distances = cdist(objective_values, np.array([[0.0, 0.0]]), metric='euclidean')
closest_index = np.argmin(distances)
closest_point = objective_values[closest_index]
distance_to_origin = np.linalg.norm(closest_point)

print("Closest Point:", closest_point)
print("Distance to Origin:", distance_to_origin)

best_individual = moead.external_archive[closest_index]
print('Best individual', best_individual['position'], best_individual['objective_values'])

u = np.repeat(best_individual['position'], 10)
t = np.linspace(0,149,150)

plt.plot(t,u)
plt.xlabel('vreme (u danima)')
plt.ylabel('u')
plt.title('Upravljanje')
plt.grid(True)
plt.show()

print(time_passed(start, end))

export = moead.plot_states(best_individual['states'])

 
with open('states_moead.csv', 'w') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)
     
    write.writerow(export)

hv = hypervolume(objective_values)

reference_point = np.array([10.0, 5.0])
volume = hv.compute(reference_point)
print('HV: ', volume)