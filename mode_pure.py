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

    expanded_u = np.repeat(u, 10).astype(float)
    expanded_u += penaltyN
    expanded_u += penaltyT

    integrated_u = trapz(expanded_u, t_span)

    return integrated_u

# Define the MODE class
class MODE:
    def __init__(self, num_individuals, num_dimensions, search_space, max_generations, scaling_factor, crossover_prob, init_cond):
        self.num_individuals = num_individuals
        self.num_dimensions = num_dimensions
        self.search_space = search_space
        self.max_generations = max_generations
        self.scaling_factor = scaling_factor
        self.crossover_prob = crossover_prob
        self.init_cond = init_cond
        self.population = []
        self.archive = []
        self.distances = []
        self.lossT = []
        self.lossu = []
        self.all_archive = []

    def initialize_population(self, num_individuals):
        u_possible = [0, 0.1]
        for _ in range(num_individuals):
            individual = {
                # 'position': pos1,
                'position': [u_possible[round(random.uniform(self.search_space[0], self.search_space[1]))] for _ in range(self.num_dimensions)],
                'objective_values': None,
                'states': None
            }
            self.population.append(individual)

    def evaluate_objective_functions(self, individual):
        x = individual['position']
        objective1, states = objective_function1(x, self.init_cond)
        objective2 = objective_function2(x, states)
        return [objective1, objective2], states

    def is_dominating(self, individual1, individual2):
        objective_values1 = individual1['objective_values']
        objective_values2 = individual2['objective_values']
        return all(obj1 <= obj2 for obj1, obj2 in zip(objective_values1, objective_values2)) and any(obj1 < obj2 for obj1, obj2 in zip(objective_values1, objective_values2))

    def generate_mutant_vector(self, donor_indices, scaling_factor):
        donor_vectors = [self.population[i]['position'] for i in donor_indices]
        base_vector = donor_vectors[0]
        differential_vector = np.subtract(donor_vectors[1], donor_vectors[2])
        mutant_vector = base_vector + scaling_factor * differential_vector
        mutant_vector = [min(max(p, self.search_space[0]), self.search_space[1]) for p in mutant_vector]
        # mutant_vector = [0 if p<0.05 else 0.1 for p in mutant_vector]
        return mutant_vector

    def perform_crossover(self, target_vector, mutant_vector, crossover_rate):
        crossover_mask = [random.random() < crossover_rate for _ in range(self.num_dimensions)]
        trial_vector = [mutant_vector[i] if crossover_mask[i] else target_vector[i] for i in range(self.num_dimensions)]

        return trial_vector

    def select_next_generation(self, target_index, trial_vector):
        target_individual = self.population[target_index]        
        objectives, states = self.evaluate_objective_functions({'position':trial_vector})
        trial_individual = {'position': trial_vector, 'objective_values': objectives, 'states': states}


        if self.is_dominating(trial_individual, target_individual):
            self.population[target_index] = trial_individual
        elif not self.is_dominating(target_individual, trial_individual):
            tournament = [target_individual, trial_individual]
            selected_individual = random.choice(tournament)
            self.population[target_index] = selected_individual


    def update_archive(self):
        new_archive = []
        for particle in self.population:
            if not any(all(particle['objective_values'][i] > p['objective_values'][i] for i in range(len(p['objective_values']))) for p in self.archive):
                new_archive.append(particle)
        self.archive = new_archive

    def plot_pareto_front(self):
        objective1_values = [individual['objective_values'][0] for individual in self.archive]
        objective2_values = [individual['objective_values'][1] for individual in self.archive]

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

    def init_archive(self):
        for individual in self.population:
            if not any(all(individual['objective_values'][i] > p['objective_values'][i] for i in range(2)) for p in self.population):
                self.archive.append(individual)


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

    def run(self):
        self.initialize_population(self.num_individuals)  


        for i, individual in enumerate(self.population):
            individual['objective_values'] = self.evaluate_objective_functions(individual)[0]
        self.init_archive()

        for individual in self.archive:
            self.all_archive.append(individual)
      

        for _ in range(self.max_generations):
            for target_index, target_individual in enumerate(self.population):
                self.population[target_index]['objective_values'] = self.evaluate_objective_functions(target_individual)[0]
                self.population[target_index]['states'] = self.evaluate_objective_functions(target_individual)[1]

                donor_indices = random.sample(range(self.num_individuals), 3)
                mutant_vector = self.generate_mutant_vector(donor_indices, self.scaling_factor)
                trial_vector = self.perform_crossover(target_individual['position'], mutant_vector, self.crossover_prob)
                self.select_next_generation(target_index, trial_vector)
            
            self.update_archive()
            for individual in self.archive:
                self.all_archive.append(individual) 

            # Closest distance
            objective_values = [particle['objective_values'] for particle in self.archive]


            distances = cdist(objective_values, np.array([[0.0, 0.0]]), metric='euclidean')
            closest_index = np.argmin(distances)
            closest_point = objective_values[closest_index]
            distance_to_origin = np.linalg.norm(closest_point)

            self.distances.append(distance_to_origin)

            best_individual = self.archive[closest_index]

            self.lossT.append(best_individual['objective_values'][0])
            self.lossu.append(best_individual['objective_values'][1])

            # # Diversity Maintenance
            # threshold = 0.1
            # crowding_distances = self.calculate_crowding_distances(self.population)
            # sorted_front = [x for _, x in sorted(zip(crowding_distances, self.population), key=lambda pair: pair[0], reverse=True)]
            # self.population = []
            # self.population = sorted_front[:int((1-threshold)*self.num_individuals)]
            # self.initialize_population(int(0.1*self.num_individuals))


# Usage example
num_individuals = 150
num_dimensions = 15
search_space = (0, 1)
max_generations = 30
scaling_factor = 0.2
crossover_prob = 0.2

N0 = 0.9
T0 = 0.25
I0 = 0.25

init_cond = [N0, T0, I0]

mode = MODE(num_individuals, num_dimensions, search_space, max_generations, scaling_factor, crossover_prob, init_cond)
start = time.time()
mode.run()
end = time.time()

mode.plot_pareto_front()

# # Print the non-dominated solutions in the archive
# print("Archive Solutions:")
# for individual in mode.population:
#     print(individual['position'], "Objective Values:", individual['objective_values'])

objective_values = [particle['objective_values'] for particle in mode.archive]

distances = cdist(objective_values, np.array([[0.0, 0.0]]), metric='euclidean')
closest_index = np.argmin(distances)
closest_point = objective_values[closest_index]
distance_to_origin = np.linalg.norm(closest_point)

print("Closest Point:", closest_point)
print("Distance to Origin:", distance_to_origin)

best_individual = mode.archive[closest_index]
print('Best individual', best_individual)

u = np.repeat(best_individual['position'], 10)
t = np.linspace(0,149,150)

plt.plot(t,u)
plt.xlabel('vreme (u danima)')
plt.ylabel('u')
plt.title('Upravljanje')
plt.grid(True)
plt.show()

print(time_passed(start, end))

export = mode.plot_states(best_individual['states'])

 
with open('states_mode.csv', 'w') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)
     
    write.writerow(export)

hv = hypervolume(objective_values)

reference_point = np.array([10.0, 5.0])
volume = hv.compute(reference_point)
print('HV: ', volume)
