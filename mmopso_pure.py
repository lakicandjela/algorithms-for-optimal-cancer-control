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

    # sol = solve_ivp(lambda t, y: df.system(t, y, df.input_func(u, t)), [0, 149], init_cond, method='RK45', t_eval=np.arange(0, 150, 10))

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

# Define the MMOPSO class
class MMOPSO:
    def __init__(self, num_particles, num_dimensions, search_space, max_iterations, init_cond):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.search_space = search_space
        self.max_iterations = max_iterations
        self.init_cond = init_cond
        self.particles = []
        self.global_best = {'position': None, 'objective_values': None, 'states': None}
        self.pareto_front = []
        self.lossT = []
        self.lossu = []
        self.distances = []

    def initialize_particles(self, num_particles):
        u_possible = [0, 0.1]
        for _ in range(num_particles):
            particle = {
                'position': [u_possible[round(random.uniform(self.search_space[0], self.search_space[1]))] for _ in range(self.num_dimensions)],
                'velocity': [0.0] * self.num_dimensions,
                'personal_best_position': None,
                'personal_best_objective_values': None,
                'personal_best_states': None
            }
            self.particles.append(particle)

    def evaluate_objective_functions(self, particle):
        x = particle['position']
        objective1, states = objective_function1(x, self.init_cond)
        objective2 = objective_function2(x, states)
        return [objective1, objective2], states

    def is_dominating(self, y, y_prime):
        return all(obj1 <= obj2 for obj1, obj2 in zip(y, y_prime)) and any(obj1 < obj2 for obj1, obj2 in zip(y, y_prime))

    def update_personal_best(self, particle):
        # if particle['personal_best_position'] is None or all(particle['personal_best_objective_values'][i] > particle['objective_values'][i] for i in range(len(particle['objective_values']))):
        if particle['personal_best_position'] is None or self.is_dominating(particle['objective_values'], particle['personal_best_objective_values']):
            particle['personal_best_position'] = particle['position']
            particle['personal_best_objective_values'] = particle['objective_values']
            particle['personal_best_states'] = particle['states']

    def update_global_best(self):
        for particle in self.particles:
            # if self.global_best['position'] is None or all(particle['personal_best_objective_values'][i] < self.global_best['objective_values'][i] for i in range(len(self.global_best['objective_values']))):
            if self.global_best['position'] is None or self.is_dominating(particle['personal_best_objective_values'], self.global_best['objective_values']):
                self.global_best['position'] = particle['personal_best_position']
                self.global_best['objective_values'] = particle['personal_best_objective_values']
                self.global_best['states'] = particle['personal_best_states']
        self.lossT.append(self.global_best['objective_values'][0])
        self.lossu.append(self.global_best['objective_values'][1])

    def update_pareto_front(self):
        self.pareto_front = []
        for particle in self.particles:
            if not any(all(particle['personal_best_objective_values'][i] > p['personal_best_objective_values'][i] for i in range(len(p['personal_best_objective_values']))) for p in self.pareto_front):
                self.pareto_front.append(particle)

    def update_particle_velocity(self, particle, inertia_weight, cognitive_weight, social_weight):
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        cognitive_term = cognitive_weight * r1 * (np.array(particle['personal_best_position']) - np.array(particle['position']))
        social_term = social_weight * r2 * (np.array(self.global_best['position']) - np.array(particle['position']))
        particle['velocity'] = list(inertia_weight * np.array(particle['velocity']) + cognitive_term + social_term)

    def update_particle_position(self, particle):
        new_position = [p + v for p, v in zip(particle['position'], particle['velocity'])]
        new_position = [min(max(p, self.search_space[0]), self.search_space[1]) for p in new_position]
        particle['position'] = new_position

    def calculate_crowding_distances(self, particles):
        num_objectives = 2
        num_solutions = self.num_particles

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

    def run(self):
        self.initialize_particles(self.num_particles)
        for _ in range(self.max_iterations):
            for particle in self.particles:
                particle['objective_values'], particle['states']  = self.evaluate_objective_functions(particle)
                self.update_personal_best(particle)

            self.update_global_best()
            self.update_pareto_front()

            inertia_weight = 0.5
            cognitive_weight = 1.0
            social_weight = 1.0

            for particle in self.particles:
                self.update_particle_velocity(particle, inertia_weight, cognitive_weight, social_weight)
                self.update_particle_position(particle)

            self.update_pareto_front()  # Re-compute pareto front to consider new solutions

            # Diversity Maintenance
            threshold = 0.1
            crowding_distances = self.calculate_crowding_distances(self.particles)
            sorted_front = [x for _, x in sorted(zip(crowding_distances, self.particles), key=lambda pair: pair[0], reverse=True)]
            self.particles = []
            self.particles = sorted_front[:int((1-threshold)*self.num_particles)]
            self.initialize_particles(int(0.1*self.num_particles))

            objective_values = [particle['personal_best_objective_values'] for particle in self.pareto_front]


            distances = cdist(objective_values, np.array([[0.0, 0.0]]), metric='euclidean')
            closest_index = np.argmin(distances)
            closest_point = objective_values[closest_index]
            distance_to_origin = np.linalg.norm(closest_point)

            self.distances.append(distance_to_origin)

        print(self.global_best['objective_values'], self.global_best['position'])

    def plot_pareto_front(self):
        objective1_values = [particle['personal_best_objective_values'][0] for particle in self.pareto_front]
        objective2_values = [particle['personal_best_objective_values'][1] for particle in self.pareto_front]

        plt.scatter(objective1_values, objective2_values)
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Pareto Front')
        plt.show()

    def plot_states(self):


        plt.plot(self.distances)
        plt.xlabel('Generacija')
        plt.ylabel('Najbliža udaljenost od izvora')
        plt.title('Udaljenosti')
        plt.grid(True)
        plt.show()

        plt.figure()

        plt.plot(self.global_best['states'][1],label='Tumorske ćelije')
        plt.plot(self.global_best['states'][0], label='Normalne ćelije')
        plt.plot(self.global_best['states'][2], label='Imune ćelije')
        plt.xlabel('vreme (u danima)')
        plt.title('Stanje u organizmu')
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure()

        plt.plot(self.lossT,label='Loss T')
        plt.plot(self.lossu, label='Loss u')
        plt.xlabel('Generacija')
        plt.grid(True)
        plt.legend()
        plt.show()

        export = [self.distances, list(self.global_best['states'][1]), list(self.global_best['states'][0]), list(self.global_best['states'][2]), self.lossT, self.lossu]

        return export

# Usage example
N0 = 0.9
T0 = 0.25
I0 = 0.25

num_particles = 150
num_dimensions = 15
search_space = (0, 1)
max_iterations = 100
init_cond =[N0, T0, I0]

mmopso = MMOPSO(num_particles, num_dimensions, search_space, max_iterations, init_cond)
start = time.time()
mmopso.run()
end = time.time()

mmopso.plot_pareto_front()


objective_values = [particle['personal_best_objective_values'] for particle in mmopso.pareto_front]

distances = cdist(objective_values, np.array([[0.0, 0.0]]), metric='euclidean')
closest_index = np.argmin(distances)
closest_point = objective_values[closest_index]
distance_to_origin = np.linalg.norm(closest_point)

print("Closest Point:", closest_point)
print("Distance to Origin:", distance_to_origin)

u = np.repeat(mmopso.global_best['position'], 10)
t = np.linspace(0,149,150)

plt.plot(t,u)
plt.xlabel('vreme (u danima)')
plt.ylabel('u')
plt.title('Upravljanje')
plt.grid(True)
plt.show()

print(time_passed(start, end))

export = mmopso.plot_states()

with open('states_mmopso.csv', 'w') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)
     
    write.writerow(export)

hv = hypervolume(objective_values)

reference_point = np.array([10.0, 5.0])
volume = hv.compute(reference_point)
print('HV: ', volume)
