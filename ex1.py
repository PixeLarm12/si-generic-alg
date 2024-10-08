import numpy as np
import random
import matplotlib.pyplot as plt

def fitness(individual, numbers):
    sum_A = sum(numbers[i] for i in range(len(individual)) if individual[i] == 1)
    sum_B = sum(numbers[i] for i in range(len(individual)) if individual[i] == 0)
    return abs(sum_A - sum_B)

def tournament_selection(population, fitness_values, k=3):
    selected = random.sample(range(len(population)), k)
    selected_fitness = [fitness_values[i] for i in selected]
    winner = selected[np.argmin(selected_fitness)]
    return population[winner]

def crossover(parent1, parent2, crossover_prob=0.9):
    if random.random() < crossover_prob:
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(individual, mutation_prob=0.02):
    for i in range(len(individual)):
        if random.random() < mutation_prob:
            individual[i] = 1 - individual[i]
    return individual

def initialize_population(pop_size, n):
    return [np.random.randint(0, 2, size=n) for _ in range(pop_size)]

def genetic_algorithm(numbers, pop_size=50, generations=150, crossover_prob=0.9, mutation_prob=0.02):
    n = len(numbers)
    population = initialize_population(pop_size, n)
    fitness_history = []
    best_fitness_history = []

    fitness_values = [fitness(individual, numbers) for individual in population]
    fitness_history.append(np.mean(fitness_values))
    best_fitness_history.append(min(fitness_values))

    for generation in range(generations):
        new_population = []

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)

            child1, child2 = crossover(parent1, parent2, crossover_prob)

            child1 = mutate(child1, mutation_prob)
            child2 = mutate(child2, mutation_prob)

            new_population.append(child1)
            new_population.append(child2)

        population = new_population[:pop_size]

        fitness_values = [fitness(individual, numbers) for individual in population]
        fitness_history.append(np.mean(fitness_values))
        best_fitness_history.append(min(fitness_values))

    return fitness_history, best_fitness_history

def plot_results(fitness_history, best_fitness_history):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, label='Fitness médio da população')
    plt.plot(best_fitness_history, label='Melhor fitness por geração')
    plt.title('Convergência da População')
    plt.xlabel('Gerações')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()

N = 30
numbers = np.random.randint(1, 100, size=N)

fitness_history, best_fitness_history = genetic_algorithm(numbers)

print(f"Fitness médio da população inicial: {fitness_history[0]:.4f}")
print(f"Fitness médio da população final: {fitness_history[-1]:.4f}")

plot_results(fitness_history, best_fitness_history)
