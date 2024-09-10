import numpy as np
import random
import matplotlib.pyplot as plt

# Função para calcular a diferença absoluta entre a soma dos dois conjuntos A e B
def fitness(individual, numbers):
    sum_A = sum(numbers[i] for i in range(len(individual)) if individual[i] == 1)
    sum_B = sum(numbers[i] for i in range(len(individual)) if individual[i] == 0)
    return abs(sum_A - sum_B)

# Função para seleção por torneio (k=3)
def tournament_selection(population, fitness_values, k=3):
    selected = random.sample(range(len(population)), k)
    selected_fitness = [fitness_values[i] for i in selected]
    winner = selected[np.argmin(selected_fitness)]
    return population[winner]

# Função para crossover de um ponto
def crossover(parent1, parent2, crossover_prob=0.9):
    if random.random() < crossover_prob:
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()

# Função para mutação
def mutate(individual, mutation_prob=0.02):
    for i in range(len(individual)):
        if random.random() < mutation_prob:
            individual[i] = 1 - individual[i]  # Inverte o bit
    return individual

# Função para inicializar a população
def initialize_population(pop_size, n):
    return [np.random.randint(0, 2, size=n) for _ in range(pop_size)]

# Função principal do algoritmo genético
def genetic_algorithm(numbers, pop_size=50, generations=150, crossover_prob=0.9, mutation_prob=0.02):
    n = len(numbers)
    population = initialize_population(pop_size, n)
    fitness_history = []
    best_fitness_history = []

    # Avaliar fitness da população inicial
    fitness_values = [fitness(individual, numbers) for individual in population]
    fitness_history.append(np.mean(fitness_values))
    best_fitness_history.append(min(fitness_values))

    # Evolução ao longo das gerações
    for generation in range(generations):
        new_population = []

        # Criar nova geração
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)

            child1, child2 = crossover(parent1, parent2, crossover_prob)

            child1 = mutate(child1, mutation_prob)
            child2 = mutate(child2, mutation_prob)

            new_population.append(child1)
            new_population.append(child2)

        # Truncar se exceder o tamanho da população
        population = new_population[:pop_size]

        # Avaliar nova população
        fitness_values = [fitness(individual, numbers) for individual in population]
        fitness_history.append(np.mean(fitness_values))
        best_fitness_history.append(min(fitness_values))

    return fitness_history, best_fitness_history

# Função para gerar o gráfico de convergência
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

# Parâmetros do problema
N = 30
numbers = np.random.randint(1, 100, size=N)

# Executar o algoritmo genético
fitness_history, best_fitness_history = genetic_algorithm(numbers)

# Resultados
print(f"Fitness médio da população inicial: {fitness_history[0]:.4f}")
print(f"Fitness médio da população final: {fitness_history[-1]:.4f}")

# Exibir gráfico de convergência
plot_results(fitness_history, best_fitness_history)
