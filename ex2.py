import numpy as np
import random
import matplotlib.pyplot as plt

# Função para calcular a distância total de uma rota
def calculate_distance(route, distance_matrix):
    dist = 0
    for i in range(len(route) - 1):
        dist += distance_matrix[route[i]][route[i+1]]
    # Volta à cidade de origem
    dist += distance_matrix[route[-1]][route[0]]
    return dist

# Função para avaliar o fitness (quanto menor a distância, melhor)
def fitness(individual, distance_matrix):
    return calculate_distance(individual, distance_matrix)

# Função para seleção por torneio (k=3)
def tournament_selection(population, fitness_values, k=3):
    selected = random.sample(range(len(population)), k)
    selected_fitness = [fitness_values[i] for i in selected]
    winner = selected[np.argmin(selected_fitness)]
    return population[winner]

# Função para crossover de um ponto
def crossover(parent1, parent2, crossover_prob=0.9):
    if random.random() < crossover_prob:
        point = random.randint(1, len(parent1) - 2)
        child1 = parent1[:point] + [city for city in parent2 if city not in parent1[:point]]
        child2 = parent2[:point] + [city for city in parent1 if city not in parent2[:point]]
        return child1, child2
    return parent1.copy(), parent2.copy()

# Função para mutação (swap de duas cidades)
def mutate(individual, mutation_prob=0.02):
    for i in range(len(individual)):
        if random.random() < mutation_prob:
            j = random.randint(1, len(individual) - 1)  # Evitar mutar a cidade de origem
            individual[i], individual[j] = individual[j], individual[i]  # Troca de cidades
    return individual

# Função para inicializar a população (cada indivíduo é uma permutação das cidades)
def initialize_population(pop_size, n_cities):
    return [random.sample(range(1, n_cities), n_cities - 1) for _ in range(pop_size)]

# Função principal do algoritmo genético
def genetic_algorithm(distance_matrix, pop_size=50, generations=250, crossover_prob=0.9, mutation_prob=0.02):
    n_cities = len(distance_matrix)
    
    # Inicializa a população com permutações das cidades (exceto a cidade de origem 0)
    population = initialize_population(pop_size, n_cities)
    population = [[0] + individual for individual in population]  # Adiciona a cidade de origem (0)
    
    fitness_history = []
    best_fitness_history = []

    # Avaliar fitness da população inicial
    fitness_values = [fitness(individual, distance_matrix) for individual in population]
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
        fitness_values = [fitness(individual, distance_matrix) for individual in population]
        fitness_history.append(np.mean(fitness_values))
        best_fitness_history.append(min(fitness_values))

    # Encontrar o melhor indivíduo
    best_individual_idx = np.argmin(fitness_values)
    best_individual = population[best_individual_idx]
    best_distance = fitness_values[best_individual_idx]

    return fitness_history, best_fitness_history, best_individual, best_distance

# Função para gerar o gráfico de convergência
def plot_results(fitness_history, best_fitness_history):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, label='Fitness médio da população')
    plt.plot(best_fitness_history, label='Melhor fitness por geração')
    plt.title('Convergência da População')
    plt.xlabel('Gerações')
    plt.ylabel('Distância')
    plt.legend()
    plt.grid(True)
    plt.show()

# Parâmetros do problema
# Suponha uma matriz de distâncias (exemplo com 12 cidades)
distance_matrix_uk12 = np.array([
    #   0    1    2    3    4    5    6    7    8    9   10   11
    [  0,  29,  20,  21,  17,  28,  23,  16,  27,  33,  31,  36],  # 0
    [ 29,   0,  15,  19,  28,  30,  40,  16,  20,  22,  35,  26],  # 1
    [ 20,  15,   0,  21,  33,  25,  28,  39,  26,  32,  24,  29],  # 2
    [ 21,  19,  21,   0,  31,  26,  29,  21,  37,  27,  33,  19],  # 3
    [ 17,  28,  33,  31,   0,  21,  18,  24,  16,  20,  28,  27],  # 4
    [ 28,  30,  25,  26,  21,   0,  16,  20,  23,  25,  30,  22],  # 5
    [ 23,  40,  28,  29,  18,  16,   0,  33,  30,  22,  17,  21],  # 6
    [ 16,  16,  39,  21,  24,  20,  33,   0,  30,  26,  29,  32],  # 7
    [ 27,  20,  26,  37,  16,  23,  30,  30,   0,  25,  20,  23],  # 8
    [ 33,  22,  32,  27,  20,  25,  22,  26,  25,   0,  21,  19],  # 9
    [ 31,  35,  24,  33,  28,  30,  17,  29,  20,  21,   0,  15],  # 10
    [ 36,  26,  29,  19,  27,  22,  21,  32,  23,  19,  15,   0],  # 11
])

# Executar o algoritmo genético
fitness_history, best_fitness_history, best_route, best_distance = genetic_algorithm(distance_matrix_uk12)

# Resultados
print(f"Distância média da população inicial: {fitness_history[0]:.4f}")
print(f"Distância média da população final: {fitness_history[-1]:.4f}")
print(f"Melhor rota encontrada: {best_route}")
print(f"Menor distância percorrida: {best_distance:.4f} km")

# Exibir gráfico de convergência
plot_results(fitness_history, best_fitness_history)
