import matplotlib.pyplot as plt
import numpy as np
import time
import csv

def readCSV(path):
    """
        Function that reads a given csv file and transforms it into a 2D-Matrix of names and affinities.
    """
    with open(path, newline='') as csvfile:
        readFile = csv.reader(csvfile, delimiter=';', quotechar='|')
        next(readFile), next(readFile)

        matrix     = [[value for value in row[:]] for row in readFile]         
        names      = [[str  (value) for value in row[:1]] for row in matrix]
        affinities = [[float(value) for value in row[1:]] for row in matrix]

        return names, affinities 

names, affinities = readCSV('Dataset/RondeTafel.csv')

# Arthur[0], Lancelot[1], Gawain[2], Geraint[3], Percival[4], Bors the Younger[5], Lamorak[6], Kay Sir Gareth[7], Bedivere[8], Gaheris[9], Galahad[10], Tristan[11]
knights = [0,1,2,3,4,5,6,7,8,9,10,11] #Genotype Template

class individual:
    def __init__(self, genotype=None):
        """
            Initializes an object of the class with a random(shuffled) genotype if None specified
        """
        if genotype == None:
            self.genotype = knights.copy() 
            np.random.shuffle(self.genotype)
        else:
            self.genotype = genotype

    def affinityCalc(self, knightA, knightB): 
        """
            Function that supports the fitness function that has been split-off for a clearer view.
        """
        return affinities[knightA][knightB] * affinities[knightB][knightA]

    def fitness(self):
        """
            Function that calculates the fitness of each individual based on the algorithm given on canvas.

            Returns fitness of an individual.
        """
        affinitySum = 0

        for knightIndex in range(len(self.genotype)):
            nextKnightIndex = (knightIndex + 1) % len(self.genotype) 

            affinitySum += self.affinityCalc(self.genotype[knightIndex], self.genotype[nextKnightIndex])

        return affinitySum
    
class evolutionaryAlgorithm: 
    def __init__(self, populationSize, mutationPercentage = 1, retain=0.20, randomSelect=0.05):
        """
            Constructor that initializes a population of individuals and the rest of the required parameters.
        """
        self.population = [individual() for _ in range(populationSize)]
        self.populationSize = populationSize
        self.retain = retain
        self.randomSelect = randomSelect
        self.mutationPercentage = mutationPercentage

        self.rankPopulation()
    
    def rankPopulation(self):
        """
            Function that sorts the population from Strongest to Weakest.
        """
        self.population = sorted(self.population, key=lambda x: x.fitness(), reverse=True)
    
    def crossover(self, parentA, parentB): # Order Based Crossover
        """
            Function that performs 'Order-Based Crossover' with given parents and also checks for childMutation.

            returns a new child.
        """
        n = len(parentA.genotype)
        start, end = sorted(np.random.choice(n, 2, replace=False))

        childGenotype = [-1] * n
        childGenotype[start:end + 1] = parentA.genotype[start:end + 1]

        remaining_genes = [gene for gene in parentB.genotype if gene not in childGenotype]
        remaining_index = 0

        for i in range(n):
            if childGenotype[i] == -1:
                childGenotype[i] = remaining_genes[remaining_index]
                remaining_index += 1

        if np.random.randint(0,100) < self.mutationPercentage: # Check if the child has mutated by chance.
            childGenotype = self.mutate(childGenotype)

        return individual(genotype=childGenotype)
    
    def mutate(self, genotype):
        """
            Function that performs 'Swap Mutation' to a given individuals genotype.

            Returns the mutated genotype.
        """
        n = len(genotype)

        pointA, pointB = np.random.choice(n, 2, replace=False)

        # Swap the two knights
        genotype[pointA], genotype[pointB] = genotype[pointB], genotype[pointA]
        return genotype

    def evolve(self):
        """
            Function that evolves the current population in-to a new generation by firstly performing a elitist selection strategy to find the parents for the new generation.
            Then the function performs the pre-defined recombination operators with the chosen parents until the new generation has reached the wanted population size. 
        """
        # pre-defined amount of best individuals
        retainedEliteParentsCount  = int(len(self.population) * self.retain)
        # pre-defined amount of random individuals from population that are not the elite
        retainedRandomParentsCount = int(len(self.population) * self.randomSelect)

        # Select pre-defined amount of best individuals from population to be the parents for the new population.
        chosenParentsPopulation = self.population[0:retainedEliteParentsCount] 
        # Select pre-defined amount of random individuals from population that are not the elite to be the parents for the new population.
        chosenParentsPopulation.extend(np.random.choice(self.population[retainedEliteParentsCount:len(self.population)-1], retainedRandomParentsCount, replace=False)) 

        self.population.clear() # Empty old population to make way for new generation.
        self.population.extend(chosenParentsPopulation) # Add chosen parents to new generation.

        while len(self.population) < self.populationSize:
            parentA, parentB = sorted(np.random.choice(len(chosenParentsPopulation), 2, replace=False))
            self.population.extend([self.crossover(chosenParentsPopulation[parentA], chosenParentsPopulation[parentB])])

    def train(self, epochs):
        """
            Function that trains x(epochs) amount(s) of new generations and also keeps a list of the best individual at each epoch. 

            return: list of best individual at each epoch.   
        """
        best_individuals = [self.population[0]]  # To store best_individuals at each epoch

        for _ in range(epochs):
            self.evolve()
            self.rankPopulation()
            best_individuals.append(self.population[0])

        return best_individuals

def winningOrderPrint(genotype):
    """
        Function that prints the sum of affinities of two knights of winning order starting from Arthur.
    """
    print_list = []

    startIndex = genotype.index(0)
    for index in range(len(genotype)):
        knightIndex     = (startIndex  + index) % len(genotype)
        nextKnightIndex = (knightIndex + 1    ) % len(genotype)

        knightA_B = affinities[genotype[knightIndex]][genotype[nextKnightIndex]]
        knightB_A = affinities[genotype[nextKnightIndex]][genotype[knightIndex]]

        print_list.append(f"{names[genotype[knightIndex]]}, ({knightA_B} x {knightB_A}) = {round(knightA_B * knightB_A, 2)}")

    print(f"\n{print_list}")

if __name__ == "__main__":
    np.random.seed(0)

    # Create Population
    population = evolutionaryAlgorithm(populationSize=100, mutationPercentage=2)
    
    # Train Population
    startTime = time.perf_counter()
    best_individuals = population.train(epochs=10)
    print(f"\nTraining took {(time.perf_counter() - startTime):0.2f} seconds\n")

    # Log strongest individuals per generation
    for generation, best_individual in enumerate(best_individuals):
        print(f"Generation {generation}: Best Fitness = {round(best_individual.fitness(), 2)}, Best Genotype = {best_individual.genotype}")

    winningOrderPrint(best_individuals[len(best_individuals) - 1].genotype)

    fitness = [best_individual.fitness() for best_individual in best_individuals]
    # Plotting the fitness values against epochs
    plt.plot(range(0, len(fitness)), fitness, marker='o')
    plt.title('Best fitness at each Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Fitness Value')
    plt.show()