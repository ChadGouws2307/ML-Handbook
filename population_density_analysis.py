# Bringing it all Together

# @ Author: Chad Gouws
# Date: 02/04/2019


# Import virtual environment libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Define function to calculate density
def calculate_density(population, area):

    # Convert to Numpy Arrays
    population = np.array(population)
    area = np.array(area)

    # Perform density calculation
    density = list(np.round(population/area*1000000, 2))

    return density


# Define function to arrange data
def arrange_data(feature1, feature2):

    # Call calculate_density function
    density = calculate_density(feature1, feature2)

    # Sort data in ascending and descending order
    ascending = sorted(density)
    descending = sorted(feature2, reverse=True)

    return descending, ascending


# Define function to perform descriptive statistics
def descriptive_stats(subject):

    # Perform functions and methods related to descriptive stats
    biggest = max(subject)
    smallest = min(subject)
    mean = np.round(np.mean(subject), 2)
    median = np.median(subject)
    std_dev = np.round(np.std(subject), 2)

    # Check for data skewness
    pearson_sk = 3*(mean - median)/std_dev

    if pearson_sk > 0:
        skewness = 'POSITIVELY SKEWED'

    else:
        skewness = 'NEGATIVELY SKEWED'

    # Set results in a list
    stats = [smallest, mean, median, biggest, std_dev, skewness]

    # Return stats list to function call
    return stats


def create_graph(area, population):

    # Create histogram
    plt.hist(population, bins=10)
    plt.title('Histogram of Population Sizes in Southern Africa')
    plt.xlabel('Population [millions]')
    plt.show()

    # Create scatter plot
    np_population = np.array(population)
    np_area = np.array(area)
    density = np_population/np_area*1000000

    sc = plt.scatter(area, population, s=120, c=density, vmin=0, vmax=500, cmap='gist_rainbow', alpha=0.8)
    plt.colorbar(sc)
    plt.xlabel('Area [km^2]')
    plt.ylabel('Population [millions]')
    plt.title('Population Density Southern Africa')
    plt.show()


# ----------------------------------------------------------------------------------
# Perform Process (read CSV, call functions, show results)

# Read text file using Pandas
df = pd.read_csv('southern_africa.txt', sep=',', index_col=0)
print(df)

# Convert DataFrame data to list
country_area = list(df['area'])
country_population = list(df['population'])

# Arrange data in ascending or descending order
descending_area, ascending_density = arrange_data(country_population, country_area)
print('\nDescending Area: ', descending_area)
print('Ascending Density: ', ascending_density)

# Perform descriptive stats and check for skewness
area_stats = descriptive_stats(country_area)
print('\nDescriptive Statistics of Area: ', area_stats)
population_stats = descriptive_stats(country_population)
print('\nArea data is', area_stats[-1])
print('Population data is '+population_stats[-1]+'\n')

# Plot data
create_graph(country_area, country_population)

# Calculate population density
country_density = calculate_density(country_population, country_area)
df['density'] = country_density
print(df)

# Loop to create list of countries with density > 100 people/km^2
dense_countries = []
for i in range(0, len(country_density)):
    if country_density[i] > 100:
        dense_countries.append(df['country'][i])
    else:
        pass

print('\nDense Countries:', dense_countries)

# Write data to CSV file
df.to_csv('southern_africa.csv', sep='|')

