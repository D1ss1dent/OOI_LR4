import random
import time
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def generate_path(start_city, unvisited, pheromones, distances):  # done
    path = [start_city]
    unvisited.remove(start_city)

    while unvisited:
        next_city = select_next_city(
            path[-1], unvisited, pheromones, distances)
        path.append(next_city)
        unvisited.remove(next_city)

    path.append(start_city)

    return path


def select_next_city(current_city, unvisited, pheromones, distances):  # done
    probabilities = [calculate_probability(
        current_city, city, unvisited, pheromones, distances) for city in unvisited]
    selected_city = random.choices(
        list(unvisited), weights=probabilities, k=1)[0]
    return selected_city


def calculate_probability(current_city, next_city, unvisited, pheromones, distances, alpha=1, beta=1):  # done
    distance = distances[current_city][next_city]
    visibility = 1 / distance
    total = sum([(1 / distances[current_city][city]) ** beta *
                 pheromones[current_city][city] ** alpha for city in unvisited])
    probability = ((visibility ** beta) *
                   (pheromones[current_city][next_city] ** alpha)) / total
    return probability


def update_pheromones(pheromones, ant_paths, distances):  # done
    evaporation = 0.7

    for i in range(len(pheromones)):
        for j in range(len(pheromones[i])):
            pheromones[i][j] *= (1 - evaporation)

    ant_paths.sort(key=lambda x: calculate_distance(x, distances))
    num_best_paths = max(len(ant_paths) // 10, 1)
    best_paths = ant_paths[:num_best_paths]

    for path in best_paths:
        for i in range(len(path) - 1):
            current_city = path[i]
            next_city = path[i + 1]
            pheromones[current_city][next_city] += evaporation * \
                (1 / distances[current_city][next_city])


def calculate_distance(path, distances):  # done
    distance = 0
    for i in range(len(path) - 1):
        current_city = path[i]
        next_city = path[i + 1]
        distance += distances[current_city][next_city]
    return distance


def ant_colony(distances, cities, n_ants, n_iterations):
    pheromones = [[1 for _ in range(len(cities))] for _ in range(len(cities))]
    best_path = None
    best_distance = float('inf')
    unchanged_count = 0
    plt.ion()

    fig, ax = plt.subplots(figsize=(10, 6))

    for iteration in range(n_iterations):
        ant_paths = []
        unvisited = set(range(len(cities)))

        for _ in range(n_ants):
            start_city = cities.index('Київ')
            path = generate_path(start_city,
                                 unvisited.copy(), pheromones, distances)
            ant_paths.append(path)

        update_pheromones(pheromones, ant_paths, distances)

        shortest_path = min(
            ant_paths, key=lambda x: calculate_distance(x, distances))
        shortest_distance = calculate_distance(shortest_path, distances)

        if shortest_distance < best_distance:
            best_path = shortest_path
            best_distance = shortest_distance
            unchanged_count = 0
        else:
            unchanged_count += 1

        print("Для ітерації: ", iteration + 1)
        print("Найкоротший маршрут:", [cities[i] for i in best_path])
        print("Довжина найкоротшого маршруту:", best_distance)
        if unchanged_count > 10:
            break

        ukraine_map([cities[i] for i in best_path], cities, ax=ax)

        ax.legend(["Довжина найкоротшого маршруту: {}".format(
            best_distance)], loc='lower left')
        ax.get_legend().legendHandles[0].set_color('black')

        ax.set_title('Карта "України: Ітерація: {}"'.format(iteration + 1))
        plt.draw()
        plt.pause(1)

    plt.ioff()
    plt.show()

    return best_path, best_distance, iteration+1


def ukraine_map(best_path, cities, ax=None):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ukraine = world[world.name == 'Ukraine']

    cities_data = {
        'City': cities,
        'Latitude': [48.9345, 48.4534, 48.0159, 50.2547, 47.8388, 48.9226, 50.4501, 48.5041, 48.5671, 50.7472,
                     49.8397, 46.975, 46.4825, 49.593, 50.6199, 44.9521, 50.9077, 49.5535, 48.6208, 49.9935,
                     46.6354, 49.4249, 49.5883, 48.2917, 51.4934],
        'Longitude': [28.4801, 35.0375, 37.8028, 28.7987, 35.3396, 24.7097, 30.5234, 32.2656, 39.3242, 25.3425,
                      24.0297, 32.059, 30.5233, 34.488, 26.2516, 34.1108, 34.8405, 25.5966, 22.2947, 36.2304,
                      32.6178, 26.9945, 31.8662, 25.434, 31.2999]
    }
    cities_df = pd.DataFrame(cities_data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax.clear()

    ukraine.plot(ax=ax, color='lightblue', edgecolor='black')

    for idx, city in cities_df.iterrows():
        ax.plot(city['Longitude'], city['Latitude'], marker='o', color='red')
        ax.text(city['Longitude'] - 0.4, city['Latitude'] +
                0.15, city['City'], fontsize=7, color='red')

    for i in range(len(best_path) - 1):
        start_idx = cities.index(best_path[i])
        end_idx = cities.index(best_path[i + 1])
        start_city = cities_df.iloc[start_idx]
        end_city = cities_df.iloc[end_idx]
        ax.annotate("", xy=(end_city['Longitude'], end_city['Latitude']),
                    xytext=(start_city['Longitude'], start_city['Latitude']),
                    arrowprops=dict(arrowstyle="->", color='black'))
        plt.draw()
        plt.pause(0.001)
        time.sleep(0.001)


distances = [
    [0, 571, 812, 126, 637, 373, 266, 317, 972, 387, 369, 466, 429,
        576, 313, 801, 611, 239, 593, 720, 533, 122, 340, 312, 423],
    [571, 0, 250, 630, 89, 952, 479, 246, 401, 888, 948, 329, 463,
        183, 814, 458, 366, 818, 1172, 222, 316, 701, 326, 891, 585],
    [812, 250, 0, 880, 243, 1202, 729, 496, 151, 1138, 1198, 579, 713,
        391, 1064, 571, 488, 1068, 1422, 283, 535, 951, 576, 1141, 826],
    [126, 630, 880, 0, 719, 459, 140, 434, 987, 261, 407, 592, 555,
        494, 187, 927, 485, 325, 679, 638, 659, 208, 352, 398, 297],
    [637, 89, 243, 719, 0, 1018, 568, 314, 394, 977, 1014, 352, 486,
        277, 903, 371, 460, 884, 1238, 303, 292, 767, 415, 957, 674],
    [373, 952, 1202, 459, 1018, 0, 599, 698, 1353, 273, 135, 785, 658,
        953, 292, 1124, 944, 137, 301, 1097, 856, 251, 721, 143, 756],
    [266, 479, 729, 140, 568, 599, 0, 299, 836, 398, 544, 517, 480,
        343, 324, 852, 339, 465, 819, 487, 584, 348, 201, 538, 151],
    [317, 246, 496, 434, 314, 698, 299, 0, 647, 692, 694, 180, 337,
        251, 618, 524, 434, 564, 918, 395, 251, 447, 126, 637, 436],
    [972, 401, 151, 987, 394, 1353, 836, 647, 0, 1245, 1349, 730, 864,
        493, 1171, 722, 535, 1219, 1573, 303, 686, 1102, 727, 1292, 873],
    [387, 888, 1138, 261, 977, 273, 398, 692, 1245, 0, 152, 853, 816,
        752, 70, 1188, 743, 164, 432, 896, 920, 268, 610, 336, 555],
    [369, 948, 1198, 407, 1014, 135, 544, 694, 1349, 152, 0, 843, 743,
        898, 215, 1178, 889, 127, 278, 1042, 910, 247, 717, 278, 701],
    [466, 329, 579, 592, 352, 785, 517, 180, 730, 853, 843, 0, 134,
        488, 779, 339, 671, 713, 1067, 551, 71, 596, 368, 642, 671],
    [429, 463, 713, 555, 486, 658, 480, 337, 864, 816, 743, 134, 0,
        596, 742, 473, 779, 676, 959, 685, 205, 559, 453, 515, 634],
    [576, 183, 391, 494, 277, 953, 343, 251, 493, 752, 898, 488, 596,
        0, 678, 631, 185, 819, 1173, 144, 499, 702, 271, 892, 412],
    [313, 814, 1064, 187, 903, 292, 324, 618, 1171, 70, 215, 779, 742,
        678, 0, 1114, 669, 158, 495, 823, 846, 195, 536, 331, 481],
    [801, 458, 571, 927, 371, 1124, 852, 524, 722, 1188, 1178, 339, 473,
        631, 1114, 0, 814, 1048, 1402, 657, 279, 931, 649, 981, 959],
    [611, 366, 488, 485, 460, 944, 339, 434, 535, 743, 889, 671, 779,
        183, 669, 814, 0, 810, 1164, 185, 682, 693, 343, 883, 338],
    [239, 818, 1068, 325, 884, 137, 465, 564, 1219, 163, 127, 713, 676,
        819, 158, 1048, 810, 0, 353, 963, 780, 117, 587, 176, 622],
    [593, 1172, 1422, 679, 1238, 301, 819, 918, 1573, 432, 278, 1067, 959,
        1173, 495, 1402, 1164, 353, 0, 1317, 1134, 417, 941, 444, 976],
    [720, 222, 283, 638, 303, 1097, 487, 395, 303, 896, 1042, 551, 685,
        144, 823, 657, 185, 963, 1317, 0, 538, 846, 415, 1036, 523],
    [533, 316, 535, 659, 292, 856, 584, 251, 686, 920, 910, 71, 205,
        499, 846, 279, 682, 780, 1134, 538, 0, 663, 411, 713, 738],
    [122, 701, 951, 208, 767, 251, 348, 447, 1102, 268, 247, 596, 559,
        702, 192, 931, 693, 117, 471, 846, 663, 0, 470, 190, 505],
    [340, 326, 576, 352, 415, 721, 201, 126, 727, 610, 717, 368, 453,
        271, 536, 649, 343, 587, 941, 415, 411, 470, 0, 660, 311],
    [312, 891, 1141, 398, 957, 143, 538, 637, 1292, 336, 278, 642, 515,
        892, 331, 981, 883, 176, 444, 1036, 713, 190, 660, 0, 695],
    [423, 585, 826, 297, 674, 756, 151, 436, 873, 555, 701, 671, 634,
        412, 481, 959, 338, 622, 976, 523, 738, 505, 311, 695, 0]
]

distances = [[int(distance) for distance in row] for row in distances]

cities = [
    'Вінниця', 'Дніпро', 'Донецьк', 'Житомир', 'Запоріжжя', 'Івано-Франківськ', 'Київ', 'Кропивницький', 'Луганськ',
    'Луцьк', 'Львів', 'Миколаїв', 'Одеса', 'Полтава', 'Рівне', 'Симферопіль', 'Суми', 'Тернопіль', 'Ужгород',
    'Харків', 'Херсон', 'Хмельницький', 'Черкаси', 'Чернівці', 'Чернігів'
]


n_ants = int(input("Введіть кількість мурах: "))
n_iterations = int(input("Введіть максимальну кількість ітерацій: "))

best_path, best_distance, f_iteration = ant_colony(
    distances, cities, n_ants, n_iterations)


print(f"Кінцевий результат для {f_iteration:d} ітерації: ")

print("Найкоротший маршрут:", [cities[i] for i in best_path])
print("Довжина найкоротшого маршруту:", best_distance)
