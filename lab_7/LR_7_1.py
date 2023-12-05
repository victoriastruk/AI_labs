import numpy as np
import random
import matplotlib.pyplot as plt


class Ant:
    def __init__(self, position):
        self.current_position = position
        self.next_position = None
        self.tabu_list = []
        self.visited_count = 0
        self.travel_path = []
        self.path_length = 0


def initialize_colony(num_agents, num_cities):
    # Ініціалізація мурашиних агентів
    colony = [Ant(random.randint(0, num_cities - 1)) for _ in range(num_agents)]
    return colony


def calculate_distance(city1, city2):
    # Розрахунок відстані між двома містами
    return distances[city1][city2]


def choose_next_city(ant, pheromone_matrix, alpha, beta):
    # Вибір наступного міста для агента
    available_cities = list(set(range(len(pheromone_matrix))) - set(ant.tabu_list))

    if not available_cities:
        return

    probabilities = []

    for city in available_cities:
        pheromone_intensity = pheromone_matrix[ant.current_position][city]
        distance = calculate_distance(ant.current_position, city)
        probability = (pheromone_intensity ** alpha) / (
                    (distance + 1e-10) ** beta)  # Додана константа для уникнення ділення на нуль
        probabilities.append(probability)

    probabilities = [prob / sum(probabilities) for prob in probabilities]
    selected_city = np.random.choice(available_cities, p=probabilities)
    ant.next_position = selected_city


def update_paths(ants, pheromone_matrix, evaporation_rate):
    # Оновлення шляхів та феромонів після кожного циклу
    for ant in ants:
        ant.visited_count += 1
        ant.tabu_list.append(ant.current_position)
        ant.travel_path.append(ant.current_position)
        ant.path_length += calculate_distance(ant.current_position, ant.next_position)

    pheromone_matrix *= (1 - evaporation_rate)

    for ant in ants:
        if ant.path_length != 0:
            pheromone_matrix[ant.current_position][ant.next_position] += 1 / ant.path_length
            pheromone_matrix[ant.next_position][ant.current_position] += 1 / ant.path_length
        ant.current_position = ant.next_position


def plot_path(cities, path):
    # Відображення шляху на графіку
    x = [cities[i][0] for i in path]
    y = [cities[i][1] for i in path]

    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.scatter(x, y, color='r')
    plt.title("Best Path")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.show()


def main(num_agents, num_iterations, alpha, beta, evaporation_rate):
    colony = initialize_colony(num_agents, len(distances))
    pheromone_matrix = np.ones_like(distances) / 1000

    best_global_path = None
    best_global_length = float('inf')

    for iteration in range(num_iterations):
        for ant in colony:
            choose_next_city(ant, pheromone_matrix, alpha, beta)

        update_paths(colony, pheromone_matrix, evaporation_rate)

        best_ant = min(colony, key=lambda x: x.path_length)

        if best_ant.path_length < best_global_length:
            best_global_length = best_ant.path_length
            best_global_path = best_ant.travel_path.copy()

    return best_global_path, best_global_length


if __name__ == "__main__":
    distances = np.array([
        [0, 645, 868, 125, 748, 366, 256, 316, 1057, 382, 360, 471, 428, 593, 311, 844, 602, 232, 575, 734, 521, 120,
         343, 312, 396],
        [645, 0, 252, 664, 81, 901, 533, 294, 394, 805, 975, 343, 468, 196, 957, 446, 430, 877, 1130, 213, 376, 765,
         324, 891, 672],
        [868, 252, 0, 858, 217, 1171, 727, 520, 148, 1111, 1221, 611, 731, 390, 1045, 591, 706, 1100, 1391, 335, 560,
         988, 547, 1141, 867],
        [125, 664, 858,	0, 738,	431, 131, 407, 1182, 257, 423, 677,	557, 468, 187, 803, 477, 298, 671, 690, 624, 185, 321, 389, 271],
        [748, 81, 217, 738, 0, 1119, 607, 303, 365,	681, 833, 377, 497,	270, 925, 365, 477, 977, 1488, 287,	297, 875, 405, 957, 747],
        [366, 901, 1171, 431, 1119,	0, 561,	618, 1402, 328,	135, 747, 627, 898,	296, 1070, 908,	134, 280, 1040, 798, 246, 709, 143, 701],
        [256, 533, 727,	131, 607, 561, 0, 298, 811,	388, 550, 490, 489,	337, 318, 972, 346,	427, 806, 478, 551,	315, 190, 538, 149],
        [316, 294, 520,	407, 303, 618, 298,	0, 668,	664, 710, 174, 294,	246, 627, 570, 506,	547, 883, 387, 225,	435, 126, 637, 363],
        [1057, 394,	148, 1182, 365,	1402, 811, 668,	0, 1199, 1379, 857,	977, 474, 1129,	739, 253, 1289,	1539, 333, 806,	1177, 706, 1292, 951],
        [382, 805, 1111, 257, 681, 328, 388, 664, 1199,	0, 152,	780, 856, 725, 70, 1052, 734, 159, 413,	866, 869, 263, 578,	336, 949],
        [360,	975,	1221,	423,	833,	135,	550,	710,	1379,	152,	0,	850	,970,	891,	232,	1173,	896,	128,	261,	1028,	1141,	240,	740,	278,	690],
        [471,	343,	611,	677,	377,	747,	490,	174,	857,	780,	850,	0,	120,	420,	864,	282,	681,	754,	999,	556,	51,	590,	300,	642,	640],
        [428,	468	731	557	497	627	489	294	977	856	970	120	-	540	741	392	800	660	1009	831	171	548	420	515	529],
        [14	Полтава	593	196	390	468	270	898	337	246	474	725	891	420	540	-	665	635	261	825	1149	141	471	653	279	892	477],
        [15	Рівне	311	957	1045	187	925	296	318	627	1129	70	232	864	741	665	-	1157	664	162	484	805	834	193	508	331	458],
        [16	Сімферополь	844	446	591	803	365	1070	972	570	739	1052	1173	282	392	635	1157	-	896	1097	1363	652	221	964	696	981	1112],
        [17	Суми	602	430	706	477	477	908	346	506	253	734	896	681	800	261	664	896	-	774	1138	190	732	662	540	883	350],
        [18	Тернопіль	232	877	1100	298	977	134	427	547	1289	159	128	754	660	825	162	1097	774	-	338	987	831	112	575	176	568],
        [19	Ужгород	575	1130	1391	671	1488	280	806	883	1539	413	261	999	1009	1149	484	1363	1138	338	-	1299	1065	455	984	444	951],
        [20	Харків	734	213	335	690	287	1040	478	387	333	866	1028	556	831	141	805	652	190	987	1299	-	576	854	420	1036	608],
        [21	Херсон	521	376	560	624	297	798	551	225	806	869	1141	51	171	471	834	221	732	831	1065	576	-	641	351	713	691],
        [22	Хмельницький	120	765	988	185	875	246	315	435	1177	263	240	590	548	653	193	964	662	112	455	854	641	-	463	190	455],
        [23	Черкаси	343	324	547	321	405	709	190	126	706	578	740	300	420	279	508	696	540	575	984	420	351	463	-	660	330],
        [24	Чернівці	312	891	1141	389	957	143	538	637	1292	336	278	642	515	892	331	981	883	176	444	1036	713	190	660	-	695],
        [25	Чернігів	396	672	867	271	747	701	149	363	951	949	690	640	529	477	458	1112	350	568	951	608	691	455	330	695	-]
    ])

    cities = [(0, 0), (1, 2), (3, 5), (7, 8), (9, 1), (4, 6), (2, 10), (11, 4), (5, 9), (12, 7), (6, 3), (8, 12),
              (10, 11), (13, 13), (15, 6), (14, 9), (7, 8), (5, 10), (9, 11), (11, 2), (4, 14), (6, 13), (14, 15),
              (12, 1)]

    starting_city = 2
    num_agents = 5
    num_iterations = 100
    alpha = 1.0
    beta = 2.0
    evaporation_rate = 0.5

    best_path, best_length = main(num_agents, num_iterations, alpha, beta, evaporation_rate)

    print(f"Best Path: {best_path}")
    print(f"Best Path Length: {best_length}")

    plot_path(cities, best_path)
