import numpy as np

import os
import re
import random

from .dubins.dubins_length_only import dubins_path_length

def solve(solver_path, problem):
    file_random_suffix = random.randint(0, 1000000)

    # if directory named "TMP" is not exit, make it
    if not os.path.isdir("./TMP"):
        os.mkdir("./TMP")

    # write problem as gtsp file
    with open("./TMP/tmp_{}.gtsp".format(file_random_suffix),"w") as f:
        f.write(problem)

    # write par
    par_str = ""
    par_str += "PROBLEM_FILE = ./TMP/tmp_{}.gtsp\n".format(file_random_suffix)
    par_str += "ASCENT_CANDIDATES = 500\n"
    par_str += "CANDIDATE_SET_TYPE = POPMUSIC\n"
    par_str += "POPMUSIC_SAMPLE_SIZE = 100\n"
    par_str += "POPMUSIC_MAX_NEIGHBORS = 30\n"
    par_str += "POPMUSIC_TRIALS = 0\n"
    par_str += "INITIAL_PERIOD = 1000\n"
    par_str += "MAX_CANDIDATES = 30\n"
    par_str += "MAX_TRIALS = 10000\n"
    par_str += "OUTPUT_TOUR_FILE = ./TMP/tmp_{}.tour\n".format(file_random_suffix)
    par_str += "POPULATION_SIZE = 1\n"
    par_str += "PRECISION = 10\n"
    par_str += "RUNS = 1\n"
    par_str += "SEED = 1\n"
    par_str += "TRACE_LEVEL = 0\n" # This is verbose

    with open("./TMP/tmp_{}.par".format(file_random_suffix),"w") as f:
        f.write(par_str)
    
    os.system("{} {}".format(solver_path, "./TMP/tmp_{}.par".format(file_random_suffix)))

    # answer from tmp.tour
    with open("./TMP/tmp_{}.tour".format(file_random_suffix),"r") as f:
        solution_str = f.read()
        solution = parse_solution(solution_str)

    # remove every tmps
    os.system("rm ./TMP/tmp_{}.*".format(file_random_suffix))

    return solution


def parse_solution(solution_str):
    tour_section_start = solution_str.find("TOUR_SECTION")
    eof_start = solution_str.find("EOF")

    tour_section = solution_str[tour_section_start:eof_start].strip()

    numbers = re.findall(r'\d+', tour_section)
    numbers = [int(num) for num in numbers]

    return numbers


def glkh_dtsp_solve(waypoints, heading_num, curvature):
    glkh_solver_path = "./GLKH"

    heading = np.linspace(-np.pi, np.pi, heading_num, endpoint=False)
    waypoint_num = len(waypoints)

    # construct dybins distance matrix with waypoints which have discretized heading
    SCALE = 1000
    dubins_distance_matrix = np.zeros((waypoint_num*heading_num,waypoint_num*heading_num))

    for i in range(waypoint_num*heading_num):
        for j in range(waypoint_num*heading_num):
            if i==j:
                dubins_distance_matrix[i][j] = 10000
            else:
                start_x = waypoints[i//heading_num][0]
                start_y = waypoints[i//heading_num][1]
                start_yaw = heading[i%heading_num]
                end_x = waypoints[j//heading_num][0]
                end_y = waypoints[j//heading_num][1]
                end_yaw = heading[j%heading_num]
                length = dubins_path_length(start_x,
                                            start_y,
                                            start_yaw,
                                            end_x,
                                            end_y,
                                            end_yaw,
                                            curvature)
                dubins_distance_matrix[i][j] = length * SCALE

    # solve GTSP with glkh and dubins_distance_matrix
    problem_str = ""
    problem_str += "TYPE : AGTSP\n"
    problem_str += "DIMENSION : " + str(waypoint_num*heading_num) + "\n"
    problem_str += "GTSP_SETS : {}\n".format(waypoint_num)
    problem_str += "EDGE_WEIGHT_TYPE : EXPLICIT\n"
    problem_str += "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n"
    problem_str += "EDGE_WEIGHT_SECTION\n"
    for i in range(waypoint_num*heading_num):
        for j in range(waypoint_num*heading_num):
            problem_str += str(int(dubins_distance_matrix[i][j])) + " "
        problem_str += "\n"
    problem_str += "GTSP_SET_SECTION\n"
    for i in range(waypoint_num):
        problem_str += "{} ".format(i+1)
        for j in range(heading_num):
            problem_str += "{} ".format(i*heading_num+j+1)
        problem_str += "-1\n"

    solution = solve(glkh_solver_path, problem_str)

    solution = np.array(solution[:-1])-1 # remove -1 and convert to 0-indexed
    
    # post process solution into index of waypoints and headings
    idx = solution // heading_num
    headings = (solution % heading_num) * 2 * np.pi / heading_num - np.pi

    return idx, headings

def glkh_dtsp_solve_originfixed(waypoints, heading_num, curvature):
    glkh_solver_path = "./GLKH"

    heading = np.linspace(-np.pi, np.pi, heading_num, endpoint=False)
    waypoint_num = len(waypoints)

    # construct dybins distance matrix with waypoints which have discretized heading
    SCALE = 1000
    dubins_distance_matrix = np.zeros((waypoint_num*heading_num,waypoint_num*heading_num))

    for i in range(waypoint_num*heading_num):
        for j in range(waypoint_num*heading_num):
            if i==j:
                dubins_distance_matrix[i][j] = 10000
            else:
                start_x = waypoints[i//heading_num][0]
                start_y = waypoints[i//heading_num][1]
                if i < heading_num: # depot
                    start_yaw = 0
                else:
                    start_yaw = heading[i%heading_num]
                end_x = waypoints[j//heading_num][0]
                end_y = waypoints[j//heading_num][1]
                if j < heading_num:
                    end_yaw = 0
                else:
                    end_yaw = heading[j%heading_num]
                length = dubins_path_length(start_x,
                                            start_y,
                                            start_yaw,
                                            end_x,
                                            end_y,
                                            end_yaw,
                                            curvature)
                dubins_distance_matrix[i][j] = length * SCALE

    # solve GTSP with glkh and dubins_distance_matrix
    problem_str = ""
    problem_str += "TYPE : AGTSP\n"
    problem_str += "DIMENSION : " + str(waypoint_num*heading_num) + "\n"
    problem_str += "GTSP_SETS : {}\n".format(waypoint_num)
    problem_str += "EDGE_WEIGHT_TYPE : EXPLICIT\n"
    problem_str += "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n"
    problem_str += "EDGE_WEIGHT_SECTION\n"
    for i in range(waypoint_num*heading_num):
        for j in range(waypoint_num*heading_num):
            problem_str += str(int(dubins_distance_matrix[i][j])) + " "
        problem_str += "\n"
    problem_str += "GTSP_SET_SECTION\n"
    for i in range(waypoint_num):
        problem_str += "{} ".format(i+1)
        for j in range(heading_num):
            problem_str += "{} ".format(i*heading_num+j+1)
        problem_str += "-1\n"

    solution = solve(glkh_solver_path, problem_str)

    solution = np.array(solution[:-1])-1 # remove -1 and convert to 0-indexed
    
    # post process solution into index of waypoints and headings
    idx = solution // heading_num
    headings = (solution % heading_num) * 2 * np.pi / heading_num - np.pi

    # fix the heading of the depot
    headings[0] = 0

    return idx, headings

if __name__ == "__main__":
    waypoints = np.random.uniform(low=-10, high=10, size=(10, 2))
    heading_num = 8
    curvature = 1.5
    solution = glkh_dtsp_solve(waypoints, heading_num, curvature)

    print(solution)