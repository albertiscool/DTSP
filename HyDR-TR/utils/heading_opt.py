import numpy as np

def aa_solve(waypoints, route):
    """
    input: waypoints: numpy array(n_waypoint x 2)
    output: headings: list of list of float
    """

    # headings = [np.arctan2(waypoints[route[1]][1]-waypoints[route[0]][1],
    #                         waypoints[route[1]][0]-waypoints[route[0]][0])]
    headings = []
    for i in range(len(route)-1):
        if i%2 == 1:
            headings.append(headings[-1])
        else:
            headings.append(np.arctan2(waypoints[route[i+1]][1]-waypoints[route[i]][1],
                                       waypoints[route[i+1]][0]-waypoints[route[i]][0]))
    
    return headings 