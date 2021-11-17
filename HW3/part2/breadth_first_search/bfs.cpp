#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list) {
    list -> count = 0;
}

void vertex_set_init(vertex_set *list, int count) {
    list -> max_vertices = count;
    list -> vertices = (int*) malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
bool top_down_step(Graph g, vertex_set *frontier, vertex_set *new_frontier, int *distances)  {
    
    bool hasFrontier = false;

	#pragma omp parallel for
    for (int i = 0; i < frontier -> count; i++) {

        int node = frontier -> vertices[i];

        int start_edge = g -> outgoing_starts[node];
        int end_edge = (node == g -> num_nodes - 1)
                           ? g -> num_edges
                           : g -> outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
            int outgoing = g -> outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER) {
                // distances[outgoing] = distances[node] + 1;
                
                // int index = new_frontier -> count++;
                // new_frontier -> vertices[index] = outgoing;

				if(__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1)) {
					
					int idx = __sync_add_and_fetch(&new_frontier -> count, 1);
                
					new_frontier -> vertices[idx - 1] = outgoing;
                    hasFrontier = true;
				}
            }
        }
    }
    return hasFrontier;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph -> num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier -> vertices[frontier -> count++] = ROOT_NODE_ID;
    sol -> distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

        #ifdef VERBOSE
            double start_time = CycleTimer::currentSeconds();
        #endif

        vertex_set_clear(new_frontier);
        top_down_step(graph, frontier, new_frontier, sol->distances);

        #ifdef VERBOSE
            double end_time = CycleTimer::currentSeconds();
            printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
        #endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

/*
function bottom-up-step(frontier, next, parents)
    for v ∈ vertices do
        if parents[v] = -1 then
            for n ∈ neighbors[v] do 
                if n ∈ frontier then 
                    parents[v] ← n
                    next ← next ∪ {v}
                    break 
                end if
            end for 
        end if
    end for
end func
*/

bool bottom_up_step(Graph g, bool *frontier, bool *new_frontier, int *distances, int n, int *from, int dis) {
    
    bool hasFrontier = false;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {

        if (distances[i] == NOT_VISITED_MARKER) {
            int start_edge = g -> incoming_starts[i];
            int end_edge = (i == n - 1) ? g -> num_edges : g -> incoming_starts[i + 1];

            for (int j = start_edge; j != end_edge; j++) {
                if (frontier[from[j]]) {

                    new_frontier[i] = 1;
                    distances[i] = dis;
                    hasFrontier = true;

                    break;

                }
            }
        }
    }
    return hasFrontier;
}

void bfs_bottom_up(Graph graph, solution *sol) {
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    int n = graph -> num_nodes;
    bool hasFrontier = true;

    bool *frontier     = (bool*) calloc(n, sizeof(bool));
    bool *new_frontier = (bool*) calloc(n, sizeof(bool));
    int  *inc_edge = graph -> incoming_edges;

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        sol -> distances[i] = NOT_VISITED_MARKER;

    frontier[ROOT_NODE_ID] = true;
    sol -> distances[ROOT_NODE_ID] = 0;

    int dis = 1;
    while (hasFrontier) {

        hasFrontier = bottom_up_step(graph, frontier, new_frontier, sol -> distances, n, inc_edge, dis);

        bool *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        dis += 1;
    }
	//free(frontier);
	//free(new_frontier);
}

void bfs_hybrid(Graph graph, solution *sol) {
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.

    // Top-Down
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier_td = &list1;
    vertex_set *new_frontier_td = &list2;

    frontier -> vertices[frontier -> count++] = ROOT_NODE_ID;
    sol -> distances[ROOT_NODE_ID] = 0;

    int n = graph -> num_nodes;
    bool hasFrontier = true, changeBT = false;

    // Bottom-Up
    bool *frontier_bt     = (bool*) calloc(n, sizeof(bool));
    bool *new_frontier_bt = (bool*) calloc(n, sizeof(bool));
    int  *inc_edge = graph -> incoming_edges;

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        sol -> distances[i] = NOT_VISITED_MARKER;


    int dis = 1;
    while (hasFrontier) {

        // Top-Down
        if (!changeBT and (float) (frontier -> count) / n < 0.1) {

            vertex_set_clear(new_frontier_td);
            hasFrontier = top_down_step(graph, frontier_td, new_frontier_td, sol -> distances);

            vertex_set *tmp = frontier;
            frontier = new_frontier;
            new_frontier = tmp;

            continue;
        }
        // Bottom-Up

        changeBT = true;
        hasFrontier = bottom_up_step(graph, frontier_bt, new_frontier_bt, sol -> distances, n, inc_edge, dis);

        bool *tmp = frontier_bt;
        frontier_bt = new_frontier_bt;
        new_frontier_bt = tmp;

        dis += 1;
    }
}
