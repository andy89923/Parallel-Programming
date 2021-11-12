#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence) {
    /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

    while (!converged) {

        // compute score_new[vi] for all nodes vi:
        score_new[vi] = sum over all nodes vj reachable from incoming edges
                        { score_old[vj] / number of edges leaving vj  }

        score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

        score_new[vi] += sum over all nodes v in graph with no outgoing edges
                        { damping * score_old[v] / numNodes }

        // compute how much per-node scores have changed
        // quit once algorithm has converged

        global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
        converged = (global_diff < convergence)
    }

    */

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;

    double *solution_old = malloc(numNodes * sizeof(Vertex));
    bool *no_out = calloc(0, numNodes * sizeof(bool));

    #pragma omp parallel for
    for (int i = 0; i < numNodes; i++) {
        if (outgoing_size(g, i) == 0) no_out = 1;
    }

    #pragma omp parallel for
    for (int i = 0; i < numNodes; ++i) {
        solution_old[i] = equal_prob;
    }

    bool converge = 1;
    while (!converge) {

        #pragma omp parallel for
        for (int i = 0; i < numNodes; i++) {
            solution_old[i] = solution[i];
            solution[i] = 0.0;
        }

        #pragma omp parallel for
        for (int i = 0; i < numNodes; i++) {

            int m = outgoing_size(g, i);
            double p = solution_old[i] / m;

            const Vertex* vs = outgoing_end(g, i);

            // #pragma omp parallel for
            for (int j = 0; j < m; j++) {
                int endV = *(vs + j);
                solution[endV] += p;
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < numNodes; i++) {
            solution[i] = solution[i] * damping + (double) (1.0 - damping) / numNodes;
        }

        double from_no = 0.0;

        #pragma omp parallel for reduction(+: from_no)
        for (int i = 0; i < numNodes; i++) {
            double now = (no_out[i] == 0 ? damping * solution_old[i] / numNodes : 0.0);

            from_no += now;
        }


        double global_diff = 0.0;

        #pragma omp parallel for reduction(+: global_diff)
        for (int i = 0; i < numNodes; i++) {
            solution[i] += from_no;
            global_diff += abs(solution_old[i] - solution[i]);
        }

        converge = (global_diff < convergence);
    }
}
