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
    double equal_prob = (double) 1.0 / numNodes;
	// double dPnN = (1.0 - damping) / numNodes;
	// double dnN = damping / numNodes;

    double *solution_old = (double*) malloc(numNodes * sizeof(double));
    //bool *no_out = (bool*) calloc(numNodes, sizeof(bool));

    #pragma omp parallel for // schedule(static, 256) 
    for (int i = 0; i < numNodes; i++) {
        //if (outgoing_size(g, i) == 0) 
		//	no_out[i] = true;
		//else
		//	no_out[i] = false;

		solution[i] = equal_prob;
    }

	double dsum, from_no;
	double global_diff;
	int ss;

    bool converge = 0;
    while (!converge) {

		from_no = 0.0;

		#pragma omp parallel for reduction(+: from_no)
        for (int i = 0; i < numNodes; i++) {
            solution_old[i] = solution[i];
            solution[i] = 0.0;

			from_no += (outgoing_size(g, i) == 0 ? solution_old[i] : 0.0);
        }
	
		#pragma omp parallel for private(dsum, ss)
        for (int i = 0; i < numNodes; i++) {
			//solution_old[i] = solution[i];
			//solution[i] = 0;

            const Vertex* vs = incoming_begin(g, i);
			const Vertex* vt = incoming_end(g, i);
			
			dsum = 0.0;

            for (const Vertex* v = vs; v != vt; v++) {
				ss = outgoing_size(g, *v);
				dsum += (ss == 0 ? 0.0 : solution_old[*v] / ss);
            }
			solution[i] = dsum * damping + (1.0 - damping) / numNodes;
			solution[i] += from_no * damping / numNodes;

			// from_no += (outgoing_size(g, i) == 0 ? solution_old[i] : 0.0);
        }


        global_diff = 0.0;

        #pragma omp parallel for reduction(+: global_diff)
        for (int i = 0; i < numNodes; i++) {
            global_diff += fabs(solution_old[i] - solution[i]);
        }

        converge = (global_diff < convergence);
    }
	free(solution_old);
}
