#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <queue>

using namespace Eigen;

typedef struct {
  long n;
  long m;
  unsigned int *rowOffsets;
  unsigned int *adj;
  long n_coarse;
  long m_coarse;
  unsigned int *rowOffsetsCoarse;
  unsigned int *adjCoarse;
  int *coarseID;
  double *eweights;
} graph_t;

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 8) {

  std::ostringstream out;
  out << std::setprecision(n) << a_value;
  return out.str();
}

static int
vu_cmpfn_inc(const void *a, const void *b) {

  int *av = ((int *) a);
  int *bv = ((int *) b);
  if (*av > *bv)
    return 1;
  if (*av < *bv)
    return -1;
  if (*av == *bv) {
    if (av[1] > bv[1])
      return 1;
    if (av[1] < bv[1])
      return -1;
  }
  return 0;
}

static int
simpleCoarsening(graph_t *g, int coarseningType) {

  if (coarseningType == 0)
    return 0;

  int num_coarsening_rounds_max = 100;
  int coarse_graph_nmax = 1000;

  int *cID = (int *) malloc((g->n)*sizeof(int));
  assert(cID != NULL);
  int *toMatch = (int *) malloc((g->n)*sizeof(int));
  assert(toMatch != NULL);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (long i=0; i<g->n; i++) {
    cID[i] = i;
    toMatch[i] = 1;
  }

  int coarse_vert_count = g->n;
  int num_rounds = 0;
  while ((coarse_vert_count > coarse_graph_nmax) && 
         (num_rounds < num_coarsening_rounds_max)) {
    num_rounds++;
    int num_matched = 0;
    for (int i=0; i<g->n; i++) {
      int u = i;
      while (cID[u] != u) {
        cID[u] = cID[cID[u]];
        u = cID[u];
      }
      if (toMatch[u] == 1) {
        for (unsigned int j=g->rowOffsets[u]; 
                          j<g->rowOffsets[u+1]; j++) {
          int v = g->adj[j];
          while (v != cID[v]) {
            cID[v] = cID[cID[v]];
            v = cID[v];
          }
          if (v == u) {
            continue;
          }
          if (toMatch[v] == 1) {
            if (u < v) {
              cID[u] = u;
              cID[v] = u;
            } else {
              cID[u] = v;
              cID[v] = v;
            }
            toMatch[u] = toMatch[v] = 0;
            num_matched += 2;
            break;
          }
        }
      }
    }

    int num_unmatched = coarse_vert_count - num_matched;
    int new_coarse_vert_count = num_matched/2 + num_unmatched;
    fprintf(stderr, "prev count: %d, matched: %d, new count: %d\n", 
             coarse_vert_count, num_matched, new_coarse_vert_count);
             coarse_vert_count = new_coarse_vert_count;

    for (int i=0; i<g->n; i++) {
      toMatch[i] = 1;
    }
  }
  fprintf(stderr, "num rounds: %d\n", num_rounds);

  int *coarse_edges = (int *) malloc(2 * g->m * sizeof(int));
  assert(coarse_edges != NULL);  

  /* set correct IDs */  
  for (int i=0; i<g->n; i++) {
    int u = cID[i];
    while (u != cID[u]) {
      u = cID[cID[u]];
    }
    cID[i] = u;
  }

  int *vertIDs = (int *) malloc(g->n * sizeof(int));
  assert(vertIDs != NULL);  
  for (int i=0; i<g->n; i++) {
    vertIDs[i] = -1;
  }
  
  int new_id = 0;
  for (int i=0; i<g->n; i++) {
    if (cID[i] == i) {
      vertIDs[i] = new_id++;
    }
  }
  assert(new_id == coarse_vert_count);
  for (int i=0; i<g->n; i++) {
    if (vertIDs[i] == -1) {
      vertIDs[i] = vertIDs[cID[i]];
    }
  }

  long ecount = 0;
  for (int i=0; i<g->n; i++) {
    int u = vertIDs[i];
    for (unsigned int j=g->rowOffsets[i]; j<g->rowOffsets[i+1]; j++) {
      int v = vertIDs[g->adj[j]];
      coarse_edges[ecount++] = u;
      coarse_edges[ecount++] = v;
      // fprintf(stderr, "%d %d\n", u, v);
    }
  }
  ecount = ecount/2;
  qsort(coarse_edges, g->m, 2*sizeof(int), vu_cmpfn_inc);

  /* count the number of coarse edges */
  int m_coarse = 1;
  int prev_u = coarse_edges[0];
  int prev_v = coarse_edges[1];
    
  for (int i=1; i<ecount; i++) {
    int curr_u = coarse_edges[2*i];
    int curr_v = coarse_edges[2*i+1];
    if ((curr_u != prev_u) || (curr_v != prev_v)) {
      m_coarse++;
      prev_u = curr_u;
      prev_v = curr_v;
    }
  }

  fprintf(stderr, "m_coarse %d\n", m_coarse);

  /* Allocate coarse edge weights array */
  double *eweights;
  eweights = (double *) malloc(m_coarse * sizeof(double));
  assert(eweights != NULL);
  for (int i=0; i<m_coarse; i++) {
    eweights[i] = 1.0;
  }

  unsigned int *rowOffsetsCoarse;
  rowOffsetsCoarse = (unsigned int *)
    malloc((coarse_vert_count+1)*sizeof(unsigned int));
  assert(rowOffsetsCoarse != NULL);
  for (int i=0; i<coarse_vert_count+1; i++) {
    rowOffsetsCoarse[i] = 0;
  }

  unsigned int *adjCoarse;
  adjCoarse = (unsigned int *) malloc(m_coarse*sizeof(unsigned int));
  assert(adjCoarse != NULL);
 
  /* update coarse edge weights */
  m_coarse = 1;
  // eweights[0] = 1.0;
  prev_u = coarse_edges[0];
  prev_v = coarse_edges[1];
  adjCoarse[0] = prev_v;
  rowOffsetsCoarse[prev_u+1]++;
    
  for (int i=1; i<ecount; i++) {
    int curr_u = coarse_edges[2*i];
    int curr_v = coarse_edges[2*i+1];
    if ((curr_u != prev_u) || (curr_v != prev_v)) {
      m_coarse++;
      adjCoarse[m_coarse-1] = curr_v;
      // eweights[m_coarse] = 1.0;
      rowOffsetsCoarse[curr_u+1]++;
      prev_u = curr_u;
      prev_v = curr_v;
    } else {
      eweights[m_coarse-1] += 1.0; 
    }
  }

  for (int i=1; i<=coarse_vert_count; i++) {
    rowOffsetsCoarse[i] += rowOffsetsCoarse[i-1];
    // fprintf(stderr, "%u ", rowOffsetsCoarse[i]);
  }
  
  /*
  fprintf(stderr, "printing coarse graph:\n");
  for (int i=0; i<coarse_vert_count; i++) {
    for (int j=rowOffsetsCoarse[i]; j<rowOffsetsCoarse[i+1]; j++) {
      fprintf(stderr, "[%u %u %lf] ", i, adjCoarse[j], eweights[j]);
    }
  }
  */

  /*
  for (int i=0; i<ecount/2; i++) {
    fprintf(stderr, "%d %d\n", coarse_edges[2*i], coarse_edges[2*i+1]);
  }
  */

  free(cID);
  free(toMatch);
  free(coarse_edges);

  g->coarseID = vertIDs;
  g->n_coarse = coarse_vert_count;
  g->m_coarse = m_coarse;
  g->eweights = eweights;
  g->adjCoarse = adjCoarse;
  g->rowOffsetsCoarse = rowOffsetsCoarse;

  if (coarseningType == 1)
    return 0;

  // Optionally write to CSR and MTX files
  std::cout << "Writing csr and mtx files to current directory\n"; 
  FILE *writeBinaryPtr = fopen( "graph_coarse.csr", "wb");
  if (writeBinaryPtr == NULL) {
    fprintf(stderr, "could not open csr file for writing\n");
    exit(1);
  }
  
  FILE *outfp_mtx = fopen( "graph_coarse.mtx", "w");
  if (outfp_mtx == NULL) {
    fprintf(stderr, "could not open mtx file for writing\n");
    exit(1);
  }

  long N = g->n_coarse;
  unsigned int *rowOffsetsCoarse_noloops = (unsigned int *)
  malloc((N+1)*sizeof(unsigned int));
  assert(rowOffsetsCoarse_noloops != NULL);
  for (unsigned int i=0; i<N+1; i++) {
    rowOffsetsCoarse_noloops[i] = 0;
  }

  long num_self_loops = 0;
  for (long i=0; i<g->n_coarse; i++) {
    for (unsigned int j=g->rowOffsetsCoarse[i]; 
                      j<g->rowOffsetsCoarse[i+1]; j++) {
      unsigned int v = g->adjCoarse[j];
      if (i == v) {
        num_self_loops++;
      } else {
        rowOffsetsCoarse_noloops[i+1]++;
      }
    }
  } 
 
  for (unsigned int i=1; i<N+1; i++) {
    rowOffsetsCoarse_noloops[i] += rowOffsetsCoarse_noloops[i-1];
  }
   
  long M = g->m_coarse - num_self_loops;
  std::cout << "edge count after loop removal: " << M << std::endl;

  fprintf(outfp_mtx, "%%%%MatrixMarket matrix coordinate pattern symmetric\n");
  fprintf(outfp_mtx, "%ld %ld %ld\n", N, N, M);

  unsigned int *adjCoarse_noloops = (unsigned int *)
    malloc(M*sizeof(unsigned int));
  assert(adjCoarse_noloops != NULL);
 
  long ec = 0; 
  for (long i=0; i<g->n_coarse; i++) {
    for (unsigned int j=g->rowOffsetsCoarse[i]; 
                      j<g->rowOffsetsCoarse[i+1]; j++) {
      unsigned int v = g->adjCoarse[j];
      if (i == v) {
        num_self_loops++;
      } else {
        adjCoarse_noloops[ec++] = v;
        fprintf(outfp_mtx, "%ld %u\n", i+1, v+1);
      }
    }
  } 
  assert(ec == M);
  fclose(outfp_mtx);
 
  long undirected = 1;
  long graph_type = 0;
  long one_indexed = 0;
  long verification_graph = 0;

  fwrite ( &N, sizeof(long), 1, writeBinaryPtr );
  fwrite ( &M, sizeof(long), 1, writeBinaryPtr );
  fwrite ( &undirected, sizeof(long), 1, writeBinaryPtr );
  fwrite ( &graph_type , sizeof(long), 1, writeBinaryPtr );
  fwrite ( &one_indexed , sizeof(long), 1, writeBinaryPtr );
  fwrite ( &verification_graph , sizeof(long), 1, writeBinaryPtr );

  fwrite ( rowOffsetsCoarse_noloops, sizeof(unsigned int), (N+1), writeBinaryPtr );
  fwrite ( adjCoarse_noloops, sizeof(unsigned int), M, writeBinaryPtr );

  fclose( writeBinaryPtr );

  free(rowOffsetsCoarse_noloops);
  free(adjCoarse_noloops);

  return 0;
}

static int
loadToMatrix(SparseMatrix<double,RowMajor>& M, VectorXd& degrees, 
    graph_t *g, int coarseningType) {

  typedef Triplet<double> T;
  std::vector<T> tripletList;
  
  if (coarseningType == 0) {
    tripletList.reserve(g->m);
 
    for (int i=0; i<g->n; i++) {
      tripletList.push_back(T(i,i,0.5));
      degrees(i) = g->rowOffsets[i+1]-g->rowOffsets[i];
      double nzv = 1/(2.0*(g->rowOffsets[i+1]-g->rowOffsets[i]));
      for (unsigned int j=g->rowOffsets[i]; j<g->rowOffsets[i+1]; j++) {
        unsigned int v = g->adj[j];
        tripletList.push_back(T(i, v, nzv));
      }
    }
    M.setFromTriplets(tripletList.begin(), tripletList.end()); 
  } else {

    for (int i=0; i<g->n_coarse; i++) {
      double degree_i = 0;
      for (unsigned int j=g->rowOffsetsCoarse[i]; j<g->rowOffsetsCoarse[i+1]; j++) {
      degree_i += g->eweights[j];
    }
    degrees(i) = degree_i;
    // std::cout << degrees(i) << " ";
    }
   
    tripletList.reserve(g->m_coarse);
 
    for (unsigned int i=0; i<g->n_coarse; i++) {
      double diag_val = 0;
      double inv_2deg = 1/(2.0*degrees(i));
      for (unsigned int j=g->rowOffsetsCoarse[i]; j<g->rowOffsetsCoarse[i+1]; j++) {
        unsigned int v = g->adjCoarse[j];
        if (v == i) {
          diag_val = g->eweights[j]*inv_2deg;
        } else {
          tripletList.push_back(T(i, v, g->eweights[j]*inv_2deg));
        }
      }
      tripletList.push_back(T(i,i,diag_val+0.5));
    }
    M.setFromTriplets(tripletList.begin(), tripletList.end()); 
  }

  return 0;
}

static VectorXd 
bfs(unsigned int *row, 
    unsigned int *col,
    long N, long M,
    unsigned int start) {
  VectorXd columnOfMatrix(N);

  unsigned int s = start;
  int *visited = (int *) malloc (sizeof(int) * N);
  memset(visited, 0, sizeof(unsigned int) * N);
  std::queue<unsigned int> Q;
  Q.push(s);
  visited[s] = 1;
  columnOfMatrix(s) = 0;

  while(!Q.empty()) {
    unsigned int h = Q.front();
    Q.pop();
    for (unsigned int j=row[h]; j<row[h+1]; j++) {
      s = col[j];
      if (!visited[s]) {
        visited[s] = 1;
        Q.push(s);
        columnOfMatrix(s) = columnOfMatrix(h) + 1;
      }
    }
  }
  
  free(visited);
  return columnOfMatrix;
}


static int 
HDE(SparseMatrix<double,RowMajor>& M, graph_t *g,
    VectorXd& degrees, 
    VectorXd& secondVec, VectorXd& thirdVec) {

  auto startTimerPart = std::chrono::high_resolution_clock::now();  
  
  // Create Laplacian
  long n = g->n;
  long m = g->m;
  typedef Triplet<double> T;
  std::vector<T> LTripletList;
  LTripletList.reserve(g->m);
  for (int i=0; i<g->n; i++) {
  LTripletList.push_back(T(i,i,degrees(i)));
    for (unsigned int j=g->rowOffsets[i]; j<g->rowOffsets[i+1]; j++) {
      unsigned int v = g->adj[j];
      LTripletList.push_back(T(i,v, -1.0));
  }
  }
  SparseMatrix<double,RowMajor> L(n,n);
  L.setFromTriplets(LTripletList.begin(), LTripletList.end()); 
  auto endTimerPart = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elt = endTimerPart - startTimerPart;
  std::cout << "Laplacian load time: " << elt.count() << " s." << std::endl;

  // HDE Initialize
  startTimerPart = std::chrono::high_resolution_clock::now();  
  VectorXi min_dist(g->n);
  min_dist.setOnes();
  min_dist = min_dist * INT_MAX;

  int maxM = 10;
  VectorXd tmp(n);
  tmp.setOnes();
  tmp.normalize();
  MatrixXd dist(n,maxM+1);
  MatrixXd dist_bak(n,maxM);
  dist.col(0) = tmp;

  int start_idx = 0;
  int j = 1;
  for (int run_count=0; run_count<maxM; run_count++) {
    dist.col(j) = bfs(g->rowOffsets, g->adj, n, m, start_idx);
    dist_bak.col(j-1) = dist.col(j);

    int max = -1;
    for (unsigned int i=0; i<n; i++) {
      if (dist.col(j)(i) < min_dist[i]) {
        min_dist[i] = dist.col(j)(i);
      }
      if (max < min_dist[i]) {
        max = min_dist[i];
        start_idx = i;
      }
    }

    for (int k=0; k<j; k++) {
      double multplr = dist.col(j).dot(dist.col(k));
      dist.col(j) = dist.col(j) - multplr * dist.col(k);
    }

    double normdist = dist.col(j).norm();
    if (normdist < 0.1) {
      j --;
    } else {
      dist.col(j).normalize();
    }
    j ++;
  }

  MatrixXd LX(n,j-1);
  LX = L * dist.leftCols(j).rightCols(j-1);

  MatrixXd XtLX(j-1,j-1);
  XtLX = dist.leftCols(j).rightCols(j-1).transpose() * LX;

  SelfAdjointEigenSolver<MatrixXd> es(XtLX);
  MatrixXd init_vecs (n, 2);
  init_vecs = dist_bak.leftCols(j-1) * es.eigenvectors().leftCols(2).real() ;
  endTimerPart = std::chrono::high_resolution_clock::now();
  elt = endTimerPart - startTimerPart;
  std::cout << "HDE Initialization time " << elt.count() << " s." << std::endl;
  secondVec = init_vecs.col(0);
  thirdVec  = init_vecs.col(1);

  return 0;
}

static int 
powerIterationKoren(SparseMatrix<double,RowMajor>& M, 
    VectorXd& degrees, double eps, VectorXd& firstVec, 
    VectorXd& secondVec, VectorXd& thirdVec, 
    int coarseningType, char *inputFilename) {

  // double eps = 1e-9;
  std::cout << "Using eps " << eps << " for second eigenvector" << std::endl;

  if (coarseningType > 0) {
    std::cout << "Using coarsened graph" << std::endl;
  }

  int n = M.rows();

  VectorXd uk_hat(n);

  // Intialized vectors are passed to function
  uk_hat = secondVec;
  // uk_hat.setRandom();
  // uk_hat.normalize();

  VectorXd uk(n);

  // For D-orthonormalization
  VectorXd firstVecD(n);
  firstVecD = firstVec.cwiseProduct(degrees);
  double mult1_denom = firstVec.dot(firstVecD);

  VectorXd residual(n);

  int num_iterations1 = 0;
  
  auto startTimerPart = std::chrono::high_resolution_clock::now();  
  while (1) {

    uk = uk_hat;
    
    // D-orthonormalize
    double mult1_num = uk.dot(firstVecD);
    uk = uk - (mult1_num/mult1_denom)*firstVec;

    // Do matrix-vector product
    uk_hat = M*uk;
    uk_hat.normalize();

    num_iterations1++;
   

    // double residual_norm1 =
    //  residual.lpNorm<Infinity>()/(uk_hat.maxCoeff()-uk_hat.minCoeff());

#if 0
    double residual_dot = uk.dot(uk_hat);
    if (residual_dot >= (1-eps)) {
      break;
    }
#endif

    residual = uk-uk_hat;
    double residual_norm = residual.norm(); 
    // std::cout << residual_norm << std::endl;

    if (residual_norm < eps) {
      break;
    }
  }

  std::cout << "Num iterations for second eigenvector: " <<
    num_iterations1 << std::endl;

  // Save this eigenvector
  secondVec = uk_hat;
  auto endTimerPart = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elt = endTimerPart - startTimerPart;
  std::cout << "Second eigenvector computation time: " << elt.count() << " s." << std::endl;

  eps = 2.0*eps;
  std::cout << "Using eps " << eps << " for third eigenvector" << std::endl;

  // For D-orthonormalization
  VectorXd secondVecD(n);
  secondVecD = secondVec.cwiseProduct(degrees);
  double mult2_denom = secondVec.dot(secondVecD);

  // Initialized vectors are passed to function
  uk_hat = thirdVec;
  // uk_hat.setRandom();
  // uk_hat.normalize();

  startTimerPart = std::chrono::high_resolution_clock::now();  
  int num_iterations2 = 0;
  while (1) {

    uk = uk_hat;
    
    // D-orthonormalize
    double mult1_num = uk.dot(firstVecD);
    uk = uk - (mult1_num/mult1_denom)*firstVec;
    double mult2_num = uk.dot(secondVecD);
    uk = uk - (mult2_num/mult2_denom)*secondVec;
  

    // Do matrix-vector product
    uk_hat = M*uk;
    uk_hat.normalize();

    num_iterations2++;

#if 0
    double residual_dot = uk.dot(uk_hat);
    if (residual_dot >= (1-eps)) {
      break;
    }
#endif

    residual = uk-uk_hat;
    double residual_norm = residual.norm(); 
    // std::cout << " " << residual_norm << std::endl;

    if (residual_norm < eps) {
      break;
    }

  }
  std::cout << "Num iterations for third eigenvector: " <<
    num_iterations2 << std::endl;

  // Save this eigenvector as well
  thirdVec = uk_hat;
  endTimerPart = std::chrono::high_resolution_clock::now();
  elt = endTimerPart - startTimerPart;
  std::cout << "Third eigenvector computation time: " << elt.count() << " s." << std::endl;

  std::cout << "Dot products of eigenvectors: " 
  << firstVec.dot(secondVec) << " "
  << firstVec.dot(thirdVec) << " " << secondVec.dot(thirdVec) <<
  std::endl;

  return 0;

}

static int 
RefineTutte(SparseMatrix<double,RowMajor>& M, 
    VectorXd& secondVec, VectorXd& thirdVec,
    int numSmoothing) {

  std::cout << "Number of smoothing rounds: " << numSmoothing << std::endl; 
  auto startTimerPart = std::chrono::high_resolution_clock::now();
  VectorXd unitVec(M.cols());;
  unitVec.setOnes();

  SparseMatrix<double,RowMajor> M2 = 2*M;
  // M2.diagonal() -= unitVec;
  M2.diagonal().setZero();
  
  for (int i=0; i<numSmoothing; i++) {
    secondVec = M2*secondVec;
    thirdVec = M2*thirdVec;
  } 
  auto endTimerPart = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elt = endTimerPart - startTimerPart;
  std::cout << "RefineTutte Time: " << elt.count() << " s." << std::endl;

  return 0; 
}

static int 
writeCoords(SparseMatrix<double,RowMajor>& M, 
    VectorXd& firstVec, VectorXd& secondVec, VectorXd& thirdVec, 
    int coarseningType, int doHDE, int refineType, 
    double eps, char *inputFilename) {

  // Write coordinates to file
  std::ofstream fout;
  std::string coordFilename(inputFilename); 
  coordFilename += "_c" + std::to_string(coarseningType) 
      + "_h" + std::to_string(doHDE) 
      + "_r" + std::to_string(refineType) 
      + "_eps" + to_string_with_precision(eps) + ".nxyz";
  std::cout << "Writing coordinates to file " << coordFilename << std::endl;
  fout.open(coordFilename);
  int n = M.cols();
  for (int i=0; i<n; i++) {
    fout << secondVec(i) << "," << thirdVec(i) << std::endl;  
  }
  fout.close();

  /* print eigenvalues */
  double firstEigenVal = (M*firstVec).cwiseQuotient(firstVec).mean();
  double firstEigenValMin =
    (M*firstVec).cwiseQuotient(firstVec).minCoeff();
  double firstEigenValMax =
    (M*firstVec).cwiseQuotient(firstVec).maxCoeff();
  double secondEigenVal = (M*secondVec).cwiseQuotient(secondVec).mean();
  double secondEigenValMin =
    (M*secondVec).cwiseQuotient(secondVec).minCoeff();
  double secondEigenValMax =
    (M*secondVec).cwiseQuotient(secondVec).maxCoeff();
  double thirdEigenVal = (M*thirdVec).cwiseQuotient(thirdVec).mean();
  double thirdEigenValMin =
    (M*thirdVec).cwiseQuotient(thirdVec).minCoeff();
  double thirdEigenValMax =
    (M*thirdVec).cwiseQuotient(thirdVec).maxCoeff();

  std::cout << "First Eigenvalue  (mean, min, max): " << firstEigenVal
    << " " << firstEigenValMin << " " << firstEigenValMax << std::endl;
  std::cout << "Second Eigenvalue (mean, min, max): " << secondEigenVal
    << " " << secondEigenValMin << " " << secondEigenValMax << std::endl;
  std::cout << "Third Eigenvalue  (mean, min, max): " << thirdEigenVal
    << " " << thirdEigenValMin << " " << thirdEigenValMax << std::endl;

  return 0;   
}


int main(int argc, char **argv) {

  if (argc != 5) {
    std::cout << "Usage: "<< argv[0] << " <csr filename> "
    "<0/1/2 (none,coarsen and continue,coarsen+stop)> <0/1 (HDE)> " 
    "<0/1/2/3 (none,Koren,Tutte,Koren+Tutte)> "
    << std::endl;
  return 1;
  }

  char *inputFilename = argv[1];

  int coarseningType = atoi(argv[2]);
  if (coarseningType == 1) {
    std::cout << "Coarsening graph and continuing" << std::endl;
  } else if (coarseningType == 2) {
    std::cout << "Coarsening and stopping" << std::endl;  
  } else {
    coarseningType = 0;
  }

  int doHDE = atoi(argv[3]);
  if (doHDE) {
    std::cout << "Running High-dimensional embedding" << std::endl;
    coarseningType = 0;
  } else {
    doHDE = 0;
  }

  int refineType = atoi(argv[4]);
  if (refineType == 0) {
    std::cout << "No eigenvector computation or refinement" << std::endl;
  } else if (refineType == 1) {
    std::cout << "Computing eigenvectors using Koren's algorithm" << std::endl;
  } else if (refineType == 2) {
    std::cout << "Refining coordinates using Tutte's algorithm" << std::endl;
  } else if (refineType == 3) {
    std::cout << "Eigenvectors followed by Tutte refinement" << std::endl;
  }

  // Read CSR file
  auto startTimer = std::chrono::high_resolution_clock::now();  
  auto startTimerPart = std::chrono::high_resolution_clock::now();  
  FILE *infp = fopen(inputFilename, "rb");
  if (infp == NULL) {
    std::cout << "Error: Could not open input file. Exiting ..." <<
    std::endl; 
    return 1;   
  }
  long n, m;
  long rest[4];
  unsigned int *rowOffsets, *adj;
  fread(&n, 1, sizeof(long), infp);
  fread(&m, 1, sizeof(long), infp);
  fread(rest, 4, sizeof(long), infp);
  rowOffsets = (unsigned int *) malloc (sizeof(unsigned int) * (n+1));
  adj = (unsigned int *) malloc (sizeof(unsigned int) * m);
  fread(rowOffsets, n+1, sizeof(unsigned int), infp);
  fread(adj, m, sizeof(unsigned int), infp);
  fclose(infp);
  auto endTimerPart = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elt = endTimerPart - startTimerPart;
  std::cout << "CSR read time: " << elt.count() << " s." << std::endl;
  std::cout << "Num edges: " << m/2 << ", vertices: " << n << std::endl;

  graph_t g;
  g.n = n; g.m = m;
  g.rowOffsets = rowOffsets;
  g.adj = adj;  

  simpleCoarsening(&g, coarseningType);
  
  VectorXd secondVecc;
  VectorXd thirdVecc;
 
  if (coarseningType > 0) {
    long n_coarse = g.n_coarse;
    SparseMatrix<double,RowMajor> Mc(n_coarse,n_coarse);
    VectorXd degreesc(n_coarse);
    loadToMatrix(Mc, degreesc, &g, coarseningType);
    VectorXd firstVecc(n_coarse);
    firstVecc.setOnes();
    firstVecc.normalize();
    secondVecc.resize(n_coarse);
    thirdVecc.resize(n_coarse);
    secondVecc.setRandom();
    if (secondVecc(0) < 0) {
      secondVecc = -secondVecc;
    }
    secondVecc.normalize();
    thirdVecc.setRandom();
    if (thirdVecc(0) < 0) {
      thirdVecc = -thirdVecc;
    }
    thirdVecc.normalize();
    double epsc = 1e-9;
    powerIterationKoren(Mc, degreesc, epsc, 
      firstVecc, secondVecc, thirdVecc, coarseningType,
        inputFilename);
    if (coarseningType == 2) {
      writeCoords(Mc, firstVecc, secondVecc, thirdVecc, 
        coarseningType, doHDE, refineType, epsc, inputFilename);
    }
  }

  // Load to matrix
  startTimerPart = std::chrono::high_resolution_clock::now();  
  SparseMatrix<double,RowMajor> M(n,n);
  VectorXd degrees(n);
  // load the full graph/matrix now
  loadToMatrix(M, degrees, &g, 0);
  endTimerPart = std::chrono::high_resolution_clock::now();
  elt = endTimerPart - startTimerPart;
  std::cout << "Matrix load time: " << elt.count() << " s." << std::endl;

  // Compute second and third eigenvectors using
  // Koren's power iteration algorithm
  VectorXd firstVec(n);
  firstVec.setOnes();
  firstVec.normalize();

  VectorXd secondVec(n);
  VectorXd thirdVec(n);

  // Initialize with previously-found coarse vectors
  if (coarseningType == 1) {
    for (long i=0; i<g.n; i++) {  
      secondVec(i) = secondVecc(g.coarseID[i]);
      thirdVec(i)  = thirdVecc(g.coarseID[i]);
    }
    secondVec.normalize();
    thirdVec.normalize();
  
  // initialize with HDE vectors  
  } else if (doHDE == 1) {
    HDE(M, &g, degrees, secondVec, thirdVec); 
  
  // Random vectors   
  } else if (doHDE == 0) {
     secondVec.setRandom();
     if (secondVec(0) < 0) {
       secondVec = -secondVec;
     }
     secondVec.normalize();
     thirdVec.setRandom();
     if (thirdVec(0) < 0) {
       thirdVec = -thirdVec;
     }
     thirdVec.normalize();
  }

  int numTutteSmoothing = 500;
  double eps = 1e-5;
  if (coarseningType != 2) {
    if (refineType == 0) {
      secondVec.normalize();
      thirdVec.normalize();
    } else if (refineType == 1) {   
      powerIterationKoren(M, degrees, eps, firstVec, secondVec, thirdVec, 
      0, inputFilename);
    } else if (refineType == 2) {
      RefineTutte(M, secondVec, thirdVec, numTutteSmoothing);
    } else if (refineType == 3) {
      RefineTutte(M, secondVec, thirdVec, numTutteSmoothing);
      powerIterationKoren(M, degrees, eps, firstVec, secondVec, thirdVec, 
      0, inputFilename);
    }
    writeCoords(M, firstVec, secondVec, thirdVec, 
          coarseningType, doHDE, refineType, 0, inputFilename);  
  }

  free(g.rowOffsets);
  free(g.adj);

  if (coarseningType > 0) {
    free(g.rowOffsetsCoarse);
    free(g.adjCoarse);
    free(g.coarseID);
    free(g.eweights);
  }
  endTimerPart = std::chrono::high_resolution_clock::now();
  elt = endTimerPart - startTimer;
  std::cout << "Overall time: " << elt.count() << " s." << std::endl;

  return 0;
}
