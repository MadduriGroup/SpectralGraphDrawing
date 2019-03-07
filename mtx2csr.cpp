/****
 * Generate CSR file from MTX File
 * If the graph is directed then makes it undirected and also delete repeated edges
 * This WORKS with: 1) Zero degree vertex (missing vertex)
 * 		    2) Without nodes and edges count
 * 		    3) This also randomize the vertices
*****/


#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <ctime>

using namespace std;

int main(int argc, char *argv[])
{

    if( argc != 3 ) {
	cout<<"usage: "<<argv[0]<<" <mtx-file> <graph-name>\n";
	exit(1);
    }

    FILE *inFilePtr = fopen( argv[1], "r");
   
    if( inFilePtr == NULL ) {
	cout<<"Can not openfile :"<< argv[1]<<endl;
	exit(1);
    }
    
    long N = 0, M = 0;
    long undirected , verification_graph = 0, graph_type = 0;
    long one_indexed = 0;
    unsigned int *adj;
    unsigned int *num_edges;

    int curr_vertex = 1, temp_vertex = 0;

    char line [ 200 ];
    int lineCount = 0;

    int symmetric = 0;
 
    //This is the adj of the graph
    vector< vector<unsigned int> > AdjVector;
 
/*****************************************************************************************/
/* Count the number of edges for each vertex  */

    while ( fgets ( line, sizeof line, inFilePtr  ) != NULL ) {
	if( line[0] == '%' ) {
	    if( line[1] == '%' ) {
		string str ( line );
		string str2 ("symmetric");
		size_t found;

		found = str.find( str2 );
		if ( found != string::npos ) {
		    symmetric = 1; 
		} else { 
		    symmetric =0;
		    undirected = symmetric ;
		}
	    } 
	    continue;
	}
          

	lineCount ++;

	if( lineCount == 1 ) {
 	  
	    sscanf( line, "%ld %ld %ld\n", &N,&N,&M);
    //        fprintf( stderr, "%ld %ld %ld\n", N,N,M);

            AdjVector.resize(N); 
	}
	else {
	    sscanf( line, "%d %d\n", &temp_vertex, &curr_vertex);
	    if( temp_vertex != curr_vertex )  {
		AdjVector[temp_vertex - 1].push_back( curr_vertex-1 );
		AdjVector[curr_vertex - 1].push_back( temp_vertex-1 );
	    }
	}
    }


    fclose ( inFilePtr );

    undirected = symmetric;

    //Sort the adj list of vertices
    for(long i = 0; i < N; i++) 
	sort( AdjVector[i].begin(), AdjVector[i].end() );

    //Get rid of the parallel edges
    for(long i = 0; i < N; i++)  {
	vector< unsigned int > adjVector;

	if( AdjVector[i].size() > 1 ) {
	    unsigned int prevVtx = AdjVector[i][0];
	    adjVector.push_back( prevVtx );

	    for(unsigned int len = 1; len < AdjVector[i].size(); len ++) {
		unsigned int currVtx = AdjVector[i][len];	
		if( currVtx != prevVtx ) {
		    prevVtx = currVtx;
		    adjVector.push_back( prevVtx );
		}
	    } 	
   	    AdjVector[i] = adjVector; 
	}  	
    }
/*
    //randomize the vertices
    vector< long > vertices; 
    for(long i = 0; i < N; i++)
	vertices.push_back( i );

    std::srand ( unsigned ( std::time(0) ) );
    std::random_shuffle ( vertices.begin(), vertices.end() );

    //vertices gives newId to oldId map
    vector< long > oldToNewId(N,0);
   
    for(long i = 0; i < N; i++) {
	oldToNewId[ vertices[i]  ] = i;
    }

    M = 0;
    for(int i = 0; i < N; i++) {
	M += AdjVector[i].size();

	for(int j = 0; j < AdjVector[i].size(); j++) {
	    AdjVector[i][j] = oldToNewId[ AdjVector[i][j] ];
	}
    } 
*/
    //Sort the adjacencies
    for(long i = 0; i < N; i++)
        sort( AdjVector[i].begin(), AdjVector[i].end() );


    M = 0;
    for(int i = 0; i < N; i++) {
        M += AdjVector[i].size();
    } 

    //cout<<"2*#edges: " << M << endl;
 
    num_edges = (unsigned int *) malloc( (N+1)*sizeof(unsigned int) );
    assert( num_edges != NULL );

    adj = (unsigned int *) malloc( M * sizeof(unsigned int) );
    assert( adj != NULL );
    num_edges[0] = 0;

    long cumEdges = 0;
    long index = 0;

    //Find oldId given newId
    for(long i = 0; i < N; i++) {
	for(unsigned int j =0; j < AdjVector[i].size(); j++) {
	    adj[ index ] = AdjVector[i][j];
	    index ++;
	}

	cumEdges += AdjVector[i].size();	
	num_edges[i+1] = cumEdges;
    }

    // Write to CSR file  
    char outFileName[256];

    sprintf(outFileName, "%s.csr", argv[2]);

    FILE *writeBinaryPtr = fopen( outFileName, "wb");
    if ( writeBinaryPtr == NULL ) {
	fprintf(stderr, "could not open file: %s\n", outFileName);
	exit(1);
    }


    fwrite ( &N, sizeof(long), 1, writeBinaryPtr );
    fwrite ( &M, sizeof(long), 1, writeBinaryPtr );
    fwrite ( &undirected, sizeof(long), 1, writeBinaryPtr );
    fwrite ( &graph_type , sizeof(long), 1, writeBinaryPtr );
    fwrite ( &one_indexed , sizeof(long), 1, writeBinaryPtr );
    fwrite ( &verification_graph , sizeof(long), 1, writeBinaryPtr );

    fwrite ( num_edges, sizeof(unsigned int), (N+1), writeBinaryPtr );
    fwrite ( adj, sizeof(unsigned int), M, writeBinaryPtr );

    //fprintf(stdout, "Done with writing to file: %s \n", outFileName);
    fclose( writeBinaryPtr );

    free(num_edges);
    free(adj);

    return 0;
}


