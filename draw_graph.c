/** @file  draw_graph.c
 * 
 *  @brief Given vertex coordinates, generate a bitmap image with edges.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "lodepng.h"

typedef struct graph {
    uint32_t n;
    int64_t  m;
    uint32_t *offsets;
    uint32_t *adj;
} graph_t;

static
double
timer()
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double) (tp.tv_sec + ((1e-6)*tp.tv_usec));
#endif
}

static
void 
draw_line(int x0, int y0, int x1, int y1, unsigned char *image, unsigned width) {

  /* Bresenham's line algorithm */
  /* Code from https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm */

  int dx = abs(x1-x0), sx = x0<x1 ? 1 : -1;
  int dy = abs(y1-y0), sy = y0<y1 ? 1 : -1; 
  int err = (dx>dy ? dx : -dy)/2;

  for(;;){
    image[4 * width * y0 + 4 * x0 + 0] = 0;
    image[4 * width * y0 + 4 * x0 + 1] = 0;
    image[4 * width * y0 + 4 * x0 + 2] = 0;
    image[4 * width * y0 + 4 * x0 + 3] = 255;
    if (x0==x1 && y0==y1) break;
    int e2 = err;
    if (e2 >-dx) { err -= dy; x0 += sx; }
    if (e2 < dy) { err += dx; y0 += sy; }
  }

}

int
main (int argc, char **argv)
{

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <binary csr filename> <vertex coords filename> <output png filename>\n", argv[0]);
        return 1;
    }

    char *input_filename = argv[1];

    // read input graph file
    double start_timer = timer();  
    FILE *infp = fopen(input_filename, "rb");
    if (infp == NULL) {
        fprintf(stderr, "Error: Could not open CSR file. Exiting ...\n");
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
    assert(rowOffsets != NULL);
    assert(adj != NULL);
    fread(rowOffsets, n+1, sizeof(unsigned int), infp);
    fread(adj, m, sizeof(unsigned int), infp);
    fclose(infp);
    double end_timer = timer();
    double elt = end_timer - start_timer;
    fprintf(stderr, "CSR file read time: %9.6lf s\n", elt);

    graph_t g;
    g.n = n;
    g.m = m/2;
    g.offsets = rowOffsets;
    g.adj = adj;
    fprintf(stderr, "Num edges: %ld, num vertices: %u\n", m/2, g.n);

    char *vertex_coords_filename = argv[2];
    infp = fopen(vertex_coords_filename, "r");
    if (infp == NULL) {
        fprintf(stderr, "Error: Could not open vertex coords file. Exiting ...\n");
        return 1;
    }

    unsigned width = 1920, height = 1080;

    double *vx = (double *) malloc(g.n * sizeof(double));
    assert(vx != NULL);
    double *vy = (double *) malloc(g.n * sizeof(double));
    assert(vy != NULL);

    double xmax = -1;
    double xmin = 1;
    double ymax = -1;
    double ymin = 1;

    for (uint32_t i=0; i<g.n; i++) {
        double x; double y;
        fscanf(infp, "%lf,%lf", &x, &y);
        vx[i] = x; vy[i] = y;
        if (x > xmax) xmax = x;
        if (x < xmin) xmin = x;
        if (y > ymax) ymax = y;
        if (y < ymin) ymin = y; 
    }
    fclose(infp);

    double aspect_ratio = (xmax-xmin)/(ymax-ymin);
    fprintf(stderr, "xmin %lf xmax %lf ymin %lf ymax %lf, aspect ratio %3.4lf\n", 
                    xmin, xmax, ymin, ymax, aspect_ratio);
    width = sqrt(100*n/aspect_ratio);
    height = aspect_ratio*width;
    fprintf(stderr, "Computed width %u, height %u\n", width, height);


    double xrange_inv = 1.0/(xmax-xmin);
    double yrange_inv = 1.0/(ymax-ymin);
    for (uint32_t i=0; i<g.n; i++) {
        vx[i] = (-xmin + vx[i])*xrange_inv*width;
        vy[i] = (-ymin + vy[i])*yrange_inv*height;
        if (vx[i] < 0) vx[i] = 0;
        if (vy[i] < 0) vy[i] = 0;
        /* handle cases where we might exceed bounds */
        if (vx[i] > (width-1)) vx[i] = width-1;
        if (vy[i] > (height-1)) vy[i] = height-1;
    }

    char *out_filename = argv[3];
    unsigned char* image = malloc(width * height * 4);

    /* white background */
    unsigned x, y;
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            image[4 * width * y + 4 * x + 0] = 255;
            image[4 * width * y + 4 * x + 1] = 255;
            image[4 * width * y + 4 * x + 2] = 255;
            image[4 * width * y + 4 * x + 3] = 255;
        }
    }

    // Commenting out SVG generation code
    // FILE *outfp = fopen("graph.svg", "w");
    // fprintf(outfp, "<svg viewBox=\"0 0 %d %d\" xmlns=\"http://www.w3.org/2000/svg\">\n<g stroke=\"black\">\n", width, height);

    for (uint32_t u=0; u<g.n; u++) {
        for (int j=g.offsets[u]; j<g.offsets[u+1]; j++) {
            uint32_t v = g.adj[j];
            if (u < v) {
                // fprintf(outfp, "<line x1=\"%u\" y1=\"%u\" x2=\"%u\" y2=\"%u\" />\n", 
                // ((unsigned) floor(vx[u])), ((unsigned) floor(vy[u])), ((unsigned) floor(vx[v])), ((unsigned) floor(vy[v])));
                draw_line(floor(vx[u]), floor(vy[u]), floor(vx[v]), floor(vy[v]), image, width);
            }
        }
    }
    // fprintf(outfp, "</g>\n</svg>\n");
    // fclose(outfp);
 
    unsigned error = lodepng_encode32_file(out_filename, image, width, height);
    /*if there's an error, display it*/
    if (error) 
        printf("error %u: %s\n", error, lodepng_error_text(error));

    free(g.offsets);
    free(g.adj);

    free(vx);
    free(vy);

    free(image);

    return 0;
}
