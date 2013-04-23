#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

typedef struct KDType {

/* defining the input positions */

    /* the buffer holding point positions  required */
    char * buffer; 
    /* number of points. required*/
    ptrdiff_t size;
    /* number of dimensions of each point. required */
    int Nd;
    /* the byte offset of the axes.  required
     * the i-th position , d-th component is
     * at i * strides[0] + d * strides[1] */
    ptrdiff_t strides[2]; 

/* the following defines how the tree is constructed */

    /* split thresh, required. 10 is good*/
    int thresh;
    /* a permutation array for sorting, required*/
    ptrdiff_t * ind; 
    /* the periodic boxsize per axis,
     * or NULL if there is no box  */
    double * boxsize;
    /* unused */
    double p;

/* the following defines the datatype of a position scalar */

    /* the byte size of each position scalar, required */
    ptrdiff_t elsize;
    /* diff returns distance of two scalars, in double,
     * if p2 is NULL, cast p1 only */
    double (* cast)(void * p1, void * data);

/* memory allocation */
    /* allocate memory */
    void * (* malloc)(size_t size);
    /* deallocate memory, size is passed in for a slab allocator */
    void (* free)(size_t size, void * ptr);

    /* data passed to cmp, middle, and diff */
    void * userdata;
} KDType;

typedef struct KDNode {
    KDType * type;
    struct KDNode * link[2];
    ptrdiff_t start;
    ptrdiff_t size;
    int dim;
    double split;
    char ext[];
} KDNode;

static KDNode * kd_alloc(KDType * type) {
    KDNode * ptr = (KDNode *)type->malloc(sizeof(KDType) + sizeof(double) * 2 * type->Nd);
    ptr->type = type;
    return ptr;
}

static inline double * kd_node_max(KDNode * node) {
    return (double*) (node->ext);
}
static inline double * kd_node_min(KDNode * node) {
    return kd_node_max(node) + node->type->Nd;
}

static inline void * kd_ptr(KDType * type, ptrdiff_t i, ptrdiff_t d) {
    i = type->ind[i];
    return & type->buffer[i * type->strides[0] + d * type->strides[1]];
}
static inline double kd_cast(KDType * type, void * p1) {
    return type->cast(p1, type->userdata);
}
static inline double kd_data(KDType * type, ptrdiff_t i, ptrdiff_t d) {
    i = type->ind[i];
    char * ptr = & type->buffer[i * type->strides[0] + d * type->strides[1]];
    return kd_cast(type, ptr);
}
static inline void * kd_swap(KDType * type, ptrdiff_t i, ptrdiff_t j) {
    ptrdiff_t t = type->ind[i];
    type->ind[i] = type->ind[j];
    type->ind[j] = t;
}

static char kd_build_split(KDNode * node, double minhint[], double maxhint[]) {
    KDType * type = node->type;
    ptrdiff_t p, q, j;
    int d;

    double * max = kd_node_max(node);
    double * min = kd_node_min(node);
    for(d = 0; d < type->Nd; d++) {
        max[d] = maxhint[d];
        min[d] = minhint[d];
    }

    if(node->size <= type->thresh) {
        node->dim = -1;
        return;
    }

    node->dim = 0;
    double longest = maxhint[0] - minhint[0];
    for(d = 1; d < type->Nd; d++) {
        double tmp = maxhint[d] - minhint[d];
        if(tmp > longest) {
            node->dim = d;
            longest = tmp;
        }
    }

    node->split = (max[node->dim] + min[node->dim]) * 0.5;

    /*
    printf("trysplit @ %g (%g %g %g %g %g %g) dim = %d, %td %td\n",
            node->split, 
            max[0], 
            max[1], 
            max[2], 
            min[0],  
            min[1],  
            min[2],  
            node->dim, node->start, node->size);
    */
    p = node->start;
    q = node->start + node->size - 1;
    while(p <= q) {
        if(kd_data(type, p, node->dim) < node->split) {
            p ++;
        } else if(kd_data(type, q, node->dim) >= node->split) {
            q --;
        } else {
            kd_swap(type, p, q); 
            p ++;
            q --;
        }
    }
    /* invariance: data[<p] < split and data[>q] >= split
     * after loop p > q.
     * thus data[0...,  p-1] < split
     * data[q + 1...., end] >= split
     * p - 1 < q + 1
     * p < q + 2
     * p > q
     * thus p = q + 1 after the loop.
     *
     * 0 -> p -1 goes to left
     * and p -> end goes to right, so that
     *  left < split
     *  and right >= split
     *  */

    /* the invariance is broken after sliding.
     * after sliding we have data[0, .... p - 1] <= split
     * and data[q +1.... end] >= split */
    if(p == node->start) {
        q = node->start;
        for(j = node->start + 1; j < node->start + node->size; j++) {
            if (kd_data(type, j, node->dim) <
                kd_data(type, node->start, node->dim)) {
                kd_swap(type, j, node->start);
            }
        }
        node->split = kd_data(type, node->start, node->dim);
        p = q + 1;
    }
    if(p == node->start + node->size) {
        p = node->start + node->size - 1;
        for(j = node->start; j < node->start + node->size- 1; j++) {
            if (kd_data(type, j, node->dim) > 
                kd_data(type, node->start + node->size - 1, node->dim)) {
                kd_swap(type, j, node->start + node->size - 1);
            }
        }
        node->split = kd_data(type, node->start + node->size - 1, node->dim);
        q = p - 1;
    }

    node->link[0] = kd_alloc(type);
    node->link[0]->start = node->start;
    node->link[0]->size = p - node->start;
    node->link[1] = kd_alloc(type);
    node->link[1]->start = p;
    node->link[1]->size = node->size - (p - node->start);
/*
    printf("will split %g (%td %td), (%td %td)\n", 
            *(double*)split, 
            node->link[0]->start, node->link[0]->size,
            node->link[1]->start, node->link[1]->size);
*/
    double midhint[type->Nd];
    for(d = 0; d < type->Nd; d++) {
        midhint[d] = maxhint[d];
    }
    midhint[node->dim] = node->split;
    kd_build_split(node->link[0], minhint, midhint);
    for(d = 0; d < type->Nd; d++) {
        midhint[d] = minhint[d];
    }
    midhint[node->dim] = node->split;
    kd_build_split(node->link[1], midhint, maxhint);
}

/* 
 * create a KD tree based on input data specified in KDType 
 * free it with kd_free
 * */
KDNode * kd_build(KDType * type) {
    ptrdiff_t i;
    double min[type->Nd];
    double max[type->Nd];    
    int d;
    for(d = 0; d < type->Nd; d++) {
        min[d] = kd_data(type, 0, d);
        max[d] = kd_data(type, 0, d);
    }
    for(i = 0; i < type->size; i++) {
        type->ind[i] = i;
        for(d = 0; d < type->Nd; d++) {
            double data = kd_data(type, i, d);
            if(min[d] > data) { min[d] = data; }
            if(max[d] < data) { max[d] = data; }
        }
    }
    KDNode * tree = kd_alloc(type);
    tree->start = 0;
    tree->size = type->size;
    kd_build_split(tree, min, max);
    return tree;
}
/**
 * free a tree
 * this is recursive
 * */
KDNode * kd_free(KDNode * node) {
    if(node->link[0]) kd_free(node->link[0]);
    if(node->link[1]) kd_free(node->link[1]);
    node->type->free(sizeof(KDNode) + node->type->elsize, node);
}

static void kd_realdiff(KDType * type, double min, double max, double * realmin, double * realmax, int d) {
    if(type->boxsize) {
        double full = type->boxsize[d];
        double half = full * 0.5;
        /* periodic */
        /* /\/\ */
        if(max <= 0 || min >= 0) {
            /* do not pass through 0 */
            min = fabs(min);
            max = fabs(max);
            if(min > max) {
                double t = min;
                min = max;
                max = t;
            }
            if(max < half) {
                /* all below half*/
                *realmin = min;
                *realmax = max;
            } else if(min > half) {
                /* all above half */
                *realmax = full - min;
                *realmin = full - max;
            } else {
                /* min below, max above */
                *realmax = half;
                *realmin = fmin(min, full - max);
            }
        } else {
            /* pass though 0 */
            min = -min;
            if(min > max) max = min;
            if(max > half) max = half;
            *realmax = max;
            *realmin = 0;
        }
    } else {
        /* simple */
        /* \/     */
        if(max <= 0 || min >= 0) {
            /* do not pass though 0 */
            min = fabs(min);
            max = fabs(max);
            if(min < max) {
                *realmin = min;
                *realmax = max;
            } else {
                *realmin = max;
                *realmax = min;
            }
        } else {
            min = fabs(min);
            max = fabs(max);
            *realmax = fmax(max, min);
            *realmin = 0;
        }
    }

}
static inline void kd_collect(KDNode * node, double * ptr) {
    /* collect all positions into a double array, 
     * so that they can be paired quickly (cache locality!)*/
    KDType * t = node->type;
    int d;
    ptrdiff_t j;

    char * base = t->buffer;
    for (j = 0; j < node->size; j++) {
        char * item = base + t->ind[j + node->start] * t->strides[0];
        for(d = 0; d < t->Nd; d++) {
            *ptr = kd_cast(t, item);
            ptr++;
            item += t->strides[1];
        }
    }

}

static void kd_enum_force(KDNode * node[2], double rmax2,
        int (*callback)(
          double r, ptrdiff_t i, ptrdiff_t j, void * data), void * data) {
    ptrdiff_t i, j;
    int d;
    KDType * t0 = node[0]->type;
    KDType * t1 = node[1]->type;
    int Nd = t0->Nd;

    double * p0base = alloca(node[0]->size * sizeof(double) * Nd);
    double * p1base = alloca(node[1]->size * sizeof(double) * Nd);
    /* collect all node[1] positions to a continue block */
    double * p1, * p0;
    double half[Nd];
    double full[Nd];
    if(t0->boxsize) {
        for(d = 0; d < Nd; d++) {
            half[d] = t0->boxsize[d] * 0.5;
            full[d] = t0->boxsize[d];
        }
    }

    kd_collect(node[0], p0base);
    kd_collect(node[1], p1base);

    double bad = rmax2 * 2 + 1;
    for (p0 = p0base, i = 0; i < node[0]->size; i++) {
        for (p1 = p1base, j = 0; j < node[1]->size; j++) {
            double r2 = 0.0;
            for (d = 0; d < Nd; d++){
                double dx = p1[d] - p0[d];
                if (dx < 0) dx = - dx;
                if (t0->boxsize) {
                    if (dx > half[d]) dx = full[d] - dx;
                }
                /*
                if (dx > maxr) {
                    r2 = bad;
                    p1 += Nd - d;
                    break;
                } */
                r2 += dx * dx;
            }
            if(r2 <= rmax2) {
                callback(pow(r2, 0.5), t0->ind[i + node[0]->start], 
                        t1->ind[j + node[1]->start], data);
            }
            p1 += Nd;
        }
        p0 += Nd;
    }
}
/*
 * enumerate two KDNode trees, up to radius max.
 *
 * for each pair i in node[0] and j in node[1],
 * if the distance is smaller than maxr,
 * call callback.
 * */
void kd_enum(KDNode * node[2], double maxr,
        int (*callback)(
          double r, ptrdiff_t i, ptrdiff_t j, void * data), void * data) {
    int Nd = node[0]->type->Nd;
    double distmax = 0, distmin = 0;
    double rmax2 = maxr * maxr;
    int d;
    double *min0 = kd_node_min(node[0]);
    double *min1 = kd_node_min(node[1]);
    double *max0 = kd_node_max(node[0]);
    double *max1 = kd_node_max(node[1]);
    for(d = 0; d < Nd; d++) {
        double min, max;
        double realmin, realmax;
        min = min0[d] - max1[d];
        max = max0[d] - min1[d];
        kd_realdiff(node[0]->type, min, max, &realmin, &realmax, d);
        distmin += realmin * realmin;
        distmax += realmax * realmax;
    }
    /*
    printf("%g %g %g \n", distmin, distmax, maxr * maxr);
    print(node[0]);
    print(node[1]);
    */
    if (distmin > rmax2) {
        /* nodes are too far, skip them */
        return;
    }
    if (distmax >= rmax2) {
        /* nodes may intersect, open them */
        int open = node[0]->size < node[1]->size;
        if(node[open]->dim < 0) {
            open = (open == 0);
        }
        if(node[open]->dim >= 0) {
            KDNode * save = node[open];
            node[open] = save->link[0];
            kd_enum(node, maxr, callback, data);
            node[open] = save->link[1];
            kd_enum(node, maxr, callback, data);
            node[open] = save;
            return;
        } else {
            /* can't open the node, need to enumerate */
        }
    } else {
        /* fully inside, fall through,
         * and enumerate  */
    }

    kd_enum_force(node, rmax2, callback, data);

}

