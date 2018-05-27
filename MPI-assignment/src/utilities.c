#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "utilities.h"

struct buff {
    size_t size;
    size_t max_size;
    double *buff;    
};

static void init_buff(struct buff *a, size_t size) {
    a->buff = (double*)malloc(size * sizeof(double));
    a->size = 0; 
    a->max_size = size;
}

static void insert_buff(struct buff *a, double el) {
    if (a->size == a->max_size) {
        a->max_size *= 2;
        a->buff = (double *)realloc(a->buff, a->max_size * sizeof(double*));
    }
    a->buff[a->size] = el;
    a->size++;
}

static void free_buff(struct buff *a) {
    free(a->buff);
    a->buff = 0;
    a->max_size = 0;
    a->size = 0;
}

struct particle * parse_input(char *filename, int *count) {

    double x;
    struct buff buff;
    struct particle *p;

    init_buff(&buff, 10);

    FILE *fp = fopen(filename, "r");
    
    while (fscanf(fp, "%lf", &x) != EOF) {
        insert_buff(&buff, x);
    }

    *count = buff.size / 6;

    p = calloc(*count, sizeof(struct particle));

    for (int i = 0; i < *count; ++i) {
        p[i].x = buff.buff[6*i];
        p[i].y = buff.buff[6*i+1];
        p[i].z = buff.buff[6*i+2];
        p[i].vx = buff.buff[6*i+3];
        p[i].vy = buff.buff[6*i+4];
        p[i].vz = buff.buff[6*i+5];
    }

    free_buff(&buff);
    fclose(fp);

    return p;
}

int write_output(char *filename, int count, struct particle *p) {
    
    FILE *fp = fopen(filename, "w+");

    for (int i = 0; i < count; ++i, ++p) {
        
        if (i)
            fprintf(fp, "\n");

        fprintf(fp, "%.15lf %.15lf %.15lf %.16lf %.16lf %.16lf", p->x, p->y, p->z, p->vx, p->vy, p->vz);
    }

    fclose(fp);

    return 0;
}

int write_log(char *filename, int count, struct particle *p, int step) {
    char name_buff[256];
    char *ext = strchr(filename, '.');
    int name_len = strlen(filename) - strlen(ext);
    
    memcpy(name_buff, filename, name_len);
    sprintf(name_buff + name_len, "-%d%s", step, ext);

    return write_output(name_buff, count, p);
}

int parse_args(int argc, char *argv[], struct cmd_args *cmd_args) {
    
    if (argc < 5) {
        printf("Error");
        exit(0);
    }

    cmd_args->particles_in = argv[1];
    cmd_args->particles_out = argv[2];
    cmd_args->stepcount = atoi(argv[3]);
    cmd_args->deltatime = atof(argv[4]);

    if (argc == 5) {
        cmd_args->verbose = 0;
    } else {
        cmd_args->verbose = 1;
    }

    return 0;
}