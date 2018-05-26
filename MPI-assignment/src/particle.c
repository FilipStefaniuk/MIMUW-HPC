#include <math.h>
#include <stdio.h>
#include "particle.h"

double norm_dist(const struct particle *p1, const struct particle *p2) {
    
    double dx, dy, dz;

    dx = p1->x - p2->x;
    dy = p1->y - p2->y;
    dz = p1->z - p2->z;

    return fmax(1e-10, sqrt(dx * dx + dy * dy + dz * dz));
}

double axilrod_teller(const struct particle  *i, const struct particle *j, const struct particle *k) {
    
    double r_ij, r_ik, r_kj;
    double rsq_ij, rsq_ik, rsq_kj;
    double mul, num;

    double tmp;

    r_ij = norm_dist(i, j);
    r_ik = norm_dist(i, k);
    r_kj = norm_dist(k, j);

    rsq_ij = r_ij * r_ij;
    rsq_ik = r_ik * r_ik;
    rsq_kj = r_kj * r_kj;

    mul = r_ij * r_ik * r_kj;

    num = 3 * (-rsq_ij + rsq_ik + rsq_kj) 
            * (rsq_ij - rsq_ik + rsq_kj) 
            * (rsq_ij + rsq_ik - rsq_kj);
    
    return (1 / pow(mul, 3.0)) + (num / (8 * pow(mul, 5.0)));
    
    // printf("%.16lf\n", tmp);
    // return tmp;
}

double calculate_h(double pos) {
    double e = 4.69041575982343e-08;
    if(fabs(pos) < 1e-10) {
        pos = 1e-10;
    }
    return e * pos;
}

void compute_force(struct particle *i, const struct particle *j, const struct particle *k) {
    
    double hx, hy, hz;
    double dx, dy, dz;
    double f1x, f2x, f1y, f2y, f1z, f2z;
    double tmp;

    hx = calculate_h(i->x);
    hy = calculate_h(i->y);
    hz = calculate_h(i->z);
    

    // printf("%.16lf %.16lf\n", i->x, hx);

    dx = (i->x + hx) - (i->x - hx);
    dy = (i->y + hy) - (i->y - hy);
    dz = (i->z + hz) - (i->z - hz);

    tmp = i->x;
    i->x = tmp + hx;
    f1x = axilrod_teller(i, j, k);

    i->x = tmp - hx; 
    f2x = axilrod_teller(i, j, k);
    i->x = tmp;

    tmp = i->y;
    i->y = tmp + hy;
    f1y = axilrod_teller(i, j, k);

    i->y = tmp - hy;
    f2y = axilrod_teller(i, j, k);
    i->y = tmp;

    tmp = i->z;
    i->z = tmp + hz;
    f1z = axilrod_teller(i, j, k);

    i->z = tmp - hz;
    f2z = axilrod_teller(i, j, k);
    i->z = tmp;
    
    // printf("%.16lf %.16lf\n", f1x, f2x);
    // printf("%.16lf\n", 2 * (f1x - f2x) / dx);


    i->fx -= 2 * (f1x - f2x) / dx;
    i->fy -= 2 * (f1y - f2y) / dy;
    i->fz -= 2 * (f1z - f2z) / dz;
}

void update_position(struct particle *p, double deltatime) {

    p->x += p->vx * deltatime + p->ax * deltatime * deltatime / 2;
    p->y += p->vy * deltatime + p->ay * deltatime * deltatime / 2;
    p->z += p->vz * deltatime + p->az * deltatime * deltatime / 2;
}

void update_velocity(struct particle *p, double deltatime) {

    p->vx += (p->ax + p->fx) * deltatime / 2;
    p->vy += (p->ay + p->fy) * deltatime / 2;
    p->vz += (p->az + p->fz) * deltatime / 2;
}

void update_acceleration(struct particle *p) {

    p->ax = p->fx;
    p->ay = p->fy;
    p->az = p->fz;

    p->fx = 0;
    p->fy = 0;
    p->fz = 0;
}