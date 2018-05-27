#ifndef __PARTICLE_H__
#define __PARTICLE_H__

struct particle {
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
    double fx, fy, fz;
};

void compute_force(struct particle *i, const struct particle *j, const struct particle *k);

void update_position(struct particle *p, double deltatime);
void update_velocity(struct particle *p, double deltatime);
void update_acceleration(struct particle *p);

#endif
