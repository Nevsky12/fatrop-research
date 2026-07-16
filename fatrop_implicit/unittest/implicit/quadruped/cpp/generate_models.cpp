#include "quadruped_helper_pinocchio.hpp"

int main(){
    PinocchioCasadi pc = PinocchioCasadi(0.05);
    pc.SimulateFalling();
    return 0;
}