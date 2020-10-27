#ifndef DECLARES_H
#define DECLARES_H
#include "xtiffio.h"
#include "geotiffio.h"
#include "stdint.h"

struct rgb_struct
{
    uint8_t red;
    uint8_t green;
    uint8_t blue;
};

extern "C" void kern(uint32* data, uint32 w, uint32 h, uint32* ans, int* kernel);
extern "C" void resampling(uint32_t * data, uint32_t w, uint32_t h, uint32_t * ans, uint32_t ans_w, uint32_t ans_h, double q);
extern "C" void singular(uint32_t * data, uint32_t w, uint32_t h, uint32_t * ans, int func);
extern "C" void intersec(uint32_t * data, uint32_t data_x0, uint32_t data_x1, uint32_t data_y0, uint32_t data_y1, uint32_t w, uint32_t h,
                        uint32_t * ans,
                        uint32_t * data1, uint32_t data1_x0, uint32_t data1_x1, uint32_t data1_y0, uint32_t data1_y1, uint32_t w1, uint32_t h1, int type);


#endif // DECLARES_H
