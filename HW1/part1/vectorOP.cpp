#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N) {
    __pp_vec_float x;
    __pp_vec_float result;
    __pp_vec_float zero = _pp_vset_float(0.f);
    __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

    //  Note: Take a careful look at this loop indexing.  This example
    //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
    //  Why is that the case?
    for (int i = 0; i < N; i += VECTOR_WIDTH) {

        // All ones
        maskAll = _pp_init_ones();

        // All zeros
        maskIsNegative = _pp_init_ones(0);

        // Load vector of values from contiguous memory addresses
        _pp_vload_float(x, values + i, maskAll); // x = values[i];

        // Set mask according to predicate
        _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

        // Execute instruction using mask ("if" clause)
        _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

        // Inverse maskIsNegative to generate "else" mask
        maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

        // Execute instruction ("else" clause)
        _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

        // Write results back to memory
        _pp_vstore_float(output + i, result, maskAll);
    }
}

void clampedExpVector(float *values, int *exponents, float *output, int N) {
    //
    // PP STUDENTS TODO: Implement your vectorized version of
    // clampedExpSerial() here.
    //
    // Your solution should work for any value of
    // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
    //

    // result = x ^ y 

    float max_num = 9.999999;

    __pp_vec_float x;
    __pp_vec_float result;
    __pp_vec_int y, zero = _pp_vset_int(0), one = _pp_vset_int(1);
    __pp_vec_float maxNum = _pp_vset_float(max_num);

    __pp_mask maskAll, maskMaxNum, maskNow, maskOver, maskNxt;


    for (int i = 0; i < N; i += VECTOR_WIDTH) {

        if (N < i + VECTOR_WIDTH) {
            maskAll = _pp_init_ones(N - i);
        }
        else {
            maskAll = _pp_init_ones();
        }
        addUserLog("Init finished");
        
        maskMaxNum = _pp_init_ones(0);
        maskOver = _pp_init_ones(0);
        maskNow = _pp_init_ones(0);
        maskNxt = _pp_init_ones(0);
        
        _pp_vload_float(x, values + i, maskAll);     // load x
        _pp_vload_int(y, exponents + i, maskAll);    // load y


        _pp_vgt_int(maskNow, y, zero, maskAll);
        _pp_vset_float(result, 1.0, maskAll); 

        while (_pp_cntbits(maskNow) > 0) {

            _pp_vmult_float(result, result, x, maskNow);

            _pp_vsub_int(y, y, one, maskNow);

            _pp_vgt_int(maskNxt, y, zero, maskNow);
            maskNow = maskNxt;
        }
        _pp_vgt_float(maskOver, result, maxNum, maskAll);

        _pp_vset_float(result, max_num,  maskOver);

        _pp_vstore_float(output + i, result, maskAll);
    }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N) {

    //
    // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
    //
    
    float sum = 0.0;
    int m = 0, tmp = VECTOR_WIDTH;
    while (tmp % 2 == 0) { m += 1; tmp >>= 1; }

    __pp_vec_float x, y;
    __pp_mask maskAll;

    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        maskAll = _pp_init_ones();

        _pp_vload_float(x, values + i, maskAll); // load x
        for (int j = 0; j < m; j++) {

            _pp_hadd_float(y, x);
            _pp_interleave_float(x, y);
        }
        sum += x.value[0];
    }

    return sum;
}
