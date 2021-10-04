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
    __pp_mask maskAll, maskMaxNum, maskExpZero, maskNow, maskOver;


    for (int i = 0; i < N; i += VECTOR_WIDTH) {

        if (N < i + VECTOR_WIDTH)
            maskAll = _pp_init_ones(N - i);


        _pp_vload_float(y, exponents + i, maskAll);  // load y

        _pp_veq_int(maskExpZero, y, zero, maskAll);  // if y == 0:
        _pp_vset_float(result, 0.0, maskExpZero);    //     result = 0 

        maskNow = _pp_mask_not(maskExpZero);
        _pp_vset_float(result, 1.0, maskNow); 
 
        while (_pp_cntbits(maskNow) > 0) {

            _pp_vmult_float(result, result, x, maskNow);

            _pp_vsub_int(y, y, one, maskNow);

            _pp_vgt_int(y, y, zero, maskNow);
        }
        _pp_vgt_float(maskOver, y, maxNum, maskAll);

        _pp_vset_float(result, y, max_num, maskOver);

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

    for (int i = 0; i < N; i += VECTOR_WIDTH) {
    

    }

    return 0.0;
}