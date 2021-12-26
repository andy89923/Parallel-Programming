__kernel void convolution(
   	const __global float* inp_dat, __global float4* oup_dat, __constant float* fil_dat,
   	const int imageHeight, constant int imageWidth, constant int half_fitr) {

	int gid = get_global_id(0);
	int idx = gid / 4;

	int poi_x = idx / imageWidth;
	int poi_y = idx % imageWidth;

	float4 ans = 0.0;

	int filter_idx = 0;
	for (int i = -half_fitr; i <= half_fitr; ++i) {

		int now_x = poi_x + i;
		if (now_x < 0 || now_x > imageHeight) continue;

		for (int j = -half_fitr; j <= half_fitr; ++j, ++filter_idx) {

			int now_y = poi_y + j;
			if (now_y < 0 || now_y > imageWidth) continue;

			int poi = now_y + now_x * imageWidth;
			
			float4 tmp = (float4)(inp_dat[poi], inp_dat[poi+1], inp_dat[poi+2], inp_dat[poi+3]);
			float4 fil = fil_dat[filter_idx];

			ans += tmp * fil;
		}
	}
	oup_dat[gid] = ans;
}