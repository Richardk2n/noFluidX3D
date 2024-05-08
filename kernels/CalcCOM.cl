kernel void CalcCOM(const global ibmPrecisionFloat* points, volatile global ibmPrecisionFloat* com){
	const uint ID = get_global_id(0);
	ibmPrecisionFloat sum = 0.0;
	for(int i=0; i<INSERT_NUM_POINTS; i++){
		sum += points[2 * ID * INSERT_NUM_POINTS + i];
	}
	com[ID] = sum / (ibmPrecisionFloat)INSERT_NUM_POINTS;
}
