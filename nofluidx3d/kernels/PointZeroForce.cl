kernel void PointZeroForce(volatile global ibmPrecisionFloat* particleForce, const global ibmPrecisionFloat* points, const ibmPrecisionFloat k){
	const uint pointID = get_global_id(0);
	if(pointID>=INSERT_NUM_POINTS) return;
	// No atomicAdd, because force on each point can be calculated independently -> probably faster!
	//if(pointID == 3){
	//	printf("PointZeroForce = - k * r = - %f * [%f, 0, %f]\n", k, points[0], points[2*INSERT_NUM_POINTS]);
	//}
	particleForce[pointID] += -k * points[0];
	particleForce[2*INSERT_NUM_POINTS + pointID] += -k * points[2*INSERT_NUM_POINTS];
}
