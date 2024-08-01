kernel void COMForce(volatile global ibmPrecisionFloat* particleForce, const global ibmPrecisionFloat* points, const ibmPrecisionFloat k, const global ibmPrecisionFloat* com){
	const uint pointID = get_global_id(0);
	if(pointID>=INSERT_NUM_POINTS) return;
	// No atomicAdd, because force on each point can be calculated independently -> probably faster!
	particleForce[pointID] += -k * com[0];
	particleForce[2*INSERT_NUM_POINTS + pointID] += -k * com[1];
}
