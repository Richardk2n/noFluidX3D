kernel void OverDamped(volatile global ibmPrecisionFloat* points, volatile global ibmPrecisionFloat* particleForce){
	const uint pointID = get_global_id(0);
	if(pointID >= INSERT_NUM_POINTS) return;
	ibmPrecisionFloat3 x = (ibmPrecisionFloat3)(points[pointID], points[INSERT_NUM_POINTS + pointID], points[2*INSERT_NUM_POINTS + pointID]);
	ibmPrecisionFloat3 f = (ibmPrecisionFloat3)(particleForce[pointID], particleForce[INSERT_NUM_POINTS + pointID], particleForce[2*INSERT_NUM_POINTS + pointID]);

	x = x + f;

	
	points[0*INSERT_NUM_POINTS + pointID] = x.x;
	points[1*INSERT_NUM_POINTS + pointID] = x.y;
	points[2*INSERT_NUM_POINTS + pointID] = x.z;
	particleForce[0*INSERT_NUM_POINTS + pointID] = 0.0;
	particleForce[1*INSERT_NUM_POINTS + pointID] = 0.0;
	particleForce[2*INSERT_NUM_POINTS + pointID] = 0.0;
}
