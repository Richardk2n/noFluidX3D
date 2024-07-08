kernel void Interaction_AdhesivePlane(volatile global ibmPrecisionFloat* particleForce, const global ibmPrecisionFloat* points, const ibmPrecisionFloat WallPos, const ibmPrecisionFloat adhesionConst, const ibmPrecisionFloat distance, const global int* pointOnSurface){
	const uint pointID = get_global_id(0);
	if(pointID>=INSERT_NUM_POINTS) return;
	if(pointOnSurface[pointID] != 1) return; // if point is not on surface dont apply force
	const ibmPrecisionFloat rm = distance;
	const ibmPrecisionFloat sigma = pow(0.5, 1.0/6.0) * rm;
	
	const ibmPrecisionFloat3 pos = (ibmPrecisionFloat3)(points[pointID], points[INSERT_NUM_POINTS + pointID], points[2*INSERT_NUM_POINTS + pointID]);
	const ibmPrecisionFloat d = pos.y - WallPos;

	const ibmPrecisionFloat force = 24.0 * adhesionConst / d * (2*pow(sigma/d, 12) - pow(sigma/d, 6));

	particleForce[INSERT_NUM_POINTS + pointID] += force;
}
