kernel void Interaction_AdhesivePlane(volatile global ibmPrecisionFloat* particleForce, const global ibmPrecisionFloat* points, const ibmPrecisionFloat WallPos, const ibmPrecisionFloat adhesionConst, const global int* pointOnSurface){
	const uint pointID = get_global_id(0);
	if(pointID>=INSERT_NUM_POINTS) return;
	if(pointOnSurface[pointID] != 1) return; // if point is not on surface dont apply force

	ibmPrecisionFloat dist = points[INSERT_NUM_POINTS + pointID] -  WallPos;
	ibmPrecisionFloat force = exp(-forceConst * dist) - adhesionConst*exp(-dist);
	particleForce[INSERT_NUM_POINTS + pointID] += force;
}
