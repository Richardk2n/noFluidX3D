kernel void Interaction_Substrate(volatile global ibmPrecisionFloat* particleForce, const global ibmPrecisionFloat* points, const ibmPrecisionFloat wallPos, const ibmPrecisionFloat forceConst){
	const uint pointID = get_global_id(0);
	if(pointID>=INSERT_NUM_POINTS) return;
	ibmPrecisionFloat disBottom = max(wallPos-points[INSERT_NUM_POINTS + pointID], 0.);
	ibmPrecisionFloat force = forceConst*disBottom;
	particleForce[INSERT_NUM_POINTS + pointID] += force;
}
