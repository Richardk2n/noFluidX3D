kernel void Interaction_PlaneAFM(volatile global ibmPrecisionFloat* particleForce, const global ibmPrecisionFloat* points, const ibmPrecisionFloat topWallPos, const ibmPrecisionFloat bottomWallPos, const ibmPrecisionFloat forceConst){
	const uint pointID = get_global_id(0);
	if(pointID>=INSERT_NUM_POINTS) return;
	ibmPrecisionFloat disTop = topWallPos - points[INSERT_NUM_POINTS + pointID];
	ibmPrecisionFloat disBottom = points[INSERT_NUM_POINTS + pointID] -  bottomWallPos;
	ibmPrecisionFloat force = exp(-forceConst * disBottom) - exp(-forceConst * disTop);
	particleForce[INSERT_NUM_POINTS + pointID] += force;
}
