kernel void Interaction_PlaneAFM(volatile global ibmPrecisionFloat* particleForce, const global ibmPrecisionFloat* points, const ibmPrecisionFloat topWallPos, const ibmPrecisionFloat bottomWallPos, const ibmPrecisionFloat forceConst){
	const uint pointID = get_global_id(0);
	if(pointID>=INSERT_NUM_POINTS) return;
	ibmPrecisionFloat disTop = max(points[INSERT_NUM_POINTS + pointID]-topWallPos, 0.);
	ibmPrecisionFloat disBottom = max(bottomWallPos-points[INSERT_NUM_POINTS + pointID], 0.);
	ibmPrecisionFloat pref = 0.1;
	ibmPrecisionFloat force = pref*(disBottom - disTop);
	particleForce[INSERT_NUM_POINTS + pointID] += force;
}
