kernel void Interaction_Plane(volatile global ibmPrecisionFloat* particleForce, const global ibmPrecisionFloat* points, const ibmPrecisionFloat wallPos){
	const uint pointID = get_global_id(0);
	if(pointID>=INSERT_NUM_POINTS) return;
	ibmPrecisionFloat dis = max(points[INSERT_NUM_POINTS + pointID]-wallPos, 0.);
	ibmPrecisionFloat force = -def_FORCE_CONST*dis;
	particleForce[INSERT_NUM_POINTS + pointID] += force;
}
