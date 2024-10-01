kernel void Interaction_RoundTip(volatile global ibmPrecisionFloat* particleForce, const global ibmPrecisionFloat* points, const ibmPrecisionFloat radius, const ibmPrecisionFloat spherePos, const ibmPrecisionFloat forceConst){
	const uint pointID = get_global_id(0);
	if(pointID>=INSERT_NUM_POINTS) return;
	// No atomicAdd, because force on each point can be calculated independently -> probably faster!
	// Force from spherical indenter
	const ibmPrecisionFloat3 pos = (ibmPrecisionFloat3)(points[pointID], points[INSERT_NUM_POINTS + pointID], points[2*INSERT_NUM_POINTS + pointID]);
	const ibmPrecisionFloat3 sphere = (ibmPrecisionFloat3)(0, spherePos, 0);

	ibmPrecisionFloat3 r = pos - sphere;
	if(pos.y > spherePos) {
		r.y = 0;
	}
	const ibmPrecisionFloat3 n = r / length(r);
	const ibmPrecisionFloat dis = length(r) - radius;
	const ibmPrecisionFloat3 force = exp(-forceConst * dis) * n;

	particleForce[pointID] += force.x;
	particleForce[INSERT_NUM_POINTS + pointID] += force.y;
	particleForce[2*INSERT_NUM_POINTS + pointID] += force.z;
}
