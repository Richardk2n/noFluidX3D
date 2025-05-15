kernel void Interaction_Sphere(volatile global ibmPrecisionFloat* particleForce, const global ibmPrecisionFloat* points, const ibmPrecisionFloat radius, const ibmPrecisionFloat spherePos){
	const uint pointID = get_global_id(0);
	if(pointID>=INSERT_NUM_POINTS) return;
	// No atomicAdd, because force on each point can be calculated independently -> probably faster!
	// Force from spherical indenter
	const ibmPrecisionFloat3 pos = (ibmPrecisionFloat3)(points[pointID], points[INSERT_NUM_POINTS + pointID], points[2*INSERT_NUM_POINTS + pointID]);
	const ibmPrecisionFloat3 sphere = (ibmPrecisionFloat3)(0, spherePos, 0);

	ibmPrecisionFloat3 r = pos - sphere;
	const ibmPrecisionFloat3 n = normalize(r);
	const ibmPrecisionFloat dis =  max(radius-length(r), 0.);
	const ibmPrecisionFloat3 force = def_FORCE_CONST * dis * n;

	particleForce[pointID] += force.x;
	particleForce[INSERT_NUM_POINTS + pointID] += force.y;
	particleForce[2*INSERT_NUM_POINTS + pointID] += force.z;
}
