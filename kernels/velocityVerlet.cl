kernel void VelocityVerlet(volatile global ibmPrecisionFloat* points, volatile global ibmPrecisionFloat* velocities, volatile global ibmPrecisionFloat* particleForce, volatile global ibmPrecisionFloat* oldParticleForce){
	const uint pointID = get_global_id(0);
	if(pointID >= INSERT_NUM_POINTS) return;
	ibmPrecisionFloat3 x = (ibmPrecisionFloat3)(points[pointID], points[INSERT_NUM_POINTS + pointID], points[2*INSERT_NUM_POINTS + pointID]);
	ibmPrecisionFloat3 v = (ibmPrecisionFloat3)(velocities[pointID], velocities[INSERT_NUM_POINTS + pointID], velocities[2*INSERT_NUM_POINTS + pointID]);
	ibmPrecisionFloat3 f = (ibmPrecisionFloat3)(particleForce[pointID], particleForce[INSERT_NUM_POINTS + pointID], particleForce[2*INSERT_NUM_POINTS + pointID]);
	ibmPrecisionFloat3 fOld = (ibmPrecisionFloat3)(oldParticleForce[pointID], oldParticleForce[INSERT_NUM_POINTS + pointID], oldParticleForce[2*INSERT_NUM_POINTS + pointID]);

	// v belongs to last timestep, we calculate v(t) = v(t-1) + 0.5(f(t) + f(t-1) and then x(t+1) = x(t) + v(t) + 0.5 * f(t)
	// 0.9 is a damping factor 
	v = 0.9*v + 0.5 * (f + fOld);
	x = x + v + 0.5 * f;
	fOld = f;
	
	points[0*INSERT_NUM_POINTS + pointID] = x.x;
	points[1*INSERT_NUM_POINTS + pointID] = x.y;
	points[2*INSERT_NUM_POINTS + pointID] = x.z;
	velocities[0*INSERT_NUM_POINTS + pointID] = v.x;
	velocities[1*INSERT_NUM_POINTS + pointID] = v.y;
	velocities[2*INSERT_NUM_POINTS + pointID] = v.z;
	oldParticleForce[0*INSERT_NUM_POINTS + pointID] = fOld.x;
	oldParticleForce[1*INSERT_NUM_POINTS + pointID] = fOld.y;
	oldParticleForce[2*INSERT_NUM_POINTS + pointID] = fOld.z;
	particleForce[0*INSERT_NUM_POINTS + pointID] = 0.0;
	particleForce[1*INSERT_NUM_POINTS + pointID] = 0.0;
	particleForce[2*INSERT_NUM_POINTS + pointID] = 0.0;
}
