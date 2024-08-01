kernel void VelocityVerlet2(volatile global ibmPrecisionFloat* points, volatile global ibmPrecisionFloat* velocities, volatile global ibmPrecisionFloat* particleForce, volatile global ibmPrecisionFloat* oldParticleForce, volatile global ibmPrecisionFloat* masses, const ibmPrecisionFloat damping){
	const uint pointID = get_global_id(0);
	if(pointID >= INSERT_NUM_POINTS) return;
	ibmPrecisionFloat3 x = (ibmPrecisionFloat3)(points[pointID], points[INSERT_NUM_POINTS + pointID], points[2*INSERT_NUM_POINTS + pointID]);
	ibmPrecisionFloat3 v = (ibmPrecisionFloat3)(velocities[pointID], velocities[INSERT_NUM_POINTS + pointID], velocities[2*INSERT_NUM_POINTS + pointID]);
	ibmPrecisionFloat3 f = (ibmPrecisionFloat3)(particleForce[pointID], particleForce[INSERT_NUM_POINTS + pointID], particleForce[2*INSERT_NUM_POINTS + pointID]);
	ibmPrecisionFloat3 fOld = (ibmPrecisionFloat3)(oldParticleForce[pointID], oldParticleForce[INSERT_NUM_POINTS + pointID], oldParticleForce[2*INSERT_NUM_POINTS + pointID]);
	// v belongs to last timestep, we calculate v(t) = 1/(1+dt/2 xi) * (v(t-1) + dt/2*(f(t)/m + ddotx(t-1))) and then x(t+1) = x(t) + dt*v(t) + 0.5 * dt^2 ddotx(t)
	v = 1.0/(1 + damping/2.0) * (v + 0.5 * (fOld + f/masses[pointID]));
	ibmPrecisionFloat3 ddotx = f/masses[pointID] - damping*v;
	x = x + v + 0.5 * ddotx;
	fOld = ddotx;
	
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
