ibmPrecisionFloat3 calcForceTip(const ibmPrecisionFloat3 pos, const ibmPrecisionFloat radius, const ibmPrecisionFloat halfAngle, const ibmPrecisionFloat3 tip, const ibmPrecisionFloat forceConst){
	ibmPrecisionFloat3 force = (ibmPrecisionFloat3)(0.0, 0.0, 0.0);
	ibmPrecisionFloat3 rho = (ibmPrecisionFloat3) (pos.x - tip.x, 0, pos.z - tip.z);

	// Spherical tip regime
	if(pos.y < (tip.y - sin(halfAngle) * length(rho))){
		const ibmPrecisionFloat3 r = pos - tip;
		const ibmPrecisionFloat3 n = r / length(r);
		const ibmPrecisionFloat dis = length(r) - radius;
		force = exp(-forceConst * dis) * n; 
	}
	else{ // angled regime
		ibmPrecisionFloat3 eRho = rho / length(rho);

		ibmPrecisionFloat3 n = -sin(halfAngle) * (ibmPrecisionFloat3)(0,1,0) + cos(halfAngle) * eRho;
		ibmPrecisionFloat dis = dot(pos, n) - dot(tip, n) - radius;
		force = exp(-forceConst * dis) * n;
	}
	return force;
	
}
ibmPrecisionFloat N1(const ibmPrecisionFloat3 xi){
	return xi.x;
}
ibmPrecisionFloat N2(const ibmPrecisionFloat3 xi){
	return xi.y;
}
ibmPrecisionFloat N3(const ibmPrecisionFloat3 xi){
	return xi.z;
}
ibmPrecisionFloat N4(const ibmPrecisionFloat3 xi){
	return 1 - xi.x - xi.y - xi.z;
}
kernel void Interaction_TipIntegral(volatile global ibmPrecisionFloat* particleForce, const global ibmPrecisionFloat* points, const global uint* tetras, const ibmPrecisionFloat radius, const ibmPrecisionFloat halfAngle, const ibmPrecisionFloat tipPosX, ibmPrecisionFloat tipPosY, ibmPrecisionFloat tipPosZ, const ibmPrecisionFloat forceConst){
	const uint tetraID = get_global_id(0);
	if(tetraID>=INSERT_NUM_TETRAS) return;
	const uint pointID1 = tetras[				  tetraID];
	const uint pointID2 = tetras[  INSERT_NUM_TETRAS+tetraID];
	const uint pointID3 = tetras[2*INSERT_NUM_TETRAS+tetraID];
	const uint pointID4 = tetras[3*INSERT_NUM_TETRAS+tetraID];
	
	const ibmPrecisionFloat3 p1 = (ibmPrecisionFloat3)(points[pointID1], points[INSERT_NUM_POINTS+pointID1], points[2*INSERT_NUM_POINTS+pointID1]);
	const ibmPrecisionFloat3 p2 = (ibmPrecisionFloat3)(points[pointID2], points[INSERT_NUM_POINTS+pointID2], points[2*INSERT_NUM_POINTS+pointID2]);
	const ibmPrecisionFloat3 p3 = (ibmPrecisionFloat3)(points[pointID3], points[INSERT_NUM_POINTS+pointID3], points[2*INSERT_NUM_POINTS+pointID3]);
	const ibmPrecisionFloat3 p4 = (ibmPrecisionFloat3)(points[pointID4], points[INSERT_NUM_POINTS+pointID4], points[2*INSERT_NUM_POINTS+pointID4]);
	//Calculate the current distances between the particles of one tetrahedron
	// r1 = vector between 1 and 4
	const ibmPrecisionFloat3 r1 = p1-p4;
	// r2 = vector between 2 and 4
	const ibmPrecisionFloat3 r2 = p2-p4;
	// r3 = vector between 3 and 4
	const ibmPrecisionFloat3 r3 = p3-p4;

	const ibmPrecisionFloat3 tipPos = (ibmPrecisionFloat3)(tipPosX, tipPosY, tipPosZ);

	// Calculate current Volume

	const ibmPrecisionFloat volume = fabs(dot(r1, cross(r2, r3)));

	// Integration points, Bower p.486
	const ibmPrecisionFloat alpha = 0.58541020;
	const ibmPrecisionFloat beta = 0.13819669;
	const ibmPrecisionFloat3 xi1 = (ibmPrecisionFloat3)(alpha, beta, beta);	
	const ibmPrecisionFloat3 xi2 = (ibmPrecisionFloat3)(beta, alpha, beta);	
	const ibmPrecisionFloat3 xi3 = (ibmPrecisionFloat3)(beta, beta, alpha);	
	const ibmPrecisionFloat3 xi4 = (ibmPrecisionFloat3)(beta, beta, beta);	

	const ibmPrecisionFloat3 x1 = xi1.x * p1 + xi1.y * p2 + xi1.z * p3 + (1.0 - xi1.x - xi1.y -xi1.z) * p4;
	const ibmPrecisionFloat3 x2 = xi2.x * p1 + xi2.y * p2 + xi2.z * p3 + (1.0 - xi2.x - xi2.y -xi2.z) * p4;
	const ibmPrecisionFloat3 x3 = xi3.x * p1 + xi3.y * p2 + xi3.z * p3 + (1.0 - xi3.x - xi3.y -xi3.z) * p4;
	const ibmPrecisionFloat3 x4 = xi4.x * p1 + xi4.y * p2 + xi4.z * p3 + (1.0 - xi4.x - xi4.y -xi4.z) * p4;
	
	// force density at integration points
	const ibmPrecisionFloat3 f1 = calcForceTip(x1, radius, halfAngle, tipPos, forceConst);
	const ibmPrecisionFloat3 f2 = calcForceTip(x2, radius, halfAngle, tipPos, forceConst);
	const ibmPrecisionFloat3 f3 = calcForceTip(x3, radius, halfAngle, tipPos, forceConst);
	const ibmPrecisionFloat3 f4 = calcForceTip(x4, radius, halfAngle, tipPos, forceConst);

	// Total forces acting on each point

	const ibmPrecisionFloat3 F1 = (ibmPrecisionFloat3)(0.25 * volume * (f1 * N1(xi1) + f2 * N1(xi2) + f3 * N1(xi3) + f4 * N1(xi4)));
	const ibmPrecisionFloat3 F2 = 0.25 * volume * (f1 * N2(xi1) + f2 * N2(xi2) + f3 * N2(xi3) + f4 * N2(xi4));
	const ibmPrecisionFloat3 F3 = 0.25 * volume * (f1 * N3(xi1) + f2 * N3(xi2) + f3 * N3(xi3) + f4 * N3(xi4));
	const ibmPrecisionFloat3 F4 = 0.25 * volume * (f1 * N4(xi1) + f2 * N4(xi2) + f3 * N4(xi3) + f4 * N4(xi4));
	
	/* DEBUG
	//if(tetraID==177745){
	if(tetraID==80790){
		printf("new timestep:\n");
		printf("positions");
		printf("p1 = [%.16f, %.16f, %.16f]\n", p1.x, p1.y, p1.z);
		printf("p2 = [%.16f, %.16f, %.16f]\n", p2.x, p2.y, p2.z);
		printf("p3 = [%.16f, %.16f, %.16f]\n", p3.x, p3.y, p3.z);
		printf("p4 = [%.16f, %.16f, %.16f]\n", p4.x, p4.y, p4.z);
		printf("forces");
		printf("F1 = [%.16f, %.16f, %.16f]\n", F1.x, F1.y, F1.z);
		printf("F2 = [%.16f, %.16f, %.16f]\n", F2.x, F2.y, F2.z);
		printf("F3 = [%.16f, %.16f, %.16f]\n", F3.x, F3.y, F3.z);
		printf("F4 = [%.16f, %.16f, %.16f]\n", F4.x, F4.y, F4.z);
	}*/

	atomicAdd(&particleForce[            	  pointID1], F1.x); // forces on point 1
	atomicAdd(&particleForce[  INSERT_NUM_POINTS+pointID1], F1.y);
	atomicAdd(&particleForce[2*INSERT_NUM_POINTS+pointID1], F1.z);
	atomicAdd(&particleForce[           	  pointID2], F2.x); // forces on point 2
	atomicAdd(&particleForce[  INSERT_NUM_POINTS+pointID2], F2.y);
	atomicAdd(&particleForce[2*INSERT_NUM_POINTS+pointID2], F2.z);
	atomicAdd(&particleForce[            	  pointID3], F3.x); // forces on point 3
	atomicAdd(&particleForce[  INSERT_NUM_POINTS+pointID3], F3.y);
	atomicAdd(&particleForce[2*INSERT_NUM_POINTS+pointID3], F3.z);
	atomicAdd(&particleForce[        	      pointID4], F4.x); // forces on point 4
	atomicAdd(&particleForce[  INSERT_NUM_POINTS+pointID4], F4.y);
	atomicAdd(&particleForce[2*INSERT_NUM_POINTS+pointID4], F4.z);
}
