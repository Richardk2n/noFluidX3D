ibmPrecisionFloat calcForceAdhesivePlane(const ibmPrecisionFloat3 pos, const ibmPrecisionFloat wallpos, const ibmPrecisionFloat adhesionConst, const ibmPrecisionFloat sigma){
	const ibmPrecisionFloat d = pos.y - wallpos;
	return 24.0 * adhesionConst / d * (2*pow(sigma/d, 12) - pow(sigma/d, 6));
}
ibmPrecisionFloat N1(const ibmPrecisionFloat3 xi){
	return xi.x;
}
ibmPrecisionFloat N2(const ibmPrecisionFloat3 xi){
	return xi.y;
}
ibmPrecisionFloat N3(const ibmPrecisionFloat3 xi){
	return 1 - xi.x - xi.y;
}

kernel void Interaction_AdhesivePlaneSurfaceIntegral(volatile global ibmPrecisionFloat* particleForce, const global ibmPrecisionFloat* points, const global uint* triangles, const ibmPrecisionFloat wallPos, const ibmPrecisionFloat adhesionConst, const ibmPrecisionFloat distance){
	const uint triangleID = get_global_id(0);
	if(triangleID>=INSERT_NUM_TRIANGLES) return;
	const ibmPrecisionFloat sigma = pow(0.5, 1.0/6.0) * distance;
	const uint pointID1 = triangles[triangleID];
	const uint pointID2 = triangles[INSERT_NUM_TRIANGLES + triangleID];
	const uint pointID3 = triangles[2*INSERT_NUM_TRIANGLES + triangleID];
	
	const ibmPrecisionFloat3 p1 = (ibmPrecisionFloat3)(points[pointID1], points[INSERT_NUM_POINTS + pointID1], points[2*INSERT_NUM_POINTS + pointID1]);
	const ibmPrecisionFloat3 p2 = (ibmPrecisionFloat3)(points[pointID2], points[INSERT_NUM_POINTS + pointID2], points[2*INSERT_NUM_POINTS + pointID2]);
	const ibmPrecisionFloat3 p3 = (ibmPrecisionFloat3)(points[pointID3], points[INSERT_NUM_POINTS + pointID3], points[2*INSERT_NUM_POINTS + pointID3]);
	
	const ibmPrecisionFloat3 r1 = p1 - p3;
	const ibmPrecisionFloat3 r2 = p2 - p3;
	const ibmPrecisionFloat area = 0.5 * length(cross(r1, r2));	

	// Xis are 2d, so z coordinate is just 0
	const ibmPrecisionFloat3 xi1 = (ibmPrecisionFloat3)(0.5, 0.0, 0.0);
	const ibmPrecisionFloat3 xi2 = (ibmPrecisionFloat3)(0.0, 0.5, 0.0);
	const ibmPrecisionFloat3 xi3 = (ibmPrecisionFloat3)(0.5, 0.5, 0.0);

	// Integration points Scheme 1 from Bower p. 485
	const ibmPrecisionFloat3 x1 = N1(xi1) * p1 + N2(xi1) * p2 + N3(xi1) * p3;
	const ibmPrecisionFloat3 x2 = N1(xi2) * p1 + N2(xi2) * p2 + N3(xi2) * p3;
	const ibmPrecisionFloat3 x3 = N1(xi3) * p1 + N2(xi3) * p2 + N3(xi3) * p3;

	// force area density at integration points
	const ibmPrecisionFloat f1 = calcForceAdhesivePlane(x1, wallPos, adhesionConst, sigma);
	const ibmPrecisionFloat f2 = calcForceAdhesivePlane(x2, wallPos, adhesionConst, sigma);
	const ibmPrecisionFloat f3 = calcForceAdhesivePlane(x3, wallPos, adhesionConst, sigma);

	// Total force on each point
	const ibmPrecisionFloat F1 = 1.0/3.0 * area * (f1 * N1(xi1) + f2 * N1(xi2) + f3 * N1(xi3));
	const ibmPrecisionFloat F2 = 1.0/3.0 * area * (f1 * N2(xi1) + f2 * N2(xi2) + f3 * N2(xi3));
	const ibmPrecisionFloat F3 = 1.0/3.0 * area * (f1 * N3(xi1) + f2 * N3(xi2) + f3 * N3(xi3));

	// DEBUG

	atomicAdd(&particleForce[  INSERT_NUM_POINTS+pointID1], F1);
	atomicAdd(&particleForce[  INSERT_NUM_POINTS+pointID2], F2);
	atomicAdd(&particleForce[  INSERT_NUM_POINTS+pointID3], F3);
}
