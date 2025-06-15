kernel void Interaction_LinearElasticStress(volatile global forcePrecisionFloat* particleForce, const global tetraPrecisionFloat* points, const global int* tetras, const global tetraPrecisionFloat* referenceEdgeVectors, const global tetraPrecisionFloat* referenceVolumes){
	const uint tetraID = get_global_id(0);
	if(tetraID>=def_tetraCount) return;

	const int pointID1 = tetras[				  tetraID];
	const int pointID2 = tetras[  def_tetraCount+tetraID];
	const int pointID3 = tetras[2*def_tetraCount+tetraID];
	const int pointID4 = tetras[3*def_tetraCount+tetraID];

	//Calculate the current distances between the particles of one tetrahedron
	const tetraPrecisionFloat3 p1 = (tetraPrecisionFloat3)(points[pointID1], points[def_pointCount+pointID1], points[2*def_pointCount+pointID1]);
	const tetraPrecisionFloat3 p2 = (tetraPrecisionFloat3)(points[pointID2], points[def_pointCount+pointID2], points[2*def_pointCount+pointID2]);
	const tetraPrecisionFloat3 p3 = (tetraPrecisionFloat3)(points[pointID3], points[def_pointCount+pointID3], points[2*def_pointCount+pointID3]);
	const tetraPrecisionFloat3 p4 = (tetraPrecisionFloat3)(points[pointID4], points[def_pointCount+pointID4], points[2*def_pointCount+pointID4]);

	//Calculate the current distances between the particles of one tetrahedron
	// r1 = vector between 1 and 4
	const tetraPrecisionFloat3 r1 = p1-p4;
	// r2 = vector between 2 and 4
	const tetraPrecisionFloat3 r2 = p2-p4;
	// r3 = vector between 3 and 4
	const tetraPrecisionFloat3 r3 = p3-p4;

	// Variables in the reference state
	const tetraPrecisionFloat3 R1 = (tetraPrecisionFloat3)(referenceEdgeVectors[				 tetraID], referenceEdgeVectors[  def_tetraCount+tetraID], referenceEdgeVectors[2*def_tetraCount+tetraID]);
	const tetraPrecisionFloat3 R2 = (tetraPrecisionFloat3)(referenceEdgeVectors[3*def_tetraCount+tetraID], referenceEdgeVectors[4*def_tetraCount+tetraID], referenceEdgeVectors[5*def_tetraCount+tetraID]);
	const tetraPrecisionFloat3 R3 = (tetraPrecisionFloat3)(referenceEdgeVectors[6*def_tetraCount+tetraID], referenceEdgeVectors[7*def_tetraCount+tetraID], referenceEdgeVectors[8*def_tetraCount+tetraID]);
	const tetraPrecisionFloat V0  = referenceVolumes[tetraID]/100;

	// Calculation of displacement vector field
	// cf. Carina MT, p. 23, eq. (5.1)
	const tetraPrecisionFloat3 u1 = r1 - R1;
	const tetraPrecisionFloat3 u2 = r2 - R2;
	const tetraPrecisionFloat3 u3 = r3 - R3;

	// Calc of A
	// cf. Carina MT, p. 73, eq. (10.4)
	// columns
	const tetraPrecisionFloat3 A_1 = u1;
	const tetraPrecisionFloat3 A_2 = u2;
	const tetraPrecisionFloat3 A_3 = u3;

	// rows
	const tetraPrecisionFloat3 A1 = (tetraPrecisionFloat3)(A_1.x, A_2.x, A_3.x);
	const tetraPrecisionFloat3 A2 = (tetraPrecisionFloat3)(A_1.y, A_2.y, A_3.y);
	const tetraPrecisionFloat3 A3 = (tetraPrecisionFloat3)(A_1.z, A_2.z, A_3.z);

	// Calculation of inverse Jacobian Matrix
	// for Jacobian Matrix cf. Carina MT, p. 74, eq. (10.7)
	const tetraPrecisionFloat divisor = dot(R1, cross(R2, R3));

	// rows
	const tetraPrecisionFloat3 B1 = cross(R2, R3)/divisor;
	const tetraPrecisionFloat3 B2 = cross(R3, R1)/divisor;
	const tetraPrecisionFloat3 B3 = cross(R1, R2)/divisor;

	// columns
	const tetraPrecisionFloat3 B_1 = (tetraPrecisionFloat3)(B1.x, B2.x, B3.x);
	const tetraPrecisionFloat3 B_2 = (tetraPrecisionFloat3)(B1.y, B2.y, B3.y);
	const tetraPrecisionFloat3 B_3 = (tetraPrecisionFloat3)(B1.z, B2.z, B3.z);

	// epsilon
	// cf. Carina MT, p. 74, eq. (10.8)
	// A and B can be transposed by transposing symmetric e -> dot products
	const tetraPrecisionFloat e11 = dot(A1, B_1);
	const tetraPrecisionFloat e22 = dot(A2, B_2);
	const tetraPrecisionFloat e33 = dot(A3, B_3);
	const tetraPrecisionFloat e12 = (dot(A1, B_2) + dot(A2, B_1))/2;
	const tetraPrecisionFloat e13 = (dot(A1, B_3) + dot(A3, B_1))/2;
	const tetraPrecisionFloat e23 = (dot(A2, B_3) + dot(A3, B_2))/2;

	const tetraPrecisionFloat Tre = e11 + e22 + e33;

	// U derrivatives
	// cf. Carina MT, p. 75, eq. (10.11)
	const tetraPrecisionFloat mod1 = def_tetraYoungsModulus*(1-3*def_tetraPoissonRatio)/(2*(1+def_tetraPoissonRatio)*(1-2*def_tetraPoissonRatio));
	const tetraPrecisionFloat mod2 = def_tetraYoungsModulus*def_tetraPoissonRatio/((1+def_tetraPoissonRatio)*(1-2*def_tetraPoissonRatio));
	const tetraPrecisionFloat mod3 = 2*def_tetraYoungsModulus/(1+def_tetraPoissonRatio);

	const tetraPrecisionFloat dUde11 = mod1*e11 + mod2*Tre;
	const tetraPrecisionFloat dUde22 = mod1*e22 + mod2*Tre;
	const tetraPrecisionFloat dUde33 = mod1*e33 + mod2*Tre;
	const tetraPrecisionFloat dUde12 = mod3*e12;
	const tetraPrecisionFloat dUde13 = mod3*e13;
	const tetraPrecisionFloat dUde23 = mod3*e23;

	const tetraPrecisionFloat3 dUde1 = (tetraPrecisionFloat3)(dUde11, dUde12, dUde13);
	const tetraPrecisionFloat3 dUde2 = (tetraPrecisionFloat3)(dUde12, dUde22, dUde23);
	const tetraPrecisionFloat3 dUde3 = (tetraPrecisionFloat3)(dUde13, dUde23, dUde33);


	// Calculate forces, force density * reference Volume of the tetrahedron
	// cf. Carina MT, p. 77, eq. (10.13)
	// dA_LM/du_aI are all unity according to (10.12)
	// de/dA derrivatives can be simplified
	// cf. Carina MT, p. 75
	const tetraPrecisionFloat3 f1 = -V0*(tetraPrecisionFloat3)(dot(dUde1, B1), dot(dUde2, B1), dot(dUde3, B1));
	const tetraPrecisionFloat3 f2 = -V0*(tetraPrecisionFloat3)(dot(dUde1, B2), dot(dUde2, B2), dot(dUde3, B2));
	const tetraPrecisionFloat3 f3 = -V0*(tetraPrecisionFloat3)(dot(dUde1, B3), dot(dUde2, B3), dot(dUde3, B3));
	
	if(tetraID == 74091) {
    	printf("%.20f\n", f3.y);
	}

	const tetraPrecisionFloat3 f4 = -(f1 + f2 + f3);


	atomicAdd(&particleForce[            	  pointID1], f1.x); // forces on point 1
	atomicAdd(&particleForce[  def_pointCount+pointID1], f1.y);
	atomicAdd(&particleForce[2*def_pointCount+pointID1], f1.z);
	atomicAdd(&particleForce[           	  pointID2], f2.x); // forces on point 2
	atomicAdd(&particleForce[  def_pointCount+pointID2], f2.y);
	atomicAdd(&particleForce[2*def_pointCount+pointID2], f2.z);
	atomicAdd(&particleForce[            	  pointID3], f3.x); // forces on point 3
	atomicAdd(&particleForce[  def_pointCount+pointID3], f3.y);
	atomicAdd(&particleForce[2*def_pointCount+pointID3], f3.z);
	atomicAdd(&particleForce[        	      pointID4], f4.x); // forces on point 4
	atomicAdd(&particleForce[  def_pointCount+pointID4], f4.y);
	atomicAdd(&particleForce[2*def_pointCount+pointID4], f4.z);
}
