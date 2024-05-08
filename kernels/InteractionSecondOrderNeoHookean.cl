kernel void Interaction_SecondOrderNeoHookean(volatile global ibmPrecisionFloat* particleForce, const global ibmPrecisionFloat* points, const global uint* tetras, const global ibmPrecisionFloat* referenceEdgeVectors, const global ibmPrecisionFloat* referenceVolumes, const global ibmPrecisionFloat* shearModulus, const global ibmPrecisionFloat* shearModulusSE, const global ibmPrecisionFloat* bulkModulus, volatile global ibmPrecisionFloat* vonMises, volatile global ibmPrecisionFloat* pressure){
	const uint tetraID = get_global_id(0);
	if(tetraID>=INSERT_NUM_TETRAS) return;

	const uint pointID1 = tetras[				  tetraID];
	const uint pointID2 = tetras[  INSERT_NUM_TETRAS+tetraID];
	const uint pointID3 = tetras[2*INSERT_NUM_TETRAS+tetraID];
	const uint pointID4 = tetras[3*INSERT_NUM_TETRAS+tetraID];

	//Calculate the current distances between the particles of one tetrahedron
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

	// Variables in the reference state
	const ibmPrecisionFloat3 R1 = (ibmPrecisionFloat3)(referenceEdgeVectors[				 tetraID], referenceEdgeVectors[  INSERT_NUM_TETRAS+tetraID], referenceEdgeVectors[2*INSERT_NUM_TETRAS+tetraID]);
	const ibmPrecisionFloat3 R2 = (ibmPrecisionFloat3)(referenceEdgeVectors[3*INSERT_NUM_TETRAS+tetraID], referenceEdgeVectors[4*INSERT_NUM_TETRAS+tetraID], referenceEdgeVectors[5*INSERT_NUM_TETRAS+tetraID]);
	const ibmPrecisionFloat3 R3 = (ibmPrecisionFloat3)(referenceEdgeVectors[6*INSERT_NUM_TETRAS+tetraID], referenceEdgeVectors[7*INSERT_NUM_TETRAS+tetraID], referenceEdgeVectors[8*INSERT_NUM_TETRAS+tetraID]);
	const ibmPrecisionFloat V0  = referenceVolumes[tetraID];

	// Calculation of displacement vector field
	// cf. Carina MT, p. 23, eq. (5.1)
	const ibmPrecisionFloat3 u1 = r1 - R1;
	const ibmPrecisionFloat3 u2 = r2 - R2;
	const ibmPrecisionFloat3 u3 = r3 - R3;
	
	// Calculation of inverse Jacobian Matrix
	// for Jacobian Matrix cf. Carina MT, p. 74, eq. (10.7)
	const ibmPrecisionFloat divisor = dot(cross(R1, R2), R3);

	const ibmPrecisionFloat3 invJ1 = cross(R2, R3)/divisor;
	const ibmPrecisionFloat3 invJ2 = cross(R3, R1)/divisor;
	const ibmPrecisionFloat3 invJ3 = cross(R1, R2)/divisor;

	// Derivative of displacement field with respect to reference position
	// cf. Carina MT, p. 72, eq. (10.3)
	// for definition of displacement field cf. Carina MT, p. 57, eq. (10.1) and p. 72, eq. (10.2)
	const ibmPrecisionFloat3 du1dR = invJ1*u1.x + invJ2*u2.x + invJ3*u3.x;
	const ibmPrecisionFloat3 du2dR = invJ1*u1.y + invJ2*u2.y + invJ3*u3.y;
	const ibmPrecisionFloat3 du3dR = invJ1*u1.z + invJ2*u2.z + invJ3*u3.z;

    	// Calculation of deformation gradient tensor components
	// cf. Carina MT, p. 23, eq. (5.3)
	const ibmPrecisionFloat3 F1 = du1dR + (ibmPrecisionFloat3)(1, 0, 0);
	const ibmPrecisionFloat3 F2 = du2dR + (ibmPrecisionFloat3)(0, 1, 0);
	const ibmPrecisionFloat3 F3 = du3dR + (ibmPrecisionFloat3)(0, 0, 1);

	// Calculation of Green Strain (Left Cauchy-Green tensor F * F^t)
	const ibmPrecisionFloat3 B1 = (ibmPrecisionFloat3)(dot(F1, F1), dot(F1, F2), dot(F1, F3));
	const ibmPrecisionFloat3 B2 = (ibmPrecisionFloat3)(dot(F2, F1), dot(F2, F2), dot(F2, F3));
	const ibmPrecisionFloat3 B3 = (ibmPrecisionFloat3)(dot(F3, F1), dot(F3, F2), dot(F3, F3));

	// Calculation of Jacobian Determinant from Green Strain
	const ibmPrecisionFloat J = sqrt(dot(cross(B1,B2),B3));
	// Trace of Green Strain and Trace of square of Green Strain
	const ibmPrecisionFloat trB = B1.x + B2.y + B3.z;	

	// Identity tensor
	const ibmPrecisionFloat3 E1 = (ibmPrecisionFloat3)(1,0,0);
	const ibmPrecisionFloat3 E2 = (ibmPrecisionFloat3)(0,1,0);
	const ibmPrecisionFloat3 E3 = (ibmPrecisionFloat3)(0,0,1);

	// Deviatoric strain tensor
	const ibmPrecisionFloat3 Bdev1 = B1 - trB/3.0 * E1;
	const ibmPrecisionFloat3 Bdev2 = B2 - trB/3.0 * E2;
	const ibmPrecisionFloat3 Bdev3 = B3 - trB/3.0 * E3;

	// Second order "NEO Hookean" material, i.e. second order polynomial hyperelastic model only dependent on I1 and not I2

	const ibmPrecisionFloat a = 1.0 / pow(J, 5.0/3.0) * (shearModulus[tetraID] + shearModulusSE[tetraID] * (trB/pow(J, 2.0/3.0) - 3.0));
	const ibmPrecisionFloat b = bulkModulus[tetraID] * (J - 1.0);

	const ibmPrecisionFloat3 sigma1 = a * Bdev1 + b * E1;
	const ibmPrecisionFloat3 sigma2 = a * Bdev2 + b * E2;
	const ibmPrecisionFloat3 sigma3 = a * Bdev3 + b * E3;

	const ibmPrecisionFloat trSigma = sigma1.x + sigma2.y +sigma3.z;
	const ibmPrecisionFloat3 devSigma1 = sigma1 - trSigma/3.0 * E1;
	const ibmPrecisionFloat3 devSigma2 = sigma2 - trSigma/3.0 * E2;
	const ibmPrecisionFloat3 devSigma3 = sigma3 - trSigma/3.0 * E3;
	const ibmPrecisionFloat vonMisesStress = sqrt(3./2. * (dot(devSigma1, devSigma1) + dot(devSigma2, devSigma2) + dot(devSigma3, devSigma3)));
	vonMises[tetraID] = vonMisesStress;
	pressure[tetraID] = trSigma / 3.0;
	
	// surface normals (cyclic: i.e. n1 is normal on surface 2,3,4)
	ibmPrecisionFloat3 n1 = cross(r2, r3);
	ibmPrecisionFloat3 n2 = cross(r3, r1);
	ibmPrecisionFloat3 n3 = cross(r1, r2);
	ibmPrecisionFloat3 n4 = cross(r3-r1, r3-r2);
	// normalize:
	n1 = n1/length(n1);
	n2 = n2/length(n2);
	n3 = n3/length(n3);
	n4 = n4/length(n4);
	// orientation: outward facing: // inward??????
	if(dot(n1, -r1) < 0) n1 *= -1;
	if(dot(n2, -r2) < 0) n2 *= -1;
	if(dot(n3, -r3) < 0) n3 *= -1;
	if(dot(n4, r1)  < 0) n4 *= -1;

	// surface areas (cyclic: i.e. A1 = surface given by points 2,3,4, belonging to normal n1)
	const ibmPrecisionFloat A1 = 0.5 * length( cross(r2, r3) );
	const ibmPrecisionFloat A2 = 0.5 * length( cross(r3, r1) );
	const ibmPrecisionFloat A3 = 0.5 * length( cross(r1, r2) );
	const ibmPrecisionFloat A4 = 0.5 * length( cross(r3-r1, r3-r2) );

	// force on surface areas given by stress * normal * area
	const ibmPrecisionFloat3 fA1 = A1 * (ibmPrecisionFloat3)(dot(sigma1, n1), dot(sigma2, n1), dot(sigma3, n1));
	const ibmPrecisionFloat3 fA2 = A2 * (ibmPrecisionFloat3)(dot(sigma1, n2), dot(sigma2, n2), dot(sigma3, n2));
	const ibmPrecisionFloat3 fA3 = A3 * (ibmPrecisionFloat3)(dot(sigma1, n3), dot(sigma2, n3), dot(sigma3, n3));
	const ibmPrecisionFloat3 fA4 = A4 * (ibmPrecisionFloat3)(dot(sigma1, n4), dot(sigma2, n4), dot(sigma3, n4));
	
	// force components on individual nodes given by 1/3 of the integral of stress*normal over adjacent surfaces
	const ibmPrecisionFloat3 f1 = -1.0/3.0 * (fA2 + fA3 + fA4);
	const ibmPrecisionFloat3 f2 = -1.0/3.0 * (fA3 + fA4 + fA1);
	const ibmPrecisionFloat3 f3 = -1.0/3.0 * (fA4 + fA1 + fA2);
	const ibmPrecisionFloat3 f4 = -1.0/3.0 * (fA1 + fA2 + fA3);
	/*
	if(tetraID == 0){
		printf("R1: [%f, %f, %f]\n", R1.x, R1.y, R1.z);
		printf("R2: [%f, %f, %f]\n", R2.x, R2.y, R2.z);
		printf("R3: [%f, %f, %f]\n", R3.x, R3.y, R3.z);
		printf("r1: [%f, %f, %f]\n", r1.x, r1.y, r1.z);
		printf("r2: [%f, %f, %f]\n", r2.x, r2.y, r2.z);
		printf("r3: [%f, %f, %f]\n", r3.x, r3.y, r3.z);
		printf("Green strain tensor:\n");
		printf("  %f\t%f\t%f\n  %f\t%f\t%f\n  %f\t%f\t%f\n", B1.x, B1.y, B1.z, B2.x, B2.y, B2.z, B3.x, B3.y, B3.z);
		printf("Trace of Green strain = %f\n", trB);
		printf("Square of green strain:\n");
		printf("  %f\t%f\t%f\n  %f\t%f\t%f\n  %f\t%f\t%f\n", Bsq1.x, Bsq1.y, Bsq1.z, Bsq2.x, Bsq2.y, Bsq2.z, Bsq3.x, Bsq3.y, Bsq3.z);
		printf("Trace of Green strain squared = %f\n", trBsq);
		printf("Jacobi Determinant = %f\n", J);
		printf("Mu1 = %f\n", mu1[tetraID]);
		printf("Mu2 = %f\n", mu2[tetraID]);
		printf("Kappa = %f\n", kappa[tetraID]);
		printf("Cauchy Stress tensor:\n");
		printf("  %f\t%f\t%f\n  %f\t%f\t%f\n  %f\t%f\t%f\n", sigma1.x, sigma1.y, sigma1.z, sigma2.x, sigma2.y, sigma2.z, sigma3.x, sigma3.y, sigma3.z);
		printf("normal 1: [%f, %f, %f]\n", n1.x, n1.y, n1.z);
		printf("normal 2: [%f, %f, %f]\n", n2.x, n2.y, n2.z);
		printf("normal 3: [%f, %f, %f]\n", n3.x, n3.y, n3.z);
		printf("normal 4: [%f, %f, %f]\n", n4.x, n4.y, n4.z);
		printf("Area of surfaces : [%f, %f, %f, %f]\n", A1, A2, A3, A4);
		printf("Force on Area 1: [%f, %f, %f]\n", fA1.x, fA1.y, fA1.z);
		printf("Force on Area 2: [%f, %f, %f]\n", fA2.x, fA2.y, fA2.z);
		printf("Force on Area 3: [%f, %f, %f]\n", fA3.x, fA3.y, fA3.z);
		printf("Force on Area 4: [%f, %f, %f]\n", fA4.x, fA4.y, fA4.z);
		printf("Force on vertex 1: [%f, %f, %f]\n", f1.x, f1.y, f1.z);
		printf("Force on vertex 2: [%f, %f, %f]\n", f2.x, f2.y, f2.z);
		printf("Force on vertex 3: [%f, %f, %f]\n", f3.x, f3.y, f3.z);
		printf("Force on vertex 4: [%f, %f, %f]\n", f4.x, f4.y, f4.z);
		printf("\n\n\n");
	}
	*/
	atomicAdd(&particleForce[            	  pointID1], f1.x); // forces on point 1
	atomicAdd(&particleForce[  INSERT_NUM_POINTS+pointID1], f1.y);
	atomicAdd(&particleForce[2*INSERT_NUM_POINTS+pointID1], f1.z);
	atomicAdd(&particleForce[           	  pointID2], f2.x); // forces on point 2
	atomicAdd(&particleForce[  INSERT_NUM_POINTS+pointID2], f2.y);
	atomicAdd(&particleForce[2*INSERT_NUM_POINTS+pointID2], f2.z);
	atomicAdd(&particleForce[            	  pointID3], f3.x); // forces on point 3
	atomicAdd(&particleForce[  INSERT_NUM_POINTS+pointID3], f3.y);
	atomicAdd(&particleForce[2*INSERT_NUM_POINTS+pointID3], f3.z);
	atomicAdd(&particleForce[        	      pointID4], f4.x); // forces on point 4
	atomicAdd(&particleForce[  INSERT_NUM_POINTS+pointID4], f4.y);
	atomicAdd(&particleForce[2*INSERT_NUM_POINTS+pointID4], f4.z);
}

