kernel void Interaction_FiniteStrainViscoplastic(volatile global ibmPrecisionFloat* particleForce, const global ibmPrecisionFloat* points, const global uint* tetras, const global ibmPrecisionFloat* referenceEdgeVectors, const global ibmPrecisionFloat* referenceVolumes, const global ibmPrecisionFloat* mu1, const global ibmPrecisionFloat* mu2, const global ibmPrecisionFloat* kappa, const global ibmPrecisionFloat* flowExponent, const global ibmPrecisionFloat* yieldStress, const global ibmPrecisionFloat* plasticFlowRate, const global ibmPrecisionFloat* hardeningThreshhold, const global ibmPrecisionFloat* hardeningExponent, volatile global ibmPrecisionFloat* accumulatedStrain, volatile global ibmPrecisionFloat* plasticDeformationTensor, volatile global ibmPrecisionFloat* vonMises, volatile global ibmPrecisionFloat* pressure){
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

	// Calculate the elastic part of the deformation gradient tensor by Fe = F*Fp^-1
	// inverse is the product of the inverse determinant and the cofactor matrix, notation follows wikipedia
	ibmPrecisionFloat a = plasticDeformationTensor[0*INSERT_NUM_TETRAS + tetraID];
	ibmPrecisionFloat b = plasticDeformationTensor[1*INSERT_NUM_TETRAS + tetraID];
	ibmPrecisionFloat c = plasticDeformationTensor[2*INSERT_NUM_TETRAS + tetraID];
	ibmPrecisionFloat d = plasticDeformationTensor[3*INSERT_NUM_TETRAS + tetraID];
	ibmPrecisionFloat e = plasticDeformationTensor[4*INSERT_NUM_TETRAS + tetraID];
	ibmPrecisionFloat f = plasticDeformationTensor[5*INSERT_NUM_TETRAS + tetraID];
	ibmPrecisionFloat g = plasticDeformationTensor[6*INSERT_NUM_TETRAS + tetraID];
	ibmPrecisionFloat h = plasticDeformationTensor[7*INSERT_NUM_TETRAS + tetraID];
	ibmPrecisionFloat i = plasticDeformationTensor[8*INSERT_NUM_TETRAS + tetraID];
	const ibmPrecisionFloat3 Fp1 = (ibmPrecisionFloat3)(a, b, c);
	const ibmPrecisionFloat3 Fp2 = (ibmPrecisionFloat3)(d, e, f);
	const ibmPrecisionFloat3 Fp3 = (ibmPrecisionFloat3)(g, h, i);
	const ibmPrecisionFloat detFp = dot(cross(Fp1, Fp2), Fp3);

	const ibmPrecisionFloat3 invFp1 = 1./detFp * (ibmPrecisionFloat3)(e*i - f*h, c*h - b*i, b*f - c*e);
	const ibmPrecisionFloat3 invFp2 = 1./detFp * (ibmPrecisionFloat3)(f*g - d*i, a*i - c*g, c*d - a*f);
	const ibmPrecisionFloat3 invFp3 = 1./detFp * (ibmPrecisionFloat3)(d*h - e*g, b*g - a*h, a*e - b*d);

	const ibmPrecisionFloat3 Fe1 = (ibmPrecisionFloat3)(F1.x*invFp1.x + F1.y*invFp2.x + F1.z*invFp3.x, F1.x*invFp1.y + F1.y*invFp2.y + F1.z*invFp3.y, F1.x*invFp1.z + F1.y*invFp2.z + F1.z*invFp3.z);
	const ibmPrecisionFloat3 Fe2 = (ibmPrecisionFloat3)(F2.x*invFp1.x + F2.y*invFp2.x + F2.z*invFp3.x, F2.x*invFp1.y + F2.y*invFp2.y + F2.z*invFp3.y, F2.x*invFp1.z + F2.y*invFp2.z + F2.z*invFp3.z);
	const ibmPrecisionFloat3 Fe3 = (ibmPrecisionFloat3)(F3.x*invFp1.x + F3.y*invFp2.x + F3.z*invFp3.x, F3.x*invFp1.y + F3.y*invFp2.y + F3.z*invFp3.y, F3.x*invFp1.z + F3.y*invFp2.z + F3.z*invFp3.z);

	// Calculation of elastic Green Strain (Left Cauchy-Green tensor Fe * Fe^t)
	const ibmPrecisionFloat3 B1 = (ibmPrecisionFloat3)(dot(Fe1, Fe1), dot(Fe1, Fe2), dot(Fe1, Fe3));
	const ibmPrecisionFloat3 B2 = (ibmPrecisionFloat3)(dot(Fe2, Fe1), dot(Fe2, Fe2), dot(Fe2, Fe3));
	const ibmPrecisionFloat3 B3 = (ibmPrecisionFloat3)(dot(Fe3, Fe1), dot(Fe3, Fe2), dot(Fe3, Fe3));

	// Calculation of Jacobian Determinant from Green Strain
	const ibmPrecisionFloat J = sqrt(dot(cross(B1,B2),B3));
	
	// Square of Green Strain
	const ibmPrecisionFloat3 Bsq1 = (ibmPrecisionFloat3)(B1.x*B1.x + B1.y*B2.x + B1.z*B3.x, B1.x*B1.y + B1.y*B2.y + B1.z*B3.y, B1.x*B1.z + B1.y*B2.z + B1.z*B3.z);
	const ibmPrecisionFloat3 Bsq2 = (ibmPrecisionFloat3)(B2.x*B1.x + B2.y*B2.x + B2.z*B3.x, B2.x*B1.y + B2.y*B2.y + B2.z*B3.y, B2.x*B1.z + B2.y*B2.z + B2.z*B3.z);
	const ibmPrecisionFloat3 Bsq3 = (ibmPrecisionFloat3)(B3.x*B1.x + B3.y*B2.x + B3.z*B3.x, B3.x*B1.y + B3.y*B2.y + B3.z*B3.y, B3.x*B1.z + B3.y*B2.z + B3.z*B3.z);

	// Trace of Green Strain and Trace of square of Green Strain
	const ibmPrecisionFloat trB = B1.x + B2.y + B3.z;
	const ibmPrecisionFloat trBsq = Bsq1.x + Bsq2.y + Bsq3.z;

	// Identity tensor
	const ibmPrecisionFloat3 E1 = (ibmPrecisionFloat3)(1,0,0);
	const ibmPrecisionFloat3 E2 = (ibmPrecisionFloat3)(0,1,0);
	const ibmPrecisionFloat3 E3 = (ibmPrecisionFloat3)(0,0,1);

	// Cauchy Stress tensor SJM PhD thesis p.16 from Bower

	const ibmPrecisionFloat3 sigma1 = mu1[tetraID]*pow(J,-5.0/3.0) * (B1 - 1./3.*trB*E1) + kappa[tetraID]*(J-1)*E1 + mu2[tetraID]*pow(J,-7.0/3.0) * (trB*B1 - 1.0/3.0*trB*trB*E1 - Bsq1 + 1.0/3.0*trBsq*E1);
	const ibmPrecisionFloat3 sigma2 = mu1[tetraID]*pow(J,-5.0/3.0) * (B2 - 1./3.*trB*E2) + kappa[tetraID]*(J-1)*E2 + mu2[tetraID]*pow(J,-7.0/3.0) * (trB*B2 - 1.0/3.0*trB*trB*E2 - Bsq2 + 1.0/3.0*trBsq*E2);
	const ibmPrecisionFloat3 sigma3 = mu1[tetraID]*pow(J,-5.0/3.0) * (B3 - 1./3.*trB*E3) + kappa[tetraID]*(J-1)*E3 + mu2[tetraID]*pow(J,-7.0/3.0) * (trB*B3 - 1.0/3.0*trB*trB*E3 - Bsq3 + 1.0/3.0*trBsq*E3);
	
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
	
	// Calculate time evolution of the plastic Deformation Tensor	
	const ibmPrecisionFloat trSigma = sigma1.x + sigma2.y +sigma3.z;
	const ibmPrecisionFloat3 devSigma1 = sigma1 - trSigma/3.0 * E1;
	const ibmPrecisionFloat3 devSigma2 = sigma2 - trSigma/3.0 * E2;
	const ibmPrecisionFloat3 devSigma3 = sigma3 - trSigma/3.0 * E3;
	const ibmPrecisionFloat vonMisesStress = sqrt(3./2. * (dot(devSigma1, devSigma1) + dot(devSigma2, devSigma2) + dot(devSigma3, devSigma3)));
	const ibmPrecisionFloat sigma0 = yieldStress[tetraID] * pow((1 + accumulatedStrain[tetraID]/hardeningThreshhold[tetraID]), hardeningExponent[tetraID]);
	vonMises[tetraID] = vonMisesStress;
	pressure[tetraID] = trSigma / 3.0;

	const ibmPrecisionFloat plasticStretchRate = plasticFlowRate[tetraID] * pow(vonMisesStress/sigma0, flowExponent[tetraID]);
	
	// update accumulated plastic strain
	accumulatedStrain[tetraID] += plasticStretchRate;
	
	// Calculate plastic Stretch Rate tensor Dp

	ibmPrecisionFloat3 Dp1, Dp2, Dp3;
	if(vonMisesStress != 0){
		Dp1 = plasticStretchRate * 3./2. * devSigma1/vonMisesStress;
		Dp2 = plasticStretchRate * 3./2. * devSigma2/vonMisesStress;
		Dp3 = plasticStretchRate * 3./2. * devSigma3/vonMisesStress;
	}else{
		Dp1 = (ibmPrecisionFloat3)(0,0,0);
		Dp2 = (ibmPrecisionFloat3)(0,0,0);
		Dp3 = (ibmPrecisionFloat3)(0,0,0);
	}

	// Calculate Plastic deformation rate FpDot = Fe^-1 * Dp * F
	// Inverse of elastic Deformation tensor
	a = Fe1.x;
	b = Fe1.y;
	c = Fe1.z;
	d = Fe2.x;
	e = Fe2.y;
	f = Fe2.z;
	g = Fe3.x;
	h = Fe3.y;
	i = Fe3.z;
	
	const ibmPrecisionFloat detFe = dot(cross(Fe1, Fe2), Fe3);
	const ibmPrecisionFloat3 invFe1 = 1./detFe * (ibmPrecisionFloat3)(e*i - f*h, c*h - b*i, b*f - c*e);
	const ibmPrecisionFloat3 invFe2 = 1./detFe * (ibmPrecisionFloat3)(f*g - d*i, a*i - c*g, c*d - a*f);
	const ibmPrecisionFloat3 invFe3 = 1./detFe * (ibmPrecisionFloat3)(d*h - e*g, b*g - a*h, a*e - b*d);

	// tmp = Fe^-1 * Dp
	const ibmPrecisionFloat3 tmp1 = (ibmPrecisionFloat3)(invFe1.x*Dp1.x + invFe1.y*Dp2.x + invFe1.z*Dp3.x, invFe1.x*Dp1.y + invFe1.y*Dp2.y + invFe1.z*Dp3.y, invFe1.x*Dp1.z + invFe1.y*Dp2.z + invFe1.z*Dp3.z);
	const ibmPrecisionFloat3 tmp2 = (ibmPrecisionFloat3)(invFe2.x*Dp1.x + invFe2.y*Dp2.x + invFe2.z*Dp3.x, invFe2.x*Dp1.y + invFe2.y*Dp2.y + invFe2.z*Dp3.y, invFe2.x*Dp1.z + invFe2.y*Dp2.z + invFe2.z*Dp3.z);
	const ibmPrecisionFloat3 tmp3 = (ibmPrecisionFloat3)(invFe3.x*Dp1.x + invFe3.y*Dp2.x + invFe3.z*Dp3.x, invFe3.x*Dp1.y + invFe3.y*Dp2.y + invFe3.z*Dp3.y, invFe3.x*Dp1.z + invFe3.y*Dp2.z + invFe3.z*Dp3.z);

	const ibmPrecisionFloat3 dotFp1 = (ibmPrecisionFloat3)(tmp1.x*F1.x + tmp1.y*F2.x + tmp1.z*F3.x, tmp1.x*F1.y + tmp1.y*F2.y + tmp1.z*F3.y, tmp1.x*F1.z + tmp1.y*F2.z + tmp1.z*F3.z);
	const ibmPrecisionFloat3 dotFp2 = (ibmPrecisionFloat3)(tmp2.x*F1.x + tmp2.y*F2.x + tmp2.z*F3.x, tmp2.x*F1.y + tmp2.y*F2.y + tmp2.z*F3.y, tmp2.x*F1.z + tmp2.y*F2.z + tmp2.z*F3.z);
	const ibmPrecisionFloat3 dotFp3 = (ibmPrecisionFloat3)(tmp3.x*F1.x + tmp3.y*F2.x + tmp3.z*F3.x, tmp3.x*F1.y + tmp3.y*F2.y + tmp3.z*F3.y, tmp3.x*F1.z + tmp3.y*F2.z + tmp3.z*F3.z);

	// Update plastic deformation tensor
	plasticDeformationTensor[0*INSERT_NUM_TETRAS + tetraID] += dotFp1.x;
	plasticDeformationTensor[1*INSERT_NUM_TETRAS + tetraID] += dotFp1.y;
	plasticDeformationTensor[2*INSERT_NUM_TETRAS + tetraID] += dotFp1.z;
	plasticDeformationTensor[3*INSERT_NUM_TETRAS + tetraID] += dotFp2.x;
	plasticDeformationTensor[4*INSERT_NUM_TETRAS + tetraID] += dotFp2.y;
	plasticDeformationTensor[5*INSERT_NUM_TETRAS + tetraID] += dotFp2.z;
	plasticDeformationTensor[6*INSERT_NUM_TETRAS + tetraID] += dotFp3.x;
	plasticDeformationTensor[7*INSERT_NUM_TETRAS + tetraID] += dotFp3.y;
	plasticDeformationTensor[8*INSERT_NUM_TETRAS + tetraID] += dotFp3.z;

	// Finally add forces to ibm
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
	/*if(tetraID == 0){
		printf("Plastic deformation tensor:\n");
		printf("  %f\t%f\t%f\n  %f\t%f\t%f\n  %f\t%f\t%f\n", Fp1.x, Fp1.y, Fp1.z, Fp2.x, Fp2.y, Fp2.z, Fp3.x, Fp3.y, Fp3.z);
		printf("Inverse plastic deformation tensor:\n");
		printf("  %f\t%f\t%f\n  %f\t%f\t%f\n  %f\t%f\t%f\n", invFp1.x, invFp1.y, invFp1.z, invFp2.x, invFp2.y, invFp2.z, invFp3.x, invFp3.y, invFp3.z);
		printf("Elastic deformation tensor:\n");
		printf("  %f\t%f\t%f\n  %f\t%f\t%f\n  %f\t%f\t%f\n", Fe1.x, Fe1.y, Fe1.z, Fe2.x, Fe2.y, Fe2.z, Fe3.x, Fe3.y, Fe3.z);
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
		printf("deviatoric stress tensor:\n");
		printf("  %f\t%f\t%f\n  %f\t%f\t%f\n  %f\t%f\t%f\n", devSigma1.x, devSigma1.y, devSigma1.z, devSigma2.x, devSigma2.y, devSigma2.z, devSigma3.x, devSigma3.y, devSigma3.z);
		printf("vonMisesStress = %f\n", vonMisesStress);
		printf("Plastic Stretch rate = %f\n", plasticStretchRate);
		printf("Inverse elastic deformation tensor:\n");
		printf("  %f\t%f\t%f\n  %f\t%f\t%f\n  %f\t%f\t%f\n", invFe1.x, invFe1.y, invFe1.z, invFe2.x, invFe2.y, invFe2.z, invFe3.x, invFe3.y, invFe3.z);
		printf("tmp Matrix:\n");
		printf("  %f\t%f\t%f\n  %f\t%f\t%f\n  %f\t%f\t%f\n", tmp1.x, tmp1.y, tmp1.z, tmp2.x, tmp2.y, tmp2.z, tmp3.x, tmp3.y, tmp3.z);
		printf("Rate of plastic deformation tensor:\n");
		printf("  %f\t%f\t%f\n  %f\t%f\t%f\n  %f\t%f\t%f\n", dotFp1.x, dotFp1.y, dotFp1.z, dotFp2.x, dotFp2.y, dotFp2.z, dotFp3.x, dotFp3.y, dotFp3.z);
		printf("Viscoplastic parameters:\n");
		printf("Y = %f\n", yieldStress[tetraID]);
		printf("m = %f\n", flowExponent[tetraID]);
		printf("eps0dot = %f\n", plasticFlowRate[tetraID]);
		printf("1/n = %f\n", hardeningExponent[tetraID]);
		printf("eps0 = %f\n", hardeningThreshhold[tetraID]);
		printf("Accumulated plastic strain = %f\n", accumulatedStrain[tetraID]); 
		printf("\n\n\n");
	}*/

}
