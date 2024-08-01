kernel void Interaction_LinearViscoelastic(volatile global ibmPrecisionFloat* particleForce,
					   const global ibmPrecisionFloat* points,
					   const global uint* tetras,
					   const global ibmPrecisionFloat* referenceEdgeVectors,
					   const global ibmPrecisionFloat* referenceVolumes,
					   const global ibmPrecisionFloat* youngsModulus,
					   const global ibmPrecisionFloat* poissonRatio,
					   const global ibmPrecisionFloat* weights,
					   const global ibmPrecisionFloat* taus,
					   volatile global ibmPrecisionFloat* deviatoricStressTensorsTransient,
					   volatile global ibmPrecisionFloat* oldDeviatoricStrainTensor,
					   const int numberWeights,
					   volatile global ibmPrecisionFloat* vonMises,
					   volatile global ibmPrecisionFloat* pressure){
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

	// Identity tensor
	const ibmPrecisionFloat3 E1 = (ibmPrecisionFloat3)(1, 0, 0);
	const ibmPrecisionFloat3 E2 = (ibmPrecisionFloat3)(0, 1, 0);
	const ibmPrecisionFloat3 E3 = (ibmPrecisionFloat3)(0, 0, 1);

	// Infinitesimal (Cauchy) strain

	const ibmPrecisionFloat3 eps1 = 0.5 * ( du1dR + (ibmPrecisionFloat3)(du1dR.x, du2dR.x, du3dR.x) );
	const ibmPrecisionFloat3 eps2 = 0.5 * ( du2dR + (ibmPrecisionFloat3)(du1dR.y, du2dR.y, du3dR.y) );
	const ibmPrecisionFloat3 eps3 = 0.5 * ( du3dR + (ibmPrecisionFloat3)(du1dR.z, du2dR.z, du3dR.z) );

	const ibmPrecisionFloat trEps = eps1.x + eps2.y + eps3.z;

	const ibmPrecisionFloat3 devEps1 = eps1 - 1./3. * trEps * E1;
	const ibmPrecisionFloat3 devEps2 = eps2 - 1./3. * trEps * E2;
	const ibmPrecisionFloat3 devEps3 = eps3 - 1./3. * trEps * E3;

	// Stress consist of elastic, transient and volumetric part
	// elastic and volumetric are the same as purely linear elastic solid
	// transient part follows Prony series notation to generalized Maxwell model

	// G0 equals the shear modulus in SW notation of generalized Maxwell
	const ibmPrecisionFloat shearModulus = youngsModulus[tetraID] / (2 * (1 + poissonRatio[tetraID]));

	// bulk Modulus K equals kappa from Hyperelastic models
	const ibmPrecisionFloat kappa = youngsModulus[tetraID] / (3 * (1 - 2*poissonRatio[tetraID]));

	// Elastic weight (w_inf) equals 1 - sum(transientWeights)
	ibmPrecisionFloat weightElastic = 1.0;
	for(int j=0; j<numberWeights; j++){
		weightElastic -= weights[j * INSERT_NUM_TETRAS + tetraID];
	}


	const ibmPrecisionFloat3 sigmaElastic1 = 2 * shearModulus * weightElastic * devEps1;
	const ibmPrecisionFloat3 sigmaElastic2 = 2 * shearModulus * weightElastic * devEps2;
	const ibmPrecisionFloat3 sigmaElastic3 = 2 * shearModulus * weightElastic * devEps3;

	const ibmPrecisionFloat3 sigmaVolumetric1 = kappa * trEps * E1;
	const ibmPrecisionFloat3 sigmaVolumetric2 = kappa * trEps * E2;
	const ibmPrecisionFloat3 sigmaVolumetric3 = kappa * trEps * E3;

	const ibmPrecisionFloat3 devEpsOld1 = (ibmPrecisionFloat3)(oldDeviatoricStrainTensor[0*INSERT_NUM_TETRAS + tetraID], oldDeviatoricStrainTensor[1*INSERT_NUM_TETRAS + tetraID], oldDeviatoricStrainTensor[2*INSERT_NUM_TETRAS + tetraID]); 
	const ibmPrecisionFloat3 devEpsOld2 = (ibmPrecisionFloat3)(oldDeviatoricStrainTensor[3*INSERT_NUM_TETRAS + tetraID], oldDeviatoricStrainTensor[4*INSERT_NUM_TETRAS + tetraID], oldDeviatoricStrainTensor[5*INSERT_NUM_TETRAS + tetraID]); 
	const ibmPrecisionFloat3 devEpsOld3 = (ibmPrecisionFloat3)(oldDeviatoricStrainTensor[6*INSERT_NUM_TETRAS + tetraID], oldDeviatoricStrainTensor[7*INSERT_NUM_TETRAS + tetraID], oldDeviatoricStrainTensor[8*INSERT_NUM_TETRAS + tetraID]); 

	// Update "old" deviatoric Strain
	oldDeviatoricStrainTensor[0*INSERT_NUM_TETRAS + tetraID] = devEps1.x;
	oldDeviatoricStrainTensor[1*INSERT_NUM_TETRAS + tetraID] = devEps1.y;
	oldDeviatoricStrainTensor[2*INSERT_NUM_TETRAS + tetraID] = devEps1.z;
	oldDeviatoricStrainTensor[3*INSERT_NUM_TETRAS + tetraID] = devEps2.x;
	oldDeviatoricStrainTensor[4*INSERT_NUM_TETRAS + tetraID] = devEps2.y;
	oldDeviatoricStrainTensor[5*INSERT_NUM_TETRAS + tetraID] = devEps2.z;
	oldDeviatoricStrainTensor[6*INSERT_NUM_TETRAS + tetraID] = devEps3.x;
	oldDeviatoricStrainTensor[7*INSERT_NUM_TETRAS + tetraID] = devEps3.y;
	oldDeviatoricStrainTensor[8*INSERT_NUM_TETRAS + tetraID] = devEps3.z;


	const ibmPrecisionFloat3 devEpsDot1 = devEps1 - devEpsOld1;
	const ibmPrecisionFloat3 devEpsDot2 = devEps2 - devEpsOld2;
	const ibmPrecisionFloat3 devEpsDot3 = devEps3 - devEpsOld3;

	ibmPrecisionFloat3 sigmaTransient1 = (ibmPrecisionFloat3) (0,0,0);
	ibmPrecisionFloat3 sigmaTransient2 = (ibmPrecisionFloat3) (0,0,0);
	ibmPrecisionFloat3 sigmaTransient3 = (ibmPrecisionFloat3) (0,0,0);

	// Add contribution from all maxwell elements
	for(int j=0; j<numberWeights; j++){
		const ibmPrecisionFloat3 maxwellStress1 = exp(-1./taus[j * INSERT_NUM_TETRAS + tetraID]) * (ibmPrecisionFloat3)(deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 0*INSERT_NUM_TETRAS + tetraID],
																deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 1*INSERT_NUM_TETRAS + tetraID],
																deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 2*INSERT_NUM_TETRAS + tetraID]) +
							 2 * shearModulus * weights[j * INSERT_NUM_TETRAS + tetraID] * devEpsDot1;
		const ibmPrecisionFloat3 maxwellStress2 = exp(-1./taus[j * INSERT_NUM_TETRAS + tetraID]) * (ibmPrecisionFloat3)(deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 3*INSERT_NUM_TETRAS + tetraID],
																deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 4*INSERT_NUM_TETRAS + tetraID],
																deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 5*INSERT_NUM_TETRAS + tetraID]) +
							 2 * shearModulus * weights[j * INSERT_NUM_TETRAS + tetraID] * devEpsDot2;
		const ibmPrecisionFloat3 maxwellStress3 = exp(-1./taus[j * INSERT_NUM_TETRAS + tetraID]) * (ibmPrecisionFloat3)(deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 6*INSERT_NUM_TETRAS + tetraID],
																deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 7*INSERT_NUM_TETRAS + tetraID],
																deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 8*INSERT_NUM_TETRAS + tetraID]) +
							 2 * shearModulus * weights[j * INSERT_NUM_TETRAS + tetraID] * devEpsDot3;
		sigmaTransient1 += maxwellStress1;
		sigmaTransient2 += maxwellStress2;
		sigmaTransient3 += maxwellStress3;
		deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 0*INSERT_NUM_TETRAS + tetraID] = maxwellStress1.x;
		deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 1*INSERT_NUM_TETRAS + tetraID] = maxwellStress1.y;
		deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 2*INSERT_NUM_TETRAS + tetraID] = maxwellStress1.z;
		deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 3*INSERT_NUM_TETRAS + tetraID] = maxwellStress2.x;
		deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 4*INSERT_NUM_TETRAS + tetraID] = maxwellStress2.y;
		deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 5*INSERT_NUM_TETRAS + tetraID] = maxwellStress2.z;
		deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 6*INSERT_NUM_TETRAS + tetraID] = maxwellStress3.x;
		deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 7*INSERT_NUM_TETRAS + tetraID] = maxwellStress3.y;
		deviatoricStressTensorsTransient[9*j*INSERT_NUM_TETRAS + 8*INSERT_NUM_TETRAS + tetraID] = maxwellStress3.z;
	}

	// Finally calculate Cauchy Stress tensor 
	const ibmPrecisionFloat3 sigma1 = sigmaElastic1 + sigmaTransient1 + sigmaVolumetric1;
	const ibmPrecisionFloat3 sigma2 = sigmaElastic2 + sigmaTransient2 + sigmaVolumetric2;
	const ibmPrecisionFloat3 sigma3 = sigmaElastic3 + sigmaTransient3 + sigmaVolumetric3;

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

