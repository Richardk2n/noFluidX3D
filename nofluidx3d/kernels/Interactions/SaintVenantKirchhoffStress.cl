typedef Double3x3 tetraPrecisionFloat3x3;

kernel void Interaction_SaintVenantKirchhoffStress(volatile global forcePrecisionFloat* particleForce, const global tetraPrecisionFloat* points, const global int* tetras, const global tetraPrecisionFloat* referenceEdgeVectors, const global tetraPrecisionFloat* referenceVolumes){
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

	const tetraPrecisionFloat3x3 r = fromRows(r1, r2, r3);

	// Variables in the reference state
	const tetraPrecisionFloat3 R1 = (tetraPrecisionFloat3)(referenceEdgeVectors[				 tetraID], referenceEdgeVectors[  def_tetraCount+tetraID], referenceEdgeVectors[2*def_tetraCount+tetraID]);
	const tetraPrecisionFloat3 R2 = (tetraPrecisionFloat3)(referenceEdgeVectors[3*def_tetraCount+tetraID], referenceEdgeVectors[4*def_tetraCount+tetraID], referenceEdgeVectors[5*def_tetraCount+tetraID]);
	const tetraPrecisionFloat3 R3 = (tetraPrecisionFloat3)(referenceEdgeVectors[6*def_tetraCount+tetraID], referenceEdgeVectors[7*def_tetraCount+tetraID], referenceEdgeVectors[8*def_tetraCount+tetraID]);
	const tetraPrecisionFloat V0  = referenceVolumes[tetraID];

	// Calculation of displacement vector field replaced by use of JT
	// cf. Carina MT, p. 23, eq. (5.1)
	const tetraPrecisionFloat3x3 JT = fromRows(R1, R2, R3);

	// Young's modulus and Poisson ratio
	// cf. Carina MT, p. 40, eqs. (5.48) and (5.49)
	const tetraPrecisionFloat E = def_tetraYoungsModulus;
	const tetraPrecisionFloat nu = def_tetraPoissonRatio;

	/* Saint Venant-Kirchhoff ---> */
	// some terms to shorten the force calculation
	const tetraPrecisionFloat kappa = nu/(1 - 2*nu);
	const tetraPrecisionFloat mu = E/(2*(1 + nu));

	// Calculation of inverse Jacobian Matrix
	// for Jacobian Matrix cf. Carina MT, p. 74, eq. (10.7)
	const tetraPrecisionFloat3x3 invJT = invert(JT);

	// Derivative of displacement field with respect to reference position
	// cf. Carina MT, p. 72, eq. (10.3)
	// for definition of displacement field cf. Carina MT, p. 57, eq. (10.1) and p. 72, eq. (10.2)
	// The displacement field was replaced by the r matrix to avoid creating an unnecessary unit tensor
	// dudR = invJT*u = invJT*(r - JT) = invJT*r - uT;
	// Calculation of deformation gradient tensor components
	// cf. Carina MT, p. 23, eq. (5.3)
	const tetraPrecisionFloat3x3 FT = multiply(invJT, r); // FT = dudRP

	// T: Trace of B (left Cauchy-Green deformation tensor) = F*F_transpose
	// cf. Carina MT, p. 28, eq. (5.13)
	const tetraPrecisionFloat3x3 B = multiply(transpose(FT), FT); // symmetric
	const tetraPrecisionFloat3x3 BP = substract(B, unitTensor(tetraPrecisionFloat3x3));

	// A term
	const tetraPrecisionFloat3x3 H = multiply(transpose(invJT), FT);

	// Force component calculation
	// cf. Carina MT, p. 87, eq. (11.3) and pp. A-XI to A-XXII
	const tetraPrecisionFloat3x3 term1 = multiply(H, kappa*Tr(BP));
	const tetraPrecisionFloat3x3 term2 = multiply(H, BP);

	const tetraPrecisionFloat3x3 F = multiply(add(term1, term2), -V0*mu);

	const tetraPrecisionFloat3 f1 = getRow(F, 1);
	const tetraPrecisionFloat3 f2 = getRow(F, 2);
	const tetraPrecisionFloat3 f3 = getRow(F, 3);
	/* <--- Saint Venant-Kirchhoff */

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
