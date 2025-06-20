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

	const double E = def_tetraYoungsModulus;
	const double nu = def_tetraPoissonRatio;

	/* Saint Venant-Kirchhoff ---> */
	// some terms to shorten the force calculation
	const tetraPrecisionFloat nuR = nu/(1 - 2*nu);
	const double e = 6*V0; // TODO name
	const double post = 2*(1 + nu)*pow(e,4);
	const double pre = -E*V0/post;

	// Matrices
	const tetraPrecisionFloat3x3 C = fromColumns(cross(R2, R3), cross(R3, R1), cross(R1, R2));
	const tetraPrecisionFloat3x3 D = multiply(C, r);
	const tetraPrecisionFloat3x3 tmp = multiply(unitTensor(tetraPrecisionFloat3x3), -pow(e,2));
	const tetraPrecisionFloat3x3 G = multiply(transpose(D), D); // symmetric // rTCTCr
	const tetraPrecisionFloat3x3 GP = add(multiply(transpose(D), D), tmp);
	const tetraPrecisionFloat3x3 H = multiply(transpose(C), D); // CTCr

	// G' = G - uT*pow(e,2)
	// G' + uT*pow(e,2) = G
	// HG = HG' + H*pow(e,2)
	// Tr G' = Tr(G) - 3*pow(e,2)

	// D' = D + uT*e
	// D'T = DT + uT*e
	// D'TD'' = DTD + De + DTe + e**2

	// Force component calculation
	// cf. Carina MT, p. 87, eq. (11.3) and pp. A-XI to A-XXII
	const tetraPrecisionFloat3x3 term1 = multiply(H, nuR*Tr(GP));
	const tetraPrecisionFloat3x3 term2 = multiply(H, GP);

	const tetraPrecisionFloat3x3 F = multiply(add(term1, term2), pre);

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
