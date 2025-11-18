typedef Double3x3 tetraPrecisionFloat3x3;

kernel void Interaction_MooneyRivlinStress(volatile global ibmPrecisionFloat* particleForce, const global ibmPrecisionFloat* points, const global uint* tetras, const global ibmPrecisionFloat* referenceEdgeVectors, const global ibmPrecisionFloat* referenceVolumes, const global ibmPrecisionFloat* mu1, const global ibmPrecisionFloat* mu2, const global ibmPrecisionFloat* kappa, volatile global ibmPrecisionFloat* vonMises, volatile global ibmPrecisionFloat* pressure){
	const uint tetraID = get_global_id(0);
	if(tetraID>=def_tetraCount) return;

	const uint pointID0 = tetras[				  tetraID];
	const uint pointID1 = tetras[  def_tetraCount+tetraID];
	const uint pointID2 = tetras[2*def_tetraCount+tetraID];
	const uint pointID3 = tetras[3*def_tetraCount+tetraID];

	//Calculate the current distances between the particles of one tetrahedron
	const tetraPrecisionFloat3 p0 = (tetraPrecisionFloat3)(points[pointID0], points[def_pointCount+pointID0], points[2*def_pointCount+pointID0]);
	const tetraPrecisionFloat3 p1 = (tetraPrecisionFloat3)(points[pointID1], points[def_pointCount+pointID1], points[2*def_pointCount+pointID1]);
	const tetraPrecisionFloat3 p2 = (tetraPrecisionFloat3)(points[pointID2], points[def_pointCount+pointID2], points[2*def_pointCount+pointID2]);
	const tetraPrecisionFloat3 p3 = (tetraPrecisionFloat3)(points[pointID3], points[def_pointCount+pointID3], points[2*def_pointCount+pointID3]);

    //Calculate the current distances between the particles of one tetrahedron
	// r1 = vector between 1 and 4
	const tetraPrecisionFloat3 r1 = p1-p0;
	// r2 = vector between 2 and 4
	const tetraPrecisionFloat3 r2 = p2-p0;
	// r3 = vector between 3 and 4
	const tetraPrecisionFloat3 r3 = p3-p0;

	const tetraPrecisionFloat3x3 rT = fromColumns(r1, r2, r3);

	// Variables in the reference state
	const tetraPrecisionFloat3 R1 = (tetraPrecisionFloat3)(referenceEdgeVectors[				 tetraID], referenceEdgeVectors[  def_tetraCount+tetraID], referenceEdgeVectors[2*def_tetraCount+tetraID]);
	const tetraPrecisionFloat3 R2 = (tetraPrecisionFloat3)(referenceEdgeVectors[3*def_tetraCount+tetraID], referenceEdgeVectors[4*def_tetraCount+tetraID], referenceEdgeVectors[5*def_tetraCount+tetraID]);
	const tetraPrecisionFloat3 R3 = (tetraPrecisionFloat3)(referenceEdgeVectors[6*def_tetraCount+tetraID], referenceEdgeVectors[7*def_tetraCount+tetraID], referenceEdgeVectors[8*def_tetraCount+tetraID]);
	const tetraPrecisionFloat V0  = referenceVolumes[tetraID];

	// Calculation of displacement vector field replaced by use of JT
	// cf. Carina MT, p. 23, eq. (5.1)
	const tetraPrecisionFloat3x3 J = fromColumns(R1, R2, R3);

	// Calculation of inverse Jacobian Matrix
	// for Jacobian Matrix cf. Carina MT, p. 74, eq. (10.7)
	// Derivative of deformation gradient with respect to displacement field
	// cf. Carina MT, p. A-XXIV
	const tetraPrecisionFloat3x3 invJ = invert(J);

	// Derivative of displacement field with respect to reference position
	// cf. Carina MT, p. 72, eq. (10.3)
	// for definition of displacement field cf. Carina MT, p. 57, eq. (10.1) and p. 72, eq. (10.2)
	// The displacement field was replaced by the r matrix to avoid creating an unnecessary unit tensor
	// dudR = invJT*u = invJT*(r - JT) = invJT*r - uT = FT - uT;
	// Calculation of deformation gradient tensor components
	// cf. Carina MT, p. 23, eq. (5.3)
	// Derivative of trace of B with respect to deformation gradient tensor
	const tetraPrecisionFloat3x3 F = multiply(rT, invJ);
	const tetraPrecisionFloat3x3 FT = transpose(F);

	// T: Trace of B (left Cauchy-Green deformation tensor) = F*F_transpose
	// cf. Carina MT, p. 28, eq. (5.13)
	const tetraPrecisionFloat3x3 B = multiply(F, FT); // symmetric




	// Square of Green Strain
	const tetraPrecisionFloat3x3 Bsq = multiply(B, B); // symmetric

	// Trace of Green Strain and Trace of square of Green Strain
	const tetraPrecisionFloat trB = Tr(B);
	const tetraPrecisionFloat trBsq = Tr(Bsq);

	// Calculation of Jacobian Determinant from Green Strain
	// Also called J in Carina MT -> renamed to Jr(atio) to not conflict with tensor named J
	const tetraPrecisionFloat Jr = det(rT)/(6*V0);

	// U: strain energy density, I1: first invariant propto trace of B
	// cf. Carina MT, p. A-XXIII
	// J terms have been resorted for efficiency
	// A factor 2 has ben resorted for efficiency
	// T: Trace of B (left Cauchy-Green deformation tensor) = F*F_transpose
	// cf. Carina MT, p. 28, eq. (5.13)
	const tetraPrecisionFloat dUdI1 = mu1[tetraID]/pow(Jr, 5/(tetraPrecisionFloat)3);
	const tetraPrecisionFloat dUdK = mu2[tetraID]/pow(Jr, 7/(tetraPrecisionFloat)3);
	const tetraPrecisionFloat dUdJ = kappa[tetraID]*(Jr - 1);
	const tetraPrecisionFloat dI1dJ = -trB/3;

	// Identity tensor
	const tetraPrecisionFloat3x3 uT = unitTensor(tetraPrecisionFloat3x3);

	// Cauchy Stress tensor SJM PhD thesis p.16 from Bower
	const tetraPrecisionFloat3x3 term1 = multiply(B, dUdI1);
	const tetraPrecisionFloat3x3 term2 = multiply(uT, dUdJ + dI1dJ*dUdI1);
	const tetraPrecisionFloat3x3 inner1 = substract(multiply(B, trB), Bsq);
	const tetraPrecisionFloat3x3 inner2 = multiply(uT, (trBsq - trB*trB)/3);
	const tetraPrecisionFloat3x3 term3 = multiply(add(inner1, inner2), dUdK);
	const tetraPrecisionFloat3x3 sigma = add(add(term1, term2), term3); // symmetric
	
	const tetraPrecisionFloat trSigma = Tr(sigma);
	const tetraPrecisionFloat3x3 devSigma = substract(sigma, multiply(uT, trSigma/3));
	
	const ibmPrecisionFloat vonMisesStress = sqrt(3/(tetraPrecisionFloat)2 * Tr(multiply(devSigma, devSigma)));
	vonMises[tetraID] = vonMisesStress;
	pressure[tetraID] = trSigma / 3;

	// surface areas (cyclic: i.e. A1 = surface given by points 2,3,0, belonging to normal n1)
	const tetraPrecisionFloat3x3 helper = multiply(invert(rT), sigma);

	// force on surface areas given by stress * normal * area
	const tetraPrecisionFloat3x3 Force = multiply(helper, -V0*Jr);

	// force components on individual nodes given by 1/3 of the integral of stress*normal over adjacent surfaces
	const tetraPrecisionFloat3 f1 = getRow(Force, 1);
	const tetraPrecisionFloat3 f2 = getRow(Force, 2);
	const tetraPrecisionFloat3 f3 = getRow(Force, 3);

	const tetraPrecisionFloat3 f0 = -(f1 + f2 + f3);

	atomicAdd(&particleForce[            	  pointID0], f0.x); // forces on point 0
	atomicAdd(&particleForce[  def_pointCount+pointID0], f0.y);
	atomicAdd(&particleForce[2*def_pointCount+pointID0], f0.z);
	atomicAdd(&particleForce[           	  pointID1], f1.x); // forces on point 1
	atomicAdd(&particleForce[  def_pointCount+pointID1], f1.y);
	atomicAdd(&particleForce[2*def_pointCount+pointID1], f1.z);
	atomicAdd(&particleForce[            	  pointID2], f2.x); // forces on point 2
	atomicAdd(&particleForce[  def_pointCount+pointID2], f2.y);
	atomicAdd(&particleForce[2*def_pointCount+pointID2], f2.z);
	atomicAdd(&particleForce[        	      pointID3], f3.x); // forces on point 3
	atomicAdd(&particleForce[  def_pointCount+pointID3], f3.y);
	atomicAdd(&particleForce[2*def_pointCount+pointID3], f3.z);
}
