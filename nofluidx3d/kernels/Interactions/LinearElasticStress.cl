kernel void Interaction_LinearElasticStress2(volatile global forcePrecisionFloat* particleForce, const global tetraPrecisionFloat* points, const global int* tetras, const global tetraPrecisionFloat* referenceEdgeVectors, const global tetraPrecisionFloat* referenceVolumes){
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
	const tetraPrecisionFloat V0  = -referenceVolumes[tetraID];

	tetraPrecisionFloat3 f1 = (tetraPrecisionFloat3)(0, 0, 0);
	tetraPrecisionFloat3 f2 = (tetraPrecisionFloat3)(0, 0, 0);
	tetraPrecisionFloat3 f3 = (tetraPrecisionFloat3)(0, 0, 0);

	const double R11 = R1.x;
	const double R12 = R1.y;
	const double R13 = R1.z;
	const double R21 = R2.x;
	const double R22 = R2.y;
	const double R23 = R2.z;
	const double R31 = R3.x;
	const double R32 = R3.y;
	const double R33 = R3.z;
	const double r11 = r1.x;
	const double r12 = r1.y;
	const double r13 = r1.z;
	const double r21 = r2.x;
	const double r22 = r2.y;
	const double r23 = r2.z;
	const double r31 = r3.x;
	const double r32 = r3.y;
	const double r33 = r3.z;

	const double E = def_tetraYoungsModulus;
	const double nu = def_tetraPoissonRatio;

	/* Saint Venant-Kirchhoff ---> */
	// some terms to shorten the force calculation
	const double c = R11*r23*R32 - R11*R23*R32 + R12*R21*r33 - R11*R22*r33 - R12*R21*R33 + R11*R22*R33;
	const double d = R12*r22*R31 - r12*R22*R31 - R12*R21*r32 + R11*R22*r32 + r12*R21*R32 - R11*r22*R32;
	const double e = R13*R22*R31 - R12*R23*R31 - R13*R21*R32 + R11*R23*R32 + R12*R21*R33 - R11*R22*R33;
	const double f = R13*R21*r31 - R11*R23*r31 - R13*r21*R31 + r11*R23*R31 + R11*r21*R33 - r11*R21*R33;
	const double g = R12*R21*r31 - R11*R22*r31 - R12*r21*R31 + r11*R22*R31 + R11*r21*R32 - r11*R21*R32;
	const double i = R13*r23*R31 - r13*R23*R31 - R13*R21*r33 + R11*R23*r33 + r13*R21*R33 - R11*r23*R33;
	const double l = R13*r23*R32 - r13*R23*R32 - R13*R22*r33 + R12*R23*r33 + r13*R22*R33 - R12*r23*R33;
	const double m = R13*R22*r32 - R12*R23*r32 - R13*r22*R32 + r12*R23*R32 + R12*r22*R33 - r12*R22*R33;
	const double h = (R13*R22 - R12*R23)*(r31 - R31) - (r21 - R21)*(R13*R32 - R12*R33) + (r11 - R11)*(R23*R32 - R22*R33);
	const double k = (R13*R21 - R11*R23)*(r32 - R32) - (r22 - R22)*(R13*R31 - R11*R33) + (r12 - R12)*(R23*R31 - R21*R33);

	// Force component calculation
	// cf. Carina MT, p. 87, eq. (11.3) and pp. A-XI to A-XXII
	f1[0] = (E*V0*((1 - 2*nu)*(R22*R31 - R21*R32)*(g)*(g*g + d*d+ pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2)\
			- 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e))+ (1 - 2*nu)*(-((g)*(-(R23*R31) + R21*R33))\
			- (R22*R31 - R21*R32)*(-(R13*R21*r31) + R11*R23*r31 + R13*r21*R31 - r11*R23*R31 - R11*r21*R33 + r11*R21*R33))*((g)*(f)+ (d)*(-(r12*R23*R31)\
			+ R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33 - R12*R21*R33 - R11*r22*R33 + R11*R22*R33)\
			- (e)*(-(R12*r22*R31) + r12*R22*R31 + R13*r23*R31 - r13*R23*R31 + R12*R21*r32 - R11*R22*r32 - r12*R21*R32 + R11*r22*R32 - R13*R21*r33 + R11*R23*r33 + r13*R21*R33 - R11*r23*R33)\
			+ (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + c)*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33))\
			+ (1 - 2*nu)*(R23*R31 - R21*R33)*(f)*(f*f + i*i- 2*(e)*(k)+ k*k)+ (1 - 2*nu)*(R23*R32 - R22*R33)*(R13*R22*r31 - R12*R23*r31 - R13*r21*R32 + r11*R23*R32 + R12*r21*R33 - r11*R22*R33)\
			*(m*m + l*l+ 2*(e)*(h)+ h*h)+ nu*((R22*R31 - R21*R32)*(g)+ (R23*R31 - R21*R33)*(f) + (R23*R32 - R22*R33)*(e) + (R23*R32 - R22*R33)*(h))*(g*g + d*d\
			+ pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2) + f*f\
			- 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e)+ m*m + i*i+ l*l - 2*(e)*(k)+ k*k+ 2*(e)*(h)+h*h)\
			+ (1 - 2*nu)*(R13*(R22*r31 - r21*R32)*(R22*R31 - R21*R32) + (R11*R22*r31 - 2*r11*R22*R31 - R11*r21*R32 + 2*r11*R21*R32)*(-(R23*R32) + R22*R33)\
			- R12*(R22*R23*r31*R31 - 2*R21*R23*r31*R32 + r21*R23*R31*R32 + R21*R22*r31*R33 - 2*r21*R22*R31*R33 + r21*R21*R32*R33))*(-((d)*(m))\
			- (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + c)*(l)\
			- (g)*(R13*(R22*(-r31 + R31) + (r21 - R21)*R32) + R12*(R23*r31 - R23*R31 - r21*R33 + R21*R33) - (r11 - R11)*(R23*R32 - R22*R33))\
			+ (e)*(-(R11*R22*r31) + r11*R22*R31 + R11*r21*R32 - r11*R21*R32 - R13*r23*R32 + r13*R23*R32 + R13*R22*r33 - r13*R22*R33 + R12*(R21*r31 - r21*R31 - R23*r33 + r23*R33)))\
			+ (1 - 2*nu)*(-((e)*(R11*R23*r31 - r11*R23*R31 - R12*R23*r32 + r12*R23*R32 + R13*(-(R21*r31) + r21*R31 + R22*r32 - r22*R32) - R11*r21*R33 + r11*R21*R33 + R12*r22*R33 - r12*R22*R33))\
			+ (-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33 - R12*R21*R33 - R11*r22*R33 + R11*R22*R33)\
			*(-(R13*R22*r32) + R12*R23*r32 + R13*r22*R32 - r12*R23*R32 - R12*r22*R33 + r12*R22*R33)+ (i)*(l) + (f)*(R13*(R22*(r31 - R31) + (-r21 + R21)*R32) + R12*(-(R23*r31) + R23*R31 + r21*R33 - R21*R33) + (r11 - R11)*(R23*R32 - R22*R33)))\
			*(-(R12*(R23*r31 - r21*R33)*(R23*R31 - R21*R33)) - (R11*R23*r31 - 2*r11*R23*R31 - R11*r21*R33 + 2*r11*R21*R33)*(R23*R32 - R22*R33)\
			+ R13*(R32*(R21*R23*r31 - 2*r21*R23*R31 + r21*R21*R33) + R22*(R23*r31*R31 - 2*R21*r31*R33 + r21*R31*R33)))))/(2.*(1 - 2*nu)*(1 + nu)*pow(e,4));

	f1[1] = (E*V0*((1 - 2*nu)*(R22*R31 - R21*R32)*(-(R12*r22*R31) + r12*R22*R31 + R12*R21*r32 - R11*R22*r32 - r12*R21*R32 + R11*r22*R32)*(g*g + d*d\
			+ pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2) - 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e))\
			+ (1 - 2*nu)*(-(R13*(r22*R31 - R21*r32)*(R22*R31 - R21*R32)) + R23*(R31*(-(R12*r22*R31) + 2*r12*R22*R31 + R12*R21*r32 - 2*R11*R22*r32) + (-2*r12*R21*R31 + R11*r22*R31 + R11*R21*r32)*R32)\
			+ (R12*R21*(r22*R31 - R21*r32) + 2*r12*R21*(-(R22*R31) + R21*R32) + R11*(r22*R22*R31 + R21*R22*r32 - 2*R21*r22*R32))*R33)*((g)*(f) + (d)*(-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33 - R12*R21*R33 - R11*r22*R33 + R11*R22*R33)\
			- (e)*(-(R12*r22*R31) + r12*R22*R31 + R13*r23*R31 - r13*R23*R31 + R12*R21*r32 - R11*R22*r32 - r12*R21*R32 + R11*r22*R32 - R13*R21*r33 + R11*R23*r33 + r13*R21*R33 - R11*r23*R33)\
			+ (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + c)*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33))\
			- (1 - 2*nu)*(R23*R31 - R21*R33)*(R13*r22*R31 - r12*R23*R31 - R13*R21*r32 + R11*R23*r32 + r12*R21*R33 - R11*r22*R33)*(f*f + i*i - 2*(e)*(k) + k*k) + (1 - 2*nu)*(R23*R32 - R22*R33)*(m)\
			*(pow(m,2) + pow(l,2) + 2*(e)*(h) + pow(h,2)) + nu*((R22*R31 - R21*R32)*(-(R12*r22*R31) + r12*R22*R31 + R12*R21*r32 - R11*R22*r32 - r12*R21*R32 + R11*r22*R32)\
			+ (R23*R31 - R21*R33)*(-(R13*R22*R31) + R12*R23*R31 + R13*R21*R32 - R11*R23*R32 - R12*R21*R33 + R11*R22*R33) + (R23*R32 - R22*R33)*(m) + (R23*R31 - R21*R33)*(k))*(pow(g,2) + pow(d,2)\
			+ pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2) + pow(f,2) - 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e)\
			+ pow(m,2) + pow(i,2)+ pow(l,2) - 2*(e)*(k) + pow(k,2) + 2*(e)*(h) + pow(h,2))+ (1 - 2*nu)*((d)*(-(R23*R32) + R22*R33) + (R22*R31 - R21*R32)*(m))\
			*(-((d)*(m)) - (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + c)*(l)\
			- (g)*(R13*(R22*(-r31 + R31) + (r21 - R21)*R32) + R12*(R23*r31 - R23*R31 - r21*R33 + R21*R33) - (r11 - R11)*(R23*R32 - R22*R33)) + (e)\
			*(-(R11*R22*r31) + r11*R22*R31 + R11*r21*R32 - r11*R21*R32 - R13*r23*R32 + r13*R23*R32 + R13*R22*r33 - r13*R22*R33 + R12*(R21*r31 - r21*R31 - R23*r33 + r23*R33))) - (1 - 2*nu)*(-((e)\
			*(R11*R23*r31 - r11*R23*R31 - R12*R23*r32 + r12*R23*R32 + R13*(-(R21*r31) + r21*R31 + R22*r32 - r22*R32) - R11*r21*R33 + r11*R21*R33 + R12*r22*R33 - r12*R22*R33))\
			+ (-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33 - R12*R21*R33 - R11*r22*R33 + R11*R22*R33)\
			*(-(R13*R22*r32) + R12*R23*r32 + R13*r22*R32 - r12*R23*R32 - R12*r22*R33 + r12*R22*R33)+(i)*(l)\
			+ (f)*(R13*(R22*(r31 - R31) + (-r21 + R21)*R32) + R12*(-(R23*r31) + R23*R31 + r21*R33 - R21*R33) + (r11 - R11)*(R23*R32 - R22*R33)))\
			*(R12*(R23*R31 - R21*R33)*(R23*r32 - r22*R33) - (2*r12*R23*R31 - R11*R23*r32 - 2*r12*R21*R33 + R11*r22*R33)*(R23*R32 - R22*R33)\
			- R13*(R32*(-2*r22*R23*R31 + R21*R23*r32 + R21*r22*R33) + R22*(R23*R31*r32 + r22*R31*R33 - 2*R21*r32*R33)))))/(2.*(1 - 2*nu)*(1 + nu)*pow(e,4));

	f1[2] = (E*V0*((1 - 2*nu)*(R22*R31 - R21*R32)*(r13*R22*R31 - R12*r23*R31 - r13*R21*R32 + R11*r23*R32 + R12*R21*r33 - R11*R22*r33)*(pow(g,2) + pow(d,2)\
			+ pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2) - 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))\
			*(e)) - (1 - 2*nu)*(r23*R23*R31*(R12*R31 - R11*R32) - R23*(R12*R21*R31 - 2*R11*R22*R31 + R11*R21*R32)*r33\
			+ R13*(R22*R31 - R21*R32)*(r23*R31 - R21*r33) - (R12*R21*(r23*R31 - R21*r33) + R11*(R22*r23*R31 - 2*R21*r23*R32 + R21*R22*r33))*R33 - 2*r13*(R22*R31 - R21*R32)*(R23*R31 - R21*R33))*((g)\
			*(f) + (d)*(-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33\
			- R12*R21*R33 - R11*r22*R33 + R11*R22*R33) - (e)*(-(R12*r22*R31) + r12*R22*R31 + R13*r23*R31 - r13*R23*R31 + R12*R21*r32 - R11*R22*r32 - r12*R21*R32 + R11*r22*R32 - R13*R21*r33\
			+ R11*R23*r33 + r13*R21*R33 - R11*r23*R33) + (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + c)\
			*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33))+(1 - 2*nu)*(R23*R31 - R21*R33)*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33)\
			*(pow(f,2) + pow(i,2) - 2*(e)*(k) + pow(k,2)) +(1 - 2*nu)*(R23*R32 - R22*R33)*(-(R13*r23*R32) + r13*R23*R32 + R13*R22*r33 - R12*R23*r33 - r13*R22*R33 + R12*r23*R33)\
			*(pow(m,2) + pow(l,2) + 2*(e)*(h) + pow(h,2)) - nu*((R22*R31 - R21*R32)*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))\
			-  (R22*R31 - R21*R32)*(e) + (R23*R31 - R21*R33)*(i) + (R23*R32 - R22*R33)*(l))*(pow(g,2) + pow(d,2) + pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2)\
			+ pow(f,2) - 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e) + pow(m,2) + pow(i,2) + pow(l,2) - 2*(e)*(k) + pow(k,2) + 2*(e)*(h)\
			+ pow(h,2)) + (1 - 2*nu)*((R23*R32 - R22*R33)*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33)\
			+ (R23*R31 - R21*R33)*(-(R13*r23*R32) + r13*R23*R32 + R13*R22*r33 - R12*R23*r33 - r13*R22*R33 + R12*r23*R33))*(-((e)\
			*(R11*R23*r31 - r11*R23*R31 - R12*R23*r32 + r12*R23*R32 + R13*(-(R21*r31) + r21*R31 + R22*r32 - r22*R32) - R11*r21*R33 + r11*R21*R33 + R12*r22*R33 - r12*R22*R33))\
			+ (-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33\
			- R12*R21*R33 - R11*r22*R33 + R11*R22*R33)*(-(R13*R22*r32) + R12*R23*r32 + R13*r22*R32 - r12*R23*R32 - R12*r22*R33 + r12*R22*R33) + (i)*(l) + (f)\
			*(R13*(R22*(r31 - R31) + (-r21 + R21)*R32) + R12*(-(R23*r31) + R23*R31 + r21*R33 - R21*R33) + (r11 - R11)*(R23*R32 - R22*R33)))\
			- (1 - 2*nu)*(R23*R32*(-2*r13*R22*R31 + R12*r23*R31 + 2*r13*R21*R32 - R11*r23*R32) + R23*(R12*R22*R31 - 2*R12*R21*R32 + R11*R22*R32)*r33 - R13*(R22*R31 - R21*R32)*(-(r23*R32) + R22*r33)\
			+ (2*r13*R22*(R22*R31 - R21*R32) + R11*R22*(r23*R32 - R22*r33) + R12*(-2*R22*r23*R31 + R21*r23*R32 + R21*R22*r33))*R33)*(-((d)*(m))\
			- (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + R11*r23*R32 - R11*R23*R32 + R12*R21*r33\
			- R11*R22*r33 - R12*R21*R33 + R11*R22*R33)*(l) -(g)*(R13*(R22*(-r31 + R31) + (r21 - R21)*R32) + R12*(R23*r31 - R23*R31 - r21*R33 + R21*R33) - (r11 - R11)*(R23*R32 - R22*R33))\
			+ (e)*(-(R11*R22*r31) + r11*R22*R31 + R11*r21*R32 - r11*R21*R32 - R13*r23*R32 + r13*R23*R32 + R13*R22*r33 - r13*R22*R33 + R12*(R21*r31 - r21*R31 - R23*r33 + r23*R33)))))\
			/(2.*(1 - 2*nu)*(1 + nu)*pow(e,4));

	f2[0] = -(E*V0*((1 - 2*nu)*(R12*R31 - R11*R32)*(g)*(pow(g,2) + pow(d,2)+ pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2)\
			- 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e)) + (1 - 2*nu)*((g)*(R13*R31 - R11*R33) + (R12*R31 - R11*R32)*(f))\
			*((g)*(f) + (d)*(-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33\
			- R12*R21*R33 - R11*r22*R33 + R11*R22*R33) - (e)*(-(R12*r22*R31) + r12*R22*R31 + R13*r23*R31 - r13*R23*R31 + R12*R21*r32 - R11*R22*r32 - r12*R21*R32 + R11*r22*R32 - R13*R21*r33\
			+ R11*R23*r33 + r13*R21*R33 - R11*r23*R33) + (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + c)\
			*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33)) + (1 - 2*nu)*(R13*R31 - R11*R33)*(f)*(pow(f,2) + pow(i,2)\
			- 2*(e)*(k) + pow(k,2)) + (1 - 2*nu)*(R13*R32 - R12*R33)*(R13*R22*r31 - R12*R23*r31 - R13*r21*R32 + r11*R23*R32 + R12*r21*R33 - r11*R22*R33)\
			*(pow(m,2) + pow(l,2) + 2*(e)*(h) + pow(h,2)) + nu*((R12*R31 - R11*R32)*(g) + (R13*R31 - R11*R33)*(f) + (R13*R32 - R12*R33)*(e) + (R13*R32 - R12*R33)*(h))\
			*(pow(g,2) + pow(d,2) + pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2)\
			+ pow(f,2) - 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e) + pow(m,2) + pow(i,2) + pow(l,2) - 2*(e)\
			*(k) + pow(k,2) + 2*(e)*(h) + pow(h,2)) + (1 - 2*nu)*(-((e)*(R11*R23*r31 - r11*R23*R31 - R12*R23*r32 + r12*R23*R32 + R13*(-(R21*r31) + r21*R31 + R22*r32 - r22*R32) - R11*r21*R33\
			+ r11*R21*R33 + R12*r22*R33 - r12*R22*R33)) + (-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33\
			- R12*R21*R33 - R11*r22*R33 + R11*R22*R33)*(-(R13*R22*r32) + R12*R23*r32 + R13*r22*R32 - r12*R23*R32 - R12*r22*R33 + r12*R22*R33) + (i)*(l) + (f)\
			*(R13*(R22*(r31 - R31) + (-r21 + R21)*R32) + R12*(-(R23*r31) + R23*R31 + r21*R33 - R21*R33) + (r11 - R11)*(R23*R32 - R22*R33))) \
			*(-(R13*R23*(R12*r31*R31 + R11*r31*R32 - 2*r11*R31*R32)) + pow(R13,2)*(R22*r31*R31 + R21*r31*R32 - 2*r21*R31*R32)\
			- R13*(R12*R21*r31 + R11*R22*r31 - 2*R12*r21*R31 + r11*R22*R31 - 2*R11*r21*R32 + r11*R21*R32)*R33\
			+ R33*(r11*R12*(-(R23*R31) + R21*R33) + R11*(2*R12*R23*r31 - r11*R23*R32 - 2*R12*r21*R33 + r11*R22*R33)))\
			+ (1 - 2*nu)*(-(pow(R12,2)*(R23*r31*R31 + R21*r31*R33 - 2*r21*R31*R33)) + R12*(R23*(R11*r31 + r11*R31)*R32 + R13*(R22*r31*R31 + R21*r31*R32 - 2*r21*R31*R32)\
			+ (R11*R22*r31 - 2*r11*R22*R31 - 2*R11*r21*R32 + r11*R21*R32)*R33) + R32*(r11*R13*(R22*R31 - R21*R32) + R11*(-2*R13*R22*r31 + 2*R13*r21*R32 - r11*R23*R32 + r11*R22*R33)))\
			*(-((d)*(m)) - (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + R11*r23*R32 - R11*R23*R32 + R12*R21*r33 - R11*R22*r33 - R12*R21*R33 + R11*R22*R33)*(l)\
			- (g)*(R13*(R22*(-r31 + R31) + (r21 - R21)*R32) + R12*(R23*r31 - R23*R31 - r21*R33 + R21*R33) - (r11 - R11)*(R23*R32 - R22*R33)) + (e)\
			*(-(R11*R22*r31) + r11*R22*R31 + R11*r21*R32 - r11*R21*R32 - R13*r23*R32 + r13*R23*R32 + R13*R22*r33 - r13*R22*R33 + R12*(R21*r31 - r21*R31 - R23*r33 + r23*R33)))))\
			/(2.*(1 - 2*nu)*(1 + nu)*pow(e,4));


	f2[1] = -(E*V0*((1 - 2*nu)*(R12*R31 - R11*R32)*(-(R12*r22*R31) + r12*R22*R31 + R12*R21*r32 - R11*R22*r32 - r12*R21*R32 + R11*r22*R32)\
			*(pow(g,2) + pow(d,2) + pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2)\
			- 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e))\
			- (1 - 2*nu)*(R13*R31 - R11*R33)*(R13*r22*R31 - r12*R23*R31 - R13*R21*r32 + R11*R23*r32 + r12*R21*R33 - R11*r22*R33)*(pow(f,2) + pow(i,2)\
			- 2*(e)*(k) + pow(k,2)) + (1 - 2*nu)*(R13*R32 - R12*R33)*(m)*(pow(m,2) + pow(l,2) + 2*(e)*(h) +  pow(h,2)) + nu*(-((R12*R31 - R11*R32)*(d))\
			- (R13*R31 - R11*R33)*(e) + (R13*R32 - R12*R33)*(m) + (R13*R31 - R11*R33)*(k))*(pow(g,2) +  pow(d,2) + pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2)\
			+ pow(f,2) - 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e) + pow(m,2) + pow(i,2) + pow(l,2)\
			- 2*(e)*(k) +  pow(k,2) + 2*(e)*(h) + pow(h,2)) + (1 - 2*nu)*((d)*(-(R13*R32) + R12*R33) - (R12*R31 - R11*R32)*(-(R13*R22*r32) + R12*R23*r32 + R13*r22*R32 - r12*R23*R32 - R12*r22*R33 + r12*R22*R33))\
			*(-((d)*(m)) - (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + R11*r23*R32 - R11*R23*R32 + R12*R21*r33\
			- R11*R22*r33 - R12*R21*R33 + R11*R22*R33)*(l) - (g)*(R13*(R22*(-r31 + R31) + (r21 - R21)*R32) + R12*(R23*r31 - R23*R31 - r21*R33 + R21*R33) - (r11 - R11)*(R23*R32 - R22*R33))\
			+ (e)*(-(R11*R22*r31) + r11*R22*R31 + R11*r21*R32 - r11*R21*R32 - R13*r23*R32 + r13*R23*R32 + R13*R22*r33 - r13*R22*R33 + R12*(R21*r31 - r21*R31 - R23*r33 + r23*R33)))\
			- (1 - 2*nu)*(-((e)*(R11*R23*r31 - r11*R23*R31 - R12*R23*r32 + r12*R23*R32 + R13*(-(R21*r31) + r21*R31 + R22*r32 - r22*R32) - R11*r21*R33 + r11*R21*R33 + R12*r22*R33 - r12*R22*R33))\
			+ (-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33\
			- R12*R21*R33 - R11*r22*R33 + R11*R22*R33)*(-(R13*R22*r32) + R12*R23*r32 + R13*r22*R32 - r12*R23*R32 - R12*r22*R33 + r12*R22*R33)\
			+ (i)*(l) + (f)*(R13*(R22*(r31 - R31) + (-r21 + R21)*R32) + R12*(-(R23*r31) + R23*R31 + r21*R33 - R21*R33) + (r11 - R11)*(R23*R32 - R22*R33)))\
			*(-(pow(R13,2)*(R22*R31*r32 - 2*r22*R31*R32 + R21*r32*R32)) + R33*(2*R11*R12*(-(R23*r32) + r22*R33) + r12*(R12*R23*R31 + R11*R23*R32 - R12*R21*R33 - R11*R22*R33))\
			+ R13*(-2*r12*R23*R31*R32 + R11*R23*r32*R32 + r12*R22*R31*R33 + R11*R22*r32*R33 + r12*R21*R32*R33 - 2*R11*r22*R32*R33 + R12*(R23*R31*r32 - 2*r22*R31*R33 + R21*r32*R33)))\
			+ (1 - 2*nu)*((g)*(f) + (d)*(-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33\
			- R12*R21*R33 - R11*r22*R33 + R11*R22*R33) - (e)*(-(R12*r22*R31) + r12*R22*R31 + R13*r23*R31 - r13*R23*R31 + R12*R21*r32 - R11*R22*r32 - r12*R21*R32 + R11*r22*R32 - R13*R21*r33\
			+ R11*R23*r33 + r13*R21*R33 - R11*r23*R33) + (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + c)\
			*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33))*(R12*R31*(-2*R13*r22*R31 + r12*R23*R31 + 2*R13*R21*r32 - R11*R23*r32) - R12*(r12*R21*R31 - 2*R11*r22*R31 + R11*R21*r32)*R33\
			+ r12*(R13*R31*(R22*R31 - R21*R32) - R11*(R23*R31*R32 + R22*R31*R33 - 2*R21*R32*R33)) + R11*(-(R13*(R22*R31*r32 - 2*r22*R31*R32 + R21*r32*R32)) + R11*(R23*r32*R32 + R22*r32*R33 - 2*r22*R32*R33)))))\
			/(2.*(1 - 2*nu)*(1 + nu)*pow(e,4));

	f2[2] = - (E*V0*((1 - 2*nu)*(R12*R31 - R11*R32)*(r13*R22*R31 - R12*r23*R31 - r13*R21*R32 + R11*r23*R32 + R12*R21*r33 - R11*R22*r33)\
			*(pow(g,2) + pow(d,2) + pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2)\
			- 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e))\
			+ (1 - 2*nu)*(R13*R31 - R11*R33)*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33)*(pow(f,2) + pow(i,2)\
			- 2*(e)*(k) + pow(k,2)) + (1 - 2*nu)*(R13*R32 - R12*R33)*(-(R13*r23*R32) + r13*R23*R32 + R13*R22*r33 - R12*R23*r33 - r13*R22*R33 + R12*r23*R33)\
			*(pow(m,2) + pow(l,2) + 2*(e)*(h) + pow(h,2)) - nu*((R12*R31 - R11*R32)*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))\
			+ (R12*R31 - R11*R32)*(-(R13*R22*R31) + R12*R23*R31 + R13*R21*R32 - R11*R23*R32 - R12*R21*R33 + R11*R22*R33) + (R13*R31 - R11*R33)*(i)\
			+ (R13*R32 - R12*R33)*(l))*(pow(g,2) + pow(d,2) + pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2)\
			+ pow(f,2) - 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e) + pow(m,2) + pow(i,2) + pow(l,2)\
			- 2*(e)*(k) + pow(k,2) + 2*(e)*(h) + pow(h,2)) + (1 - 2*nu)*((R13*R32 - R12*R33)*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33)\
			+ (R13*R31 - R11*R33)*(-(R13*r23*R32) + r13*R23*R32 + R13*R22*r33 - R12*R23*r33 - r13*R22*R33 + R12*r23*R33))*(-((e)\
			*(R11*R23*r31 - r11*R23*R31 - R12*R23*r32 + r12*R23*R32 + R13*(-(R21*r31) + r21*R31 + R22*r32 - r22*R32) - R11*r21*R33\
			+ r11*R21*R33 + R12*r22*R33 - r12*R22*R33)) + (-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33\
			- R12*R21*R33 - R11*r22*R33 + R11*R22*R33)*(-(R13*R22*r32) + R12*R23*r32 + R13*r22*R32 - r12*R23*R32 - R12*r22*R33 + r12*R22*R33) + (i)*(l) + (f)\
			*(R13*(R22*(r31 - R31) + (-r21 + R21)*R32) + R12*(-(R23*r31) + R23*R31 + r21*R33 - R21*R33) + (r11 - R11)*(R23*R32 - R22*R33))) -(1 - 2*nu)*(-((d)*(m))\
			- (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + R11*r23*R32 - R11*R23*R32 + R12*R21*r33\
			- R11*R22*r33 - R12*R21*R33 + R11*R22*R33)*(l) - (g)*(R13*(R22*(-r31 + R31) + (r21 - R21)*R32) + R12*(R23*r31 - R23*R31 - r21*R33 + R21*R33) - (r11 - R11)*(R23*R32 - R22*R33))\
			+ (e)*(-(R11*R22*r31) + r11*R22*R31 + R11*r21*R32 - r11*R21*R32 - R13*r23*R32 + r13*R23*R32 + R13*R22*r33 - r13*R22*R33\
			+ R12*(R21*r31 - r21*R31 - R23*r33 + r23*R33)))*(r13*R32*(-(R13*R22*R31) - R12*R23*R31 + R13*R21*R32 + R11*R23*R32) + 2*R11*R13*R32*(-(r23*R32) + R22*r33)\
			- r13*(-2*R12*R22*R31 + R12*R21*R32 + R11*R22*R32)*R33 + pow(R12,2)*(R23*R31*r33 - 2*r23*R31*R33 + R21*r33*R33) + R12*(2*R13*r23*R31*R32 - R13*(R22*R31 + R21*R32)*r33 - R11*(R23*R32*r33 - 2*r23*R32*R33 + R22*r33*R33)))\
			- (1 - 2*nu)*((g)*(f) + (d)*(-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33\
			- R12*R21*R33 - R11*r22*R33 + R11*R22*R33) - (e)*(-(R12*r22*R31) + r12*R22*R31 + R13*r23*R31 - r13*R23*R31 + R12*R21*r32 - R11*R22*r32 - r12*R21*R32 + R11*r22*R32 - R13*R21*r33\
			+ R11*R23*r33 + r13*R21*R33 - R11*r23*R33) + (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + c)\
			*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33))*(r13*R31*(-(R13*R22*R31) - R12*R23*R31 + R13*R21*R32 + R11*R23*R32) + r13*(R12*R21*R31 + R11*R22*R31 - 2*R11*R21*R32)*R33\
			+ R12*(2*R13*R31*(r23*R31 - R21*r33) + R11*(R23*R31*r33 - 2*r23*R31*R33 + R21*r33*R33)) + R11*(R13*(-2*r23*R31*R32 + R22*R31*r33 + R21*R32*r33) - R11*(R23*R32*r33 - 2*r23*R32*R33 + R22*r33*R33)))))\
			/(2.*(1 - 2*nu)*(1 + nu)*pow(e,4));

	f3[0] = (E*V0*((1 - 2*nu)*(R12*R21 - R11*R22)*(g)*(pow(g,2) + pow(d,2) + pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2)\
			- 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e)) + (1 - 2*nu)*((R13*R21 - R11*R23)*(g) + (R12*R21 - R11*R22)*(f))\
			*((g)*(f) + (d)*(-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33 - R12*R21*R33 - R11*r22*R33 + R11*R22*R33)\
			- (e)*(-(R12*r22*R31) + r12*R22*R31 + R13*r23*R31 - r13*R23*R31 + R12*R21*r32 - R11*R22*r32 - r12*R21*R32 + R11*r22*R32 - R13*R21*r33 + R11*R23*r33 + r13*R21*R33 - R11*r23*R33)\
			+ (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + c)*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33))\
			+ (1 - 2*nu)*(R13*R21 - R11*R23)*(f)*(pow(f,2) + pow(i,2) - 2*(e)*(k) + pow(k,2)) + (1 - 2*nu)*(R13*R22 - R12*R23)*(R13*R22*r31 - R12*R23*r31 - R13*r21*R32 + r11*R23*R32 + R12*r21*R33 - r11*R22*R33)\
			*(pow(m,2) + pow(l,2) + 2*(e)*(h) + pow(h,2)) + nu*((R12*R21 - R11*R22)*(g) + (R13*R21 - R11*R23)*(f) + (R13*R22 - R12*R23)*(e) + (R13*R22 - R12*R23)*(h))*(pow(g,2) + pow(d,2)\
			+ pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2) + pow(f,2) - 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e)\
			+ pow(m,2) + pow(i,2) + pow(l,2) - 2*(e)*(k) + pow(k,2) + 2*(e)*(h) + pow(h,2)) + (1 - 2*nu)*(-((e)*(R11*R23*r31 - r11*R23*R31 - R12*R23*r32 + r12*R23*R32 + R13*(-(R21*r31) + r21*R31 + R22*r32 - r22*R32) - R11*r21*R33 + r11*R21*R33 + R12*r22*R33 - r12*R22*R33))\
			+ (-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33 - R12*R21*R33 - R11*r22*R33 + R11*R22*R33)\
			*(-(R13*R22*r32) + R12*R23*r32 + R13*r22*R32 - r12*R23*R32 - R12*r22*R33 + r12*R22*R33) + (i)*(l) + (f)*(R13*(R22*(r31 - R31) + (-r21 + R21)*R32) + R12*(-(R23*r31) + R23*R31 + r21*R33 - R21*R33) + (r11 - R11)*(R23*R32 - R22*R33)))\
			*(pow(R13,2)*(2*R21*R22*r31 - r21*R22*R31 - r21*R21*R32) + R13*(-2*R11*R22*R23*r31 + r11*R22*R23*R31 + R11*r21*R23*R32 + r11*R21*R23*R32 + R11*r21*R22*R33 - 2*r11*R21*R22*R33\
			+ R12*(-2*R21*R23*r31 + r21*R23*R31 + r21*R21*R33)) + R23*(r11*R12*(-(R23*R31) + R21*R33) + R11*(2*R12*R23*r31 - r11*R23*R32 - 2*R12*r21*R33 + r11*R22*R33)))\
			+ (1 - 2*nu)*(pow(R12,2)*(-2*R21*R23*r31 + r21*R23*R31 + r21*R21*R33) + R12*(R23*(2*R11*R22*r31 - r11*R22*R31 - R11*r21*R32 + 2*r11*R21*R32) + R13*(2*R21*R22*r31 - r21*R22*R31 - r21*R21*R32)\
			- (R11*r21 + r11*R21)*R22*R33) + R22*(r11*R13*(R22*R31 - R21*R32) + R11*(-2*R13*R22*r31 + 2*R13*r21*R32 - r11*R23*R32 + r11*R22*R33)))*(-((d)*(m))\
			- (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + c)*(l)\
			- (g)*(R13*(R22*(-r31 + R31) + (r21 - R21)*R32) + R12*(R23*r31 - R23*R31 - r21*R33 + R21*R33) - (r11 - R11)*(R23*R32 - R22*R33)) + (e)\
			*(-(R11*R22*r31) + r11*R22*R31 + R11*r21*R32 - r11*R21*R32 - R13*r23*R32 + r13*R23*R32 + R13*R22*r33 - r13*R22*R33 + R12*(R21*r31 - r21*R31 - R23*r33 + r23*R33)))))\
			/(2.*(1 - 2*nu)*(1 + nu)*pow(e,4));

	f3[1] = (E*V0*((1 - 2*nu)*(R12*R21 - R11*R22)*(-(R12*r22*R31) + r12*R22*R31 + R12*R21*r32 - R11*R22*r32 - r12*R21*R32 + R11*r22*R32)*(pow (g, 2)\
			+ pow (d, 2) + pow (-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33), 2) - 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31\
			- R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e)) - (1 - 2*nu)*(R13*R21 - R11*R23)*(R13*r22*R31 - r12*R23*R31 - R13*R21*r32 + R11*R23*r32 + r12*R21*R33\
			- R11*r22*R33)*(pow (f, 2) + pow (i, 2) -2*(R13*R22*R31 - R12*R23*R31 - R13*R21*R32 + R11*R23*R32\
			+ R12*R21*R33 - R11*R22*R33)*(k) + pow (k, 2)) + (1 - 2*nu)*(R13*R22\
			- R12*R23)*(m)*(pow (m, 2) + pow (R13*r23*R32 - r13*R23*R32 - R13*R22*r33 + R12*R23*r33\
			+ r13*R22*R33 - R12*r23*R33, 2) + 2*(e)*(h) + pow ((R13*R22 - R12*R23)*(r31 - R31)\
			- (r21 - R21)*(R13*R32 - R12*R33) + (r11 - R11)*(R23*R32 - R22*R33), 2)) + nu*((R12*R21 - R11*R22)*(-(R12*r22*R31) + r12*R22*R31 + R12*R21*r32 - R11*R22*r32 - r12*R21*R32 + R11*r22*R32) + (R13*R21 - R11*R23)*(-(R13*R22*R31) + R12*R23*R31 + R13*R21*R32\
			- R11*R23*R32 - R12*R21*R33 + R11*R22*R33) + (R13*R22 - R12*R23)*(m) + (R13*R21 - R11*R23)*((R13*R21 - R11*R23)*(r32 - R32) - (r22 - R22)*(R13*R31 - R11*R33) + (r12 - R12)*(R23*R31\
			- R21*R33)))*(pow (g, 2) + pow (d, 2) + pow (-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31\
			- R21*R32) + (R12*R21 - R11*R22)*(r33 - R33), 2) + pow (f, 2) - 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(R13*R22\
			*R31 - R12*R23*R31 - R13*R21*R32 + R11*R23*R32 + R12*R21*R33 - R11*R22*R33) + pow (m, 2) + pow (R13*r23*R31 - r13*R23*R31 - R13*R21*r33 + R11*R23*r33 + r13*R21*R33\
			- R11*r23*R33, 2) + pow (l, 2) - 2*(e)*((R13*R21 - R11*R23)*(r32 - R32) - (r22 - R22)\
			*(R13*R31 - R11*R33) + (r12 - R12)*(R23*R31 - R21*R33)) + pow (k, 2) + 2*(R13*R22*R31 - R12*R23*R31 - R13*R21*R32 + R11*R23*R32 + R12*R21*R33 - R11*R22\
			*R33)*(h) + pow (h, 2)) - (1 - 2*nu)*(-((R13*R22*R31 - R12*R23*R31\
			- R13*R21*R32 + R11*R23*R32 + R12*R21*R33 - R11*R22*R33)*(R11*R23*r31 - r11*R23*R31 - R12*R23*r32 + r12*R23*R32 + R13*(-(R21*r31) + r21*R31 + R22*r32 - r22*R32) - R11*r21*R33 + r11*R21*R33 + R12*r22*R33 - r12*R22*R33)) + (-(r12*R23*R31) + R12*R23*R31\
			+ R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33 - R12*R21*R33 - R11*r22*R33 + R11*R22*R33)*(-(R13*R22*r32) + R12*R23*r32 + R13*r22*R32 - r12*R23*R32 - R12*r22*R33 + r12*R22*R33) + (R13*r23*R31 - r13*R23*R31 - R13*R21*r33\
			+ R11*R23*r33 + r13*R21*R33 - R11*r23*R33)*(l) + (f)*(R13*(R22*(r31 - R31) + (-r21 + R21)*R32)\
			+ R12*(-(R23*r31) + R23*R31 + r21*R33 - R21*R33) + (r11 - R11)*(R23*R32 - R22*R33)))*(pow (R13, 2)*(r22*R22*R31 - 2*R21*R22*r32 + R21*r22*R32) - R13*(R23*(R12*r22*R31 + r12*R22*R31 - 2*R12*R21*r32 - 2*R11*R22*r32 + r12*R21*R32 + R11*r22*R32) + (R12*R21*r22\
			- 2*r12*R21*R22 + R11*r22*R22)*R33) + R23*(2*R11*R12*(-(R23*r32) + r22*R33) + r12*(R12*R23*R31 + R11*R23*R32 - R12*R21*R33 - R11*R22*R33))) + (1 - 2*nu)*((g)*(R13*R21*r31\
			- R11*R23*r31 - R13*r21*R31 + r11*R23*R31 + R11*r21*R33 - r11*R21*R33) + (d)*(-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31\
			- R21*r32 + R21*R32) + r12*R21*R33 - R12*R21*R33 - R11*r22*R33 + R11*R22*R33) - (e)*(-(R12*r22*R31) + r12*R22*R31 + R13*r23*R31 - r13*R23*R31 + R12*R21*r32 - R11*R22*r32\
			- r12*R21*R32 + R11*r22*R32 - R13*R21*r33 + R11*R23*r33 + r13*R21*R33 - R11*r23*R33) + (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + R11*r23*R32 - R11*R23*R32 + R12*R21*r33 - R11*R22*r33 - R12*R21*R33\
			+ R11*R22*R33)*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33))*(R12*(2*R13*R21*(-(r22*R31) + R21*r32) + r12*R21*(R23*R31 - R21*R33) + R11*(r22*R23*R31 - 2*R21*R23*r32 + R21*r22*R33)) + r12*(R13*R21*(R22*R31 - R21*R32)\
			+ R11*(-2*R22*R23*R31 + R21*R23*R32 + R21*R22*R33)) + R11*(R13*(r22*R22*R31 - 2*R21*R22*r32 + R21*r22*R32) + R11*(2*R22*R23*r32 - r22*R23*R32 - r22*R22*R33))) + (1 - 2*nu)*((R13*R22 - R12*R23)*(-(R12*r22*R31) + r12*R22*R31 + R12*R21*r32 - R11*R22*r32 - r12*R21*R32\
			+ R11*r22*R32) - (R12*R21 - R11*R22)*(-(R13*R22*r32) + R12*R23*r32 + R13*r22*R32 - r12*R23*R32 - R12*r22*R33 + r12*R22* R33))*(-((d)*(R13*R22*r32 - R12*R23*r32 - R13*r22*R32 + r12*R23*R32\
			+ R12*r22*R33 - r12*R22*R33)) - (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + c)*(R13*r23*R32 - r13*R23*R32 - R13*R22*r33 + R12*R23*r33\
			+ r13*R22*R33 - R12*r23*R33) - (g)*(R13*(R22*(-r31 + R31) + (r21 - R21)*R32) + R12*(R23*r31 - R23*R31 - r21*R33 + R21*R33) - (r11 - R11)*(R23*R32 - R22*R33)) + (R13*R22*R31 - R12*R23*R31\
			- R13*R21*R32 + R11*R23*R32 + R12*R21*R33 - R11*R22*R33)*(-(R11*R22*r31) + r11*R22*R31 + R11*r21*R32 - r11*R21*R32 - R13*r23*R32 + r13*R23*R32 + R13*R22*r33 - r13*R22*R33 + R12*(R21*r31 - r21*R31 - R23*r33 + r23*R33)))))/(2.*(1 - 2*nu)*(1 + nu)\
			*pow(e, 4));


	f3[2] = (E*V0*((1 - 2*nu)*(R12*R21 - R11*R22)*(r13*R22*R31 - R12*r23*R31 - r13*R21*R32 + R11*r23*R32 + R12*R21*r33 - R11*R22*r33)*(pow(g,2) + pow(d,2)\
			+ pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2)\
			- 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e))\
			+ (1 - 2*nu)*(R13*R21 - R11*R23)*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33)\
			*(pow(f,2) + pow(i,2) - 2*(e)*(k) + pow(k,2)) + (1 - 2*nu)*(R13*R22 - R12*R23)*(-(R13*r23*R32) + r13*R23*R32 + R13*R22*r33 - R12*R23*r33 - r13*R22*R33 + R12*r23*R33)\
			*(pow(m,2) + pow(l,2) + 2*(e)*(h) + pow(h,2))  - nu*((R12*R21 - R11*R22)*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))\
			- (R12*R21 - R11*R22)*(e) - (R13*R21 - R11*R23)*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33)\
			- (R13*R22 - R12*R23)*(-(R13*r23*R32) + r13*R23*R32 + R13*R22*r33 - R12*R23*r33 - r13*R22*R33 + R12*r23*R33))*(pow(g,2) + pow(d,2)\
			+ pow(-((r23 - R23)*(R12*R31 - R11*R32)) + (r13 - R13)*(R22*R31 - R21*R32) + (R12*R21 - R11*R22)*(r33 - R33),2) + pow(f,2)\
			- 2*((r23 - R23)*(R12*R31 - R11*R32) - (r13 - R13)*(R22*R31 - R21*R32) - (R12*R21 - R11*R22)*(r33 - R33))*(e) + pow(m,2) + pow(i,2)\
			+ pow(l,2) - 2*(e)*(k) + pow(k,2) + 2*(e)*(h) + pow(h,2)) + (1 - 2*nu)*((R13*R22 - R12*R23)*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33)\
			+ (R13*R21 - R11*R23)*(-(R13*r23*R32) + r13*R23*R32 + R13*R22*r33 - R12*R23*r33 - r13*R22*R33 + R12*r23*R33))*(-((e)\
			*(R11*R23*r31 - r11*R23*R31 - R12*R23*r32 + r12*R23*R32 + R13*(-(R21*r31) + r21*R31 + R22*r32 - r22*R32) - R11*r21*R33 + r11*R21*R33 + R12*r22*R33 - r12*R22*R33))\
			+ (-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33 - R12*R21*R33 - R11*r22*R33 + R11*R22*R33)\
			*(-(R13*R22*r32) + R12*R23*r32 + R13*r22*R32 - r12*R23*R32 - R12*r22*R33 + r12*R22*R33) + (i)*(l)\
			+ (f)*(R13*(R22*(r31 - R31) + (-r21 + R21)*R32) + R12*(-(R23*r31) + R23*R31 + r21*R33 - R21*R33) + (r11 - R11)*(R23*R32 - R22*R33)))\
			- (1 - 2*nu)*(2*R11*R13*R22*(-(r23*R32) + R22*r33) + r13*(R13*R22*(-(R22*R31) + R21*R32) + R23*(R12*R22*R31 - 2*R12*R21*R32 + R11*R22*R32) + R22*(R12*R21 - R11*R22)*R33)\
			- pow(R12,2)*(r23*R23*R31 - 2*R21*R23*r33 + R21*r23*R33) + R12*(R13*R22*r23*R31 + R13*R21*r23*R32 + R11*r23*R23*R32 - 2*R13*R21*R22*r33 - 2*R11*R22*R23*r33 + R11*R22*r23*R33))\
			*(-((d)*(m)) - (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + c)*(l)\
			- (g)*(R13*(R22*(-r31 + R31) + (r21 - R21)*R32) + R12*(R23*r31 - R23*R31 - r21*R33 + R21*R33) - (r11 - R11)*(R23*R32 - R22*R33)) + (e)\
			*(-(R11*R22*r31) + r11*R22*R31 + R11*r21*R32 - r11*R21*R32 - R13*r23*R32 + r13*R23*R32 + R13*R22*r33 - r13*R22*R33 + R12*(R21*r31 - r21*R31 - R23*r33 + r23*R33)))\
			- (1 - 2*nu)*((g)*(f) +(d)*(-(r12*R23*R31) + R12*R23*R31 + R11*R23*r32 - R11*R23*R32 + R13*(r22*R31 - R22*R31 - R21*r32 + R21*R32) + r12*R21*R33 - R12*R21*R33 - R11*r22*R33 + R11*R22*R33)\
			- (e)*(-(R12*r22*R31) + r12*R22*R31 + R13*r23*R31 - r13*R23*R31 + R12*R21*r32 - R11*R22*r32 - r12*R21*R32 + R11*r22*R32 - R13*R21*r33 + R11*R23*r33 + r13*R21*R33 - R11*r23*R33)\
			+ (r13*R22*R31 - R13*R22*R31 - R12*r23*R31 + R12*R23*R31 - r13*R21*R32 + R13*R21*R32 + c)*(-(R13*r23*R31) + r13*R23*R31 + R13*R21*r33 - R11*R23*r33 - r13*R21*R33 + R11*r23*R33))\
			*(r13*(R13*R21*(-(R22*R31) + R21*R32) - R23*(R12*R21*R31 - 2*R11*R22*R31 + R11*R21*R32) + R21*(R12*R21 - R11*R22)*R33) - R12*(2*R13*R21*(-(r23*R31) + R21*r33) + R11*(r23*R23*R31 - 2*R21*R23*r33 + R21*r23*R33))\
			+ R11*(-(R13*(R22*r23*R31 + R21*r23*R32 - 2*R21*R22*r33)) + R11*(r23*R23*R32 - 2*R22*R23*r33 + R22*r23*R33)))))\
			/(2.*(1 - 2*nu)*(1 + nu)*pow(e,4));
	/* <--- Saint Venant-Kirchhoff */
	
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

